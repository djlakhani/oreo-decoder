"""OreoLM Knowledge Interaction Layer reasoning components."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from .contracts import KGContractSpec, KGInputBatch


@dataclass(frozen=True, slots=True)
class KGReasonerConfig:
    """Architecture dimensions for the standalone KG reasoner."""

    hidden_size: int
    num_entities: int
    num_relations: int
    num_kil_steps: int
    entity_embedding_dim: int = 128
    relation_embedding_dim: int | None = None
    rms_norm_eps: float = 1e-6
    initializer_range: float = 0.02

    def __post_init__(self) -> None:
        positive_integer_fields = (
            "hidden_size",
            "num_entities",
            "num_relations",
            "num_kil_steps",
            "entity_embedding_dim",
        )
        for name in positive_integer_fields:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if self.relation_embedding_dim is not None:
            if (
                not isinstance(self.relation_embedding_dim, int)
                or isinstance(self.relation_embedding_dim, bool)
                or self.relation_embedding_dim <= 0
            ):
                raise ValueError("relation_embedding_dim must be a positive integer.")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps must be positive.")
        if self.initializer_range <= 0:
            raise ValueError("initializer_range must be positive.")

    @property
    def resolved_relation_embedding_dim(self) -> int:
        return self.relation_embedding_dim or self.hidden_size

    @property
    def contract_spec(self) -> KGContractSpec:
        return KGContractSpec(
            num_entities=self.num_entities,
            num_relations=self.num_relations,
            num_kil_steps=self.num_kil_steps,
        )


@dataclass(frozen=True, slots=True)
class KGReasonerState:
    """Explicit per-forward state threaded between KIL calls."""

    entity_probs: Tensor
    local_entity_embeddings: Tensor


@dataclass(frozen=True, slots=True)
class KGInitializationOutput:
    state: KGReasonerState
    entity_logits: Tensor | None
    entity_link_loss: Tensor


@dataclass(frozen=True, slots=True)
class KGStepOutput:
    hidden_states: Tensor
    state: KGReasonerState
    relation_logits: Tensor
    relation_probs: Tensor
    relation_loss: Tensor
    retrieved_entity_embeddings: Tensor
    projected_entity_values: Tensor


class ProjectionMLP(nn.Module):
    """Two-layer nonlinear projection used by OreoLM's E/Q/V projections."""

    def __init__(self, input_dim: int, intermediate_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, intermediate_dim)
        self.output = nn.Linear(intermediate_dim, output_dim)

    def forward(self, values: Tensor) -> Tensor:
        return self.output(F.silu(self.input(values)))


class EntityMemory(nn.Module):
    """Global trainable value memory indexed by dense entity IDs."""

    def __init__(self, num_entities: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_entities, embedding_dim)

    @property
    def weight(self) -> Tensor:
        return self.embedding.weight

    def forward(self, entity_ids: Tensor) -> Tensor:
        return self.embedding(entity_ids)


class RelationMemory(nn.Module):
    """Global trainable relation-key memory shared by every KIL step."""

    def __init__(self, num_relations: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_relations, embedding_dim)

    @property
    def weight(self) -> Tensor:
        return self.embedding.weight


class EntityLinker(nn.Module):
    """Predict an initial local entity distribution from source-marker states."""

    def __init__(
        self,
        hidden_size: int,
        entity_embedding_dim: int,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.query_projection = ProjectionMLP(
            hidden_size,
            hidden_size,
            entity_embedding_dim,
        )
        self.query_norm = Qwen3RMSNorm(entity_embedding_dim, eps=rms_norm_eps)

    def forward(
        self,
        source_hidden_states: Tensor,
        local_entity_embeddings: Tensor,
        mention_batch: Tensor,
        node_batch: Tensor,
    ) -> tuple[Tensor, Tensor]:
        query = self.query_norm(self.query_projection(source_hidden_states))
        logits = query @ local_entity_embeddings.transpose(0, 1)
        component_mask = mention_batch[:, None] == node_batch[None, :]
        logits = logits.masked_fill(~component_mask, -torch.inf)
        return logits, F.softmax(logits, dim=-1)


class RelationRouter(nn.Module):
    """Predict the relation action for one KIL step."""

    def __init__(
        self,
        hidden_size: int,
        relation_embedding_dim: int,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.query_projection = ProjectionMLP(
            hidden_size,
            hidden_size,
            relation_embedding_dim,
        )
        self.query_norm = Qwen3RMSNorm(relation_embedding_dim, eps=rms_norm_eps)

    def forward(
        self,
        relation_hidden_states: Tensor,
        relation_keys: Tensor,
    ) -> tuple[Tensor, Tensor]:
        query = self.query_norm(self.query_projection(relation_hidden_states))
        logits = query @ relation_keys.transpose(0, 1)
        return logits, F.softmax(logits, dim=-1)


class GlobalInverseDegreeWalker(nn.Module):
    """Fixed inverse-out-degree walk with global per-mention normalization."""

    def forward(
        self,
        previous_states: Tensor,
        relation_probs: Tensor,
        edge_src: Tensor,
        edge_dst: Tensor,
        edge_type: Tensor,
    ) -> Tensor:
        mention_count, node_count = previous_states.shape
        if edge_src.numel() == 0:
            return previous_states

        out_degree = torch.zeros(
            node_count,
            dtype=previous_states.dtype,
            device=previous_states.device,
        )
        out_degree = out_degree.index_add(
            0,
            edge_src,
            torch.ones_like(edge_src, dtype=previous_states.dtype),
        )
        inverse_degree = out_degree.index_select(0, edge_src).reciprocal()

        source_mass = previous_states.index_select(1, edge_src)
        relation_mass = relation_probs.index_select(1, edge_type)
        edge_scores = source_mass * relation_mass * inverse_degree.unsqueeze(0)

        normalizer = edge_scores.sum(dim=1, keepdim=True)
        has_outgoing_mass = normalizer > 0
        normalized_scores = torch.where(
            has_outgoing_mass,
            edge_scores / normalizer.clamp_min(torch.finfo(edge_scores.dtype).tiny),
            torch.zeros_like(edge_scores),
        )

        next_states = torch.zeros(
            (mention_count, node_count),
            dtype=previous_states.dtype,
            device=previous_states.device,
        ).index_add(1, edge_dst, normalized_scores)

        return torch.where(has_outgoing_mass, next_states, previous_states)


class ValueProjection(nn.Module):
    """Map an aggregated raw entity embedding into Qwen hidden space."""

    def __init__(self, entity_embedding_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.projection = ProjectionMLP(
            entity_embedding_dim,
            hidden_size,
            hidden_size,
        )

    def forward(self, entity_embeddings: Tensor) -> Tensor:
        return self.projection(entity_embeddings)


class KnowledgeFusion(nn.Module):
    """Add projected knowledge at target markers, followed by Qwen RMSNorm."""

    def __init__(self, hidden_size: int, rms_norm_eps: float) -> None:
        super().__init__()
        self.norm = Qwen3RMSNorm(hidden_size, eps=rms_norm_eps)

    def forward(
        self,
        hidden_states: Tensor,
        projected_values: Tensor,
        mention_batch: Tensor,
        target_positions: Tensor,
    ) -> Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        target_states = hidden_states[mention_batch, target_positions]
        fused_states = self.norm(target_states + projected_values)

        flat_hidden = hidden_states.reshape(batch_size * sequence_length, hidden_size)
        flat_targets = mention_batch * sequence_length + target_positions
        return flat_hidden.index_copy(0, flat_targets, fused_states).reshape_as(hidden_states)


class KnowledgeInteractionStep(nn.Module):
    """Step-specific Q-Proj, V-Proj, and residual-fusion parameters."""

    def __init__(self, config: KGReasonerConfig) -> None:
        super().__init__()
        self.relation_router = RelationRouter(
            config.hidden_size,
            config.resolved_relation_embedding_dim,
            config.rms_norm_eps,
        )
        self.value_projection = ValueProjection(
            config.entity_embedding_dim,
            config.hidden_size,
        )
        self.fusion = KnowledgeFusion(config.hidden_size, config.rms_norm_eps)


class KGReasoner(nn.Module):
    """Stateful-in-data, stateless-in-module OreoLM KG reasoner.

    `initialize` is called once before the first KIL. `forward` performs exactly
    one reasoning/fusion step so Qwen layers can be run between calls.
    """

    def __init__(self, config: KGReasonerConfig) -> None:
        super().__init__()
        self.config = config
        self.entity_memory = EntityMemory(
            config.num_entities,
            config.entity_embedding_dim,
        )
        self.relation_memory = RelationMemory(
            config.num_relations,
            config.resolved_relation_embedding_dim,
        )
        self.entity_linker = EntityLinker(
            config.hidden_size,
            config.entity_embedding_dim,
            config.rms_norm_eps,
        )
        self.walker = GlobalInverseDegreeWalker()
        self.steps = nn.ModuleList(
            KnowledgeInteractionStep(config) for _ in range(config.num_kil_steps)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use Qwen-style normal initialization for learned matrices."""

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def initialize(
        self,
        hidden_states: Tensor,
        kg: KGInputBatch,
    ) -> KGInitializationOutput:
        """Retrieve local entity values and construct the initial distribution."""

        local_entity_embeddings = self.entity_memory(kg.kg_node_ids)
        source_hidden_states = hidden_states[
            kg.kg_mention_batch,
            kg.kg_src_positions,
        ]

        requires_linker = (
            kg.kg_initial_node_indices is None
            and kg.kg_initial_entity_probs is None
        ) or kg.kg_entity_link_targets is not None

        entity_logits: Tensor | None = None
        predicted_probs: Tensor | None = None
        if requires_linker:
            entity_logits, predicted_probs = self.entity_linker(
                source_hidden_states,
                local_entity_embeddings,
                kg.kg_mention_batch,
                kg.kg_node_batch,
            )

        if kg.kg_initial_node_indices is not None:
            entity_probs = F.one_hot(
                kg.kg_initial_node_indices,
                num_classes=kg.kg_node_ids.numel(),
            ).to(dtype=local_entity_embeddings.dtype)
        elif kg.kg_initial_entity_probs is not None:
            entity_probs = kg.kg_initial_entity_probs.to(
                dtype=local_entity_embeddings.dtype,
            )
            component_mask = (
                kg.kg_mention_batch[:, None] == kg.kg_node_batch[None, :]
            )
            entity_probs = entity_probs.masked_fill(~component_mask, 0.0)
            entity_probs = entity_probs / entity_probs.sum(
                dim=-1,
                keepdim=True,
            ).clamp_min(torch.finfo(entity_probs.dtype).tiny)
        else:
            if predicted_probs is None:
                raise RuntimeError("Entity predictor was not run for learned initialization.")
            entity_probs = predicted_probs

        entity_link_loss = _masked_entity_link_loss(
            entity_logits,
            kg.kg_entity_link_targets,
            kg.kg_entity_link_mask,
            zero_reference=source_hidden_states,
        )
        return KGInitializationOutput(
            state=KGReasonerState(
                entity_probs=entity_probs,
                local_entity_embeddings=local_entity_embeddings,
            ),
            entity_logits=entity_logits,
            entity_link_loss=entity_link_loss,
        )

    def forward(
        self,
        hidden_states: Tensor,
        kg: KGInputBatch,
        state: KGReasonerState,
        step_index: int,
    ) -> KGStepOutput:
        """Perform one relation prediction, graph walk, and target fusion."""

        if not 0 <= step_index < len(self.steps):
            raise IndexError(
                f"step_index must lie in [0, {len(self.steps)}), got {step_index}."
            )

        step = self.steps[step_index]
        relation_hidden_states = hidden_states[
            kg.kg_mention_batch,
            kg.kg_rel_positions,
        ]
        relation_logits, relation_probs = step.relation_router(
            relation_hidden_states,
            self.relation_memory.weight,
        )

        next_entity_probs = self.walker(
            state.entity_probs,
            relation_probs,
            kg.kg_edge_src,
            kg.kg_edge_dst,
            kg.kg_edge_type,
        )
        retrieved_entity_embeddings = (
            next_entity_probs @ state.local_entity_embeddings
        )
        projected_entity_values = step.value_projection(
            retrieved_entity_embeddings,
        )
        fused_hidden_states = step.fusion(
            hidden_states,
            projected_entity_values,
            kg.kg_mention_batch,
            kg.kg_tgt_positions,
        )

        relation_loss = _masked_relation_loss(
            relation_logits,
            kg.kg_relation_targets,
            kg.kg_relation_target_mask,
            step_index=step_index,
            zero_reference=relation_hidden_states,
        )
        return KGStepOutput(
            hidden_states=fused_hidden_states,
            state=KGReasonerState(
                entity_probs=next_entity_probs,
                local_entity_embeddings=state.local_entity_embeddings,
            ),
            relation_logits=relation_logits,
            relation_probs=relation_probs,
            relation_loss=relation_loss,
            retrieved_entity_embeddings=retrieved_entity_embeddings,
            projected_entity_values=projected_entity_values,
        )


def _masked_entity_link_loss(
    logits: Tensor | None,
    targets: Tensor | None,
    mask: Tensor | None,
    *,
    zero_reference: Tensor,
) -> Tensor:
    if logits is None or targets is None or mask is None:
        return zero_reference.sum() * 0.0
    losses = F.cross_entropy(logits, targets, reduction="none")
    loss_mask = mask.to(dtype=losses.dtype)
    return (losses * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)


def _masked_relation_loss(
    logits: Tensor,
    targets: Tensor | None,
    mask: Tensor | None,
    *,
    step_index: int,
    zero_reference: Tensor,
) -> Tensor:
    if targets is None or mask is None:
        return zero_reference.sum() * 0.0
    step_mask = mask[step_index]
    target_distribution = targets[step_index]
    losses = -(target_distribution * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    loss_mask = step_mask.to(dtype=losses.dtype)
    return (losses * loss_mask).sum() / loss_mask.sum().clamp_min(1.0)
