"""Typed and validated tensor contract for Oreo-Qwen batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .tokens import OreoSpecialTokenIds


class ContractError(ValueError):
    """Raised when a batch violates the documented Oreo-Qwen contract."""


@dataclass(frozen=True, slots=True)
class KGContractSpec:
    """Static dimensions needed to validate a batch-local graph."""

    num_entities: int
    num_relations: int
    num_kil_steps: int
    probability_atol: float = 1e-5

    def __post_init__(self) -> None:
        for name in ("num_entities", "num_relations", "num_kil_steps"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if self.probability_atol <= 0:
            raise ValueError("probability_atol must be positive.")


@dataclass(kw_only=True, slots=True)
class KGInputBatch:
    """The flattened, disjoint-union graph portion of a model batch."""

    kg_mention_batch: Tensor
    kg_src_positions: Tensor
    kg_rel_positions: Tensor
    kg_tgt_positions: Tensor

    kg_node_ids: Tensor
    kg_node_batch: Tensor
    kg_edge_src: Tensor
    kg_edge_dst: Tensor
    kg_edge_type: Tensor

    kg_initial_node_indices: Tensor | None = None
    kg_initial_entity_probs: Tensor | None = None

    kg_entity_link_targets: Tensor | None = None
    kg_entity_link_mask: Tensor | None = None
    kg_relation_targets: Tensor | None = None
    kg_relation_target_mask: Tensor | None = None

    def as_model_kwargs(self) -> dict[str, Tensor]:
        """Flatten the typed object into the public `model(**batch)` API."""

        result: dict[str, Tensor] = {}
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        return result

    def validate(
        self,
        *,
        spec: KGContractSpec,
        batch_size: int,
        sequence_length: int,
        device: torch.device,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        special_token_ids: OreoSpecialTokenIds | None = None,
    ) -> None:
        """Validate shapes, ownership, marker positions, and supervision."""

        named_tensors = self.as_model_kwargs()
        for name, tensor in named_tensors.items():
            _require_tensor(name, tensor)
            if tensor.device != device:
                raise ContractError(
                    f"{name} is on {tensor.device}, but language tensors are on {device}."
                )

        integer_fields = (
            "kg_mention_batch",
            "kg_src_positions",
            "kg_rel_positions",
            "kg_tgt_positions",
            "kg_node_ids",
            "kg_node_batch",
            "kg_edge_src",
            "kg_edge_dst",
            "kg_edge_type",
        )
        for name in integer_fields:
            _require_long(name, getattr(self, name))

        mention_count = self.kg_mention_batch.numel()
        node_count = self.kg_node_ids.numel()
        edge_count = self.kg_edge_src.numel()

        for name in (
            "kg_mention_batch",
            "kg_src_positions",
            "kg_rel_positions",
            "kg_tgt_positions",
        ):
            _require_shape(name, getattr(self, name), (mention_count,))
        for name in ("kg_node_ids", "kg_node_batch"):
            _require_shape(name, getattr(self, name), (node_count,))
        for name in ("kg_edge_src", "kg_edge_dst", "kg_edge_type"):
            _require_shape(name, getattr(self, name), (edge_count,))

        if mention_count == 0:
            raise ContractError("A KG batch must contain at least one mention.")
        if node_count == 0:
            raise ContractError("A KG batch must contain at least one node.")

        _require_range("kg_mention_batch", self.kg_mention_batch, 0, batch_size)
        _require_range("kg_node_batch", self.kg_node_batch, 0, batch_size)
        _require_range("kg_node_ids", self.kg_node_ids, 0, spec.num_entities)
        _require_range("kg_src_positions", self.kg_src_positions, 0, sequence_length)
        _require_range("kg_rel_positions", self.kg_rel_positions, 0, sequence_length)
        _require_range("kg_tgt_positions", self.kg_tgt_positions, 0, sequence_length)
        _require_range("kg_edge_src", self.kg_edge_src, 0, node_count)
        _require_range("kg_edge_dst", self.kg_edge_dst, 0, node_count)
        _require_range("kg_edge_type", self.kg_edge_type, 0, spec.num_relations)

        node_counts = torch.bincount(self.kg_node_batch, minlength=batch_size)
        if torch.any(node_counts.index_select(0, self.kg_mention_batch) == 0):
            raise ContractError("Every mention must have nodes in its batch component.")

        if edge_count:
            source_batches = self.kg_node_batch.index_select(0, self.kg_edge_src)
            destination_batches = self.kg_node_batch.index_select(0, self.kg_edge_dst)
            if not torch.equal(source_batches, destination_batches):
                raise ContractError("KG edges may not cross batch-example components.")

        target_keys = self.kg_mention_batch * sequence_length + self.kg_tgt_positions
        if torch.unique(target_keys).numel() != mention_count:
            raise ContractError("Fusion target (batch, position) pairs must be unique.")

        if attention_mask is not None:
            mention_attention = attention_mask[self.kg_mention_batch]
            for name, positions in (
                ("kg_src_positions", self.kg_src_positions),
                ("kg_rel_positions", self.kg_rel_positions),
                ("kg_tgt_positions", self.kg_tgt_positions),
            ):
                active = mention_attention.gather(1, positions.unsqueeze(1)).squeeze(1)
                if not torch.all(active.bool()):
                    raise ContractError(f"{name} contains a padding position.")

        if special_token_ids is not None:
            if input_ids is None:
                raise ContractError("Marker-token validation requires input_ids.")
            for name, positions, expected_id in (
                ("kg_src_positions", self.kg_src_positions, special_token_ids.source_entity),
                ("kg_rel_positions", self.kg_rel_positions, special_token_ids.relation),
                ("kg_tgt_positions", self.kg_tgt_positions, special_token_ids.target_entity),
            ):
                actual = input_ids[self.kg_mention_batch, positions]
                if not torch.all(actual == expected_id):
                    raise ContractError(
                        f"{name} does not point exclusively to token ID {expected_id}."
                    )

        self._validate_initial_state(spec, mention_count, node_count)
        self._validate_entity_supervision(mention_count, node_count)
        self._validate_relation_supervision(spec, mention_count)

    def _validate_initial_state(
        self,
        spec: KGContractSpec,
        mention_count: int,
        node_count: int,
    ) -> None:
        hard = self.kg_initial_node_indices
        soft = self.kg_initial_entity_probs
        if hard is not None and soft is not None:
            raise ContractError(
                "Supply either kg_initial_node_indices or kg_initial_entity_probs, not both."
            )

        if hard is not None:
            _require_long("kg_initial_node_indices", hard)
            _require_shape("kg_initial_node_indices", hard, (mention_count,))
            _require_range("kg_initial_node_indices", hard, 0, node_count)
            self._require_nodes_match_mentions("kg_initial_node_indices", hard)

        if soft is not None:
            _require_floating("kg_initial_entity_probs", soft)
            _require_shape("kg_initial_entity_probs", soft, (mention_count, node_count))
            _require_finite_nonnegative("kg_initial_entity_probs", soft)

            component_mask = self.kg_mention_batch[:, None] == self.kg_node_batch[None, :]
            outside_mass = soft.masked_select(~component_mask)
            if outside_mass.numel() and torch.any(outside_mass.abs() > spec.probability_atol):
                raise ContractError(
                    "kg_initial_entity_probs assigns mass outside a mention's component."
                )
            _require_probability_rows(
                "kg_initial_entity_probs",
                soft,
                atol=spec.probability_atol,
            )

    def _validate_entity_supervision(self, mention_count: int, node_count: int) -> None:
        targets = self.kg_entity_link_targets
        mask = self.kg_entity_link_mask
        if (targets is None) != (mask is None):
            raise ContractError(
                "kg_entity_link_targets and kg_entity_link_mask must be supplied together."
            )
        if targets is None:
            return

        _require_long("kg_entity_link_targets", targets)
        _require_bool("kg_entity_link_mask", mask)
        _require_shape("kg_entity_link_targets", targets, (mention_count,))
        _require_shape("kg_entity_link_mask", mask, (mention_count,))
        _require_range("kg_entity_link_targets", targets, 0, node_count)
        self._require_nodes_match_mentions("kg_entity_link_targets", targets)

    def _validate_relation_supervision(
        self,
        spec: KGContractSpec,
        mention_count: int,
    ) -> None:
        targets = self.kg_relation_targets
        mask = self.kg_relation_target_mask
        if (targets is None) != (mask is None):
            raise ContractError(
                "kg_relation_targets and kg_relation_target_mask must be supplied together."
            )
        if targets is None:
            return

        _require_floating("kg_relation_targets", targets)
        _require_bool("kg_relation_target_mask", mask)
        _require_shape(
            "kg_relation_targets",
            targets,
            (spec.num_kil_steps, mention_count, spec.num_relations),
        )
        _require_shape(
            "kg_relation_target_mask",
            mask,
            (spec.num_kil_steps, mention_count),
        )
        _require_finite_nonnegative("kg_relation_targets", targets)
        selected_rows = targets[mask]
        if selected_rows.numel():
            _require_probability_rows(
                "masked kg_relation_targets",
                selected_rows,
                atol=spec.probability_atol,
            )

    def _require_nodes_match_mentions(self, name: str, node_indices: Tensor) -> None:
        node_owners = self.kg_node_batch.index_select(0, node_indices)
        if not torch.equal(node_owners, self.kg_mention_batch):
            raise ContractError(f"{name} contains a node owned by another batch example.")


@dataclass(kw_only=True, slots=True)
class OreoQwenBatch:
    """Language tensors plus an optional typed KG batch."""

    input_ids: Tensor | None = None
    inputs_embeds: Tensor | None = None
    attention_mask: Tensor | None = None
    labels: Tensor | None = None
    kg: KGInputBatch | None = None

    def as_model_kwargs(self) -> dict[str, Any]:
        """Return the flat keyword interface expected by the future model."""

        result: dict[str, Any] = {}
        for name in ("input_ids", "inputs_embeds", "attention_mask", "labels"):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        if self.kg is not None:
            result.update(self.kg.as_model_kwargs())
        return result

    def validate(
        self,
        *,
        spec: KGContractSpec | None = None,
        special_token_ids: OreoSpecialTokenIds | None = None,
        expected_delimiter_token_id: int | None = None,
    ) -> None:
        """Validate language tensors and, when present, the local KG."""

        if (self.input_ids is None) == (self.inputs_embeds is None):
            raise ContractError("Supply exactly one of input_ids and inputs_embeds.")

        if self.input_ids is not None:
            _require_tensor("input_ids", self.input_ids)
            _require_long("input_ids", self.input_ids)
            if self.input_ids.ndim != 2:
                raise ContractError(
                    f"input_ids must have rank 2, got shape {tuple(self.input_ids.shape)}."
                )
            batch_size, sequence_length = self.input_ids.shape
            device = self.input_ids.device
        else:
            _require_tensor("inputs_embeds", self.inputs_embeds)
            if not self.inputs_embeds.is_floating_point() or self.inputs_embeds.ndim != 3:
                raise ContractError(
                    "inputs_embeds must be a floating tensor with shape [B, L, H]."
                )
            batch_size, sequence_length = self.inputs_embeds.shape[:2]
            device = self.inputs_embeds.device

        for name, tensor in (
            ("attention_mask", self.attention_mask),
            ("labels", self.labels),
        ):
            if tensor is not None:
                _require_tensor(name, tensor)
                if tensor.device != device:
                    raise ContractError(
                        f"{name} is on {tensor.device}, but language input is on {device}."
                    )
                _require_shape(name, tensor, (batch_size, sequence_length))

        if self.attention_mask is not None:
            if self.attention_mask.dtype != torch.bool and not _is_integer(self.attention_mask):
                raise ContractError("attention_mask must have boolean or integer dtype.")
            if torch.any((self.attention_mask != 0) & (self.attention_mask != 1)):
                raise ContractError("attention_mask values must be 0 or 1.")

        if self.labels is not None:
            _require_long("labels", self.labels)
            if self.attention_mask is not None:
                if torch.any(self.labels[self.attention_mask == 0] != -100):
                    raise ContractError("Padding positions in labels must be -100.")

        if expected_delimiter_token_id is not None:
            self._validate_suffix_labels(
                expected_delimiter_token_id,
                special_token_ids=special_token_ids,
            )

        if self.kg is not None:
            if spec is None:
                raise ContractError("A KGContractSpec is required when KG inputs are present.")
            self.kg.validate(
                spec=spec,
                batch_size=batch_size,
                sequence_length=sequence_length,
                device=device,
                input_ids=self.input_ids,
                attention_mask=self.attention_mask,
                special_token_ids=special_token_ids,
            )

    def _validate_suffix_labels(
        self,
        delimiter_token_id: int,
        *,
        special_token_ids: OreoSpecialTokenIds | None,
    ) -> None:
        if self.input_ids is None or self.labels is None:
            raise ContractError("Suffix-label validation requires input_ids and labels.")

        active = (
            self.attention_mask.bool()
            if self.attention_mask is not None
            else torch.ones_like(self.input_ids, dtype=torch.bool)
        )
        positions = torch.arange(self.input_ids.shape[1], device=self.input_ids.device)

        for row in range(self.input_ids.shape[0]):
            matches = (self.input_ids[row] == delimiter_token_id) & active[row]
            if matches.sum().item() != 1:
                raise ContractError(
                    "Each supervised example must contain exactly one active delimiter token."
                )
            delimiter_position = torch.nonzero(matches, as_tuple=False).item()
            prompt = active[row] & (positions <= delimiter_position)
            suffix = active[row] & (positions > delimiter_position)
            if not torch.any(suffix):
                raise ContractError("A supervised example must contain a nonempty suffix.")
            if torch.any(self.labels[row, prompt] != -100):
                raise ContractError("Prompt and delimiter labels must all be -100.")
            if torch.any(self.labels[row, suffix] == -100):
                raise ContractError("Every active suffix token must contribute to loss.")
            if not torch.equal(self.labels[row, suffix], self.input_ids[row, suffix]):
                raise ContractError("Supervised suffix labels must equal their input token IDs.")
            if (
                special_token_ids is not None
                and delimiter_token_id == special_token_ids.reconstruct
            ):
                first_suffix_position = torch.nonzero(suffix, as_tuple=False)[0].item()
                if self.input_ids[row, first_suffix_position].item() != special_token_ids.masks[0]:
                    raise ContractError(
                        "A reconstruction suffix must begin with <|kg_mask_0|>."
                    )


def _require_tensor(name: str, value: Any) -> None:
    if not isinstance(value, Tensor):
        raise ContractError(f"{name} must be a torch.Tensor.")


def _require_shape(name: str, tensor: Tensor, expected: tuple[int, ...]) -> None:
    if tuple(tensor.shape) != expected:
        raise ContractError(
            f"{name} must have shape {expected}, got {tuple(tensor.shape)}."
        )


def _require_long(name: str, tensor: Tensor) -> None:
    if tensor.dtype != torch.long:
        raise ContractError(f"{name} must have dtype torch.long, got {tensor.dtype}.")


def _require_bool(name: str, tensor: Tensor) -> None:
    if tensor.dtype != torch.bool:
        raise ContractError(f"{name} must have dtype torch.bool, got {tensor.dtype}.")


def _require_floating(name: str, tensor: Tensor) -> None:
    if not tensor.is_floating_point():
        raise ContractError(f"{name} must have floating-point dtype, got {tensor.dtype}.")


def _require_range(name: str, tensor: Tensor, lower: int, upper: int) -> None:
    if tensor.numel() == 0:
        return
    if torch.any(tensor < lower) or torch.any(tensor >= upper):
        raise ContractError(f"{name} values must lie in [{lower}, {upper}).")


def _require_finite_nonnegative(name: str, tensor: Tensor) -> None:
    if not torch.all(torch.isfinite(tensor)):
        raise ContractError(f"{name} must contain only finite values.")
    if torch.any(tensor < 0):
        raise ContractError(f"{name} must be nonnegative.")


def _require_probability_rows(name: str, tensor: Tensor, *, atol: float) -> None:
    expected = torch.ones(
        tensor.shape[:-1],
        dtype=tensor.dtype,
        device=tensor.device,
    )
    if not torch.allclose(tensor.sum(dim=-1), expected, atol=atol, rtol=0):
        raise ContractError(f"Every row of {name} must sum to one.")


def _is_integer(tensor: Tensor) -> bool:
    return tensor.dtype in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
