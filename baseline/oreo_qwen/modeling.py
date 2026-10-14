"""Qwen3 backbone and causal-LM wrapper with OreoLM KIL insertions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from transformers import Qwen3Config
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import (
    create_causal_mask,
    create_sliding_window_causal_mask,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Model,
    Qwen3PreTrainedModel,
)
from transformers.utils import can_return_tuple
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from .configuration import OreoQwen3Config
from .contracts import KGInputBatch, OreoQwenBatch
from .reasoner import KGReasoner
from .tokens import OreoSpecialTokenIds, register_oreo_special_tokens


@dataclass
class OreoBaseModelOutputWithPast(BaseModelOutputWithPast):
    """Qwen backbone outputs plus losses and optional KIL diagnostics."""

    entity_link_loss: Tensor | None = None
    relation_loss: Tensor | None = None
    entity_link_logits: Tensor | None = None
    entity_states: tuple[Tensor, ...] | None = None
    relation_probs: tuple[Tensor, ...] | None = None


@dataclass
class OreoCausalLMOutputWithPast(CausalLMOutputWithPast):
    """Causal-LM outputs with each objective exposed separately."""

    lm_loss: Tensor | None = None
    entity_link_loss: Tensor | None = None
    relation_loss: Tensor | None = None
    entity_link_logits: Tensor | None = None
    entity_states: tuple[Tensor, ...] | None = None
    relation_probs: tuple[Tensor, ...] | None = None


class OreoQwen3Model(Qwen3Model):
    """Qwen3 backbone that inserts a KG reasoning step at each boundary."""

    config_class = OreoQwen3Config

    def __init__(self, config: OreoQwen3Config) -> None:
        super().__init__(config)
        self.reasoner = KGReasoner(config.reasoner_config)

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | dict[str, Tensor] | None = None,
        position_ids: Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: Tensor | None = None,
        use_cache: bool | None = None,
        output_kg_diagnostics: bool | None = None,
        kg_mention_batch: Tensor | None = None,
        kg_src_positions: Tensor | None = None,
        kg_rel_positions: Tensor | None = None,
        kg_tgt_positions: Tensor | None = None,
        kg_node_ids: Tensor | None = None,
        kg_node_batch: Tensor | None = None,
        kg_edge_src: Tensor | None = None,
        kg_edge_dst: Tensor | None = None,
        kg_edge_type: Tensor | None = None,
        kg_initial_node_indices: Tensor | None = None,
        kg_initial_entity_probs: Tensor | None = None,
        kg_entity_link_targets: Tensor | None = None,
        kg_entity_link_mask: Tensor | None = None,
        kg_relation_targets: Tensor | None = None,
        kg_relation_target_mask: Tensor | None = None,
        **kwargs: Any,
    ) -> OreoBaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        kg = _build_kg_input(
            kg_mention_batch=kg_mention_batch,
            kg_src_positions=kg_src_positions,
            kg_rel_positions=kg_rel_positions,
            kg_tgt_positions=kg_tgt_positions,
            kg_node_ids=kg_node_ids,
            kg_node_batch=kg_node_batch,
            kg_edge_src=kg_edge_src,
            kg_edge_dst=kg_edge_dst,
            kg_edge_type=kg_edge_type,
            kg_initial_node_indices=kg_initial_node_indices,
            kg_initial_entity_probs=kg_initial_entity_probs,
            kg_entity_link_targets=kg_entity_link_targets,
            kg_entity_link_mask=kg_entity_link_mask,
            kg_relation_targets=kg_relation_targets,
            kg_relation_target_mask=kg_relation_target_mask,
        )

        past_seen_tokens = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        run_kg = kg is not None and past_seen_tokens == 0

        if run_kg and self.config.validate_kg_inputs:
            validation_attention_mask = (
                attention_mask if isinstance(attention_mask, Tensor) else None
            )
            OreoQwenBatch(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=validation_attention_mask,
                kg=kg,
            ).validate(
                spec=self.config.reasoner_config.contract_spec,
                special_token_ids=(
                    self.config.oreo_special_token_ids
                    if input_ids is not None
                    else None
                ),
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            position_ids = (
                torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
                + past_seen_tokens
            ).unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = (
                    create_sliding_window_causal_mask(**mask_kwargs)
                )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        diagnostics_enabled = (
            self.config.output_kg_diagnostics
            if output_kg_diagnostics is None
            else output_kg_diagnostics
        )

        initialization = None
        reasoner_state = None
        relation_losses: list[Tensor] = []
        entity_states: list[Tensor] = []
        relation_probabilities: list[Tensor] = []
        step_by_boundary = {
            boundary: step
            for step, boundary in enumerate(self.config.kil_layer_boundaries)
        }

        for layer_index, decoder_layer in enumerate(
            self.layers[: self.config.num_hidden_layers]
        ):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[
                    self.config.layer_types[layer_index]
                ],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

            completed_layers = layer_index + 1
            if run_kg and completed_layers in step_by_boundary:
                if initialization is None:
                    initialization = self.reasoner.initialize(hidden_states, kg)
                    reasoner_state = initialization.state
                    if diagnostics_enabled:
                        entity_states.append(reasoner_state.entity_probs)

                step_output = self.reasoner(
                    hidden_states,
                    kg,
                    reasoner_state,
                    step_by_boundary[completed_layers],
                )
                hidden_states = step_output.hidden_states
                reasoner_state = step_output.state
                relation_losses.append(step_output.relation_loss)
                if diagnostics_enabled:
                    entity_states.append(reasoner_state.entity_probs)
                    relation_probabilities.append(step_output.relation_probs)

        hidden_states = self.norm(hidden_states)
        relation_loss = (
            torch.stack(relation_losses).sum() if relation_losses else None
        )

        return OreoBaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            entity_link_loss=(
                initialization.entity_link_loss
                if initialization is not None
                else None
            ),
            relation_loss=relation_loss,
            entity_link_logits=(
                initialization.entity_logits if initialization is not None else None
            ),
            entity_states=tuple(entity_states) if diagnostics_enabled and run_kg else None,
            relation_probs=(
                tuple(relation_probabilities)
                if diagnostics_enabled and run_kg
                else None
            ),
        )


class OreoQwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    """Training and generation wrapper for Oreo-Qwen3."""

    config_class = OreoQwen3Config
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_gather_output"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
    _fsdp_plan = {"lm_head": "keep_full_weight"}

    def __init__(self, config: OreoQwen3Config) -> None:
        super().__init__(config)
        self.model = OreoQwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | dict[str, Tensor] | None = None,
        position_ids: Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: Tensor | None = None,
        labels: Tensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | Tensor = 0,
        output_kg_diagnostics: bool | None = None,
        kg_mention_batch: Tensor | None = None,
        kg_src_positions: Tensor | None = None,
        kg_rel_positions: Tensor | None = None,
        kg_tgt_positions: Tensor | None = None,
        kg_node_ids: Tensor | None = None,
        kg_node_batch: Tensor | None = None,
        kg_edge_src: Tensor | None = None,
        kg_edge_dst: Tensor | None = None,
        kg_edge_type: Tensor | None = None,
        kg_initial_node_indices: Tensor | None = None,
        kg_initial_entity_probs: Tensor | None = None,
        kg_entity_link_targets: Tensor | None = None,
        kg_entity_link_mask: Tensor | None = None,
        kg_relation_targets: Tensor | None = None,
        kg_relation_target_mask: Tensor | None = None,
        **kwargs: Any,
    ) -> OreoCausalLMOutputWithPast:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_kg_diagnostics=output_kg_diagnostics,
            kg_mention_batch=kg_mention_batch,
            kg_src_positions=kg_src_positions,
            kg_rel_positions=kg_rel_positions,
            kg_tgt_positions=kg_tgt_positions,
            kg_node_ids=kg_node_ids,
            kg_node_batch=kg_node_batch,
            kg_edge_src=kg_edge_src,
            kg_edge_dst=kg_edge_dst,
            kg_edge_type=kg_edge_type,
            kg_initial_node_indices=kg_initial_node_indices,
            kg_initial_entity_probs=kg_initial_entity_probs,
            kg_entity_link_targets=kg_entity_link_targets,
            kg_entity_link_mask=kg_entity_link_mask,
            kg_relation_targets=kg_relation_targets,
            kg_relation_target_mask=kg_relation_target_mask,
            **kwargs,
        )

        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(outputs.last_hidden_state[:, slice_indices, :])

        lm_loss = None
        if labels is not None:
            lm_loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        has_entity_supervision = kg_entity_link_targets is not None
        has_relation_supervision = kg_relation_targets is not None
        loss = lm_loss
        if has_entity_supervision and outputs.entity_link_loss is not None:
            weighted = self.config.entity_link_loss_weight * outputs.entity_link_loss
            loss = weighted if loss is None else loss + weighted
        if has_relation_supervision and outputs.relation_loss is not None:
            weighted = self.config.relation_loss_weight * outputs.relation_loss
            loss = weighted if loss is None else loss + weighted

        return OreoCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            lm_loss=lm_loss,
            entity_link_loss=outputs.entity_link_loss,
            relation_loss=outputs.relation_loss,
            entity_link_logits=outputs.entity_link_logits,
            entity_states=outputs.entity_states,
            relation_probs=outputs.relation_probs,
        )

    @classmethod
    def from_qwen3_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        oreo_config_overrides: Mapping[str, Any],
        qwen_config: Qwen3Config | None = None,
        **from_pretrained_kwargs: Any,
    ) -> "OreoQwen3ForCausalLM":
        """Load native Qwen weights while newly initializing Oreo parameters.

        ``oreo_config_overrides`` must at least set the intended entity and
        relation vocabulary sizes. Other Oreo configuration fields, including
        KIL boundaries, may be overridden in the same mapping.
        """

        missing = {"num_entities", "num_relations"} - set(oreo_config_overrides)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"oreo_config_overrides is missing: {names}.")
        if qwen_config is None:
            config_loading_keys = (
                "cache_dir",
                "force_download",
                "local_files_only",
                "revision",
                "subfolder",
                "token",
            )
            config_loading_kwargs = {
                key: from_pretrained_kwargs[key]
                for key in config_loading_keys
                if key in from_pretrained_kwargs
            }
            qwen_config = Qwen3Config.from_pretrained(
                pretrained_model_name_or_path,
                **config_loading_kwargs,
            )
        config = OreoQwen3Config.from_qwen3_config(
            qwen_config,
            **dict(oreo_config_overrides),
        )
        return cls.from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            **from_pretrained_kwargs,
        )


def register_oreo_tokens_and_resize(
    model: Qwen3PreTrainedModel,
    tokenizer: Any,
) -> OreoSpecialTokenIds:
    """Register Oreo tokens, resize embeddings, and persist their IDs.

    This helper is intentionally explicit rather than part of model
    construction: tokenizer mutation must happen before input serialization,
    and loading an already prepared checkpoint must not add tokens again.
    """

    token_ids = register_oreo_special_tokens(tokenizer)
    model.resize_token_embeddings(len(tokenizer))
    if isinstance(model.config, OreoQwen3Config):
        model.config.set_oreo_special_token_ids(token_ids)
    else:
        # The benchmark's native-Qwen control uses the same marked text and
        # therefore needs the same resized vocabulary, even though it has no
        # reasoner and does not consume these IDs architecturally.
        for name, value in token_ids.as_config_dict().items():
            setattr(model.config, name, value)
    return token_ids


_KG_REQUIRED_ARGUMENTS = (
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


def _build_kg_input(**values: Tensor | None) -> KGInputBatch | None:
    if all(value is None for value in values.values()):
        return None
    missing = [name for name in _KG_REQUIRED_ARGUMENTS if values[name] is None]
    if missing:
        raise ValueError(
            "A partial KG batch was supplied; missing required fields: "
            + ", ".join(missing)
            + "."
        )
    return KGInputBatch(**values)


__all__ = [
    "OreoBaseModelOutputWithPast",
    "OreoCausalLMOutputWithPast",
    "OreoQwen3ForCausalLM",
    "OreoQwen3Model",
    "register_oreo_tokens_and_resize",
]
