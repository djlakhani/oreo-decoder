"""Serializable configuration for the Oreo-Qwen3 architecture."""

from __future__ import annotations

from typing import Any

from transformers import Qwen3Config

from .reasoner import KGReasonerConfig
from .tokens import OreoSpecialTokenIds


class OreoQwen3Config(Qwen3Config):
    """Qwen3 configuration extended with Knowledge Interaction Layer settings.

    A KIL boundary is expressed as a count of *completed* Qwen decoder layers.
    For example, boundary ``9`` inserts a KIL after ``layers[8]`` and before
    ``layers[9]``.
    """

    model_type = "oreo_qwen3"

    def __init__(
        self,
        *,
        num_entities: int = 1,
        num_relations: int = 1,
        kil_layer_boundaries: list[int] | tuple[int, ...] | None = None,
        entity_embedding_dim: int = 128,
        relation_embedding_dim: int | None = None,
        entity_link_loss_weight: float = 1.0,
        relation_loss_weight: float = 1.0,
        output_kg_diagnostics: bool = False,
        validate_kg_inputs: bool = True,
        oreo_source_entity_token_id: int | None = None,
        oreo_relation_token_id: int | None = None,
        oreo_target_entity_token_id: int | None = None,
        oreo_reconstruct_token_id: int | None = None,
        oreo_answer_token_id: int | None = None,
        oreo_mask_token_ids: list[int] | tuple[int, ...] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if kil_layer_boundaries is None:
            kil_layer_boundaries = self._default_kil_boundaries(
                self.num_hidden_layers
            )

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.kil_layer_boundaries = list(kil_layer_boundaries)
        self.entity_embedding_dim = entity_embedding_dim
        self.relation_embedding_dim = relation_embedding_dim
        self.entity_link_loss_weight = entity_link_loss_weight
        self.relation_loss_weight = relation_loss_weight
        self.output_kg_diagnostics = output_kg_diagnostics
        self.validate_kg_inputs = validate_kg_inputs

        self.oreo_source_entity_token_id = oreo_source_entity_token_id
        self.oreo_relation_token_id = oreo_relation_token_id
        self.oreo_target_entity_token_id = oreo_target_entity_token_id
        self.oreo_reconstruct_token_id = oreo_reconstruct_token_id
        self.oreo_answer_token_id = oreo_answer_token_id
        self.oreo_mask_token_ids = (
            None if oreo_mask_token_ids is None else list(oreo_mask_token_ids)
        )

        self._validate_oreo_fields()

    @staticmethod
    def _default_kil_boundaries(num_hidden_layers: int) -> tuple[int, ...]:
        if num_hidden_layers == 28:
            return (9, 19)
        if num_hidden_layers < 2:
            raise ValueError(
                "Oreo-Qwen requires at least two decoder layers so a KIL can "
                "be followed by another Qwen layer."
            )
        if num_hidden_layers == 2:
            return (1,)
        first = max(1, num_hidden_layers // 3)
        second = min(num_hidden_layers - 1, (2 * num_hidden_layers) // 3)
        return (first,) if first == second else (first, second)

    def _validate_oreo_fields(self) -> None:
        for name in ("num_entities", "num_relations", "entity_embedding_dim"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")

        if self.relation_embedding_dim is not None:
            value = self.relation_embedding_dim
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError("relation_embedding_dim must be a positive integer.")

        boundaries = self.kil_layer_boundaries
        if not boundaries:
            raise ValueError("kil_layer_boundaries must contain at least one boundary.")
        if any(not isinstance(item, int) or isinstance(item, bool) for item in boundaries):
            raise ValueError("Every KIL boundary must be an integer.")
        if boundaries != sorted(set(boundaries)):
            raise ValueError("KIL boundaries must be unique and strictly increasing.")
        if boundaries[0] < 1 or boundaries[-1] >= self.num_hidden_layers:
            raise ValueError(
                "KIL boundaries must lie between 1 and num_hidden_layers - 1."
            )

        for name in ("entity_link_loss_weight", "relation_loss_weight"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative number.")

        if not isinstance(self.output_kg_diagnostics, bool):
            raise ValueError("output_kg_diagnostics must be a bool.")
        if not isinstance(self.validate_kg_inputs, bool):
            raise ValueError("validate_kg_inputs must be a bool.")

        token_fields = (
            self.oreo_source_entity_token_id,
            self.oreo_relation_token_id,
            self.oreo_target_entity_token_id,
            self.oreo_reconstruct_token_id,
            self.oreo_answer_token_id,
            self.oreo_mask_token_ids,
        )
        if all(value is None for value in token_fields):
            return
        if any(value is None for value in token_fields):
            raise ValueError(
                "Oreo special-token IDs must be either entirely absent or entirely set."
            )
        # The value object checks integer type, nonnegativity, count, and uniqueness.
        token_ids = self.oreo_special_token_ids
        if token_ids is None:
            raise RuntimeError("Expected complete Oreo special-token IDs.")
        if max(token_ids.all_ids) >= self.vocab_size:
            raise ValueError("Every Oreo special-token ID must be below vocab_size.")

    @property
    def num_kil_steps(self) -> int:
        return len(self.kil_layer_boundaries)

    @property
    def reasoner_config(self) -> KGReasonerConfig:
        return KGReasonerConfig(
            hidden_size=self.hidden_size,
            num_entities=self.num_entities,
            num_relations=self.num_relations,
            num_kil_steps=self.num_kil_steps,
            entity_embedding_dim=self.entity_embedding_dim,
            relation_embedding_dim=self.relation_embedding_dim,
            rms_norm_eps=self.rms_norm_eps,
            initializer_range=self.initializer_range,
        )

    @property
    def oreo_special_token_ids(self) -> OreoSpecialTokenIds | None:
        fields = (
            self.oreo_source_entity_token_id,
            self.oreo_relation_token_id,
            self.oreo_target_entity_token_id,
            self.oreo_reconstruct_token_id,
            self.oreo_answer_token_id,
            self.oreo_mask_token_ids,
        )
        if all(value is None for value in fields):
            return None
        if any(value is None for value in fields):
            raise ValueError("Oreo special-token IDs are only partially configured.")
        return OreoSpecialTokenIds(
            source_entity=self.oreo_source_entity_token_id,
            relation=self.oreo_relation_token_id,
            target_entity=self.oreo_target_entity_token_id,
            reconstruct=self.oreo_reconstruct_token_id,
            answer=self.oreo_answer_token_id,
            masks=tuple(self.oreo_mask_token_ids),
        )

    def set_oreo_special_token_ids(self, token_ids: OreoSpecialTokenIds) -> None:
        """Persist resolved tokenizer IDs after model embedding resize."""

        for name, value in token_ids.as_config_dict().items():
            setattr(self, name, value)
        self._validate_oreo_fields()

    @classmethod
    def from_qwen3_config(
        cls,
        config: Qwen3Config,
        **oreo_overrides: Any,
    ) -> "OreoQwen3Config":
        """Copy a native Qwen3 blueprint and add Oreo-specific settings."""

        values = config.to_dict()
        values.pop("model_type", None)
        values.pop("architectures", None)
        values.update(oreo_overrides)
        return cls(**values)


__all__ = ["OreoQwen3Config"]
