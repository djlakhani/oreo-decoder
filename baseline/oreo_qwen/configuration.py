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
        num_relations: int = 1,
        kil_layer_boundaries: list[int] | tuple[int, ...] | None = None,
        knowledge_embedding_dim: int = 384,
        adapter_bottleneck_dim: int = 512,
        knowledge_embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        relation_loss_weight: float = 1.0,
        output_kg_diagnostics: bool = False,
        validate_kg_inputs: bool = True,
        oreo_relation_token_id: int | None = None,
        oreo_target_entity_token_id: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        if kil_layer_boundaries is None:
            kil_layer_boundaries = self._default_kil_boundaries(
                self.num_hidden_layers
            )

        self.num_relations = num_relations
        self.kil_layer_boundaries = list(kil_layer_boundaries)
        self.knowledge_embedding_dim = knowledge_embedding_dim
        self.adapter_bottleneck_dim = adapter_bottleneck_dim
        self.knowledge_embedding_model_name = knowledge_embedding_model_name
        self.relation_loss_weight = relation_loss_weight
        self.output_kg_diagnostics = output_kg_diagnostics
        self.validate_kg_inputs = validate_kg_inputs

        self.oreo_relation_token_id = oreo_relation_token_id
        self.oreo_target_entity_token_id = oreo_target_entity_token_id

        self._validate_oreo_fields()

    @staticmethod
    def _default_kil_boundaries(num_hidden_layers: int) -> tuple[int, ...]:
        if num_hidden_layers == 36:
            # One late KIL, leaving the final two Qwen3-4B decoder blocks to
            # integrate the knowledge written to and broadcast from T_ENT.
            return (34,)
        if num_hidden_layers < 2:
            raise ValueError(
                "Oreo-Qwen requires at least two decoder layers so a KIL can "
                "be followed by another Qwen layer."
            )
        if num_hidden_layers == 2:
            return (1,)
        return (max(1, num_hidden_layers - 2),)

    def _validate_oreo_fields(self) -> None:
        for name in (
            "num_relations",
            "knowledge_embedding_dim",
            "adapter_bottleneck_dim",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")

        if (
            not isinstance(self.knowledge_embedding_model_name, str)
            or not self.knowledge_embedding_model_name
        ):
            raise ValueError("knowledge_embedding_model_name must be a nonempty string.")

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

        for name in ("relation_loss_weight",):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative number.")

        if not isinstance(self.output_kg_diagnostics, bool):
            raise ValueError("output_kg_diagnostics must be a bool.")
        if not isinstance(self.validate_kg_inputs, bool):
            raise ValueError("validate_kg_inputs must be a bool.")

        token_fields = (
            self.oreo_relation_token_id,
            self.oreo_target_entity_token_id,
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
            num_relations=self.num_relations,
            num_kil_steps=self.num_kil_steps,
            knowledge_embedding_dim=self.knowledge_embedding_dim,
            adapter_bottleneck_dim=self.adapter_bottleneck_dim,
            rms_norm_eps=self.rms_norm_eps,
            initializer_range=self.initializer_range,
        )

    @property
    def oreo_special_token_ids(self) -> OreoSpecialTokenIds | None:
        fields = (
            self.oreo_relation_token_id,
            self.oreo_target_entity_token_id,
        )
        if all(value is None for value in fields):
            return None
        if any(value is None for value in fields):
            raise ValueError("Oreo special-token IDs are only partially configured.")
        return OreoSpecialTokenIds(
            relation=self.oreo_relation_token_id,
            target_entity=self.oreo_target_entity_token_id,
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
