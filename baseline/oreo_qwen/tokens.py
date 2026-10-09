"""OreoLM special-token definitions and tokenizer registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SOURCE_ENTITY_TOKEN = "<|s_ent|>"
REL_TOKEN = "<|rel|>"
TARGET_ENTITY_TOKEN = "<|t_ent|>"
KG_RECONSTRUCT_TOKEN = "<|kg_reconstruct|>"
KG_ANSWER_TOKEN = "<|kg_answer|>"
KG_MASK_TOKENS = tuple(f"<|kg_mask_{index}|>" for index in range(16))

OREO_SPECIAL_TOKENS = (
    SOURCE_ENTITY_TOKEN,
    REL_TOKEN,
    TARGET_ENTITY_TOKEN,
    KG_RECONSTRUCT_TOKEN,
    KG_ANSWER_TOKEN,
    *KG_MASK_TOKENS,
)


@dataclass(frozen=True, slots=True)
class OreoSpecialTokenIds:
    """Resolved tokenizer IDs for every Oreo-Qwen special token."""

    source_entity: int
    relation: int
    target_entity: int
    reconstruct: int
    answer: int
    masks: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(self.masks) != len(KG_MASK_TOKENS):
            raise ValueError(
                f"Expected {len(KG_MASK_TOKENS)} mask token IDs, got {len(self.masks)}."
            )

        ids = self.all_ids
        if any(not isinstance(token_id, int) or isinstance(token_id, bool) for token_id in ids):
            raise TypeError("Every Oreo special-token ID must be an integer.")
        if any(token_id < 0 for token_id in ids):
            raise ValueError("Oreo special-token IDs must be nonnegative.")
        if len(set(ids)) != len(ids):
            raise ValueError("Every Oreo special token must resolve to a distinct ID.")

    @property
    def all_ids(self) -> tuple[int, ...]:
        return (
            self.source_entity,
            self.relation,
            self.target_entity,
            self.reconstruct,
            self.answer,
            *self.masks,
        )

    def as_config_dict(self) -> dict[str, int | list[int]]:
        """Return JSON-serializable fields for a future OreoQwen3Config."""

        return {
            "oreo_source_entity_token_id": self.source_entity,
            "oreo_relation_token_id": self.relation,
            "oreo_target_entity_token_id": self.target_entity,
            "oreo_reconstruct_token_id": self.reconstruct,
            "oreo_answer_token_id": self.answer,
            "oreo_mask_token_ids": list(self.masks),
        }

    @classmethod
    def from_tokenizer(cls, tokenizer: Any) -> "OreoSpecialTokenIds":
        converted = tokenizer.convert_tokens_to_ids(list(OREO_SPECIAL_TOKENS))
        if isinstance(converted, int):
            converted = [converted]
        if converted is None or len(converted) != len(OREO_SPECIAL_TOKENS):
            raise ValueError("Tokenizer did not return an ID for every Oreo special token.")

        ids = tuple(converted)
        if any(token_id is None for token_id in ids):
            raise ValueError("Tokenizer returned None for an Oreo special token.")

        unknown_id = getattr(tokenizer, "unk_token_id", None)
        if unknown_id is not None and any(token_id == unknown_id for token_id in ids):
            raise ValueError("At least one Oreo special token still maps to the unknown token.")

        return cls(
            source_entity=ids[0],
            relation=ids[1],
            target_entity=ids[2],
            reconstruct=ids[3],
            answer=ids[4],
            masks=ids[5:],
        )


def register_oreo_special_tokens(tokenizer: Any) -> OreoSpecialTokenIds:
    """Register Oreo tokens without replacing a tokenizer's existing specials.

    The caller remains responsible for resizing model embeddings after token
    registration. That operation belongs to the model-integration phase.
    """

    special_tokens = {"extra_special_tokens": list(OREO_SPECIAL_TOKENS)}
    try:
        tokenizer.add_special_tokens(
            special_tokens,
            replace_extra_special_tokens=False,
        )
    except TypeError:
        # Transformers 4 used the older keyword and field name.
        tokenizer.add_special_tokens(
            {"additional_special_tokens": list(OREO_SPECIAL_TOKENS)},
            replace_additional_special_tokens=False,
        )

    return OreoSpecialTokenIds.from_tokenizer(tokenizer)

