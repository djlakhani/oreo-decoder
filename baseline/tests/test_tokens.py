from __future__ import annotations

import unittest

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from oreo_qwen.tokens import (
    KG_MASK_TOKENS,
    OREO_SPECIAL_TOKENS,
    OreoSpecialTokenIds,
    register_oreo_special_tokens,
)


def make_tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(
        WordLevel(
            vocab={"<unk>": 0, "<eos>": 1, "hello": 2},
            unk_token="<unk>",
        )
    )
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<eos>",
        pad_token="<eos>",
    )


class SpecialTokenTests(unittest.TestCase):
    def test_registration_is_complete_distinct_and_idempotent(self) -> None:
        tokenizer = make_tokenizer()
        tokenizer.add_special_tokens(
            {"extra_special_tokens": ["<|existing|>"]},
            replace_extra_special_tokens=False,
        )
        original_size = len(tokenizer)

        token_ids = register_oreo_special_tokens(tokenizer)

        self.assertEqual(len(KG_MASK_TOKENS), 16)
        self.assertEqual(len(OREO_SPECIAL_TOKENS), 21)
        self.assertEqual(len(token_ids.all_ids), 21)
        self.assertEqual(len(set(token_ids.all_ids)), 21)
        self.assertEqual(len(tokenizer), original_size + 21)
        self.assertIn("<|existing|>", tokenizer.all_special_tokens)

        for token, token_id in zip(OREO_SPECIAL_TOKENS, token_ids.all_ids):
            self.assertEqual(tokenizer.encode(token, add_special_tokens=False), [token_id])

        second_ids = register_oreo_special_tokens(tokenizer)
        self.assertEqual(second_ids, token_ids)
        self.assertEqual(len(tokenizer), original_size + 21)

    def test_config_mapping_is_json_serializable(self) -> None:
        token_ids = register_oreo_special_tokens(make_tokenizer())
        config = token_ids.as_config_dict()

        self.assertEqual(config["oreo_source_entity_token_id"], token_ids.source_entity)
        self.assertEqual(config["oreo_mask_token_ids"], list(token_ids.masks))

    def test_duplicate_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "distinct"):
            OreoSpecialTokenIds(
                source_entity=1,
                relation=1,
                target_entity=2,
                reconstruct=3,
                answer=4,
                masks=tuple(range(5, 21)),
            )


if __name__ == "__main__":
    unittest.main()

