from __future__ import annotations

from types import MethodType

import pytest
import torch

from oreo_qwen.embedding import (
    FrozenKnowledgeEmbedder,
    PersistentEmbeddingCache,
    _last_token_pool,
    format_entity_text,
    format_relation_text,
)


def test_persistent_cache_round_trip_and_namespace_isolation(tmp_path) -> None:
    path = tmp_path / "knowledge.sqlite3"
    with PersistentEmbeddingCache(
        path, namespace="model-a", output_dim=2, dtype=torch.float16
    ) as cache:
        cache.put_many(
            (
                (("entity", 1), "one", torch.tensor([1.0, 2.0])),
                (("entity", 2), "two", torch.tensor([3.0, 4.0])),
            )
        )
        torch.testing.assert_close(
            cache.get(("entity", 1), "one"),
            torch.tensor([1.0, 2.0], dtype=torch.float16),
        )
        assert cache.item_count == 2
        with pytest.raises(ValueError, match="conflicting text"):
            cache.get(("entity", 1), "different")

    with PersistentEmbeddingCache(
        path, namespace="model-b", output_dim=2, dtype=torch.float16
    ) as other:
        assert other.get(("entity", 1), "one") is None
        assert other.item_count == 0


def test_last_token_pool_handles_left_and_right_padding() -> None:
    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    left_mask = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1]])
    torch.testing.assert_close(_last_token_pool(hidden, left_mask), hidden[:, -1])

    right_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    expected = torch.stack((hidden[0, 2], hidden[1, 1]))
    torch.testing.assert_close(_last_token_pool(hidden, right_mask), expected)


def test_on_demand_cache_encodes_each_key_once() -> None:
    embedder = FrozenKnowledgeEmbedder.__new__(FrozenKnowledgeEmbedder)
    embedder.output_dim = 2
    embedder.cache_dtype = torch.float32
    embedder._cache = {}
    calls: list[list[str]] = []

    def fake_encode(_self, texts):
        calls.append(list(texts))
        return torch.tensor([[float(len(text)), 1.0] for text in texts])

    embedder._encode_texts = MethodType(fake_encode, embedder)
    first = embedder.resolve(["Q1", "Q2", "Q1"], ["one", "two", "one"])
    second = embedder.resolve(["Q2", "Q3"], ["two", "three"])

    self_expected = torch.tensor([[3.0, 1.0], [3.0, 1.0], [3.0, 1.0]])
    torch.testing.assert_close(first, self_expected)
    torch.testing.assert_close(second, torch.tensor([[3.0, 1.0], [5.0, 1.0]]))
    assert calls == [["one", "two"], ["three"]]
    assert embedder.cached_items == 3


def test_knowledge_text_formatting_is_typed_and_stable() -> None:
    assert format_entity_text(
        "Albert Einstein",
        description="Physicist",
        aliases=("Einstein",),
        types=("human",),
    ) == (
        "Entity: Albert Einstein\nDescription: Physicist\n"
        "Types: human\nAliases: Einstein"
    )
    assert format_relation_text("place of birth", description="birth location") == (
        "Relation: place of birth\nDescription: birth location"
    )
