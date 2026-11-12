from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from oreo_qwen import KGContractSpec
from oreo_qwen.data import (
    CollationError,
    OreoSemanticCollator,
    Wikidata5MReconstructionDataset,
)
from oreo_qwen.tokens import OreoSpecialTokenIds
from oreo_qwen.wikidata5m import Wikidata5MStore, build_index
from oreo_qwen.wikidata5m.preprocessing import (
    PreprocessConfig,
    build_preprocessed_corpus,
)


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 1
    unk_token_id = 99_999

    def __init__(self) -> None:
        self.vocab = {"<pad>": 0, "<eos>": 1, "<|rel|>": 2, "<|t_ent|>": 3}

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self.vocab.get(tokens, self.unk_token_id)
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]

    def _encode(self, text: str) -> list[int]:
        pieces = re.findall(r"<\|rel\|>|<\|t_ent\|>|\S+", text)
        result: list[int] = []
        for piece in pieces:
            if piece not in self.vocab:
                self.vocab[piece] = len(self.vocab)
            result.append(self.vocab[piece])
        return result

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, **_kwargs
    ):
        assert tokenize
        text = " ".join(
            f"<{message['role']}> {message['content']}" for message in messages
        )
        if add_generation_prompt:
            text += " <assistant>"
        return self._encode(text)

    def __call__(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return {"input_ids": self._encode(text)}


def _write_source(root: Path) -> Path:
    aliases = root / "wikidata5m_alias"
    graph = root / "wikidata5m_transductive"
    aliases.mkdir(parents=True)
    graph.mkdir(parents=True)
    (aliases / "wikidata5m_entity.txt").write_text(
        "Q101\tMira Chen\tM. Chen\n"
        "Q102\tNorthbridge University\tNorthbridge\n"
        "Q103\tSolace Foundation\n"
        "Q104\tIlan Rao\n"
        "Q105\tDana Lee\n"
        "Q900\tAurora Institute\tAurora\n",
        encoding="utf-8",
    )
    (aliases / "wikidata5m_relation.txt").write_text(
        "P26\tworked with\nP108\tfounded by\nP463\tsupported by\nP749\tpartner\n",
        encoding="utf-8",
    )
    (root / "wikidata5m_text.txt").write_text(
        "Q900\tAurora Institute was founded by Mira Chen and Northbridge University. "
        "Mira Chen worked with Ilan Rao, while Northbridge University hired Dana Lee. "
        "Solace Foundation supported Aurora Institute. The annual lecture returned.\n",
        encoding="utf-8",
    )
    (graph / "wikidata5m_transductive_train.txt").write_text(
        "Q101\tP108\tQ900\n"
        "Q102\tP749\tQ900\n"
        "Q103\tP463\tQ900\n"
        "Q104\tP26\tQ101\n"
        "Q105\tP26\tQ102\n",
        encoding="utf-8",
    )
    (graph / "wikidata5m_transductive_valid.txt").write_text("", encoding="utf-8")
    (graph / "wikidata5m_transductive_test.txt").write_text("", encoding="utf-8")
    return root


@pytest.fixture
def dataset(tmp_path: Path) -> Wikidata5MReconstructionDataset:
    source = _write_source(tmp_path / "Wikidata5M")
    index = build_index(source)
    corpus = tmp_path / "corpus"
    with Wikidata5MStore(index) as store:
        build_preprocessed_corpus(
            store,
            corpus,
            PreprocessConfig(shard_entity_count=2, max_context_blanks=0),
        )
    return Wikidata5MReconstructionDataset(
        corpus, index_dir=index, validate_checksums=True
    )


def test_dataset_streams_records_and_embedding_requests(dataset) -> None:
    assert len(dataset) == 1
    item = next(iter(dataset))
    assert item.record.example_id == "wiki5m:Q900"
    assert item.passage_text.startswith("Aurora Institute")
    assert len(item.node_embedding_requests) == len(item.record.local_graph.nodes)
    assert [request.dense_id for request in item.node_embedding_requests] == [
        node.entity_id for node in item.record.local_graph.nodes
    ]
    assert all(request.text.startswith("Entity: ") for request in item.node_embedding_requests)

    relation_requests = dataset.relation_embedding_requests()
    assert len(relation_requests) == dataset.relation_count
    assert [request.dense_id for request in relation_requests] == list(
        range(dataset.relation_count)
    )
    assert all(request.text.startswith("Relation: ") for request in relation_requests)


def test_collator_builds_language_graph_and_supervision(dataset) -> None:
    item = next(iter(dataset))
    tokenizer = _Tokenizer()
    collator = OreoSemanticCollator(
        tokenizer,
        num_relations=dataset.relation_count,
        pad_to_multiple_of=8,
    )
    semantic = collator([item, item])

    assert semantic.input_ids.shape[0] == 2
    assert semantic.input_ids.shape[1] % 8 == 0
    assert semantic.example_ids == ("wiki5m:Q900", "wiki5m:Q900")
    assert torch.all(
        semantic.labels[0, : semantic.assistant_content_start_positions[0]] == -100
    )
    active_suffix = (
        torch.arange(semantic.input_ids.shape[1])
        >= semantic.assistant_content_start_positions[:, None]
    ) & semantic.attention_mask.bool()
    assert torch.equal(semantic.labels[active_suffix], semantic.input_ids[active_suffix])

    mentions_per_item = len(item.record.marker_mention_ids)
    nodes_per_item = len(item.record.local_graph.nodes)
    assert semantic.kg.kg_mention_batch.tolist() == [0] * mentions_per_item + [
        1
    ] * mentions_per_item
    assert semantic.kg.kg_node_batch.tolist() == [0] * nodes_per_item + [1] * nodes_per_item
    assert torch.all(
        semantic.kg.kg_initial_node_indices[mentions_per_item:] >= nodes_per_item
    )
    assert semantic.kg.kg_relation_target_mask.any()
    selected = semantic.kg.kg_relation_targets[
        semantic.kg.kg_relation_target_mask
    ]
    torch.testing.assert_close(selected.sum(dim=-1), torch.ones(selected.shape[0]))

    embedding_dim = 4
    materialized = semantic.materialize(
        node_embeddings=torch.randn(2 * nodes_per_item, embedding_dim),
        relation_embeddings=torch.randn(dataset.relation_count, embedding_dim),
    )
    materialized.validate(
        spec=KGContractSpec(
            num_relations=dataset.relation_count,
            num_kil_steps=1,
            knowledge_embedding_dim=embedding_dim,
        ),
        special_token_ids=OreoSpecialTokenIds(relation=2, target_entity=3),
    )


def test_collator_refuses_semantic_truncation(dataset) -> None:
    item = next(iter(dataset))
    collator = OreoSemanticCollator(
        _Tokenizer(), num_relations=dataset.relation_count, max_length=4
    )
    with pytest.raises(CollationError, match="semantic truncation"):
        collator([item])

    dropping = OreoSemanticCollator(
        _Tokenizer(),
        num_relations=dataset.relation_count,
        max_length=4,
        overflow_policy="drop",
    )
    assert dropping([item]) is None
