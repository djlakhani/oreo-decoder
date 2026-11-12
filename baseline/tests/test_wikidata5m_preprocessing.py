from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from oreo_qwen.wikidata5m import Wikidata5MStore, build_index
from oreo_qwen.wikidata5m.preprocessing import (
    CorpusValidationError,
    PreprocessConfig,
    build_preprocessed_corpus,
    preprocess_passage,
    render_example,
    validate_preprocessed_corpus,
    validate_preprocessed_record,
)
from oreo_qwen.wikidata5m.preprocessing.aliases import (
    match_graph_aliases,
    normalize_alias,
    normalized_text,
)
from oreo_qwen.wikidata5m.preprocessing.cli import main as preprocessing_main


def _write_preprocessing_fixture(root: Path) -> Path:
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
        "Q106\tGhost Entity\n"
        "Q107\tShared Name\tAurora\n"
        "Q108\tShared Name\n"
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
        "Solace Foundation supported Aurora Institute. The annual lecture drew a crowd; "
        "the annual lecture returned.\n"
        "Q101\tMira Chen is a researcher.\n"
        "Q106\tNothing relevant appears here.\n",
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
def preprocessing_store(tmp_path: Path):
    source = _write_preprocessing_fixture(tmp_path / "Wikidata5M")
    index = build_index(source)
    with Wikidata5MStore(index) as store:
        yield source, store


def test_single_passage_record_and_rendering(preprocessing_store) -> None:
    _, store = preprocessing_store
    result = preprocess_passage(store, "Q900")
    assert result.accepted
    assert result.record is not None
    record = result.record
    passage = store.require_passage("Q900")
    validate_preprocessed_record(record, passage, store=store)

    assert record.target_groups[0].qid == "Q900"
    assert record.target_groups[0].kind == "primary_kg"
    assert len(record.target_groups) == 2
    assert len(record.target_groups[0].visible_supporter_qids) >= 2
    assert all(item.split == "train" for item in record.local_graph.edges)
    assert [item.sentinel_index for item in record.masked_occurrences] == list(
        range(len(record.masked_occurrences))
    )
    assert [item.start_char for item in record.masked_occurrences] == sorted(
        item.start_char for item in record.masked_occurrences
    )

    rendered = render_example(record, passage)
    for occurrence in record.masked_occurrences:
        assert occurrence.sentinel_token in rendered.corrupted_passage
        assert f"{occurrence.sentinel_token} {occurrence.exact_text}" in rendered.reconstruction_suffix
    marked_mira = "Mira Chen <|rel|> <|t_ent|>"
    marked_northbridge = "Northbridge University <|rel|> <|t_ent|>"
    assert marked_mira in rendered.corrupted_passage or (
        marked_northbridge in rendered.corrupted_passage
    )
    assert "[BLANK_" in rendered.corrupted_passage
    assert any(item.kind == "context_only" for item in record.masked_occurrences)
    for item in record.masked_occurrences:
        if item.kind == "context_only":
            assert normalize_alias(item.exact_text) in normalize_alias(
                rendered.corrupted_passage
            )


def test_matching_offsets_survive_unicode_normalization() -> None:
    original = "The Cafe\u0301 Society"
    view = normalized_text(original)
    position = view.text.index("café")
    assert original[view.starts[position] : view.ends[position + 3]] == "Cafe\u0301"
    assert normalize_alias("  CAFÉ\tSociety ") == "café society"


def test_rejection_reasons_are_explicit(preprocessing_store) -> None:
    _, store = preprocessing_store
    no_support = preprocess_passage(store, "Q101")
    assert no_support.rejection_reason == "insufficient_primary_support"

    hub = preprocess_passage(
        store,
        "Q900",
        PreprocessConfig(max_incoming_degree=2),
    )
    assert hub.rejection_reason == "hub_primary"

    missing_subject = preprocess_passage(store, "Q106")
    assert missing_subject.rejection_reason == "no_subject_mention"

    overflow = preprocess_passage(store, "Q900", PreprocessConfig(max_sentinels=1))
    assert overflow.rejection_reason == "subject_sentinel_overflow"


def test_alias_ambiguity_is_rejected_but_subject_wins(preprocessing_store) -> None:
    _, store = preprocessing_store
    ambiguous = match_graph_aliases(
        "Shared Name appeared.",
        {"Q107": 1, "Q108": 1},
        "Q900",
        store,
        max_aliases_per_entity=64,
    )
    assert ambiguous.matches == ()
    assert ambiguous.diagnostics["ambiguous_alias_matches"] == 1

    subject = match_graph_aliases(
        "Aurora appeared.",
        {"Q900": 0, "Q107": 1},
        "Q900",
        store,
        max_aliases_per_entity=64,
    )
    assert [item.qid for item in subject.matches] == ["Q900"]


def test_inspect_cli_returns_rendered_record(preprocessing_store, capsys) -> None:
    _, store = preprocessing_store
    assert preprocessing_main(["inspect", str(store.index_dir), "--qid", "Q900"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["accepted"] is True
    assert payload["record"]["example_id"] == "wiki5m:Q900"
    assert payload["rendered"]["corrupted_passage"]


def test_corpus_is_deterministic_and_validated(preprocessing_store, tmp_path: Path) -> None:
    _, store = preprocessing_store
    config = PreprocessConfig(shard_entity_count=3)
    first = build_preprocessed_corpus(store, tmp_path / "first", config)
    second = build_preprocessed_corpus(store, tmp_path / "second", config)
    parallel = build_preprocessed_corpus(
        store, tmp_path / "parallel", config, workers=2
    )
    first_manifest = validate_preprocessed_corpus(first, store=store, full=True)
    second_manifest = validate_preprocessed_corpus(second, store=store, full=True)
    parallel_manifest = validate_preprocessed_corpus(parallel, store=store, full=True)
    assert first_manifest["records"] == second_manifest["records"] == 1
    assert [item["sha256"] for item in first_manifest["shards"]] == [
        item["sha256"] for item in second_manifest["shards"]
    ]
    assert [item["sha256"] for item in first_manifest["shards"]] == [
        item["sha256"] for item in parallel_manifest["shards"]
    ]
    assert first_manifest["statistics"]["inspected"] == store.entity_count

    shard_mtimes = {
        item["path"]: (first / item["path"]).stat().st_mtime_ns
        for item in first_manifest["shards"]
    }
    build_preprocessed_corpus(store, first, config)
    assert shard_mtimes == {
        item["path"]: (first / item["path"]).stat().st_mtime_ns
        for item in first_manifest["shards"]
    }

    shard = first / first_manifest["shards"][0]["path"]
    shard.write_bytes(shard.read_bytes() + b"corrupt")
    with pytest.raises(CorpusValidationError, match="truncated|checksum"):
        validate_preprocessed_corpus(first)


@pytest.mark.skipif(
    not os.environ.get("WIKIDATA5M_ROOT"),
    reason="set WIKIDATA5M_ROOT to run the real-data preprocessing smoke test",
)
def test_real_wikidata5m_bounded_preprocessing(tmp_path: Path) -> None:
    source = Path(os.environ["WIKIDATA5M_ROOT"])
    with Wikidata5MStore(source / "indexed", source_root=source) as store:
        stop = min(store.entity_count, 10)
        output = build_preprocessed_corpus(
            store,
            tmp_path / "real-smoke",
            PreprocessConfig(shard_entity_count=10),
            stop_entity_id=stop,
        )
        validate_preprocessed_corpus(output, store=store, full=True)
