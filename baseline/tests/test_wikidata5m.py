from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from oreo_qwen.wikidata5m import (
    CorruptIndexError,
    EdgeDirection,
    GraphSplit,
    IndexBuildError,
    MalformedSourceError,
    MissingDataError,
    QueryDirection,
    StaleIndexError,
    UnknownEntityError,
    UnknownRelationError,
    Wikidata5MStore,
    build_index,
    validate_index,
)
from oreo_qwen.wikidata5m.cli import main


def _write_fixture(root: Path) -> Path:
    alias_dir = root / "wikidata5m_alias"
    graph_dir = root / "wikidata5m_transductive"
    alias_dir.mkdir(parents=True)
    graph_dir.mkdir(parents=True)
    (alias_dir / "wikidata5m_entity.txt").write_text(
        "Q10\tTen\tTen\n"
        "Q2\tTwo\tDeux\n"
        "Q1\tOne\n",
        encoding="utf-8",
    )
    (alias_dir / "wikidata5m_relation.txt").write_text(
        "P31\tinstance of\n"
        "P5\tfive relation\n",
        encoding="utf-8",
    )
    (root / "wikidata5m_text.txt").write_text(
        "Q2\tTwo has café context.\n"
        "Q1\tOne has context with a tab\tinside.\n",
        encoding="utf-8",
    )
    (graph_dir / "wikidata5m_transductive_train.txt").write_text(
        "Q10\tP31\tQ2\n"
        "Q2\tP5\tQ1\n"
        "Q2\tP31\tQ2\n"
        "Q20\tP99\tQ1\n"
        "Q10\tP31\tQ2\n",
        encoding="utf-8",
    )
    (graph_dir / "wikidata5m_transductive_valid.txt").write_text(
        "Q1\tP31\tQ10\n", encoding="utf-8"
    )
    (graph_dir / "wikidata5m_transductive_test.txt").write_text(
        "Q2\tP31\tQ10\n", encoding="utf-8"
    )
    return root


@pytest.fixture
def built_index(tmp_path: Path) -> tuple[Path, Path]:
    source = _write_fixture(tmp_path / "Wikidata5M")
    return source, build_index(source)


def test_build_layout_manifest_and_stats(built_index: tuple[Path, Path]) -> None:
    source, index = built_index
    assert index == source / "indexed"
    expected = {
        "manifest.json",
        "metadata.sqlite",
        "graph/out_offsets.npy",
        "graph/out_neighbors.npy",
        "graph/out_relations.npy",
        "graph/out_splits.npy",
        "graph/in_offsets.npy",
        "graph/in_neighbors.npy",
        "graph/in_relations.npy",
        "graph/in_splits.npy",
    }
    actual = {
        str(path.relative_to(index)) for path in index.rglob("*") if path.is_file()
    }
    assert actual == expected
    assert not list(index.glob(".build-*"))

    manifest = validate_index(index, full=True)
    assert manifest["schema_version"] == 1
    assert manifest["counts"] == {
        "edges": 7,
        "edges_by_split": {"test": 1, "train": 5, "valid": 1},
        "entities": 4,
        "entity_alias_rows": 3,
        "entity_aliases": 5,
        "passages": 2,
        "relation_alias_rows": 2,
        "relation_aliases": 2,
        "relations": 3,
    }


def test_numeric_mappings_passages_and_aliases(
    built_index: tuple[Path, Path],
) -> None:
    _, index = built_index
    with Wikidata5MStore(index) as store:
        assert [(item.entity_id, item.qid) for item in store.iter_entities()] == [
            (0, "Q1"),
            (1, "Q2"),
            (2, "Q10"),
            (3, "Q20"),
        ]
        assert [(item.relation_id, item.pid) for item in store.iter_relations()] == [
            (0, "P5"),
            (1, "P31"),
            (2, "P99"),
        ]
        assert store.entity_id("Q10") == 2
        assert store.entity_qid(1) == "Q2"
        assert store.relation_id("P31") == 1
        assert store.relation_pid(0) == "P5"
        assert store.passage("Q2") == "Two has café context."
        assert store.passage("Q1") == "One has context with a tab\tinside."
        assert store.passage("Q20") is None
        with pytest.raises(MissingDataError):
            store.require_passage("Q20")
        assert store.entity_aliases("Q10") == ("Ten", "Ten")
        assert store.entity_aliases("Q2") == ("Two", "Deux")
        assert store.entity_aliases("Q20") == ()
        assert store.relation_aliases("P31") == ("instance of",)
        assert store.relation_aliases("P99") == ()
        assert list(store.iter_passages()) == [
            ("Q1", "One has context with a tab\tinside."),
            ("Q2", "Two has café context."),
        ]


def test_unknown_identifiers_raise_typed_errors(
    built_index: tuple[Path, Path],
) -> None:
    _, index = built_index
    with Wikidata5MStore(index) as store:
        with pytest.raises(UnknownEntityError):
            store.entity_id("Q999")
        with pytest.raises(UnknownEntityError):
            store.entity_qid(99)
        with pytest.raises(UnknownRelationError):
            store.relation_id("P404")
        with pytest.raises(UnknownRelationError):
            store.relation_pid(99)
        with pytest.raises(UnknownEntityError):
            store.passage("Q999")


def test_directions_splits_self_loop_and_duplicate_edges(
    built_index: tuple[Path, Path],
) -> None:
    _, index = built_index
    with Wikidata5MStore(index) as store:
        all_splits = [GraphSplit.TRAIN, GraphSplit.VALID, GraphSplit.TEST]
        edges = store.edges("Q2", direction=QueryDirection.BOTH, splits=all_splits)
        assert [
            (
                edge.head_qid,
                edge.relation_pid,
                edge.tail_qid,
                edge.split.value,
                edge.direction,
            )
            for edge in edges
        ] == [
            ("Q2", "P5", "Q1", "train", EdgeDirection.OUTGOING),
            ("Q2", "P31", "Q2", "train", EdgeDirection.SELF),
            ("Q2", "P31", "Q10", "test", EdgeDirection.OUTGOING),
            ("Q10", "P31", "Q2", "train", EdgeDirection.INCOMING),
            ("Q10", "P31", "Q2", "train", EdgeDirection.INCOMING),
        ]

        incoming = store.edges(
            "Q2", direction="incoming", splits=["train"], relations=["P31"]
        )
        assert [edge.direction for edge in incoming] == [
            EdgeDirection.INCOMING,
            EdgeDirection.SELF,
            EdgeDirection.INCOMING,
        ]
        assert [edge.head_qid for edge in incoming] == ["Q10", "Q2", "Q10"]

        with pytest.raises(ValueError, match="explicitly"):
            store.edges("Q2", direction="both", splits=None)
        with pytest.raises(ValueError, match="at least one"):
            store.edges("Q2", direction="both", splits=[])


def test_dense_and_decoded_queries_are_equivalent(
    built_index: tuple[Path, Path],
) -> None:
    _, index = built_index
    with Wikidata5MStore(index) as store:
        dense = store.edges_dense(
            store.entity_id("Q2"),
            direction="outgoing",
            splits=["train", "test"],
            relation_ids=[store.relation_id("P31")],
        )
        decoded = store.edges(
            "Q2",
            direction="outgoing",
            splits=["train", "test"],
            relations=["P31"],
        )
        assert len(dense) == len(decoded) == 2
        assert [store.entity_qid(int(value)) for value in dense.head_ids] == [
            edge.head_qid for edge in decoded
        ]
        assert [store.entity_qid(int(value)) for value in dense.tail_ids] == [
            edge.tail_qid for edge in decoded
        ]
        assert [store.relation_pid(int(value)) for value in dense.relation_ids] == [
            edge.relation_pid for edge in decoded
        ]


def test_deterministic_arrays_and_iteration(tmp_path: Path) -> None:
    source = _write_fixture(tmp_path / "Wikidata5M")
    first = build_index(source, tmp_path / "first")
    second = build_index(source, tmp_path / "second")
    for name in (
        "out_offsets",
        "out_neighbors",
        "out_relations",
        "out_splits",
        "in_offsets",
        "in_neighbors",
        "in_relations",
        "in_splits",
    ):
        assert (first / "graph" / f"{name}.npy").read_bytes() == (
            second / "graph" / f"{name}.npy"
        ).read_bytes()
    with Wikidata5MStore(first) as store:
        first_iteration = list(
            store.iter_edges(splits=["train", "valid", "test"])
        )
        second_iteration = list(
            store.iter_edges(splits=["train", "valid", "test"])
        )
    assert first_iteration == second_iteration
    assert len(first_iteration) == 7


def test_existing_index_requires_force(built_index: tuple[Path, Path]) -> None:
    source, index = built_index
    with pytest.raises(IndexBuildError, match="already exists"):
        build_index(source)
    rebuilt = build_index(source, force=True)
    assert rebuilt == index
    validate_index(index)


def test_stale_source_is_rejected(built_index: tuple[Path, Path]) -> None:
    source, index = built_index
    passage_path = source / "wikidata5m_text.txt"
    stat = passage_path.stat()
    os.utime(passage_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1))
    with pytest.raises(StaleIndexError, match="modification time"):
        Wikidata5MStore(index)


def test_full_validation_detects_same_size_source_change(
    built_index: tuple[Path, Path],
) -> None:
    source, index = built_index
    passage_path = source / "wikidata5m_text.txt"
    stat = passage_path.stat()
    contents = passage_path.read_bytes()
    passage_path.write_bytes(contents.replace(b"cafe", b"cafe", 1))
    # Change one UTF-8-safe ASCII byte while preserving size and quick stat data.
    changed = passage_path.read_bytes().replace(b"Two has", b"Two haz", 1)
    passage_path.write_bytes(changed)
    os.utime(passage_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    validate_index(index, full=False)
    with pytest.raises(StaleIndexError, match="content changed"):
        validate_index(index, full=True)


def test_corrupt_array_size_and_checksum_are_rejected(
    built_index: tuple[Path, Path],
) -> None:
    _, index = built_index
    path = index / "graph" / "out_neighbors.npy"
    original = path.read_bytes()
    path.write_bytes(original[:-1])
    with pytest.raises(CorruptIndexError, match="size"):
        validate_index(index)


def test_full_validation_detects_artifact_checksum(
    built_index: tuple[Path, Path],
) -> None:
    _, index = built_index
    path = index / "graph" / "out_neighbors.npy"
    contents = bytearray(path.read_bytes())
    contents[-1] ^= 1
    path.write_bytes(contents)
    with pytest.raises(CorruptIndexError, match="checksum"):
        validate_index(index, full=True)


def test_malformed_and_duplicate_sources_fail(tmp_path: Path) -> None:
    malformed = _write_fixture(tmp_path / "malformed")
    (malformed / "wikidata5m_transductive" / "wikidata5m_transductive_train.txt").write_text(
        "Q1\tP31\n", encoding="utf-8"
    )
    with pytest.raises(MalformedSourceError, match="exactly three"):
        build_index(malformed)

    duplicate = _write_fixture(tmp_path / "duplicate")
    entity_file = duplicate / "wikidata5m_alias" / "wikidata5m_entity.txt"
    entity_file.write_text(entity_file.read_text(encoding="utf-8") + "Q1\tAgain\n")
    with pytest.raises(MalformedSourceError, match="duplicate Q"):
        build_index(duplicate)


def test_cli_stats_validate_and_inspect(
    built_index: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    _, index = built_index
    assert main(["stats", str(index)]) == 0
    assert json.loads(capsys.readouterr().out)["entities"] == 4

    assert main(["validate", str(index)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"

    assert main(
        [
            "inspect",
            str(index),
            "--qid",
            "Q2",
            "--edges",
            "--direction",
            "outgoing",
            "--split",
            "train",
        ]
    ) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["qid"] == "Q2"
    assert len(inspected["edges"]) == 2


@pytest.mark.skipif(
    os.environ.get("WIKIDATA5M_RUN_INTEGRATION") != "1",
    reason="set WIKIDATA5M_RUN_INTEGRATION=1 for the full dataset build",
)
def test_real_wikidata5m_build_and_open() -> None:
    source = Path(
        os.environ.get(
            "WIKIDATA5M_ROOT",
            "/Users/diyalakhani/Documents/Diya/ucla/thesis/KGReaLM/datasets/Wikidata5M",
        )
    )
    index = build_index(source, force=True)
    with Wikidata5MStore(index) as store:
        assert store.entity_count > 4_000_000
        assert store.relation_count > 100
        first = next(store.iter_entities())
        assert store.entity_qid(first.entity_id) == first.qid
