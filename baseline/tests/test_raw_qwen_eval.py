from __future__ import annotations

import json
from pathlib import Path

import pytest

from oreo_qwen.raw_eval import (
    iter_eval_examples,
    prepare_random_sample,
    recovery_parse,
    score_generation,
    strict_parse,
    validate_eval_dataset,
)
from oreo_qwen.raw_eval.cli import main as raw_eval_main
from oreo_qwen.wikidata5m import Wikidata5MStore, build_index


def _write_fixture(root: Path) -> Path:
    aliases = root / "wikidata5m_alias"
    graph = root / "wikidata5m_transductive"
    aliases.mkdir(parents=True)
    graph.mkdir(parents=True)
    (aliases / "wikidata5m_entity.txt").write_text(
        "Q101\tMira Chen\n"
        "Q102\tNorthbridge University\n"
        "Q103\tSolace Foundation\n"
        "Q104\tIlan Rao\n"
        "Q900\tAurora Institute\tAurora\n",
        encoding="utf-8",
    )
    (aliases / "wikidata5m_relation.txt").write_text(
        "P26\tworked with\nP108\tfounded by\nP463\tsupported by\n",
        encoding="utf-8",
    )
    (root / "wikidata5m_text.txt").write_text(
        "Q900\tAurora Institute hired Mira Chen and Northbridge University. "
        "Mira Chen worked with Ilan Rao. Solace Foundation supported Aurora Institute. "
        "The annual lecture began, and the annual lecture returned.\n",
        encoding="utf-8",
    )
    (graph / "wikidata5m_transductive_train.txt").write_text(
        "Q101\tP108\tQ900\n"
        "Q102\tP108\tQ900\n"
        "Q103\tP463\tQ900\n"
        "Q104\tP26\tQ101\n",
        encoding="utf-8",
    )
    (graph / "wikidata5m_transductive_valid.txt").write_text("", encoding="utf-8")
    (graph / "wikidata5m_transductive_test.txt").write_text("", encoding="utf-8")
    return root


@pytest.fixture
def eval_store(tmp_path: Path):
    source = _write_fixture(tmp_path / "Wikidata5M")
    with Wikidata5MStore(build_index(source)) as store:
        yield source, store


def test_prepare_retains_gold_identity_outside_prompt(eval_store, tmp_path: Path) -> None:
    _, store = eval_store
    output = prepare_random_sample(
        store,
        tmp_path / "eval-inputs",
        sample_size=1,
        seed=17,
        shard_size=1,
    )
    manifest = validate_eval_dataset(output)
    assert manifest["records"] == 1
    example = next(iter_eval_examples(output / manifest["shards"][0]["path"]))
    assert example.passage_qid == "Q900"
    assert "<|s_ent|>" not in example.user_prompt
    assert "<|rel|>" not in example.user_prompt
    assert "[BLANK_0]" in example.user_prompt
    assert all("Q900" not in value for value in (example.system_prompt, example.user_prompt))
    primary = [target for target in example.targets if target.kind == "primary_kg"]
    assert primary
    assert all(target.gold_qid == "Q900" for target in primary)
    assert all(target.gold_entity_id == store.entity_id("Q900") for target in primary)
    assert all("aurora institute" in target.accepted_normalized_aliases for target in primary)


def test_strict_and_recovery_parsers() -> None:
    exact = "[BLANK_0] Paris\n[BLANK_1] France"
    strict = strict_parse(exact, 2)
    assert strict.valid
    assert strict.answers == {0: "Paris", 1: "France"}

    malformed = "- [BLANK_0]: Paris\n- [BLANK_1]: France"
    assert not strict_parse(malformed, 2).valid
    recovered = recovery_parse(malformed, 2)
    assert recovered.valid
    assert recovered.answers == {0: "Paris", 1: "France"}


def test_alias_scoring_and_format_are_separate(eval_store, tmp_path: Path) -> None:
    _, store = eval_store
    output = prepare_random_sample(store, tmp_path / "inputs", sample_size=1, seed=3)
    manifest = validate_eval_dataset(output)
    example = next(iter_eval_examples(output / manifest["shards"][0]["path"]))
    answers = []
    for target in example.targets:
        if target.gold_qid == "Q900" and "aurora" in target.accepted_normalized_aliases:
            answer = "Aurora"
        else:
            answer = target.exact_text
        answers.append(f"{target.blank_token} {answer}")
    score = score_generation(example, "\n".join(answers))
    assert score["strict_parse"]["valid"]
    q900 = [item for item in score["strict_targets"] if item["gold_qid"] == "Q900"]
    assert q900
    assert all(item["alias_aware"] for item in q900)
    assert any(not item["exact"] for item in q900)


def test_cli_validate_and_inspect(eval_store, tmp_path: Path, capsys) -> None:
    _, store = eval_store
    output = prepare_random_sample(store, tmp_path / "inputs", sample_size=1, seed=9)
    assert raw_eval_main(["validate", str(output)]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["records"] == 1

    assert raw_eval_main(["inspect", str(output), "--count", "1"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["example_id"] == "wiki5m:Q900"
    assert "exact_text" not in inspected["targets"][0]
