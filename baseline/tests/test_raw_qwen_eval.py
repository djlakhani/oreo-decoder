from __future__ import annotations

import gzip
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
from oreo_qwen.raw_eval.artifacts import file_sha256
from oreo_qwen.raw_eval.review import generate_review
from oreo_qwen.raw_eval.schema import PREDICTION_SCHEMA_VERSION, stable_json
from oreo_qwen.raw_eval.scoring import aggregate_prediction_records
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


def test_alias_substring_scoring_is_separate_and_lenient(
    eval_store, tmp_path: Path
) -> None:
    _, store = eval_store
    output = prepare_random_sample(store, tmp_path / "inputs", sample_size=1, seed=4)
    manifest = validate_eval_dataset(output)
    example = next(iter_eval_examples(output / manifest["shards"][0]["path"]))
    answers = []
    substring_indices: set[int] = set()
    for target in example.targets:
        if target.gold_qid == "Q900" and "aurora institute" in target.accepted_normalized_aliases:
            answer = "instit"
            substring_indices.add(target.blank_index)
        else:
            answer = target.exact_text
        answers.append(f"{target.blank_token} {answer}")
    score = score_generation(example, "\n".join(answers))
    assert substring_indices
    scored = {
        item["blank_index"]: item
        for item in score["strict_targets"]
        if item["blank_index"] in substring_indices
    }
    assert all(not item["alias_aware"] for item in scored.values())
    assert all(item["alias_substring_aware"] for item in scored.values())


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


def test_html_review_uses_prediction_sidecars(eval_store, tmp_path: Path) -> None:
    _, store = eval_store
    inputs = prepare_random_sample(store, tmp_path / "inputs", sample_size=1, seed=5)
    input_manifest = validate_eval_dataset(inputs)
    example = next(iter_eval_examples(inputs / input_manifest["shards"][0]["path"]))
    generation = "\n".join(
        f"{target.blank_token} {target.exact_text}" for target in example.targets
    )
    score = score_generation(example, generation)
    record = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "prompt_version": example.prompt_version,
        "example_id": example.example_id,
        "passage_qid": example.passage_qid,
        "passage_entity_id": example.passage_entity_id,
        "datastore_manifest_sha256": example.datastore_manifest_sha256,
        "source_record_sha256": example.source_record_sha256,
        "targets": example.to_dict()["targets"],
        "status": "generated",
        "input_tokens": 100,
        "generated_tokens": 20,
        "prompt_text": example.user_prompt,
        "raw_generated_text": generation,
        "generated_text": generation,
        "score": score,
    }
    run = tmp_path / "run"
    prediction = run / "predictions" / "predictions-00000.jsonl.gz"
    prediction.parent.mkdir(parents=True)
    with gzip.open(prediction, "wt", encoding="utf-8") as handle:
        handle.write(stable_json(record) + "\n")
    manifest = {
        "complete": True,
        "configuration": {},
        "shards": [
            {
                "path": "predictions/predictions-00000.jsonl.gz",
                "records": 1,
                "size": prediction.stat().st_size,
                "sha256": file_sha256(prediction),
            }
        ],
    }
    (run / "manifest.json").write_text(stable_json(manifest) + "\n", encoding="utf-8")
    rescored = {
        "scorer_version": "test",
        "run_manifest_sha256": file_sha256(run / "manifest.json"),
        "metrics": aggregate_prediction_records([record]),
    }
    metrics_path = run / "rescored_metrics.json"
    metrics_path.write_text(stable_json(rescored) + "\n", encoding="utf-8")

    report = generate_review(metrics_path)
    page = report.read_text(encoding="utf-8")
    assert "All KG blanks" in page
    assert "Q900" in page
    assert "Passage with blanks" in page
    assert "Aurora Institute" in page
    assert "Model response" in page
