"""Deterministically sample accepted passages and materialize raw-Qwen inputs."""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from ..wikidata5m.preprocessing import PreprocessConfig, normalize_alias
from ..wikidata5m.preprocessing.pipeline import preprocess_passage
from ..wikidata5m.store import Wikidata5MStore
from .artifacts import atomic_write_json, file_sha256, write_eval_shard
from .prompting import SYSTEM_PROMPT, build_user_prompt, render_native_passage
from .schema import (
    EVAL_SCHEMA_VERSION,
    PROMPT_VERSION,
    EvalArtifactError,
    EvalExample,
    EvalTarget,
    sha256_text,
    stable_json,
    validate_eval_example,
)

PREPARE_MANIFEST_VERSION = 1


def _accepted_aliases(
    store: Wikidata5MStore,
    local_candidate_qids: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    owners: dict[str, set[str]] = {}
    aliases_by_qid: dict[str, set[str]] = {}
    for qid in local_candidate_qids:
        aliases = {
            normalized
            for alias in store.entity_aliases(qid)
            if (normalized := normalize_alias(alias))
        }
        aliases_by_qid[qid] = aliases
        for alias in aliases:
            owners.setdefault(alias, set()).add(qid)
    return {
        qid: tuple(sorted(alias for alias in aliases if owners[alias] == {qid}))
        for qid, aliases in aliases_by_qid.items()
    }


def materialize_eval_example(
    store: Wikidata5MStore,
    record: Any,
    passage: str,
) -> EvalExample:
    corrupted = render_native_passage(record, passage)
    accepted = _accepted_aliases(store, record.evaluation.local_candidate_qids)
    targets = tuple(
        EvalTarget(
            blank_index=occurrence.sentinel_index,
            blank_token=f"[BLANK_{occurrence.sentinel_index}]",
            exact_text=occurrence.exact_text,
            normalized_exact_text=normalize_alias(occurrence.exact_text),
            kind=occurrence.kind,
            gold_qid=occurrence.gold_qid,
            gold_entity_id=occurrence.gold_entity_id,
            accepted_normalized_aliases=(
                ()
                if occurrence.gold_qid is None
                else accepted.get(occurrence.gold_qid, ())
            ),
        )
        for occurrence in record.masked_occurrences
    )
    example = EvalExample(
        schema_version=EVAL_SCHEMA_VERSION,
        prompt_version=PROMPT_VERSION,
        example_id=record.example_id,
        passage_qid=record.passage.qid,
        passage_entity_id=record.passage.entity_id,
        datastore_manifest_sha256=record.evaluation.datastore_manifest_sha256,
        source_record_sha256=sha256_text(record.to_json()),
        corrupted_passage=corrupted,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_user_prompt(corrupted, len(targets)),
        targets=targets,
    )
    validate_eval_example(example)
    return example


def _configuration(
    *,
    store: Wikidata5MStore,
    sample_size: int,
    seed: int,
    shard_size: int,
    preprocess_config: PreprocessConfig,
) -> dict[str, Any]:
    return {
        "schema_version": EVAL_SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "sample_size": sample_size,
        "seed": seed,
        "selection": "numpy_pcg64_random_permutation_until_k_accepted",
        "shard_size": shard_size,
        "preprocess_config": preprocess_config.to_dict(),
        "preprocess_config_fingerprint": preprocess_config.fingerprint,
        "datastore_manifest_sha256": file_sha256(store.index_dir / "manifest.json"),
        "entity_count": store.entity_count,
    }


def prepare_random_sample(
    store: Wikidata5MStore,
    output_dir: str | Path,
    *,
    sample_size: int,
    seed: int,
    shard_size: int = 250,
    preprocess_config: PreprocessConfig | None = None,
    force: bool = False,
    checkpoint_interval: int = 1_000,
) -> Path:
    """Sample uniformly from accepted passages without building the full corpus."""

    if sample_size <= 0 or shard_size <= 0 or checkpoint_interval <= 0:
        raise ValueError("sample size, shard size, and checkpoint interval must be positive")
    config = preprocess_config or PreprocessConfig()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    work = output / ".prepare-work"
    work.mkdir(exist_ok=True)
    accepted_path = work / "accepted.jsonl"
    state_path = work / "state.json"
    requested = _configuration(
        store=store,
        sample_size=sample_size,
        seed=seed,
        shard_size=shard_size,
        preprocess_config=config,
    )

    if force:
        accepted_path.unlink(missing_ok=True)
        state_path.unlink(missing_ok=True)
        (output / "manifest.json").unlink(missing_ok=True)
        (output / "sample_manifest.json").unlink(missing_ok=True)
        shutil.rmtree(output / "shards", ignore_errors=True)
    state: dict[str, Any] = {
        "configuration": requested,
        "cursor": 0,
        "rejections": {},
    }
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("configuration") != requested:
            raise EvalArtifactError("existing prepare state has different configuration")
    elif (output / "manifest.json").exists() and not force:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("configuration") == requested:
            return output
        raise EvalArtifactError("existing evaluation dataset has different configuration")

    examples: list[EvalExample] = []
    existing_qids: set[str] = set()
    if accepted_path.exists():
        with accepted_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                example = EvalExample.from_json(line)
                if example.passage_qid not in existing_qids:
                    examples.append(example)
                    existing_qids.add(example.passage_qid)
    if len(examples) > sample_size:
        raise EvalArtifactError("prepare work file contains too many accepted examples")

    permutation = np.random.Generator(np.random.PCG64(seed)).permutation(
        store.entity_count
    )
    cursor = int(state.get("cursor", 0))
    rejections: Counter[str] = Counter(state.get("rejections", {}))
    with accepted_path.open("a", encoding="utf-8") as accepted_handle:
        while cursor < len(permutation) and len(examples) < sample_size:
            entity_id = int(permutation[cursor])
            cursor += 1
            qid = store.entity_qid(entity_id)
            if qid in existing_qids:
                continue
            result = preprocess_passage(store, qid, config)
            if result.record is None:
                assert result.rejection_reason is not None
                rejections[result.rejection_reason] += 1
            else:
                passage = store.require_passage(qid)
                example = materialize_eval_example(store, result.record, passage)
                accepted_handle.write(example.to_json() + "\n")
                accepted_handle.flush()
                os.fsync(accepted_handle.fileno())
                examples.append(example)
                existing_qids.add(qid)
            if cursor % checkpoint_interval == 0 or result.record is not None:
                state = {
                    "configuration": requested,
                    "cursor": cursor,
                    "accepted": len(examples),
                    "rejections": dict(sorted(rejections.items())),
                }
                atomic_write_json(state_path, state)

    if len(examples) < sample_size:
        raise EvalArtifactError(
            f"only found {len(examples)} accepted examples after scanning all entities"
        )
    examples = examples[:sample_size]
    shard_dir = output / "shards"
    shard_dir.mkdir(exist_ok=True)
    shard_entries: list[dict[str, Any]] = []
    for shard_index, start in enumerate(range(0, len(examples), shard_size)):
        stop = min(start + shard_size, len(examples))
        path = shard_dir / f"inputs-{shard_index:05d}.jsonl.gz"
        info = write_eval_shard(path, examples[start:stop])
        shard_entries.append(
            {
                "path": str(path.relative_to(output)),
                "start": start,
                "stop": stop,
                **info,
            }
        )

    sample_manifest = {
        "schema_version": EVAL_SCHEMA_VERSION,
        "prompt_version": PROMPT_VERSION,
        "sample_size": len(examples),
        "seed": seed,
        "examples": [
            {
                "sample_index": index,
                "example_id": example.example_id,
                "passage_qid": example.passage_qid,
                "source_record_sha256": example.source_record_sha256,
            }
            for index, example in enumerate(examples)
        ],
    }
    atomic_write_json(output / "sample_manifest.json", sample_manifest)
    manifest = {
        "manifest_version": PREPARE_MANIFEST_VERSION,
        "complete": True,
        "configuration": requested,
        "records": len(examples),
        "scanned_entities": cursor,
        "rejections": dict(sorted(rejections.items())),
        "sample_manifest_sha256": file_sha256(output / "sample_manifest.json"),
        "shards": shard_entries,
    }
    atomic_write_json(output / "manifest.json", manifest)
    accepted_path.unlink(missing_ok=True)
    state_path.unlink(missing_ok=True)
    try:
        work.rmdir()
    except OSError:
        pass
    return output
