"""Batched, resumable raw-Qwen inference over materialized eval shards."""

from __future__ import annotations

import gzip
import json
import os
import platform
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from .artifacts import atomic_write_json, file_sha256, iter_eval_examples, load_manifest
from .schema import PREDICTION_SCHEMA_VERSION, EvalArtifactError, EvalExample, stable_json
from .scoring import aggregate_prediction_records, score_generation

RUN_MANIFEST_VERSION = 1


def _prediction_path(output: Path, input_path: Path) -> Path:
    suffix = input_path.name.removeprefix("inputs-")
    return output / "predictions" / f"predictions-{suffix}"


def _write_prediction_shard(path: Path, records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    count = 0
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                for record in records:
                    zipped.write((stable_json(record) + "\n").encode("utf-8"))
                    count += 1
        validated = sum(1 for _ in iter_prediction_records(temporary))
        if validated != count:
            raise EvalArtifactError("prediction shard record count is wrong")
        info = {
            "records": count,
            "size": temporary.stat().st_size,
            "sha256": file_sha256(temporary),
        }
        os.replace(temporary, path)
        return info
    finally:
        temporary.unlink(missing_ok=True)


def iter_prediction_records(path: str | Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise EvalArtifactError(
                    f"invalid prediction JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise EvalArtifactError("prediction record root must be an object")
            if record.get("schema_version") != PREDICTION_SCHEMA_VERSION:
                raise EvalArtifactError("unsupported prediction-record schema")
            yield record


def _messages(example: EvalExample) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": example.system_prompt},
        {"role": "user", "content": example.user_prompt},
    ]


def _environment(model_name: str, revision: str | None) -> dict[str, Any]:
    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": properties.name,
            "total_memory": properties.total_memory,
            "capability": list(torch.cuda.get_device_capability(0)),
            "device_count": torch.cuda.device_count(),
        }
    return {
        "model_name": model_name,
        "requested_revision": revision,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": gpu,
    }


def _base_prediction(example: EvalExample) -> dict[str, Any]:
    return {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "prompt_version": example.prompt_version,
        "example_id": example.example_id,
        "passage_qid": example.passage_qid,
        "passage_entity_id": example.passage_entity_id,
        "datastore_manifest_sha256": example.datastore_manifest_sha256,
        "source_record_sha256": example.source_record_sha256,
        # Gold identities and aliases are retained only in the prediction
        # artifact, never inserted into the prompt sent to Qwen.
        "targets": example.to_dict()["targets"],
    }


def _generate_shard(
    examples: Sequence[EvalExample],
    *,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
    max_input_tokens: int,
    max_new_tokens: int,
    progress_callback: Callable[[int], None] | None = None,
) -> list[dict[str, Any]]:
    prepared: list[tuple[EvalExample, str, list[int]]] = []
    predictions: dict[str, dict[str, Any]] = {}
    for example in examples:
        prompt = tokenizer.apply_chat_template(
            _messages(example),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if len(input_ids) > max_input_tokens:
            predictions[example.example_id] = {
                **_base_prediction(example),
                "status": "input_too_long",
                "input_tokens": len(input_ids),
                "generated_tokens": None,
                "prompt_text": prompt,
                "raw_generated_text": None,
                "generated_text": None,
                "score": None,
            }
        else:
            prepared.append((example, prompt, input_ids))

    skipped = len(examples) - len(prepared)
    if skipped and progress_callback is not None:
        progress_callback(skipped)

    for start in range(0, len(prepared), batch_size):
        batch = prepared[start : start + batch_size]
        padded = tokenizer.pad(
            [
                {"input_ids": item[2], "attention_mask": [1] * len(item[2])}
                for item in batch
            ],
            padding=True,
            return_tensors="pt",
        )
        padded = {key: value.to(device) for key, value in padded.items()}
        width = int(padded["input_ids"].shape[1])
        started = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                **padded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        elapsed = time.perf_counter() - started
        for row, (example, prompt, unpadded_ids) in enumerate(batch):
            generated_ids = output_ids[row, width:].tolist()
            if tokenizer.eos_token_id in generated_ids:
                generated_ids = generated_ids[
                    : generated_ids.index(tokenizer.eos_token_id) + 1
                ]
            else:
                while generated_ids and generated_ids[-1] == tokenizer.pad_token_id:
                    generated_ids.pop()
            raw_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            score = score_generation(example, text)
            predictions[example.example_id] = {
                **_base_prediction(example),
                "status": "generated",
                "input_tokens": len(unpadded_ids),
                "generated_tokens": len(generated_ids),
                "batch_elapsed_seconds": elapsed,
                "prompt_text": prompt,
                "raw_generated_text": raw_text,
                "generated_text": text,
                "score": score,
            }
        if progress_callback is not None:
            progress_callback(len(batch))
    return [predictions[example.example_id] for example in examples]


def _print_progress(completed: int, total: int) -> None:
    width = 30
    ratio = 1.0 if total == 0 else min(1.0, completed / total)
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    ending = "\n" if completed >= total else "\r"
    print(
        f"Qwen inference [{bar}] {completed:,}/{total:,} ({ratio:.1%})",
        end=ending,
        file=sys.stderr,
        flush=True,
    )


def run_inference(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    model_name: str = "Qwen/Qwen3-0.6B",
    revision: str | None = None,
    batch_size: int = 4,
    max_input_tokens: int = 8_192,
    max_new_tokens: int = 512,
    local_files_only: bool = True,
    seed: int = 0,
) -> Path:
    if batch_size <= 0 or max_input_tokens <= 0 or max_new_tokens <= 0:
        raise ValueError("batch and token limits must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("raw-Qwen evaluation requires a CUDA GPU")
    inputs = Path(input_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    input_manifest = load_manifest(inputs)
    configuration = {
        "model_name": model_name,
        "revision": revision,
        "batch_size": batch_size,
        "max_input_tokens": max_input_tokens,
        "max_new_tokens": max_new_tokens,
        "local_files_only": local_files_only,
        "thinking": False,
        "do_sample": False,
        "seed": seed,
        "input_manifest_sha256": file_sha256(inputs / "manifest.json"),
    }
    existing_path = output / "manifest.json"
    progress_path = output / ".progress.json"
    existing: dict[str, Any] | None = None
    resume_path = existing_path if existing_path.exists() else progress_path
    if resume_path.exists():
        existing = json.loads(resume_path.read_text(encoding="utf-8"))
        if existing.get("configuration") != configuration:
            raise EvalArtifactError("existing run uses different inference configuration")

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Loading model and tokenizer: {model_name}", file=sys.stderr, flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        local_files_only=local_files_only,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=revision,
        local_files_only=local_files_only,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()

    prior = {
        entry["input_path"]: entry
        for entry in ([] if existing is None else existing.get("shards", []))
    }
    shard_entries: list[dict[str, Any]] = []
    total_records = int(input_manifest["records"])
    completed_records = 0
    last_reported = 0

    def advance_progress(count: int) -> None:
        nonlocal completed_records, last_reported
        completed_records += count
        if completed_records >= total_records or completed_records - last_reported >= 10:
            _print_progress(completed_records, total_records)
            last_reported = completed_records

    _print_progress(0, total_records)
    for input_entry in input_manifest["shards"]:
        input_relative = str(input_entry["path"])
        input_path = inputs / input_relative
        if file_sha256(input_path) != input_entry["sha256"]:
            raise EvalArtifactError(f"input shard checksum failed: {input_path}")
        destination = _prediction_path(output, input_path)
        old = prior.get(input_relative)
        if old is not None and destination.is_file() and (
            destination.stat().st_size == old.get("size")
            and file_sha256(destination) == old.get("sha256")
        ):
            shard_entries.append(old)
            advance_progress(int(old["records"]))
            continue
        examples = list(iter_eval_examples(input_path))
        records = _generate_shard(
            examples,
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size,
            max_input_tokens=max_input_tokens,
            max_new_tokens=max_new_tokens,
            progress_callback=advance_progress,
        )
        info = _write_prediction_shard(destination, records)
        entry = {
            "input_path": input_relative,
            "input_sha256": input_entry["sha256"],
            "path": str(destination.relative_to(output)),
            **info,
        }
        shard_entries.append(entry)
        partial_records = list(_iter_run_records(output, shard_entries))
        atomic_write_json(output / "partial_metrics.json", aggregate_prediction_records(partial_records))
        atomic_write_json(
            progress_path,
            {
                "manifest_version": RUN_MANIFEST_VERSION,
                "complete": False,
                "configuration": configuration,
                "shards": shard_entries,
            },
        )

    all_records = list(_iter_run_records(output, shard_entries))
    metrics = aggregate_prediction_records(all_records)
    atomic_write_json(output / "final_metrics.json", metrics)
    environment = _environment(model_name, revision)
    environment["resolved_model_commit"] = getattr(model.config, "_commit_hash", None)
    environment["resolved_tokenizer_commit"] = tokenizer.init_kwargs.get("_commit_hash")
    atomic_write_json(output / "environment.json", environment)
    manifest = {
        "manifest_version": RUN_MANIFEST_VERSION,
        "complete": True,
        "configuration": configuration,
        "input_records": input_manifest["records"],
        "prediction_records": len(all_records),
        "final_metrics_sha256": file_sha256(output / "final_metrics.json"),
        "environment_sha256": file_sha256(output / "environment.json"),
        "shards": shard_entries,
    }
    atomic_write_json(output / "manifest.json", manifest)
    progress_path.unlink(missing_ok=True)
    return output


def _iter_run_records(
    output: Path, entries: Sequence[Mapping[str, Any]]
) -> Iterator[dict[str, Any]]:
    for entry in entries:
        yield from iter_prediction_records(output / str(entry["path"]))
