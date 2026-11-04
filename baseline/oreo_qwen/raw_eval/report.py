"""Standalone rescoring of saved raw-Qwen generations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json, file_sha256, load_manifest
from .inference import iter_prediction_records
from .schema import EvalArtifactError
from .scoring import aggregate_prediction_records, rescore_prediction_record

SCORER_VERSION = "raw_qwen_reconstruction_scorer_v1"


def rescore_run(run_dir: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    run = Path(run_dir).expanduser().resolve()
    manifest = load_manifest(run)
    if manifest.get("complete") is not True:
        raise EvalArtifactError("cannot authoritatively score an incomplete run")
    rescored: list[dict[str, Any]] = []
    for entry in manifest["shards"]:
        path = run / entry["path"]
        if path.stat().st_size != entry["size"] or file_sha256(path) != entry["sha256"]:
            raise EvalArtifactError(f"prediction shard checksum failed: {path}")
        for record in iter_prediction_records(path):
            score = rescore_prediction_record(record)
            rescored.append({**record, "score": score})
    metrics = {
        "scorer_version": SCORER_VERSION,
        "run_manifest_sha256": file_sha256(run / "manifest.json"),
        "metrics": aggregate_prediction_records(rescored),
    }
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else run / "rescored_metrics.json"
    )
    atomic_write_json(destination, metrics)
    return metrics
