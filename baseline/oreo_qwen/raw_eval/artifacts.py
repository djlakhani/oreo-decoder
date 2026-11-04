"""Deterministic JSON/JSONL artifact helpers."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Iterator

from .schema import EvalArtifactError, EvalExample, stable_json


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (stable_json(value) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_eval_shard(path: Path, examples: Iterable[EvalExample]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    count = 0
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
                for example in examples:
                    zipped.write((example.to_json() + "\n").encode("utf-8"))
                    count += 1
        validated = sum(1 for _ in iter_eval_examples(temporary))
        if validated != count:
            raise EvalArtifactError("temporary eval shard record count is wrong")
        checksum = file_sha256(temporary)
        size = temporary.stat().st_size
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {"records": count, "size": size, "sha256": checksum}


def iter_eval_examples(path: str | Path) -> Iterator[EvalExample]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                yield EvalExample.from_json(line)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise EvalArtifactError(
                    f"invalid evaluation record at {path}:{line_number}: {error}"
                ) from error


def load_manifest(directory: str | Path) -> dict[str, Any]:
    path = Path(directory) / "manifest.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvalArtifactError(f"cannot read manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvalArtifactError("manifest root must be an object")
    return value


def validate_eval_dataset(directory: str | Path) -> dict[str, Any]:
    root = Path(directory).expanduser().resolve()
    manifest = load_manifest(root)
    if manifest.get("complete") is not True:
        raise EvalArtifactError("evaluation-input build is incomplete")
    total = 0
    expected_start = 0
    for entry in manifest.get("shards", []):
        if int(entry["start"]) != expected_start:
            raise EvalArtifactError("evaluation shard ranges are not contiguous")
        expected_start = int(entry["stop"])
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root):
            raise EvalArtifactError("evaluation shard path escapes dataset root")
        if not path.is_file() or path.stat().st_size != entry["size"]:
            raise EvalArtifactError(f"missing or truncated evaluation shard: {path}")
        if file_sha256(path) != entry["sha256"]:
            raise EvalArtifactError(f"evaluation shard checksum failed: {path}")
        count = sum(1 for _ in iter_eval_examples(path))
        if count != entry["records"] or count != int(entry["stop"]) - int(entry["start"]):
            raise EvalArtifactError(f"evaluation shard count is wrong: {path}")
        total += count
    if expected_start != manifest.get("records") or total != manifest.get("records"):
        raise EvalArtifactError("evaluation manifest record count is wrong")
    return manifest

