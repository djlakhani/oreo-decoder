"""Schemas for native raw-Qwen reconstruction evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping

EVAL_SCHEMA_VERSION = "raw_qwen_reconstruction_eval_v1"
PREDICTION_SCHEMA_VERSION = "raw_qwen_reconstruction_prediction_v1"
PROMPT_VERSION = "native_blank_lines_one_shot_v1"


class RawEvalError(RuntimeError):
    """Base error for raw-Qwen evaluation artifacts."""


class EvalArtifactError(RawEvalError):
    """An evaluation input or prediction artifact is invalid."""


@dataclass(frozen=True, slots=True)
class EvalTarget:
    blank_index: int
    blank_token: str
    exact_text: str
    normalized_exact_text: str
    kind: str
    gold_qid: str | None
    gold_entity_id: int | None
    accepted_normalized_aliases: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvalExample:
    schema_version: str
    prompt_version: str
    example_id: str
    passage_qid: str
    passage_entity_id: int
    datastore_manifest_sha256: str
    source_record_sha256: str
    corrupted_passage: str
    system_prompt: str
    user_prompt: str
    targets: tuple[EvalTarget, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return stable_json(self.to_dict())

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EvalExample":
        return cls(
            schema_version=str(raw["schema_version"]),
            prompt_version=str(raw["prompt_version"]),
            example_id=str(raw["example_id"]),
            passage_qid=str(raw["passage_qid"]),
            passage_entity_id=int(raw["passage_entity_id"]),
            datastore_manifest_sha256=str(raw["datastore_manifest_sha256"]),
            source_record_sha256=str(raw["source_record_sha256"]),
            corrupted_passage=str(raw["corrupted_passage"]),
            system_prompt=str(raw["system_prompt"]),
            user_prompt=str(raw["user_prompt"]),
            targets=tuple(
                EvalTarget(
                    blank_index=int(item["blank_index"]),
                    blank_token=str(item["blank_token"]),
                    exact_text=str(item["exact_text"]),
                    normalized_exact_text=str(item["normalized_exact_text"]),
                    kind=str(item["kind"]),
                    gold_qid=item.get("gold_qid"),
                    gold_entity_id=(
                        None
                        if item.get("gold_entity_id") is None
                        else int(item["gold_entity_id"])
                    ),
                    accepted_normalized_aliases=tuple(
                        str(value) for value in item["accepted_normalized_aliases"]
                    ),
                )
                for item in raw["targets"]
            ),
        )

    @classmethod
    def from_json(cls, payload: str) -> "EvalExample":
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise EvalArtifactError("evaluation example JSON root must be an object")
        example = cls.from_dict(raw)
        validate_eval_example(example)
        return example


def stable_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def validate_eval_example(example: EvalExample) -> None:
    if example.schema_version != EVAL_SCHEMA_VERSION:
        raise EvalArtifactError("unsupported evaluation-example schema")
    if example.prompt_version != PROMPT_VERSION:
        raise EvalArtifactError("unsupported prompt version")
    if example.example_id != f"wiki5m:{example.passage_qid}":
        raise EvalArtifactError("example ID does not match passage QID")
    if not example.corrupted_passage or not example.system_prompt or not example.user_prompt:
        raise EvalArtifactError("evaluation prompt fields cannot be empty")
    if not example.targets:
        raise EvalArtifactError("evaluation example must contain at least one target")
    for index, target in enumerate(example.targets):
        if target.blank_index != index or target.blank_token != f"[BLANK_{index}]":
            raise EvalArtifactError("blank indices and tokens must be contiguous")
        if target.blank_token not in example.corrupted_passage:
            raise EvalArtifactError("blank token is absent from corrupted passage")
        if target.exact_text == "":
            raise EvalArtifactError("target exact text cannot be empty")
        if target.kind not in ("primary_kg", "auxiliary_kg", "context_only"):
            raise EvalArtifactError("unknown target kind")
        if target.kind == "context_only":
            if target.gold_qid is not None or target.gold_entity_id is not None:
                raise EvalArtifactError("context target cannot contain entity metadata")
            if target.accepted_normalized_aliases:
                raise EvalArtifactError("context target cannot contain entity aliases")
        elif target.gold_qid is None or target.gold_entity_id is None:
            raise EvalArtifactError("KG target is missing its gold entity identity")

