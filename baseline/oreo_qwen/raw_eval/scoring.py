"""Strict parsing, recovery parsing, and deterministic reconstruction metrics."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..wikidata5m.preprocessing import normalize_alias
from .schema import EvalExample, EvalTarget

_STRICT_LINE = re.compile(r"^\[BLANK_(\d+)\] (.+)$")
_RECOVERY_LINE = re.compile(
    r"^\s*(?:[-*]\s*)?\[BLANK_(\d+)\]\s*:?[ \t]+(.+?)\s*$"
)


@dataclass(frozen=True, slots=True)
class ParseResult:
    valid: bool
    answers: Mapping[int, str]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "answers": {str(key): value for key, value in sorted(self.answers.items())},
            "error": self.error,
        }


def strict_parse(text: str, expected_count: int) -> ParseResult:
    if text.endswith("\n"):
        text = text[:-1]
    if not text or text.startswith("\n") or text.endswith("\n"):
        return ParseResult(False, {}, "empty response or extra blank line")
    lines = text.split("\n")
    if len(lines) != expected_count:
        return ParseResult(False, {}, "wrong number of response lines")
    answers: dict[int, str] = {}
    observed: list[int] = []
    for line in lines:
        match = _STRICT_LINE.fullmatch(line)
        if match is None:
            return ParseResult(False, answers, "line violates strict blank grammar")
        index = int(match.group(1))
        answer = match.group(2)
        if index in answers:
            return ParseResult(False, answers, "duplicate blank identifier")
        observed.append(index)
        answers[index] = answer
    expected = list(range(expected_count))
    if observed != expected:
        return ParseResult(False, answers, "blank identifiers are missing or out of order")
    return ParseResult(True, answers)


def recovery_parse(text: str, expected_count: int) -> ParseResult:
    answers: dict[int, str] = {}
    duplicates = False
    for line in text.splitlines():
        match = _RECOVERY_LINE.fullmatch(line)
        if match is None:
            continue
        index = int(match.group(1))
        if not 0 <= index < expected_count:
            continue
        if index in answers:
            duplicates = True
            continue
        answers[index] = match.group(2)
    valid = len(answers) == expected_count and not duplicates
    error = None if valid else "could not recover every blank exactly once"
    return ParseResult(valid, answers, error)


def _target_score(target: Any, prediction: str | None, *, parse_valid: bool) -> dict[str, Any]:
    normalized_prediction = None if prediction is None else normalize_alias(prediction)
    exact = bool(parse_valid and prediction == target.exact_text)
    normalized = bool(
        parse_valid and normalized_prediction == target.normalized_exact_text
    )
    alias_aware: bool | None
    if target.gold_qid is None:
        alias_aware = None
    else:
        alias_aware = bool(
            parse_valid
            and (
                exact
                or normalized_prediction in set(target.accepted_normalized_aliases)
            )
        )
    return {
        "blank_index": target.blank_index,
        "blank_token": target.blank_token,
        "kind": target.kind,
        "gold_qid": target.gold_qid,
        "gold_entity_id": target.gold_entity_id,
        "exact_text": target.exact_text,
        "prediction": prediction,
        "normalized_prediction": normalized_prediction,
        "parsed": prediction is not None,
        "exact": exact,
        "normalized": normalized,
        "alias_aware": alias_aware,
    }


def score_generation(example: EvalExample, generation: str) -> dict[str, Any]:
    return score_targets(example.targets, generation)


def score_targets(targets: Sequence[Any], generation: str) -> dict[str, Any]:
    strict = strict_parse(generation, len(targets))
    recovery = recovery_parse(generation, len(targets))
    strict_targets = [
        _target_score(
            target,
            strict.answers.get(target.blank_index),
            parse_valid=strict.valid,
        )
        for target in targets
    ]
    recovery_targets = [
        _target_score(
            target,
            recovery.answers.get(target.blank_index),
            parse_valid=recovery.valid,
        )
        for target in targets
    ]
    kg_strict = [item for item in strict_targets if item["alias_aware"] is not None]
    return {
        "strict_parse": strict.to_dict(),
        "recovery_parse": recovery.to_dict(),
        "strict_targets": strict_targets,
        "recovery_targets": recovery_targets,
        "complete_exact": strict.valid and all(item["exact"] for item in strict_targets),
        "complete_normalized": strict.valid
        and all(item["normalized"] for item in strict_targets),
        "complete_alias_aware": strict.valid
        and all(item["alias_aware"] for item in kg_strict)
        and all(
            item["normalized"]
            for item in strict_targets
            if item["alias_aware"] is None
        ),
    }


def rescore_prediction_record(record: Mapping[str, Any]) -> dict[str, Any] | None:
    if record.get("status") != "generated" or record.get("generated_text") is None:
        return None
    targets = tuple(
        EvalTarget(
            blank_index=int(item["blank_index"]),
            blank_token=str(item["blank_token"]),
            exact_text=str(item["exact_text"]),
            normalized_exact_text=str(item["normalized_exact_text"]),
            kind=str(item["kind"]),
            gold_qid=item.get("gold_qid"),
            gold_entity_id=(
                None if item.get("gold_entity_id") is None else int(item["gold_entity_id"])
            ),
            accepted_normalized_aliases=tuple(item["accepted_normalized_aliases"]),
        )
        for item in record["targets"]
    )
    return score_targets(targets, str(record["generated_text"]))


def aggregate_prediction_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    response_counts: Counter[str] = Counter()
    target_counts: dict[str, Counter[str]] = defaultdict(Counter)
    input_lengths: list[int] = []
    generation_lengths: list[int] = []
    for record in records:
        response_counts["examples"] += 1
        status = str(record.get("status", "unknown"))
        response_counts[f"status:{status}"] += 1
        if status != "generated":
            continue
        score = record["score"]
        response_counts["strict_format_valid"] += bool(score["strict_parse"]["valid"])
        response_counts["recovery_format_valid"] += bool(
            score["recovery_parse"]["valid"]
        )
        response_counts["complete_exact"] += bool(score["complete_exact"])
        response_counts["complete_normalized"] += bool(score["complete_normalized"])
        response_counts["complete_alias_aware"] += bool(
            score["complete_alias_aware"]
        )
        if record.get("input_tokens") is not None:
            input_lengths.append(int(record["input_tokens"]))
        if record.get("generated_tokens") is not None:
            generation_lengths.append(int(record["generated_tokens"]))
        for target in score["strict_targets"]:
            counters = target_counts[str(target["kind"])]
            counters["targets"] += 1
            counters["parsed"] += bool(target["parsed"])
            counters["exact"] += bool(target["exact"])
            counters["normalized"] += bool(target["normalized"])
            if target["alias_aware"] is not None:
                counters["alias_targets"] += 1
                counters["alias_aware"] += bool(target["alias_aware"])

    def rate(numerator: int, denominator: int) -> float | None:
        return None if denominator == 0 else numerator / denominator

    examples = response_counts["examples"]
    generated = response_counts["status:generated"]
    response_metrics = {
        "examples": examples,
        "generated": generated,
        "statuses": {
            key.removeprefix("status:"): value
            for key, value in sorted(response_counts.items())
            if key.startswith("status:")
        },
        "strict_format_valid": response_counts["strict_format_valid"],
        "strict_format_valid_rate": rate(
            response_counts["strict_format_valid"], generated
        ),
        "recovery_format_valid": response_counts["recovery_format_valid"],
        "recovery_format_valid_rate": rate(
            response_counts["recovery_format_valid"], generated
        ),
        "complete_exact": response_counts["complete_exact"],
        "complete_exact_rate": rate(response_counts["complete_exact"], examples),
        "complete_normalized": response_counts["complete_normalized"],
        "complete_normalized_rate": rate(
            response_counts["complete_normalized"], examples
        ),
        "complete_alias_aware": response_counts["complete_alias_aware"],
        "complete_alias_aware_rate": rate(
            response_counts["complete_alias_aware"], examples
        ),
    }
    by_kind: dict[str, Any] = {}
    for kind, counts in sorted(target_counts.items()):
        by_kind[kind] = {
            **dict(counts),
            "parsed_rate": rate(counts["parsed"], counts["targets"]),
            "exact_rate": rate(counts["exact"], counts["targets"]),
            "normalized_rate": rate(counts["normalized"], counts["targets"]),
            "alias_aware_rate": rate(counts["alias_aware"], counts["alias_targets"]),
        }
    return {
        "response": response_metrics,
        "targets_by_kind": by_kind,
        "input_tokens": _length_summary(input_lengths),
        "generated_tokens": _length_summary(generation_lengths),
    }


def _length_summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }
