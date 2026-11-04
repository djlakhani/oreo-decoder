"""Deterministic target and context-blank selection helpers."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Iterable

from .aliases import normalize_alias

_WORD = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", re.UNICODE)
_CONTEXT_STOPWORDS = {
    "a", "an", "and", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "was",
    "were", "with",
}


@dataclass(frozen=True, slots=True)
class Span:
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class ContextBlank:
    start_char: int
    end_char: int
    exact_text: str
    normalized_text: str


def overlaps(first: Span, second: Span) -> bool:
    return first.start < second.end and second.start < first.end


def stable_rank(seed: str, value: str) -> tuple[str, str]:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest(), value


def select_context_blanks(
    passage: str,
    *,
    excluded_spans: Iterable[Span],
    count: int,
    min_tokens: int,
    max_tokens: int,
    seed: str,
) -> tuple[ContextBlank, ...]:
    """Choose repeated content spans while leaving one occurrence visible."""

    if count <= 0:
        return ()
    excluded = list(excluded_spans)
    words = list(_WORD.finditer(passage))
    occurrences: dict[str, list[Span]] = {}
    surfaces: dict[tuple[str, int, int], str] = {}
    for start_index in range(len(words)):
        for width in range(min_tokens, max_tokens + 1):
            stop_index = start_index + width
            if stop_index > len(words):
                break
            selected = words[start_index:stop_index]
            start, stop = selected[0].start(), selected[-1].end()
            between = passage[start:stop]
            if any(mark in between for mark in ("\n", ".", "?", "!")):
                continue
            span = Span(start, stop)
            if any(overlaps(span, blocked) for blocked in excluded):
                continue
            tokens = [normalize_alias(item.group()) for item in selected]
            if not any(len("".join(ch for ch in token if ch.isalnum())) >= 3
                       and token not in _CONTEXT_STOPWORDS for token in tokens):
                continue
            normalized = " ".join(tokens)
            occurrences.setdefault(normalized, []).append(span)
            surfaces[(normalized, start, stop)] = passage[start:stop]

    candidates: list[tuple[str, Span, tuple[Span, ...]]] = []
    for normalized, spans in occurrences.items():
        unique = sorted(set(spans), key=lambda item: (item.start, item.end))
        nonoverlapping: list[Span] = []
        for span in unique:
            if not any(overlaps(span, other) for other in nonoverlapping):
                nonoverlapping.append(span)
        if len(nonoverlapping) >= 2:
            # Mask one deterministically; another occurrence remains visible.
            chosen = min(
                nonoverlapping,
                key=lambda item: stable_rank(seed, f"{normalized}:{item.start}:{item.end}"),
            )
            candidates.append((normalized, chosen, tuple(nonoverlapping)))

    chosen_candidates: list[tuple[str, Span, tuple[Span, ...]]] = []
    for normalized, span, alternatives in sorted(
        candidates,
        key=lambda item: stable_rank(seed, f"context:{item[0]}:{item[1].start}"),
    ):
        if any(overlaps(span, item[1]) for item in chosen_candidates):
            continue
        tentative = [*chosen_candidates, (normalized, span, alternatives)]
        masks = [item[1] for item in tentative]
        # A second selected span can cover the only visible backup for an
        # earlier one (for example, selecting both "annual" and "annual
        # lecture"). Recheck the leave-one-visible invariant globally.
        if any(
            not any(
                all(not overlaps(occurrence, mask) for mask in masks)
                for occurrence in candidate_alternatives
            )
            for _, _, candidate_alternatives in tentative
        ):
            continue
        chosen_candidates.append((normalized, span, alternatives))
        if len(chosen_candidates) == count:
            break

    chosen_blanks = [
        ContextBlank(
            start_char=span.start,
            end_char=span.end,
            exact_text=surfaces[(normalized, span.start, span.end)],
            normalized_text=normalized,
        )
        for normalized, span, _ in chosen_candidates
    ]
    return tuple(sorted(chosen_blanks, key=lambda item: (item.start_char, item.end_char)))
