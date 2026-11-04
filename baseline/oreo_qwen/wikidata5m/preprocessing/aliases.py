"""Conservative, graph-guided alias matching with source character offsets."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Iterable, Mapping

from ..store import Wikidata5MStore
from .schema import qid_number

NORMALIZATION_VERSION = "nfkc_casefold_whitespace_v1"
STOP_ALIAS_VERSION = "conservative_english_v1"

# These aliases are far more likely to be ordinary prose than entity names.  The
# list is intentionally small and versioned in source; ambiguity is rejected
# rather than guessed.
STOP_ALIASES = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "by",
        "for",
        "from",
        "he",
        "her",
        "his",
        "i",
        "in",
        "it",
        "its",
        "of",
        "on",
        "or",
        "she",
        "the",
        "their",
        "they",
        "to",
        "was",
        "were",
        "with",
    }
)


@dataclass(frozen=True, slots=True)
class NormalizedText:
    text: str
    starts: tuple[int, ...]
    ends: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class AliasMatch:
    qid: str
    start_char: int
    end_char: int
    surface: str
    normalized_alias: str
    graph_depth: int


@dataclass(frozen=True, slots=True)
class MatchResult:
    matches: tuple[AliasMatch, ...]
    diagnostics: Mapping[str, int]


def normalized_text(value: str) -> NormalizedText:
    """Return a comparison string plus a map back to original code-point spans."""

    out: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    index = 0
    while index < len(value):
        stop = index + 1
        while stop < len(value) and unicodedata.combining(value[stop]):
            stop += 1
        folded = unicodedata.normalize("NFKC", value[index:stop]).casefold()
        for character in folded:
            if character.isspace():
                if out and out[-1] != " ":
                    out.append(" ")
                    starts.append(index)
                    ends.append(stop)
                elif out and out[-1] == " ":
                    ends[-1] = stop
            else:
                out.append(character)
                starts.append(index)
                ends.append(stop)
        index = stop
    if out and out[-1] == " ":
        out.pop()
        starts.pop()
        ends.pop()
    return NormalizedText("".join(out), tuple(starts), tuple(ends))


def normalize_alias(value: str) -> str:
    return normalized_text(value.strip()).text


def _looks_name_like(alias: str) -> bool:
    letters = [character for character in alias if character.isalpha()]
    if not letters:
        return False
    return any(character.isupper() for character in alias) or len(alias.split()) > 1


def _prepared_aliases(
    store: Wikidata5MStore,
    qid: str,
    *,
    is_subject: bool,
    limit: int,
) -> tuple[tuple[str, str], ...]:
    prepared: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw in store.entity_aliases(qid):
        alias = raw.strip()
        normalized = normalize_alias(alias)
        if not normalized or normalized in seen or normalized in STOP_ALIASES:
            continue
        if not is_subject and not _looks_name_like(alias):
            continue
        seen.add(normalized)
        prepared.append((alias, normalized))
        if len(prepared) == limit:
            break
    return tuple(prepared)


def _is_boundary(text: str, start: int, stop: int) -> bool:
    before = text[start - 1] if start else ""
    after = text[stop] if stop < len(text) else ""
    return (not before or not before.isalnum()) and (not after or not after.isalnum())


def match_graph_aliases(
    passage: str,
    candidate_depths: Mapping[str, int],
    subject_qid: str,
    store: Wikidata5MStore,
    *,
    max_aliases_per_entity: int,
) -> MatchResult:
    """Resolve aliases for only the supplied local graph candidates.

    Exact-span ambiguity is resolved only for the passage subject. Other
    ambiguous aliases are discarded. Overlaps use subject, longest span,
    shallowest graph depth, earliest position, then numeric QID precedence.
    """

    view = normalized_text(passage)
    by_alias: dict[str, set[str]] = {}
    for qid in sorted(candidate_depths, key=qid_number):
        for _, alias in _prepared_aliases(
            store,
            qid,
            is_subject=qid == subject_qid,
            limit=max_aliases_per_entity,
        ):
            by_alias.setdefault(alias, set()).add(qid)

    raw_matches: list[AliasMatch] = []
    ambiguous = 0
    for alias in sorted(by_alias, key=lambda item: (-len(item), item)):
        qids = by_alias[alias]
        cursor = 0
        while True:
            position = view.text.find(alias, cursor)
            if position < 0:
                break
            stop = position + len(alias)
            cursor = position + 1
            if not _is_boundary(view.text, position, stop):
                continue
            original_start = view.starts[position]
            original_stop = view.ends[stop - 1]
            if subject_qid in qids:
                selected = subject_qid
            elif len(qids) == 1:
                selected = next(iter(qids))
            else:
                ambiguous += 1
                continue
            raw_matches.append(
                AliasMatch(
                    qid=selected,
                    start_char=original_start,
                    end_char=original_stop,
                    surface=passage[original_start:original_stop],
                    normalized_alias=alias,
                    graph_depth=candidate_depths[selected],
                )
            )

    def priority(match: AliasMatch) -> tuple[int, int, int, int, int]:
        return (
            0 if match.qid == subject_qid else 1,
            -(match.end_char - match.start_char),
            match.graph_depth,
            match.start_char,
            qid_number(match.qid),
        )

    kept: list[AliasMatch] = []
    # Pick winners globally, then restore passage order. This makes a short
    # early alias lose to a longer overlapping entity name.
    for candidate in sorted(raw_matches, key=priority):
        if any(
            candidate.start_char < other.end_char
            and other.start_char < candidate.end_char
            for other in kept
        ):
            continue
        kept.append(candidate)
    kept.sort(key=lambda item: (item.start_char, item.end_char, qid_number(item.qid)))
    return MatchResult(
        tuple(kept),
        {
            "alias_candidates": len(by_alias),
            "raw_alias_matches": len(raw_matches),
            "ambiguous_alias_matches": ambiguous,
            "overlap_drops": len(raw_matches) - len(kept),
        },
    )


def mentioned_qids(matches: Iterable[AliasMatch]) -> set[str]:
    return {match.qid for match in matches}
