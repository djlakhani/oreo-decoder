"""Render an untokenized preprocessing record into model-facing text."""

from __future__ import annotations

from dataclasses import dataclass

from .schema import PreprocessedRecord, RenderedExample, passage_sha256

KG_MARKERS = " <|rel|> <|t_ent|>"


@dataclass(frozen=True, slots=True)
class _Edit:
    start: int
    end: int
    replacement: str
    kind: str


def render_example(record: PreprocessedRecord, passage_text: str) -> RenderedExample:
    """Apply masks and post-mention marker insertions without changing offsets."""

    if record.passage.text_sha256 != passage_sha256(passage_text):
        raise ValueError("passage text does not match preprocessing record")
    edits: list[_Edit] = [
        _Edit(item.start_char, item.end_char, item.sentinel_token, "mask")
        for item in record.masked_occurrences
    ]
    mention_by_id = {item.mention_id: item for item in record.mentions}
    edits.extend(
        _Edit(
            mention_by_id[mention_id].end_char,
            mention_by_id[mention_id].end_char,
            KG_MARKERS,
            "marker",
        )
        for mention_id in record.marker_mention_ids
    )
    # An insertion at the same boundary as a following replacement belongs to
    # the visible mention on its left and must be emitted first.
    edits.sort(key=lambda item: (item.start, 0 if item.kind == "marker" else 1, item.end))

    chunks: list[str] = []
    cursor = 0
    for edit in edits:
        if edit.start < cursor:
            raise ValueError("render edits overlap")
        chunks.append(passage_text[cursor:edit.start])
        chunks.append(edit.replacement)
        cursor = edit.end
    chunks.append(passage_text[cursor:])

    suffix = "\n".join(
        f"{item.sentinel_token} {item.exact_text}"
        for item in record.masked_occurrences
    )
    return RenderedExample(
        corrupted_passage="".join(chunks),
        reconstruction_suffix=suffix,
    )
