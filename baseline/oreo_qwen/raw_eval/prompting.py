"""Native-blank rendering and the versioned raw-Qwen prompt."""

from __future__ import annotations

from ..wikidata5m.preprocessing.schema import PreprocessedRecord, passage_sha256

SYSTEM_PROMPT = """You reconstruct missing text spans in a passage.

The passage contains numbered blanks such as [BLANK_0] and [BLANK_1].
Return exactly one answer line for every blank, in ascending numerical order.

Use this exact format:
[BLANK_0] missing text
[BLANK_1] missing text

Rules:
- Copy only the text that belongs in each blank.
- Do not repeat the passage.
- Do not explain your answers.
- Do not use bullets, JSON, Markdown, or code fences.
- Do not omit, rename, duplicate, or reorder blank identifiers.
- Use exactly one line per blank."""

_DEMONSTRATION = """FORMAT EXAMPLE

Passage:
The project began in [BLANK_0] and its code name was [BLANK_1].

Correct response:
[BLANK_0] 2024
[BLANK_1] Oriole"""


def render_native_passage(record: PreprocessedRecord, passage: str) -> str:
    """Replace selected spans with native textual blanks and add no KG markers."""

    if record.passage.text_sha256 != passage_sha256(passage):
        raise ValueError("passage text does not match preprocessing record")
    chunks: list[str] = []
    cursor = 0
    for occurrence in record.masked_occurrences:
        if occurrence.start_char < cursor:
            raise ValueError("masked occurrences overlap")
        chunks.append(passage[cursor : occurrence.start_char])
        chunks.append(f"[BLANK_{occurrence.sentinel_index}]")
        cursor = occurrence.end_char
    chunks.append(passage[cursor:])
    return "".join(chunks)


def build_user_prompt(corrupted_passage: str, blank_count: int) -> str:
    identifiers = "\n".join(f"[BLANK_{index}]" for index in range(blank_count))
    return (
        f"{_DEMONSTRATION}\n\n"
        "TASK\n\n"
        f"Passage:\n{corrupted_passage}\n\n"
        f"Required blank identifiers:\n{identifiers}\n\n"
        "Response:"
    )
