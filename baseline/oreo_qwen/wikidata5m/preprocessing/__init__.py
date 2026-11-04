"""Wikidata5M passage preprocessing public API."""

from .aliases import NORMALIZATION_VERSION, normalize_alias
from .pipeline import (
    build_preprocessed_corpus,
    preprocess_passage,
    validate_preprocessed_corpus,
)
from .render import render_example
from .schema import (
    CorpusValidationError,
    PreprocessConfig,
    PreprocessedRecord,
    PreprocessingError,
    PreprocessResult,
    RecordValidationError,
    RenderedExample,
    SYSTEM_INSTRUCTION,
    validate_preprocessed_record,
)

__all__ = [
    "CorpusValidationError",
    "NORMALIZATION_VERSION",
    "PreprocessConfig",
    "PreprocessedRecord",
    "PreprocessingError",
    "PreprocessResult",
    "RecordValidationError",
    "RenderedExample",
    "SYSTEM_INSTRUCTION",
    "build_preprocessed_corpus",
    "normalize_alias",
    "preprocess_passage",
    "render_example",
    "validate_preprocessed_corpus",
    "validate_preprocessed_record",
]
