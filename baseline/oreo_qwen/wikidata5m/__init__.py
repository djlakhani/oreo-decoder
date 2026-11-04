"""Persistent, queryable storage for Wikidata5M."""

from .builder import build_index
from .errors import (
    CorruptIndexError,
    IndexBuildError,
    MalformedSourceError,
    MissingDataError,
    StaleIndexError,
    UnknownEntityError,
    UnknownRelationError,
    Wikidata5MError,
)
from .store import Wikidata5MStore, validate_index
from .preprocessing import (
    PreprocessConfig,
    PreprocessedRecord,
    PreprocessResult,
    build_preprocessed_corpus,
    preprocess_passage,
    render_example,
    validate_preprocessed_corpus,
    validate_preprocessed_record,
)
from .types import (
    DenseEdgeBatch,
    Edge,
    EdgeDirection,
    EntityMapping,
    GraphSplit,
    QueryDirection,
    RelationMapping,
)

__all__ = [
    "CorruptIndexError",
    "DenseEdgeBatch",
    "Edge",
    "EdgeDirection",
    "EntityMapping",
    "GraphSplit",
    "IndexBuildError",
    "MalformedSourceError",
    "MissingDataError",
    "QueryDirection",
    "RelationMapping",
    "StaleIndexError",
    "UnknownEntityError",
    "UnknownRelationError",
    "Wikidata5MError",
    "Wikidata5MStore",
    "PreprocessConfig",
    "PreprocessedRecord",
    "PreprocessResult",
    "build_index",
    "build_preprocessed_corpus",
    "preprocess_passage",
    "render_example",
    "validate_index",
    "validate_preprocessed_corpus",
    "validate_preprocessed_record",
]
