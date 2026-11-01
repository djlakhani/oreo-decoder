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
    "build_index",
    "validate_index",
]
