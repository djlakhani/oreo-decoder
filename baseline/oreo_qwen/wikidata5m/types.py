"""Public types for querying a Wikidata5M index."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum

import numpy as np


class GraphSplit(str, Enum):
    """A Wikidata5M transductive graph split."""

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


SPLIT_TO_CODE = {
    GraphSplit.TRAIN: 0,
    GraphSplit.VALID: 1,
    GraphSplit.TEST: 2,
}
CODE_TO_SPLIT = {value: key for key, value in SPLIT_TO_CODE.items()}


class QueryDirection(str, Enum):
    """Which adjacency lists should be queried."""

    OUTGOING = "outgoing"
    INCOMING = "incoming"
    BOTH = "both"


class EdgeDirection(IntEnum):
    """The orientation of a returned edge relative to the queried entity."""

    OUTGOING = 0
    INCOMING = 1
    SELF = 2


@dataclass(frozen=True)
class Edge:
    """A decoded graph edge."""

    head_qid: str
    relation_pid: str
    tail_qid: str
    split: GraphSplit
    direction: EdgeDirection


@dataclass(frozen=True)
class DenseEdgeBatch:
    """Vectorized dense-ID graph results.

    All arrays have equal length. Filtering can require materialized arrays, so
    callers should treat them as read-only even when NumPy reports otherwise.
    """

    head_ids: np.ndarray
    relation_ids: np.ndarray
    tail_ids: np.ndarray
    split_codes: np.ndarray
    direction_codes: np.ndarray

    def __len__(self) -> int:
        return int(self.relation_ids.shape[0])


@dataclass(frozen=True)
class EntityMapping:
    """A deterministic QID/dense-ID mapping entry."""

    entity_id: int
    qid: str


@dataclass(frozen=True)
class RelationMapping:
    """A deterministic PID/dense-ID mapping entry."""

    relation_id: int
    pid: str
