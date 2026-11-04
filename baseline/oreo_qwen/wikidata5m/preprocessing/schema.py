"""Versioned schemas and validation for Wikidata5M reconstruction examples."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from ..errors import Wikidata5MError

SCHEMA_VERSION = "wiki5m_reconstruction_v1"
SYSTEM_INSTRUCTION = (
    "Reconstruct each masked span. Return only each sentinel followed by its "
    "original text, in sentinel order."
)


class PreprocessingError(Wikidata5MError):
    """Base class for preprocessing failures."""


class RecordValidationError(PreprocessingError):
    """A generated reconstruction record violates the v1 schema."""


class CorpusValidationError(PreprocessingError):
    """A sharded preprocessing corpus is stale or corrupt."""


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    """All behavior that contributes to deterministic v1 output."""

    schema_version: str = SCHEMA_VERSION
    graph_split: str = "train"
    max_incoming_degree: int = 1_000
    max_aliases_per_entity: int = 64
    max_auxiliary_targets: int = 2
    required_primary_supporters: int = 2
    required_auxiliary_supporters: int = 1
    max_context_blanks: int = 2
    min_context_ngram_tokens: int = 1
    max_context_ngram_tokens: int = 4
    max_sentinels: int = 16
    shard_entity_count: int = 10_000
    audit_example_limit: int = 100

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported preprocessing schema: {self.schema_version}")
        if self.graph_split != "train":
            raise ValueError("Wikidata5M reconstruction preprocessing is train-only")
        positive = (
            "max_incoming_degree",
            "max_aliases_per_entity",
            "required_primary_supporters",
            "required_auxiliary_supporters",
            "min_context_ngram_tokens",
            "max_context_ngram_tokens",
            "max_sentinels",
            "shard_entity_count",
            "audit_example_limit",
        )
        for name in positive:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        nonnegative = ("max_auxiliary_targets", "max_context_blanks")
        for name in nonnegative:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.min_context_ngram_tokens > self.max_context_ngram_tokens:
            raise ValueError("minimum context n-gram length exceeds maximum")
        if self.max_sentinels > 16:
            raise ValueError("the tokenizer contract provides only 16 mask sentinels")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class PassageMetadata:
    qid: str
    entity_id: int
    text_sha256: str
    character_count: int


@dataclass(frozen=True, slots=True)
class ResolvedMention:
    mention_id: int
    qid: str
    entity_id: int
    start_char: int
    end_char: int
    surface: str
    matched_normalized_alias: str
    graph_depth: int
    masked: bool


@dataclass(frozen=True, slots=True)
class LocalNode:
    local_node_id: int
    qid: str
    entity_id: int


@dataclass(frozen=True, slots=True)
class LocalEdge:
    source_local_node_id: int
    destination_local_node_id: int
    relation_id: int
    relation_pid: str
    split: str
    depth: int


@dataclass(frozen=True, slots=True)
class LocalGraph:
    nodes: tuple[LocalNode, ...]
    edges: tuple[LocalEdge, ...]


@dataclass(frozen=True, slots=True)
class TargetGroup:
    target_group_id: int
    kind: str
    qid: str
    entity_id: int
    occurrence_mention_ids: tuple[int, ...]
    visible_supporter_qids: tuple[str, ...]
    support_edge_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class MaskedOccurrence:
    sentinel_index: int
    sentinel_token: str
    start_char: int
    end_char: int
    exact_text: str
    kind: str
    target_group_id: int | None
    gold_qid: str | None
    gold_entity_id: int | None


@dataclass(frozen=True, slots=True)
class EvaluationTarget:
    sentinel_index: int
    exact_text: str
    gold_qid: str | None
    kind: str


@dataclass(frozen=True, slots=True)
class EvaluationMetadata:
    targets: tuple[EvaluationTarget, ...]
    local_candidate_qids: tuple[str, ...]
    datastore_manifest_sha256: str


@dataclass(frozen=True, slots=True)
class PreprocessedRecord:
    schema_version: str
    example_id: str
    passage: PassageMetadata
    mentions: tuple[ResolvedMention, ...]
    local_graph: LocalGraph
    target_groups: tuple[TargetGroup, ...]
    masked_occurrences: tuple[MaskedOccurrence, ...]
    marker_mention_ids: tuple[int, ...]
    evaluation: EvaluationMetadata

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "PreprocessedRecord":
        passage = PassageMetadata(**raw["passage"])
        mentions = tuple(ResolvedMention(**item) for item in raw["mentions"])
        graph_raw = raw["local_graph"]
        graph = LocalGraph(
            nodes=tuple(LocalNode(**item) for item in graph_raw["nodes"]),
            edges=tuple(LocalEdge(**item) for item in graph_raw["edges"]),
        )
        groups = tuple(
            TargetGroup(
                **{
                    **item,
                    "occurrence_mention_ids": tuple(item["occurrence_mention_ids"]),
                    "visible_supporter_qids": tuple(item["visible_supporter_qids"]),
                    "support_edge_indices": tuple(item["support_edge_indices"]),
                }
            )
            for item in raw["target_groups"]
        )
        occurrences = tuple(
            MaskedOccurrence(**item) for item in raw["masked_occurrences"]
        )
        evaluation_raw = raw["evaluation"]
        evaluation = EvaluationMetadata(
            targets=tuple(
                EvaluationTarget(**item) for item in evaluation_raw["targets"]
            ),
            local_candidate_qids=tuple(evaluation_raw["local_candidate_qids"]),
            datastore_manifest_sha256=evaluation_raw["datastore_manifest_sha256"],
        )
        return cls(
            schema_version=raw["schema_version"],
            example_id=raw["example_id"],
            passage=passage,
            mentions=mentions,
            local_graph=graph,
            target_groups=groups,
            masked_occurrences=occurrences,
            marker_mention_ids=tuple(raw["marker_mention_ids"]),
            evaluation=evaluation,
        )

    @classmethod
    def from_json(cls, payload: str) -> "PreprocessedRecord":
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise RecordValidationError("record JSON root must be an object")
        return cls.from_dict(raw)


@dataclass(frozen=True, slots=True)
class RenderedExample:
    corrupted_passage: str
    reconstruction_suffix: str
    system_instruction: str = SYSTEM_INSTRUCTION


@dataclass(frozen=True, slots=True)
class PreprocessResult:
    passage_qid: str
    passage_entity_id: int
    record: PreprocessedRecord | None = None
    rejection_reason: str | None = None
    diagnostics: Mapping[str, int] = field(default_factory=dict)
    inspection: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (self.record is None) == (self.rejection_reason is None):
            raise ValueError("result must contain exactly one of record or rejection reason")

    @property
    def accepted(self) -> bool:
        return self.record is not None


def passage_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def qid_number(qid: str) -> int:
    if len(qid) < 2 or qid[0] != "Q" or not qid[1:].isdigit():
        raise RecordValidationError(f"invalid QID: {qid!r}")
    return int(qid[1:])


def validate_preprocessed_record(
    record: PreprocessedRecord,
    passage_text: str,
    *,
    store: Any | None = None,
    expected_datastore_manifest_sha256: str | None = None,
) -> None:
    """Validate one full-passage record against its canonical source text."""

    if record.schema_version != SCHEMA_VERSION:
        raise RecordValidationError("record schema version is unsupported")
    if record.example_id != f"wiki5m:{record.passage.qid}":
        raise RecordValidationError("example ID does not match passage QID")
    if record.passage.character_count != len(passage_text):
        raise RecordValidationError("passage character count is stale")
    if record.passage.text_sha256 != passage_sha256(passage_text):
        raise RecordValidationError("passage text hash is stale")
    if expected_datastore_manifest_sha256 is not None and (
        record.evaluation.datastore_manifest_sha256
        != expected_datastore_manifest_sha256
    ):
        raise RecordValidationError("record datastore fingerprint is stale")

    mentions = record.mentions
    if tuple(item.mention_id for item in mentions) != tuple(range(len(mentions))):
        raise RecordValidationError("mention IDs must be contiguous in passage order")
    if tuple((m.start_char, m.end_char) for m in mentions) != tuple(
        sorted((m.start_char, m.end_char) for m in mentions)
    ):
        raise RecordValidationError("mentions are not ordered by character span")
    previous_end = 0
    for mention in mentions:
        if mention.start_char < previous_end or mention.end_char <= mention.start_char:
            raise RecordValidationError("mention spans overlap or are empty")
        if passage_text[mention.start_char : mention.end_char] != mention.surface:
            raise RecordValidationError("mention surface does not match passage")
        if mention.graph_depth not in (0, 1, 2):
            raise RecordValidationError("mention graph depth must be 0, 1, or 2")
        previous_end = mention.end_char
        if store is not None:
            if store.entity_id(mention.qid) != mention.entity_id:
                raise RecordValidationError("mention QID/dense-ID mapping is stale")

    occurrences = record.masked_occurrences
    if not occurrences or len(occurrences) > 16:
        raise RecordValidationError("record must contain between 1 and 16 masks")
    if tuple(item.sentinel_index for item in occurrences) != tuple(
        range(len(occurrences))
    ):
        raise RecordValidationError("sentinel indices must be contiguous")
    if tuple((m.start_char, m.end_char) for m in occurrences) != tuple(
        sorted((m.start_char, m.end_char) for m in occurrences)
    ):
        raise RecordValidationError("masked occurrences are not in passage order")
    previous_end = 0
    for occurrence in occurrences:
        if occurrence.sentinel_token != f"<|kg_mask_{occurrence.sentinel_index}|>":
            raise RecordValidationError("sentinel token does not match sentinel index")
        if occurrence.start_char < previous_end or occurrence.end_char <= occurrence.start_char:
            raise RecordValidationError("masked occurrence spans overlap or are empty")
        if passage_text[occurrence.start_char : occurrence.end_char] != occurrence.exact_text:
            raise RecordValidationError("mask answer does not match source passage")
        if occurrence.kind not in ("primary_kg", "auxiliary_kg", "context_only"):
            raise RecordValidationError("unknown mask kind")
        if occurrence.kind == "context_only":
            if any(
                value is not None
                for value in (
                    occurrence.target_group_id,
                    occurrence.gold_qid,
                    occurrence.gold_entity_id,
                )
            ):
                raise RecordValidationError("context-only mask has KG metadata")
        elif occurrence.gold_qid is None or occurrence.gold_entity_id is None:
            raise RecordValidationError("KG mask is missing entity metadata")
        previous_end = occurrence.end_char

    masked_mentions = {
        mention.mention_id
        for mention in mentions
        if mention.masked
    }
    marker_ids = record.marker_mention_ids
    expected_markers = tuple(
        mention.mention_id for mention in mentions if not mention.masked
    )
    if marker_ids != expected_markers:
        raise RecordValidationError("marker IDs must be every visible mention in order")

    groups = record.target_groups
    if (
        not groups
        or len(groups) > 3
        or groups[0].kind != "primary_kg"
        or groups[0].qid != record.passage.qid
        or any(group.kind != "auxiliary_kg" for group in groups[1:])
    ):
        raise RecordValidationError("target group 0 must be the primary target")
    if tuple(group.target_group_id for group in groups) != tuple(range(len(groups))):
        raise RecordValidationError("target group IDs must be contiguous")
    all_target_qids = {group.qid for group in groups}
    visible_mention_qids = {mention.qid for mention in mentions if not mention.masked}
    for group in groups:
        if group.kind not in ("primary_kg", "auxiliary_kg"):
            raise RecordValidationError("unknown target group kind")
        if not group.occurrence_mention_ids:
            raise RecordValidationError("target group has no mention occurrence")
        if not set(group.occurrence_mention_ids).issubset(masked_mentions):
            raise RecordValidationError("target group contains a visible occurrence")
        expected_occurrences = tuple(
            mention.mention_id for mention in mentions if mention.qid == group.qid
        )
        if group.occurrence_mention_ids != expected_occurrences:
            raise RecordValidationError("target group does not contain every QID occurrence")
        required_supporters = 2 if group.kind == "primary_kg" else 1
        if len(group.visible_supporter_qids) < required_supporters:
            raise RecordValidationError("target group has insufficient visible support")
        if set(group.visible_supporter_qids) & all_target_qids:
            raise RecordValidationError("a masked target cannot be a visible supporter")
        if not set(group.visible_supporter_qids).issubset(visible_mention_qids):
            raise RecordValidationError("supporter QID has no visible mention")
        for mention_id in group.occurrence_mention_ids:
            if not 0 <= mention_id < len(mentions) or mentions[mention_id].qid != group.qid:
                raise RecordValidationError("target group occurrence has the wrong QID")

    groups_by_id = {group.target_group_id: group for group in groups}
    for occurrence in occurrences:
        if occurrence.kind == "context_only":
            if any(
                occurrence.start_char < mention.end_char
                and mention.start_char < occurrence.end_char
                for mention in mentions
            ):
                raise RecordValidationError("context-only mask overlaps an entity mention")
            continue
        if occurrence.target_group_id not in groups_by_id:
            raise RecordValidationError("KG mask refers to an unknown target group")
        group = groups_by_id[occurrence.target_group_id]
        if (
            occurrence.kind != group.kind
            or occurrence.gold_qid != group.qid
            or occurrence.gold_entity_id != group.entity_id
        ):
            raise RecordValidationError("KG mask does not match its target group")
        if not any(
            mention.qid == group.qid
            and mention.start_char == occurrence.start_char
            and mention.end_char == occurrence.end_char
            for mention in mentions
        ):
            raise RecordValidationError("KG mask does not correspond to a resolved mention")

    graph = record.local_graph
    if tuple(node.local_node_id for node in graph.nodes) != tuple(range(len(graph.nodes))):
        raise RecordValidationError("local node IDs must be contiguous")
    if tuple(node.entity_id for node in graph.nodes) != tuple(
        sorted(node.entity_id for node in graph.nodes)
    ):
        raise RecordValidationError("local nodes must be sorted by global entity ID")
    edge_keys = tuple(
        (edge.source_local_node_id, edge.destination_local_node_id, edge.relation_id)
        for edge in graph.edges
    )
    if edge_keys != tuple(sorted(set(edge_keys))):
        raise RecordValidationError("local graph edges must be sorted and unique")
    for edge in graph.edges:
        if edge.split != "train" or edge.depth not in (1, 2):
            raise RecordValidationError("local graph edge has invalid split or depth")
        if not (0 <= edge.source_local_node_id < len(graph.nodes)) or not (
            0 <= edge.destination_local_node_id < len(graph.nodes)
        ):
            raise RecordValidationError("local graph edge endpoint is out of range")

    for group in groups:
        indexed_supporters: set[str] = set()
        for edge_index in group.support_edge_indices:
            if not 0 <= edge_index < len(graph.edges):
                raise RecordValidationError("support edge index is out of range")
            edge = graph.edges[edge_index]
            destination = graph.nodes[edge.destination_local_node_id]
            source = graph.nodes[edge.source_local_node_id]
            if destination.qid != group.qid or source.qid not in group.visible_supporter_qids:
                raise RecordValidationError("support edge does not enter its target group")
            indexed_supporters.add(source.qid)
        if indexed_supporters != set(group.visible_supporter_qids):
            raise RecordValidationError("support edge indices do not cover every supporter")

    evaluation = record.evaluation
    if len(evaluation.targets) != len(occurrences):
        raise RecordValidationError("evaluation target count does not match masks")
    for target, occurrence in zip(evaluation.targets, occurrences):
        if (
            target.sentinel_index != occurrence.sentinel_index
            or target.exact_text != occurrence.exact_text
            or target.gold_qid != occurrence.gold_qid
            or target.kind != occurrence.kind
        ):
            raise RecordValidationError("evaluation target does not match mask")
    expected_candidates = tuple(sorted((node.qid for node in graph.nodes), key=qid_number))
    if evaluation.local_candidate_qids != expected_candidates:
        raise RecordValidationError("local candidate QIDs do not match graph nodes")

    if store is not None:
        if store.entity_id(record.passage.qid) != record.passage.entity_id:
            raise RecordValidationError("passage QID/dense-ID mapping is stale")
        for node in graph.nodes:
            if store.entity_id(node.qid) != node.entity_id:
                raise RecordValidationError("local node QID/dense-ID mapping is stale")
        for edge in graph.edges:
            if store.relation_id(edge.relation_pid) != edge.relation_id:
                raise RecordValidationError("edge PID/dense-ID mapping is stale")
