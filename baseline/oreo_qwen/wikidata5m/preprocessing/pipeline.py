"""Passage preprocessing and deterministic sharded-corpus construction."""

from __future__ import annotations

import gzip
import json
import os
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from ..errors import MissingDataError
from ..store import Wikidata5MStore
from .aliases import (
    NORMALIZATION_VERSION,
    STOP_ALIAS_VERSION,
    match_graph_aliases,
    mentioned_qids,
)
from .schema import (
    CorpusValidationError,
    EvaluationMetadata,
    EvaluationTarget,
    LocalEdge,
    LocalGraph,
    LocalNode,
    MaskedOccurrence,
    PassageMetadata,
    PreprocessConfig,
    PreprocessedRecord,
    PreprocessResult,
    ResolvedMention,
    SCHEMA_VERSION,
    TargetGroup,
    file_sha256,
    passage_sha256,
    qid_number,
    RecordValidationError,
    validate_preprocessed_record,
)
from .selection import Span, select_context_blanks, stable_rank

CORPUS_MANIFEST_VERSION = 1
IMPLEMENTATION_VERSION = 1


def _manifest_hash(store: Wikidata5MStore) -> str:
    return file_sha256(store.index_dir / "manifest.json")


def _incoming(store: Wikidata5MStore, entity_id: int) -> tuple[tuple[int, int], ...]:
    batch = store.edges_dense(entity_id, direction="incoming", splits=["train"])
    # The index preserves duplicate triples. Preprocessing needs graph facts, so
    # exact duplicates are collapsed deterministically after degree gating.
    return tuple(
        sorted(
            {
                (int(batch.head_ids[index]), int(batch.relation_ids[index]))
                for index in range(len(batch))
                if int(batch.head_ids[index]) != entity_id
            }
        )
    )


def _reject(
    qid: str,
    entity_id: int,
    reason: str,
    diagnostics: Mapping[str, int],
    inspection: Mapping[str, Any] | None = None,
) -> PreprocessResult:
    return PreprocessResult(
        passage_qid=qid,
        passage_entity_id=entity_id,
        rejection_reason=reason,
        diagnostics=dict(diagnostics),
        inspection={} if inspection is None else dict(inspection),
    )


def preprocess_passage(
    store: Wikidata5MStore,
    passage_qid: str,
    config: PreprocessConfig | None = None,
) -> PreprocessResult:
    """Convert one indexed Wikidata5M passage into one reconstruction record."""

    config = config or PreprocessConfig()
    entity_id = store.entity_id(passage_qid)
    diagnostics: Counter[str] = Counter()
    try:
        passage = store.require_passage(passage_qid)
    except MissingDataError:
        return _reject(passage_qid, entity_id, "missing_passage", diagnostics)

    primary_batch = store.edges_dense(
        entity_id, direction="incoming", splits=[config.graph_split]
    )
    diagnostics["primary_raw_incoming_degree"] = len(primary_batch)
    if len(primary_batch) > config.max_incoming_degree:
        return _reject(passage_qid, entity_id, "hub_primary", diagnostics)
    primary_edges = _incoming(store, entity_id)
    primary_source_ids = {source_id for source_id, _ in primary_edges}
    primary_source_qids = {store.entity_qid(item) for item in primary_source_ids}

    first_depths = {passage_qid: 0}
    first_depths.update({qid: 1 for qid in primary_source_qids})
    first = match_graph_aliases(
        passage,
        first_depths,
        passage_qid,
        store,
        max_aliases_per_entity=config.max_aliases_per_entity,
    )
    diagnostics.update({f"phase1_{key}": value for key, value in first.diagnostics.items()})
    first_qids = mentioned_qids(first.matches)
    if passage_qid not in first_qids:
        return _reject(passage_qid, entity_id, "no_subject_mention", diagnostics)

    matched_primary_ids = sorted(
        (store.entity_id(qid) for qid in primary_source_qids & first_qids)
    )
    nested_edges_by_target: dict[int, tuple[tuple[int, int], ...]] = {}
    nested_hubs: set[int] = set()
    nested_source_ids: set[int] = set()
    nested_raw_degrees: list[int] = []
    for target_id in matched_primary_ids:
        raw = store.edges_dense(
            target_id, direction="incoming", splits=[config.graph_split]
        )
        diagnostics["nested_entities_examined"] += 1
        nested_raw_degrees.append(len(raw))
        if len(raw) > config.max_incoming_degree:
            nested_hubs.add(target_id)
            diagnostics["nested_hub"] += 1
            continue
        edges = _incoming(store, target_id)
        nested_edges_by_target[target_id] = edges
        nested_source_ids.update(source_id for source_id, _ in edges)

    candidate_depths = dict(first_depths)
    for source_id in nested_source_ids:
        candidate_depths.setdefault(store.entity_qid(source_id), 2)
    final = match_graph_aliases(
        passage,
        candidate_depths,
        passage_qid,
        store,
        max_aliases_per_entity=config.max_aliases_per_entity,
    )
    diagnostics.update({f"phase2_{key}": value for key, value in final.diagnostics.items()})
    final_qids = mentioned_qids(final.matches)
    if passage_qid not in final_qids:
        return _reject(passage_qid, entity_id, "no_subject_mention", diagnostics)

    mentions: list[ResolvedMention] = []
    for mention_id, match in enumerate(final.matches):
        mentions.append(
            ResolvedMention(
                mention_id=mention_id,
                qid=match.qid,
                entity_id=store.entity_id(match.qid),
                start_char=match.start_char,
                end_char=match.end_char,
                surface=match.surface,
                matched_normalized_alias=match.normalized_alias,
                graph_depth=match.graph_depth,
                masked=False,
            )
        )
    mention_ids_by_qid: dict[str, list[int]] = {}
    for mention in mentions:
        mention_ids_by_qid.setdefault(mention.qid, []).append(mention.mention_id)
    if len(mention_ids_by_qid.get(passage_qid, [])) > config.max_sentinels:
        return _reject(
            passage_qid,
            entity_id,
            "subject_sentinel_overflow",
            diagnostics,
        )

    primary_supporters = {
        store.entity_qid(source_id)
        for source_id, _ in primary_edges
        if store.entity_qid(source_id) in final_qids
    }
    if len(primary_supporters) < config.required_primary_supporters:
        return _reject(
            passage_qid,
            entity_id,
            "insufficient_primary_support",
            diagnostics,
            {
                "primary_supporter_qids": sorted(primary_supporters, key=qid_number),
                "nested_degrees": nested_raw_degrees,
            },
        )

    nested_supporters: dict[str, set[str]] = {}
    for target_id, edges in nested_edges_by_target.items():
        target_qid = store.entity_qid(target_id)
        nested_supporters[target_qid] = {
            store.entity_qid(source_id)
            for source_id, _ in edges
            if store.entity_qid(source_id) in final_qids
            and source_id != target_id
        }

    selected_aux: list[str] = []
    eligible_aux = [
        qid
        for qid in primary_supporters
        if store.entity_id(qid) not in nested_hubs and nested_supporters.get(qid)
    ]
    for candidate in sorted(
        eligible_aux, key=lambda qid: stable_rank(passage_qid, f"aux:{qid}")
    ):
        tentative = {*selected_aux, candidate}
        if len(primary_supporters - tentative) < config.required_primary_supporters:
            continue
        if any(
            len(
                nested_supporters.get(aux_qid, set())
                - {passage_qid, *tentative}
            )
            < config.required_auxiliary_supporters
            for aux_qid in tentative
        ):
            continue
        masked_count = sum(
            len(mention_ids_by_qid[qid]) for qid in {passage_qid, *tentative}
        )
        if masked_count > config.max_sentinels:
            continue
        selected_aux.append(candidate)
        if len(selected_aux) == config.max_auxiliary_targets:
            break

    target_qids = {passage_qid, *selected_aux}
    kg_mention_ids = {
        mention_id
        for qid in target_qids
        for mention_id in mention_ids_by_qid.get(qid, [])
    }
    if not kg_mention_ids:
        return _reject(passage_qid, entity_id, "no_subject_mention", diagnostics)
    if len(kg_mention_ids) > config.max_sentinels:
        return _reject(passage_qid, entity_id, "subject_sentinel_overflow", diagnostics)

    # Keep only graph facts whose source and target were resolved in the text.
    raw_local_edges: set[tuple[int, int, int, int]] = set()
    for source_id, relation_id in primary_edges:
        if store.entity_qid(source_id) in final_qids:
            raw_local_edges.add((source_id, entity_id, relation_id, 1))
    for target_id, edges in nested_edges_by_target.items():
        if store.entity_qid(target_id) not in final_qids:
            continue
        for source_id, relation_id in edges:
            if store.entity_qid(source_id) in final_qids:
                raw_local_edges.add((source_id, target_id, relation_id, 2))

    node_ids = {entity_id}
    for source_id, target_id, _, _ in raw_local_edges:
        node_ids.update((source_id, target_id))
    ordered_node_ids = sorted(node_ids)
    local_id = {global_id: index for index, global_id in enumerate(ordered_node_ids)}
    nodes = tuple(
        LocalNode(index, store.entity_qid(global_id), global_id)
        for index, global_id in enumerate(ordered_node_ids)
    )
    edges = tuple(
        LocalEdge(
            source_local_node_id=local_id[source_id],
            destination_local_node_id=local_id[target_id],
            relation_id=relation_id,
            relation_pid=store.relation_pid(relation_id),
            split="train",
            depth=depth,
        )
        for source_id, target_id, relation_id, depth in sorted(
            raw_local_edges,
            key=lambda item: (local_id[item[0]], local_id[item[1]], item[2], item[3]),
        )
    )

    group_specs = [("primary_kg", passage_qid), *[("auxiliary_kg", qid) for qid in selected_aux]]
    groups: list[TargetGroup] = []
    for group_id, (kind, qid) in enumerate(group_specs):
        visible = (
            primary_supporters - target_qids
            if kind == "primary_kg"
            else nested_supporters[qid] - target_qids
        )
        edge_indices = tuple(
            index
            for index, edge in enumerate(edges)
            if nodes[edge.destination_local_node_id].qid == qid
            and nodes[edge.source_local_node_id].qid in visible
        )
        groups.append(
            TargetGroup(
                target_group_id=group_id,
                kind=kind,
                qid=qid,
                entity_id=store.entity_id(qid),
                occurrence_mention_ids=tuple(mention_ids_by_qid[qid]),
                visible_supporter_qids=tuple(sorted(visible, key=qid_number)),
                support_edge_indices=edge_indices,
            )
        )

    remaining = config.max_sentinels - len(kg_mention_ids)
    context = select_context_blanks(
        passage,
        excluded_spans=(Span(item.start_char, item.end_char) for item in mentions),
        count=min(config.max_context_blanks, remaining),
        min_tokens=config.min_context_ngram_tokens,
        max_tokens=config.max_context_ngram_tokens,
        seed=passage_qid,
    )
    masks_raw: list[tuple[int, int, str, str, int | None, str | None, int | None]] = []
    group_by_qid = {item.qid: item for item in groups}
    for mention in mentions:
        if mention.mention_id in kg_mention_ids:
            group = group_by_qid[mention.qid]
            masks_raw.append(
                (
                    mention.start_char,
                    mention.end_char,
                    mention.surface,
                    group.kind,
                    group.target_group_id,
                    mention.qid,
                    mention.entity_id,
                )
            )
    for blank in context:
        masks_raw.append(
            (blank.start_char, blank.end_char, blank.exact_text, "context_only", None, None, None)
        )
    masks_raw.sort(key=lambda item: (item[0], item[1]))
    occurrences = tuple(
        MaskedOccurrence(
            sentinel_index=index,
            sentinel_token=f"[BLANK_{index}]",
            start_char=item[0],
            end_char=item[1],
            exact_text=item[2],
            kind=item[3],
            target_group_id=item[4],
            gold_qid=item[5],
            gold_entity_id=item[6],
        )
        for index, item in enumerate(masks_raw)
    )
    masked_ids = set(kg_mention_ids)
    mentions = [
        ResolvedMention(
            mention_id=item.mention_id,
            qid=item.qid,
            entity_id=item.entity_id,
            start_char=item.start_char,
            end_char=item.end_char,
            surface=item.surface,
            matched_normalized_alias=item.matched_normalized_alias,
            graph_depth=item.graph_depth,
            masked=item.mention_id in masked_ids,
        )
        for item in mentions
    ]
    marker_ids = tuple(item.mention_id for item in mentions if not item.masked)
    manifest_hash = _manifest_hash(store)
    record = PreprocessedRecord(
        schema_version=SCHEMA_VERSION,
        example_id=f"wiki5m:{passage_qid}",
        passage=PassageMetadata(
            qid=passage_qid,
            entity_id=entity_id,
            text_sha256=passage_sha256(passage),
            character_count=len(passage),
        ),
        mentions=tuple(mentions),
        local_graph=LocalGraph(nodes=nodes, edges=edges),
        target_groups=tuple(groups),
        masked_occurrences=occurrences,
        marker_mention_ids=marker_ids,
        evaluation=EvaluationMetadata(
            targets=tuple(
                EvaluationTarget(item.sentinel_index, item.exact_text, item.gold_qid, item.kind)
                for item in occurrences
            ),
            local_candidate_qids=tuple(node.qid for node in nodes),
            datastore_manifest_sha256=manifest_hash,
        ),
    )
    try:
        validate_preprocessed_record(
            record,
            passage,
            store=store,
            expected_datastore_manifest_sha256=manifest_hash,
        )
    except RecordValidationError as error:
        return _reject(
            passage_qid,
            entity_id,
            "record_validation_failed",
            diagnostics,
            {"validation_error": str(error)},
        )
    diagnostics["resolved_mentions"] = len(mentions)
    diagnostics["local_edges"] = len(edges)
    diagnostics["kg_masks"] = len(kg_mention_ids)
    diagnostics["context_masks"] = len(context)
    return PreprocessResult(
        passage_qid=passage_qid,
        passage_entity_id=entity_id,
        record=record,
        diagnostics=dict(diagnostics),
        inspection={
            "primary_supporter_qids": sorted(primary_supporters, key=qid_number),
            "selected_auxiliary_qids": selected_aux,
            "nested_hub_qids": sorted(
                (store.entity_qid(item) for item in nested_hubs), key=qid_number
            ),
            "nested_degrees": nested_raw_degrees,
        },
    )


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _write_gzip_jsonl(path: Path, records: Iterable[PreprocessedRecord]) -> tuple[int, str]:
    count = 0
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            for record in records:
                handle.write(_json_bytes(record.to_dict()))
                count += 1
    return count, file_sha256(path)


def _validate_temporary_shard(path: Path, expected_count: int) -> None:
    count = 0
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                PreprocessedRecord.from_json(line)
                count += 1
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, KeyError, ValueError) as error:
        raise CorpusValidationError(f"temporary shard validation failed: {error}") from error
    if count != expected_count:
        raise CorpusValidationError("temporary shard record count is wrong")


def _histogram(values: Iterable[int]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items(), key=lambda item: int(item[0])))


def _shard_statistics(
    records: list[PreprocessedRecord],
    *,
    inspected: int,
    primary_degrees: list[int],
    nested_degrees: list[int],
    diagnostics: Counter[str],
) -> dict[str, Any]:
    return {
        "inspected": inspected,
        "accepted": len(records),
        "rejected": inspected - len(records),
        "resolved_mentions": sum(len(record.mentions) for record in records),
        "local_nodes": sum(len(record.local_graph.nodes) for record in records),
        "local_edges": sum(len(record.local_graph.edges) for record in records),
        "targets": sum(len(record.target_groups) for record in records),
        "sentinels": sum(len(record.masked_occurrences) for record in records),
        "diagnostics": dict(sorted(diagnostics.items())),
        "histograms": {
            "primary_degree": _histogram(primary_degrees),
            "nested_degree": _histogram(nested_degrees),
            "mentions": _histogram(len(record.mentions) for record in records),
            "auxiliaries": _histogram(len(record.target_groups) - 1 for record in records),
            "context_blanks": _histogram(
                sum(item.kind == "context_only" for item in record.masked_occurrences)
                for record in records
            ),
            "local_nodes": _histogram(len(record.local_graph.nodes) for record in records),
            "local_edges": _histogram(len(record.local_graph.edges) for record in records),
            "sentinels": _histogram(len(record.masked_occurrences) for record in records),
        },
    }


def _aggregate_statistics(entries: list[dict[str, Any]]) -> dict[str, Any]:
    scalar_names = (
        "inspected", "accepted", "rejected", "resolved_mentions", "local_nodes",
        "local_edges", "targets", "sentinels",
    )
    result: dict[str, Any] = {
        name: sum(int(entry.get("statistics", {}).get(name, 0)) for entry in entries)
        for name in scalar_names
    }
    diagnostics: Counter[str] = Counter()
    histograms: dict[str, Counter[str]] = {}
    for entry in entries:
        statistics = entry.get("statistics", {})
        diagnostics.update(statistics.get("diagnostics", {}))
        for name, values in statistics.get("histograms", {}).items():
            histograms.setdefault(name, Counter()).update(values)
    result["diagnostics"] = dict(sorted(diagnostics.items()))
    result["histograms"] = {
        name: dict(sorted(values.items(), key=lambda item: int(item[0])))
        for name, values in sorted(histograms.items())
    }
    return result


def _build_shard_entry(
    store: Wikidata5MStore,
    output: Path,
    config: PreprocessConfig,
    shard_start: int,
    shard_stop: int,
) -> dict[str, Any]:
    """Build one independent dense-ID shard for serial or process workers."""

    shard_dir = output / "shards"
    name = f"entities-{shard_start:08d}-{shard_stop - 1:08d}.jsonl.gz"
    destination = shard_dir / name
    records: list[PreprocessedRecord] = []
    rejections: Counter[str] = Counter()
    rejection_samples: list[dict[str, str]] = []
    primary_degrees: list[int] = []
    nested_degrees: list[int] = []
    diagnostics: Counter[str] = Counter()
    for entity_id in range(shard_start, shard_stop):
        qid = store.entity_qid(entity_id)
        if store.passage(qid) is None:
            rejections["missing_passage"] += 1
            if len(rejection_samples) < config.audit_example_limit:
                rejection_samples.append({"qid": qid, "reason": "missing_passage"})
            continue
        result = preprocess_passage(store, qid, config)
        if "primary_raw_incoming_degree" in result.diagnostics:
            primary_degrees.append(int(result.diagnostics["primary_raw_incoming_degree"]))
        nested_degrees.extend(
            int(item) for item in result.inspection.get("nested_degrees", [])
        )
        diagnostics.update(result.diagnostics)
        if result.record is None:
            assert result.rejection_reason is not None
            rejections[result.rejection_reason] += 1
            if len(rejection_samples) < config.audit_example_limit:
                rejection_samples.append(
                    {"qid": qid, "reason": result.rejection_reason}
                )
        else:
            records.append(result.record)

    with tempfile.NamedTemporaryFile(
        prefix=f".{name}.", suffix=".tmp", dir=shard_dir, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        record_count, checksum = _write_gzip_jsonl(temporary_path, records)
        _validate_temporary_shard(temporary_path, record_count)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {
        "path": str(destination.relative_to(output)),
        "start_entity_id": shard_start,
        "stop_entity_id": shard_stop,
        "records": record_count,
        "rejections": dict(sorted(rejections.items())),
        "rejection_samples": rejection_samples,
        "mask_kind_counts": dict(
            sorted(
                Counter(
                    occurrence.kind
                    for record in records
                    for occurrence in record.masked_occurrences
                ).items()
            )
        ),
        "statistics": _shard_statistics(
            records,
            inspected=shard_stop - shard_start,
            primary_degrees=primary_degrees,
            nested_degrees=nested_degrees,
            diagnostics=diagnostics,
        ),
        "size": destination.stat().st_size,
        "sha256": checksum,
    }


def _build_shard_worker(
    index_dir: str,
    source_root: str,
    output: str,
    config_values: Mapping[str, Any],
    shard_start: int,
    shard_stop: int,
) -> dict[str, Any]:
    config = PreprocessConfig(**config_values)
    with Wikidata5MStore(
        index_dir, source_root=source_root, validate=False
    ) as worker_store:
        return _build_shard_entry(
            worker_store, Path(output), config, shard_start, shard_stop
        )


def _ordered_corpus_totals(
    entries_by_range: Mapping[tuple[int, int], dict[str, Any]],
    audit_limit: int,
) -> tuple[list[dict[str, Any]], int, dict[str, int], list[dict[str, Any]]]:
    entries = [entries_by_range[key] for key in sorted(entries_by_range)]
    rejections: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    for entry in entries:
        rejections.update(entry.get("rejections", {}))
        samples.extend(entry.get("rejection_samples", []))
    return (
        entries,
        sum(int(entry["records"]) for entry in entries),
        dict(sorted(rejections.items())),
        samples[:audit_limit],
    )


def _build_preprocessed_corpus_parallel(
    *,
    store: Wikidata5MStore,
    output: Path,
    config: PreprocessConfig,
    start_entity_id: int,
    stop_entity_id: int,
    workers: int,
    force: bool,
    base_manifest: Mapping[str, Any],
    previous_shards: Mapping[tuple[int, int], dict[str, Any]],
    progress_callback: Callable[[int, int, int], None] | None,
) -> Path:
    entries_by_range: dict[tuple[int, int], dict[str, Any]] = {}
    pending: list[tuple[int, int]] = []
    for shard_start in range(start_entity_id, stop_entity_id, config.shard_entity_count):
        shard_stop = min(shard_start + config.shard_entity_count, stop_entity_id)
        key = (shard_start, shard_stop)
        destination = output / "shards" / (
            f"entities-{shard_start:08d}-{shard_stop - 1:08d}.jsonl.gz"
        )
        previous = None if force else previous_shards.get(key)
        if previous is not None and destination.is_file() and (
            destination.stat().st_size == previous.get("size")
            and file_sha256(destination) == previous.get("sha256")
        ):
            entries_by_range[key] = previous
        else:
            pending.append(key)

    progress_path = output / ".build-manifest.json"
    total_entities = stop_entity_id - start_entity_id

    def publish() -> None:
        entries, records, rejections, samples = _ordered_corpus_totals(
            entries_by_range, config.audit_example_limit
        )
        inspected = sum(stop - start for start, stop in entries_by_range)
        progress = {
            **base_manifest,
            "complete": False,
            "records": records,
            "rejections": rejections,
            "rejection_samples": samples,
            "statistics": _aggregate_statistics(entries),
            "shards": entries,
        }
        temporary_progress = output / ".build-manifest.json.tmp"
        temporary_progress.write_bytes(_json_bytes(progress))
        os.replace(temporary_progress, progress_path)
        if progress_callback is not None:
            progress_callback(inspected, total_entities, records)

    if entries_by_range:
        publish()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _build_shard_worker,
                str(store.index_dir),
                str(store.source_root),
                str(output),
                config.to_dict(),
                shard_start,
                shard_stop,
            ): (shard_start, shard_stop)
            for shard_start, shard_stop in pending
        }
        for future in as_completed(futures):
            key = futures[future]
            entries_by_range[key] = future.result()
            publish()

    entries, records, rejections, samples = _ordered_corpus_totals(
        entries_by_range, config.audit_example_limit
    )
    manifest = {
        **base_manifest,
        "complete": True,
        "records": records,
        "rejections": rejections,
        "rejection_samples": samples,
        "statistics": _aggregate_statistics(entries),
        "shards": entries,
    }
    temporary_manifest = output / ".manifest.json.tmp"
    temporary_manifest.write_bytes(_json_bytes(manifest))
    os.replace(temporary_manifest, output / "manifest.json")
    progress_path.unlink(missing_ok=True)
    return output


def build_preprocessed_corpus(
    store: Wikidata5MStore,
    output_dir: str | Path,
    config: PreprocessConfig | None = None,
    *,
    start_entity_id: int = 0,
    stop_entity_id: int | None = None,
    force: bool = False,
    workers: int = 1,
    progress_callback: Callable[[int, int, int], None] | None = None,
) -> Path:
    """Build fixed dense-ID-range shards and a resumable corpus manifest."""

    config = config or PreprocessConfig()
    output = Path(output_dir).expanduser().resolve()
    stop = store.entity_count if stop_entity_id is None else stop_entity_id
    if not 0 <= start_entity_id <= stop <= store.entity_count:
        raise ValueError("invalid dense entity-ID range")
    if workers <= 0:
        raise ValueError("workers must be positive")
    output.mkdir(parents=True, exist_ok=True)
    shard_dir = output / "shards"
    shard_dir.mkdir(exist_ok=True)
    manifest_path = output / "manifest.json"
    progress_path = output / ".build-manifest.json"
    datastore_hash = _manifest_hash(store)
    base_manifest = {
        "manifest_version": CORPUS_MANIFEST_VERSION,
        "schema_version": SCHEMA_VERSION,
        "implementation_version": IMPLEMENTATION_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "stop_alias_version": STOP_ALIAS_VERSION,
        "config": config.to_dict(),
        "config_fingerprint": config.fingerprint,
        "datastore_manifest_sha256": datastore_hash,
        "datastore_manifest_path": str(store.index_dir / "manifest.json"),
        "source_index": str(store.index_dir),
        "entity_range": [start_entity_id, stop],
    }
    existing: dict[str, Any] | None = None
    if not force:
        resume_path = progress_path if progress_path.exists() else manifest_path
        if resume_path.exists():
            existing = json.loads(resume_path.read_text(encoding="utf-8"))
            for key in (
                "implementation_version",
                "normalization_version",
                "stop_alias_version",
                "config_fingerprint",
                "datastore_manifest_sha256",
                "entity_range",
            ):
                if existing.get(key) != base_manifest[key]:
                    raise CorpusValidationError("existing corpus was built with different inputs")
    previous_shards = {
        (int(item["start_entity_id"]), int(item["stop_entity_id"])): item
        for item in ([] if existing is None else existing.get("shards", []))
    }

    if workers > 1:
        return _build_preprocessed_corpus_parallel(
            store=store,
            output=output,
            config=config,
            start_entity_id=start_entity_id,
            stop_entity_id=stop,
            workers=workers,
            force=force,
            base_manifest=base_manifest,
            previous_shards=previous_shards,
            progress_callback=progress_callback,
        )

    shard_entries: list[dict[str, Any]] = []
    total_rejections: Counter[str] = Counter()
    total_records = 0
    rejection_samples: list[dict[str, Any]] = []
    for shard_start in range(start_entity_id, stop, config.shard_entity_count):
        shard_stop = min(shard_start + config.shard_entity_count, stop)
        name = f"entities-{shard_start:08d}-{shard_stop - 1:08d}.jsonl.gz"
        destination = shard_dir / name
        previous = previous_shards.get((shard_start, shard_stop))
        if previous is not None and destination.is_file() and (
            destination.stat().st_size == previous.get("size")
            and file_sha256(destination) == previous.get("sha256")
        ):
            shard_entries.append(previous)
            total_records += int(previous["records"])
            total_rejections.update(previous.get("rejections", {}))
            rejection_samples.extend(previous.get("rejection_samples", []))
            if progress_callback is not None:
                progress_callback(
                    shard_stop - start_entity_id,
                    stop - start_entity_id,
                    total_records,
                )
            continue
        records: list[PreprocessedRecord] = []
        rejections: Counter[str] = Counter()
        shard_rejection_samples: list[dict[str, str]] = []
        primary_degrees: list[int] = []
        nested_degrees: list[int] = []
        shard_diagnostics: Counter[str] = Counter()
        for entity_id in range(shard_start, shard_stop):
            qid = store.entity_qid(entity_id)
            if store.passage(qid) is None:
                rejections["missing_passage"] += 1
                if len(shard_rejection_samples) < config.audit_example_limit:
                    shard_rejection_samples.append(
                        {"qid": qid, "reason": "missing_passage"}
                    )
                continue
            result = preprocess_passage(store, qid, config)
            if "primary_raw_incoming_degree" in result.diagnostics:
                primary_degrees.append(int(result.diagnostics["primary_raw_incoming_degree"]))
            nested_degrees.extend(int(item) for item in result.inspection.get("nested_degrees", []))
            shard_diagnostics.update(result.diagnostics)
            if result.record is None:
                assert result.rejection_reason is not None
                rejections[result.rejection_reason] += 1
                if len(shard_rejection_samples) < config.audit_example_limit:
                    shard_rejection_samples.append(
                        {"qid": qid, "reason": result.rejection_reason}
                    )
            else:
                records.append(result.record)
        with tempfile.NamedTemporaryFile(
            prefix=f".{name}.", suffix=".tmp", dir=shard_dir, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            record_count, checksum = _write_gzip_jsonl(temporary_path, records)
            _validate_temporary_shard(temporary_path, record_count)
            os.replace(temporary_path, destination)
        finally:
            temporary_path.unlink(missing_ok=True)
        entry = {
            "path": str(destination.relative_to(output)),
            "start_entity_id": shard_start,
            "stop_entity_id": shard_stop,
            "records": record_count,
            "rejections": dict(sorted(rejections.items())),
            "rejection_samples": shard_rejection_samples,
            "mask_kind_counts": dict(
                sorted(
                    Counter(
                        occurrence.kind
                        for record in records
                        for occurrence in record.masked_occurrences
                    ).items()
                )
            ),
            "statistics": _shard_statistics(
                records,
                inspected=shard_stop - shard_start,
                primary_degrees=primary_degrees,
                nested_degrees=nested_degrees,
                diagnostics=shard_diagnostics,
            ),
            "size": destination.stat().st_size,
            "sha256": checksum,
        }
        shard_entries.append(entry)
        total_records += record_count
        total_rejections.update(rejections)
        rejection_samples.extend(shard_rejection_samples)
        progress = {
            **base_manifest,
            "complete": False,
            "records": total_records,
            "rejections": dict(sorted(total_rejections.items())),
            "rejection_samples": rejection_samples[: config.audit_example_limit],
            "statistics": _aggregate_statistics(shard_entries),
            "shards": shard_entries,
        }
        temporary_progress = output / ".build-manifest.json.tmp"
        temporary_progress.write_bytes(_json_bytes(progress))
        os.replace(temporary_progress, progress_path)
        if progress_callback is not None:
            progress_callback(
                shard_stop - start_entity_id,
                stop - start_entity_id,
                total_records,
            )

    manifest = {
        **base_manifest,
        "complete": True,
        "records": total_records,
        "rejections": dict(sorted(total_rejections.items())),
        "rejection_samples": rejection_samples[: config.audit_example_limit],
        "statistics": _aggregate_statistics(shard_entries),
        "shards": shard_entries,
    }
    temporary_manifest = output / ".manifest.json.tmp"
    temporary_manifest.write_bytes(_json_bytes(manifest))
    os.replace(temporary_manifest, manifest_path)
    progress_path.unlink(missing_ok=True)
    return output


def validate_preprocessed_corpus(
    output_dir: str | Path,
    *,
    store: Wikidata5MStore | None = None,
    full: bool = False,
) -> dict[str, Any]:
    """Validate corpus layout/checksums and optionally every source-backed record."""

    output = Path(output_dir).expanduser().resolve()
    manifest_path = output / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CorpusValidationError(f"cannot read corpus manifest: {error}") from error
    if manifest.get("manifest_version") != CORPUS_MANIFEST_VERSION or manifest.get("schema_version") != SCHEMA_VERSION:
        raise CorpusValidationError("unsupported corpus manifest")
    if manifest.get("complete") is not True:
        raise CorpusValidationError("corpus build is incomplete")
    if full and store is None:
        raise CorpusValidationError("full validation requires an open datastore")
    try:
        manifest_config = PreprocessConfig(**manifest["config"])
    except (KeyError, TypeError, ValueError) as error:
        raise CorpusValidationError(f"corpus configuration is invalid: {error}") from error
    if manifest_config.fingerprint != manifest.get("config_fingerprint"):
        raise CorpusValidationError("corpus configuration fingerprint is corrupt")
    if store is not None and manifest.get("datastore_manifest_sha256") != _manifest_hash(store):
        raise CorpusValidationError("corpus datastore fingerprint is stale")
    total = 0
    rejection_totals: Counter[str] = Counter()
    entity_range = manifest.get("entity_range")
    if not isinstance(entity_range, list) or len(entity_range) != 2:
        raise CorpusValidationError("corpus entity range is invalid")
    expected_start = int(entity_range[0])
    root = output.resolve()
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise CorpusValidationError("corpus shard table is invalid")
    for entry in shards:
        if int(entry["start_entity_id"]) != expected_start:
            raise CorpusValidationError("shard dense-ID ranges are not contiguous")
        expected_start = int(entry["stop_entity_id"])
        path = (output / entry["path"]).resolve()
        if not path.is_relative_to(root):
            raise CorpusValidationError("shard path escapes corpus directory")
        if not path.is_file() or path.stat().st_size != entry["size"]:
            raise CorpusValidationError(f"missing or truncated shard: {path}")
        if file_sha256(path) != entry["sha256"]:
            raise CorpusValidationError(f"shard checksum failed: {path}")
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                count = 0
                for line in handle:
                    record = PreprocessedRecord.from_json(line)
                    count += 1
                    if full:
                        assert store is not None
                        passage = store.require_passage(record.passage.qid)
                        validate_preprocessed_record(
                            record,
                            passage,
                            store=store,
                            expected_datastore_manifest_sha256=_manifest_hash(store),
                        )
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
            KeyError,
            ValueError,
            RecordValidationError,
        ) as error:
            raise CorpusValidationError(f"invalid shard {path}: {error}") from error
        if count != entry["records"]:
            raise CorpusValidationError(f"record count mismatch in shard: {path}")
        total += count
        rejection_totals.update(entry.get("rejections", {}))
    if expected_start != int(entity_range[1]):
        raise CorpusValidationError("shard dense-ID ranges do not cover the corpus range")
    if total != manifest.get("records"):
        raise CorpusValidationError("manifest total record count is wrong")
    if dict(sorted(rejection_totals.items())) != manifest.get("rejections"):
        raise CorpusValidationError("manifest rejection totals are wrong")
    return manifest
