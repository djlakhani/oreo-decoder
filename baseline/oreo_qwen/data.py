"""Streaming Wiki5M data and CPU-only semantic collation for Oreo-Qwen."""

from __future__ import annotations

import gzip
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable, Iterator, Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info

from .contracts import KGInputBatch, OreoQwenBatch
from .embedding import format_entity_text, format_relation_text
from .tokens import OreoSpecialTokenIds
from .wikidata5m.preprocessing.render import render_example
from .wikidata5m.preprocessing.schema import PreprocessedRecord, SCHEMA_VERSION
from .wikidata5m.store import Wikidata5MStore

ENTITY_TEXT_FORMAT_VERSION = "wiki5m_entity_aliases_v1"
RELATION_TEXT_FORMAT_VERSION = "wiki5m_relation_aliases_v1"


class DatasetError(ValueError):
    """A preprocessed corpus cannot be consumed safely."""


class CollationError(ValueError):
    """A record cannot be translated into the Oreo tensor contract."""


@dataclass(frozen=True, slots=True)
class KnowledgeEmbeddingRequest:
    """Stable text request consumed later by ``FrozenKnowledgeEmbedder``."""

    kind: str
    dense_id: int
    identifier: str
    text: str
    format_version: str

    @property
    def key(self) -> Hashable:
        return (self.format_version, self.dense_id)


@dataclass(frozen=True, slots=True)
class ReconstructionDatasetItem:
    """One v3 record plus source text needed to render and embed it."""

    record: PreprocessedRecord
    passage_text: str
    node_embedding_requests: tuple[KnowledgeEmbeddingRequest, ...]


class Wikidata5MReconstructionDataset(IterableDataset[ReconstructionDatasetItem]):
    """Stream v3 JSONL shards and join them with the immutable Wiki5M index.

    Shards are assigned across distributed ranks and DataLoader workers. The
    dataset deliberately retains no open SQLite or source-file handles, so it
    is safe to construct before workers are spawned.
    """

    def __init__(
        self,
        corpus_dir: str | Path,
        *,
        index_dir: str | Path | None = None,
        source_root: str | Path | None = None,
        shuffle_shards: bool = False,
        seed: int = 0,
        rank: int | None = None,
        world_size: int | None = None,
        validate_checksums: bool = False,
    ) -> None:
        super().__init__()
        self.corpus_dir = Path(corpus_dir).expanduser().resolve()
        manifest_path = self.corpus_dir / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise DatasetError(f"cannot read corpus manifest {manifest_path}: {error}") from error
        if not isinstance(manifest, dict):
            raise DatasetError("corpus manifest root must be an object")
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise DatasetError(
                f"expected {SCHEMA_VERSION}, got {manifest.get('schema_version')!r}"
            )
        if manifest.get("complete") is not True:
            raise DatasetError("preprocessed corpus is incomplete")

        manifest_index = manifest.get("source_index")
        if index_dir is None:
            if not isinstance(manifest_index, str):
                raise DatasetError("corpus manifest has no source_index; pass index_dir")
            index_dir = manifest_index
        self.index_dir = Path(index_dir).expanduser().resolve()
        self.source_root = (
            None if source_root is None else Path(source_root).expanduser().resolve()
        )
        index_manifest_path = self.index_dir / "manifest.json"
        try:
            index_manifest_bytes = index_manifest_path.read_bytes()
            index_manifest = json.loads(index_manifest_bytes)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise DatasetError(
                f"cannot read Wiki5M index manifest {index_manifest_path}: {error}"
            ) from error
        actual_fingerprint = hashlib.sha256(index_manifest_bytes).hexdigest()
        expected_fingerprint = manifest.get("datastore_manifest_sha256")
        if actual_fingerprint != expected_fingerprint:
            raise DatasetError("corpus and Wiki5M index fingerprints do not match")
        try:
            self.relation_count = int(index_manifest["counts"]["relations"])
        except (KeyError, TypeError, ValueError) as error:
            raise DatasetError("Wiki5M index relation count is invalid") from error

        raw_shards = manifest.get("shards")
        if not isinstance(raw_shards, list):
            raise DatasetError("corpus manifest shard table is invalid")
        shards: list[Path] = []
        corpus_root = self.corpus_dir
        for entry in raw_shards:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
                raise DatasetError("corpus manifest contains an invalid shard entry")
            path = (corpus_root / entry["path"]).resolve()
            if not path.is_relative_to(corpus_root):
                raise DatasetError(f"shard path escapes corpus directory: {entry['path']}")
            if not path.is_file():
                raise DatasetError(f"corpus shard is missing: {path}")
            if path.stat().st_size != entry.get("size"):
                raise DatasetError(f"corpus shard has the wrong size: {path}")
            if validate_checksums:
                from .wikidata5m.preprocessing.schema import file_sha256

                if file_sha256(path) != entry.get("sha256"):
                    raise DatasetError(f"corpus shard checksum failed: {path}")
            shards.append(path)
        self._shards = tuple(shards)
        self._record_count = int(manifest.get("records", 0))
        if self._record_count < 0:
            raise DatasetError("corpus record count is invalid")
        if (rank is None) != (world_size is None):
            raise ValueError("rank and world_size must be supplied together")
        if world_size is not None and (world_size <= 0 or not 0 <= rank < world_size):
            raise ValueError("rank/world_size are invalid")
        self.rank = rank
        self.world_size = world_size
        self.shuffle_shards = bool(shuffle_shards)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return self._record_count

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be nonnegative")
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[ReconstructionDatasetItem]:
        rank, world_size = self._distributed_context()
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id
        worker_count = 1 if worker is None else worker.num_workers
        consumer_id = rank * worker_count + worker_id
        consumer_count = world_size * worker_count

        shard_indices = list(range(len(self._shards)))
        if self.shuffle_shards:
            random.Random(self.seed + self.epoch).shuffle(shard_indices)
        selected = shard_indices[consumer_id::consumer_count]

        with Wikidata5MStore(
            self.index_dir, source_root=self.source_root, validate=True
        ) as store:
            for shard_index in selected:
                path = self._shards[shard_index]
                line_number = 0
                try:
                    with gzip.open(path, "rt", encoding="utf-8") as handle:
                        for line_number, line in enumerate(handle, 1):
                            record = PreprocessedRecord.from_json(line)
                            passage = store.require_passage(record.passage.qid)
                            requests = tuple(
                                _entity_embedding_request(store, node.entity_id, node.qid)
                                for node in record.local_graph.nodes
                            )
                            yield ReconstructionDatasetItem(record, passage, requests)
                except (
                    OSError,
                    UnicodeError,
                    json.JSONDecodeError,
                    KeyError,
                    TypeError,
                    ValueError,
                ) as error:
                    raise DatasetError(
                        f"cannot read {path} at record line {line_number}: {error}"
                    ) from error

    def relation_embedding_requests(self) -> tuple[KnowledgeEmbeddingRequest, ...]:
        """Return the full relation bank in dense relation-ID order."""

        with Wikidata5MStore(
            self.index_dir, source_root=self.source_root, validate=True
        ) as store:
            return tuple(
                _relation_embedding_request(store, item.relation_id, item.pid)
                for item in store.iter_relations()
            )

    def _distributed_context(self) -> tuple[int, int]:
        if self.rank is not None and self.world_size is not None:
            return self.rank, self.world_size
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank(), torch.distributed.get_world_size()
        return 0, 1


@dataclass(kw_only=True, slots=True)
class SemanticKGInputBatch:
    """CPU graph tensors plus unresolved node-embedding requests."""

    kg_mention_batch: Tensor
    kg_rel_positions: Tensor
    kg_tgt_positions: Tensor
    kg_node_batch: Tensor
    kg_edge_src: Tensor
    kg_edge_dst: Tensor
    kg_edge_type: Tensor
    kg_initial_node_indices: Tensor
    node_embedding_requests: tuple[KnowledgeEmbeddingRequest, ...]
    kg_relation_targets: Tensor | None = None
    kg_relation_target_mask: Tensor | None = None

    @property
    def node_embedding_keys(self) -> tuple[Hashable, ...]:
        return tuple(item.key for item in self.node_embedding_requests)

    @property
    def node_embedding_texts(self) -> tuple[str, ...]:
        return tuple(item.text for item in self.node_embedding_requests)


@dataclass(kw_only=True, slots=True)
class SemanticOreoBatch:
    """Collator output awaiting frozen node and relation embeddings."""

    input_ids: Tensor
    attention_mask: Tensor
    labels: Tensor | None
    assistant_content_start_positions: Tensor
    kg: SemanticKGInputBatch
    example_ids: tuple[str, ...]

    def materialize(
        self,
        *,
        node_embeddings: Tensor,
        relation_embeddings: Tensor,
        device: str | torch.device | None = None,
    ) -> OreoQwenBatch:
        """Attach frozen vectors and produce the model's validated batch type."""

        if node_embeddings.ndim != 2:
            raise CollationError("node_embeddings must have shape [V, D]")
        if node_embeddings.shape[0] != len(self.kg.node_embedding_requests):
            raise CollationError("node embedding rows do not match graph nodes")
        if relation_embeddings.ndim != 2:
            raise CollationError("relation_embeddings must have shape [R, D]")
        expected_relations = (
            self.kg.kg_relation_targets.shape[-1]
            if self.kg.kg_relation_targets is not None
            else int(self.kg.kg_edge_type.max().item()) + 1
        )
        if relation_embeddings.shape[0] != expected_relations:
            raise CollationError(
                "relation embedding rows do not match the global relation count"
            )
        if relation_embeddings.shape[1] != node_embeddings.shape[1]:
            raise CollationError("node and relation embedding widths differ")
        target_device = node_embeddings.device if device is None else torch.device(device)

        def move(value: Tensor | None) -> Tensor | None:
            return None if value is None else value.to(target_device)

        kg = KGInputBatch(
            kg_mention_batch=move(self.kg.kg_mention_batch),
            kg_rel_positions=move(self.kg.kg_rel_positions),
            kg_tgt_positions=move(self.kg.kg_tgt_positions),
            kg_node_embeddings=node_embeddings.to(target_device),
            kg_relation_embeddings=relation_embeddings.to(target_device),
            kg_node_batch=move(self.kg.kg_node_batch),
            kg_edge_src=move(self.kg.kg_edge_src),
            kg_edge_dst=move(self.kg.kg_edge_dst),
            kg_edge_type=move(self.kg.kg_edge_type),
            kg_initial_node_indices=move(self.kg.kg_initial_node_indices),
            kg_relation_targets=move(self.kg.kg_relation_targets),
            kg_relation_target_mask=move(self.kg.kg_relation_target_mask),
        )
        return OreoQwenBatch(
            input_ids=move(self.input_ids),
            attention_mask=move(self.attention_mask),
            labels=move(self.labels),
            assistant_content_start_positions=move(
                self.assistant_content_start_positions
            ),
            kg=kg,
        )


class OreoSemanticCollator:
    """Render, tokenize, align markers, and batch local graphs on CPU.

    No embedding model is loaded here. This object is therefore safe to use as
    a DataLoader ``collate_fn`` with multiprocessing workers.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        num_relations: int,
        num_kil_steps: int = 1,
        max_length: int | None = None,
        pad_to_multiple_of: int | None = None,
        training: bool = True,
        append_eos: bool = True,
        overflow_policy: str = "error",
        chat_template_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if num_relations <= 0:
            raise ValueError("num_relations must be positive")
        if num_kil_steps <= 0:
            raise ValueError("num_kil_steps must be positive")
        if max_length is not None and max_length <= 0:
            raise ValueError("max_length must be positive")
        if pad_to_multiple_of is not None and pad_to_multiple_of <= 0:
            raise ValueError("pad_to_multiple_of must be positive")
        if overflow_policy not in ("error", "drop"):
            raise ValueError("overflow_policy must be 'error' or 'drop'")
        self.tokenizer = tokenizer
        self.special_token_ids = OreoSpecialTokenIds.from_tokenizer(tokenizer)
        self.num_relations = int(num_relations)
        self.num_kil_steps = int(num_kil_steps)
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.training = bool(training)
        self.append_eos = bool(append_eos)
        self.overflow_policy = overflow_policy
        self.chat_template_kwargs = dict(chat_template_kwargs or {"enable_thinking": False})
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            raise ValueError("tokenizer must define a pad or EOS token ID")
        self.pad_token_id = int(pad_token_id)

    def __call__(
        self, items: Sequence[ReconstructionDatasetItem]
    ) -> SemanticOreoBatch | None:
        if not items:
            raise CollationError("cannot collate an empty batch")

        sequences: list[list[int]] = []
        label_rows: list[list[int]] = []
        starts: list[int] = []
        mention_batches: list[int] = []
        rel_positions: list[int] = []
        tgt_positions: list[int] = []
        node_batches: list[int] = []
        edge_src: list[int] = []
        edge_dst: list[int] = []
        edge_type: list[int] = []
        initial_nodes: list[int] = []
        embedding_requests: list[KnowledgeEmbeddingRequest] = []
        supervised_relations: list[tuple[int, ...]] = []
        example_ids: list[str] = []
        node_offset = 0

        for item in items:
            batch_index = len(sequences)
            record = item.record
            rendered = render_example(record, item.passage_text)
            prompt_ids = self._prompt_ids(
                rendered.system_instruction, rendered.corrupted_passage
            )
            assistant_start = len(prompt_ids)
            if self.training:
                answer_ids = self._encode_text(rendered.reconstruction_suffix)
                if not answer_ids:
                    raise CollationError(f"{record.example_id} has an empty answer")
                eos_id = getattr(self.tokenizer, "eos_token_id", None)
                if self.append_eos and eos_id is not None and answer_ids[-1] != eos_id:
                    answer_ids.append(int(eos_id))
                sequence = prompt_ids + answer_ids
                labels = [-100] * assistant_start + answer_ids
            else:
                sequence = prompt_ids
                labels = [-100] * len(sequence)
            if self.max_length is not None and len(sequence) > self.max_length:
                if self.overflow_policy == "drop":
                    continue
                raise CollationError(
                    f"{record.example_id} tokenizes to {len(sequence)} tokens, exceeding "
                    f"max_length={self.max_length}; semantic truncation is not allowed"
                )

            rels = [index for index, value in enumerate(prompt_ids) if value == self.special_token_ids.relation]
            tgts = [index for index, value in enumerate(prompt_ids) if value == self.special_token_ids.target_entity]
            expected_mentions = len(record.marker_mention_ids)
            if len(rels) != expected_mentions or len(tgts) != expected_mentions:
                raise CollationError(
                    f"{record.example_id} rendered {expected_mentions} mentions but tokenized "
                    f"to {len(rels)} REL and {len(tgts)} T_ENT markers"
                )
            if any(target != relation + 1 for relation, target in zip(rels, tgts, strict=True)):
                raise CollationError(f"{record.example_id} marker pairs are not adjacent")

            nodes = record.local_graph.nodes
            if len(item.node_embedding_requests) != len(nodes):
                raise CollationError(
                    f"{record.example_id} embedding requests do not match local nodes"
                )
            local_node_by_qid = {node.qid: node.local_node_id for node in nodes}
            gold_by_source = self._gold_relations_by_source(record)
            mention_by_id = {mention.mention_id: mention for mention in record.mentions}
            for mention_id, relation_position, target_position in zip(
                record.marker_mention_ids, rels, tgts, strict=True
            ):
                mention = mention_by_id[mention_id]
                try:
                    local_node = local_node_by_qid[mention.qid]
                except KeyError as error:
                    raise CollationError(
                        f"{record.example_id} marker entity {mention.qid} is absent from graph"
                    ) from error
                mention_batches.append(batch_index)
                rel_positions.append(relation_position)
                tgt_positions.append(target_position)
                initial_nodes.append(node_offset + local_node)
                supervised_relations.append(gold_by_source.get(mention.qid, ()))

            node_batches.extend([batch_index] * len(nodes))
            embedding_requests.extend(item.node_embedding_requests)
            for edge in record.local_graph.edges:
                if not 0 <= edge.relation_id < self.num_relations:
                    raise CollationError(
                        f"{record.example_id} relation ID {edge.relation_id} is out of range"
                    )
                edge_src.append(node_offset + edge.source_local_node_id)
                edge_dst.append(node_offset + edge.destination_local_node_id)
                edge_type.append(edge.relation_id)
            node_offset += len(nodes)
            sequences.append(sequence)
            label_rows.append(labels)
            starts.append(assistant_start)
            example_ids.append(record.example_id)

        if not sequences:
            return None
        sequence_length = max(map(len, sequences))
        if self.pad_to_multiple_of is not None:
            multiple = self.pad_to_multiple_of
            sequence_length = ((sequence_length + multiple - 1) // multiple) * multiple
        input_ids = torch.full(
            (len(sequences), sequence_length), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((len(sequences), sequence_length), dtype=torch.long)
        labels_tensor = torch.full(
            (len(sequences), sequence_length), -100, dtype=torch.long
        )
        for row, (sequence, labels) in enumerate(zip(sequences, label_rows, strict=True)):
            length = len(sequence)
            input_ids[row, :length] = torch.tensor(sequence, dtype=torch.long)
            attention_mask[row, :length] = 1
            labels_tensor[row, :length] = torch.tensor(labels, dtype=torch.long)

        relation_targets, relation_mask = self._relation_supervision(
            supervised_relations
        )
        semantic_kg = SemanticKGInputBatch(
            kg_mention_batch=torch.tensor(mention_batches, dtype=torch.long),
            kg_rel_positions=torch.tensor(rel_positions, dtype=torch.long),
            kg_tgt_positions=torch.tensor(tgt_positions, dtype=torch.long),
            kg_node_batch=torch.tensor(node_batches, dtype=torch.long),
            kg_edge_src=torch.tensor(edge_src, dtype=torch.long),
            kg_edge_dst=torch.tensor(edge_dst, dtype=torch.long),
            kg_edge_type=torch.tensor(edge_type, dtype=torch.long),
            kg_initial_node_indices=torch.tensor(initial_nodes, dtype=torch.long),
            node_embedding_requests=tuple(embedding_requests),
            kg_relation_targets=relation_targets,
            kg_relation_target_mask=relation_mask,
        )
        return SemanticOreoBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels_tensor if self.training else None,
            assistant_content_start_positions=torch.tensor(starts, dtype=torch.long),
            kg=semantic_kg,
            example_ids=tuple(example_ids),
        )

    def _prompt_ids(self, system_instruction: str, corrupted_passage: str) -> list[int]:
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"Passage:\n{corrupted_passage}"},
        ]
        try:
            encoded = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                **self.chat_template_kwargs,
            )
        except Exception as error:
            raise CollationError(f"tokenizer chat template failed: {error}") from error
        return _as_token_ids(encoded, "chat template")

    def _encode_text(self, text: str) -> list[int]:
        encoded = self.tokenizer(text, add_special_tokens=False)
        if isinstance(encoded, Mapping):
            encoded = encoded.get("input_ids")
        return _as_token_ids(encoded, "assistant answer")

    def _gold_relations_by_source(
        self, record: PreprocessedRecord
    ) -> dict[str, tuple[int, ...]]:
        relations: dict[str, set[int]] = {}
        graph = record.local_graph
        for group in record.target_groups:
            for edge_index in group.support_edge_indices:
                try:
                    edge = graph.edges[edge_index]
                    source_qid = graph.nodes[edge.source_local_node_id].qid
                except IndexError as error:
                    raise CollationError(
                        f"{record.example_id} contains an invalid support edge index"
                    ) from error
                relations.setdefault(source_qid, set()).add(edge.relation_id)
        return {qid: tuple(sorted(values)) for qid, values in relations.items()}

    def _relation_supervision(
        self, supervised_relations: Sequence[tuple[int, ...]]
    ) -> tuple[Tensor, Tensor]:
        mention_count = len(supervised_relations)
        targets = torch.zeros(
            (self.num_kil_steps, mention_count, self.num_relations),
            dtype=torch.float32,
        )
        mask = torch.zeros((self.num_kil_steps, mention_count), dtype=torch.bool)
        for mention_index, relation_ids in enumerate(supervised_relations):
            if not relation_ids:
                continue
            for relation_id in relation_ids:
                if not 0 <= relation_id < self.num_relations:
                    raise CollationError(
                        f"supervised relation ID {relation_id} is out of range"
                    )
            probability = 1.0 / len(relation_ids)
            targets[0, mention_index, list(relation_ids)] = probability
            mask[0, mention_index] = True
        return targets, mask


def _entity_embedding_request(
    store: Wikidata5MStore, entity_id: int, qid: str
) -> KnowledgeEmbeddingRequest:
    aliases = store.entity_aliases(qid)
    label = aliases[0] if aliases else qid
    text = format_entity_text(label, aliases=aliases[1:])
    return KnowledgeEmbeddingRequest(
        kind="entity",
        dense_id=entity_id,
        identifier=qid,
        text=text,
        format_version=ENTITY_TEXT_FORMAT_VERSION,
    )


def _relation_embedding_request(
    store: Wikidata5MStore, relation_id: int, pid: str
) -> KnowledgeEmbeddingRequest:
    aliases = store.relation_aliases(pid)
    label = aliases[0] if aliases else pid
    text = format_relation_text(label)
    return KnowledgeEmbeddingRequest(
        kind="relation",
        dense_id=relation_id,
        identifier=pid,
        text=text,
        format_version=RELATION_TEXT_FORMAT_VERSION,
    )


def _as_token_ids(value: Any, source: str) -> list[int]:
    if isinstance(value, Tensor):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise CollationError(f"{source} did not produce a flat token-ID sequence")
    return list(value)


__all__ = [
    "CollationError",
    "DatasetError",
    "ENTITY_TEXT_FORMAT_VERSION",
    "KnowledgeEmbeddingRequest",
    "OreoSemanticCollator",
    "RELATION_TEXT_FORMAT_VERSION",
    "ReconstructionDatasetItem",
    "SemanticKGInputBatch",
    "SemanticOreoBatch",
    "Wikidata5MReconstructionDataset",
]
