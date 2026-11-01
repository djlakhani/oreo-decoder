"""Read and validate a persistent Wikidata5M index."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO, Collection, Iterator, Sequence

import numpy as np

from .builder import (
    ARRAY_DTYPES,
    ENTITY_FILE,
    PASSAGE_FILE,
    RELATION_FILE,
    SCHEMA_VERSION,
    SPLIT_FILES,
)
from .errors import (
    CorruptIndexError,
    MissingDataError,
    StaleIndexError,
    UnknownEntityError,
    UnknownRelationError,
)
from .types import (
    CODE_TO_SPLIT,
    SPLIT_TO_CODE,
    DenseEdgeBatch,
    Edge,
    EdgeDirection,
    EntityMapping,
    GraphSplit,
    QueryDirection,
    RelationMapping,
)

_SOURCE_RELATIVES = {
    "entity_aliases": ENTITY_FILE,
    "relation_aliases": RELATION_FILE,
    "passages": PASSAGE_FILE,
    **{f"graph_{split}": path for split, path in SPLIT_FILES.items()},
}
_ARRAY_NAMES = (
    "out_offsets",
    "out_neighbors",
    "out_relations",
    "out_splits",
    "in_offsets",
    "in_neighbors",
    "in_relations",
    "in_splits",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_manifest(index_dir: Path) -> dict[str, object]:
    path = index_dir / "manifest.json"
    if not path.is_file():
        raise CorruptIndexError(f"index has no publication manifest: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CorruptIndexError(f"cannot read index manifest {path}: {error}") from error
    if not isinstance(manifest, dict):
        raise CorruptIndexError("manifest root must be a JSON object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise CorruptIndexError(
            f"unsupported index schema {manifest.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    return manifest


def _source_root(manifest: dict[str, object], override: str | Path | None) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()
    value = manifest.get("source_root")
    if not isinstance(value, str):
        raise CorruptIndexError("manifest source_root is missing or invalid")
    return Path(value).expanduser().resolve()


def _validate_sources(
    manifest: dict[str, object], source_root: Path, *, full: bool
) -> None:
    sources = manifest.get("sources")
    if not isinstance(sources, dict):
        raise CorruptIndexError("manifest sources table is missing or invalid")
    for name, relative in _SOURCE_RELATIVES.items():
        expected = sources.get(name)
        if not isinstance(expected, dict):
            raise CorruptIndexError(f"manifest has no source entry for {name}")
        path = source_root / relative
        if not path.is_file():
            raise StaleIndexError(f"indexed source file is missing: {path}")
        stat = path.stat()
        if stat.st_size != expected.get("size"):
            raise StaleIndexError(f"indexed source file size changed: {path}")
        if full:
            if _sha256(path) != expected.get("sha256"):
                raise StaleIndexError(f"indexed source file content changed: {path}")
        elif stat.st_mtime_ns != expected.get("mtime_ns"):
            raise StaleIndexError(f"indexed source file modification time changed: {path}")


def _safe_artifact_path(index_dir: Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise CorruptIndexError("manifest artifact path must be a string")
    root = index_dir.resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise CorruptIndexError(f"artifact path escapes the index directory: {relative}")
    return path


def _validate_artifacts(
    index_dir: Path, manifest: dict[str, object], *, full: bool
) -> None:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise CorruptIndexError("manifest artifacts table is missing or invalid")
    expected_names = {"metadata.sqlite", *{f"graph/{name}.npy" for name in _ARRAY_NAMES}}
    if set(artifacts) != expected_names:
        missing = sorted(expected_names - set(artifacts))
        extra = sorted(set(artifacts) - expected_names)
        raise CorruptIndexError(
            f"manifest artifact set is invalid; missing={missing}, extra={extra}"
        )
    for name, raw_info in artifacts.items():
        if not isinstance(raw_info, dict):
            raise CorruptIndexError(f"manifest artifact entry is invalid: {name}")
        path = _safe_artifact_path(index_dir, raw_info.get("path"))
        if not path.is_file():
            raise CorruptIndexError(f"index artifact is missing: {path}")
        if path.stat().st_size != raw_info.get("size"):
            raise CorruptIndexError(f"index artifact size is wrong: {path}")
        if full and _sha256(path) != raw_info.get("sha256"):
            raise CorruptIndexError(f"index artifact checksum failed: {path}")
        if path.suffix == ".npy":
            try:
                array = np.load(path, mmap_mode="r", allow_pickle=False)
            except Exception as error:
                raise CorruptIndexError(f"cannot load array artifact {path}: {error}") from error
            if array.dtype.name != raw_info.get("dtype"):
                raise CorruptIndexError(f"array dtype does not match manifest: {path}")
            if list(array.shape) != raw_info.get("shape"):
                raise CorruptIndexError(f"array shape does not match manifest: {path}")


def _open_readonly_database(path: Path) -> sqlite3.Connection:
    try:
        connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        return connection
    except sqlite3.Error as error:
        raise CorruptIndexError(f"cannot open metadata database {path}: {error}") from error


def _validate_database(index_dir: Path, manifest: dict[str, object]) -> None:
    path = index_dir / "metadata.sqlite"
    connection = _open_readonly_database(path)
    try:
        result = connection.execute("PRAGMA quick_check").fetchone()
        if result is None or result[0] != "ok":
            raise CorruptIndexError(f"metadata database quick check failed: {result}")
        version = connection.execute(
            "SELECT value FROM index_metadata WHERE key='schema_version'"
        ).fetchone()
        if version is None or int(version[0]) != SCHEMA_VERSION:
            raise CorruptIndexError("metadata database schema version is invalid")
        counts = manifest.get("counts")
        if not isinstance(counts, dict):
            raise CorruptIndexError("manifest counts table is missing or invalid")
        entity_count = connection.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        relation_count = connection.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
        if entity_count != counts.get("entities"):
            raise CorruptIndexError("metadata entity count does not match manifest")
        if relation_count != counts.get("relations"):
            raise CorruptIndexError("metadata relation count does not match manifest")
    except sqlite3.Error as error:
        raise CorruptIndexError(f"metadata database is invalid: {error}") from error
    finally:
        connection.close()


def _validate_graph(index_dir: Path, manifest: dict[str, object], *, full: bool) -> None:
    counts = manifest["counts"]
    assert isinstance(counts, dict)
    entity_count = int(counts["entities"])
    relation_count = int(counts["relations"])
    edge_count = int(counts["edges"])
    arrays = {
        name: np.load(index_dir / "graph" / f"{name}.npy", mmap_mode="r", allow_pickle=False)
        for name in _ARRAY_NAMES
    }
    for prefix in ("out", "in"):
        offsets = arrays[f"{prefix}_offsets"]
        if offsets.dtype != ARRAY_DTYPES["offsets"] or offsets.shape != (entity_count + 1,):
            raise CorruptIndexError(f"{prefix} offsets have an invalid dtype or shape")
        if int(offsets[0]) != 0 or int(offsets[-1]) != edge_count:
            raise CorruptIndexError(f"{prefix} offsets have invalid boundaries")
        for suffix in ("neighbors", "relations", "splits"):
            array = arrays[f"{prefix}_{suffix}"]
            if array.dtype != ARRAY_DTYPES[suffix] or array.shape != (edge_count,):
                raise CorruptIndexError(
                    f"{prefix} {suffix} array has an invalid dtype or shape"
                )
        if full:
            if np.any(offsets[1:] < offsets[:-1]):
                raise CorruptIndexError(f"{prefix} offsets are not monotonic")
            chunk_size = 5_000_000
            for start in range(0, edge_count, chunk_size):
                stop = min(start + chunk_size, edge_count)
                if np.any(arrays[f"{prefix}_neighbors"][start:stop] >= entity_count):
                    raise CorruptIndexError(f"{prefix} neighbors contain an invalid entity ID")
                if np.any(arrays[f"{prefix}_relations"][start:stop] >= relation_count):
                    raise CorruptIndexError(f"{prefix} relations contain an invalid relation ID")
                if np.any(arrays[f"{prefix}_splits"][start:stop] > 2):
                    raise CorruptIndexError(f"{prefix} splits contain an invalid split code")


def validate_index(
    index_dir: str | Path,
    *,
    source_root: str | Path | None = None,
    full: bool = False,
) -> dict[str, object]:
    """Validate an index and return its manifest.

    Quick validation checks source stat metadata, SQLite consistency, and array
    schemas. Full validation additionally hashes every source and artifact and
    scans graph-array values.
    """

    index = Path(index_dir).expanduser().resolve()
    manifest = _read_manifest(index)
    sources = _source_root(manifest, source_root)
    _validate_sources(manifest, sources, full=full)
    _validate_artifacts(index, manifest, full=full)
    _validate_database(index, manifest)
    _validate_graph(index, manifest, full=full)
    return manifest


class Wikidata5MStore:
    """Read-only access to a validated Wikidata5M hybrid index."""

    def __init__(
        self,
        index_dir: str | Path,
        *,
        source_root: str | Path | None = None,
        validate: bool = True,
    ) -> None:
        self.index_dir = Path(index_dir).expanduser().resolve()
        self.manifest = (
            validate_index(self.index_dir, source_root=source_root, full=False)
            if validate
            else _read_manifest(self.index_dir)
        )
        self.source_root = _source_root(self.manifest, source_root)
        self._connection = _open_readonly_database(self.index_dir / "metadata.sqlite")
        self._arrays = {
            name: np.load(
                self.index_dir / "graph" / f"{name}.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
            for name in _ARRAY_NAMES
        }
        self._passage_handle: BinaryIO | None = None
        self._entity_alias_handle: BinaryIO | None = None
        self._relation_alias_handle: BinaryIO | None = None

    @classmethod
    def open(
        cls,
        index_dir: str | Path,
        *,
        source_root: str | Path | None = None,
        validate: bool = True,
    ) -> "Wikidata5MStore":
        """Open a store; provided for a more descriptive call site."""

        return cls(index_dir, source_root=source_root, validate=validate)

    def close(self) -> None:
        """Close SQLite and raw-text file handles."""

        self.entity_id.cache_clear()
        self.entity_qid.cache_clear()
        self.relation_id.cache_clear()
        self.relation_pid.cache_clear()
        self._connection.close()
        for handle in (
            self._passage_handle,
            self._entity_alias_handle,
            self._relation_alias_handle,
        ):
            if handle is not None:
                handle.close()

    def __enter__(self) -> "Wikidata5MStore":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    @property
    def entity_count(self) -> int:
        return int(self.manifest["counts"]["entities"])  # type: ignore[index]

    @property
    def relation_count(self) -> int:
        return int(self.manifest["counts"]["relations"])  # type: ignore[index]

    @lru_cache(maxsize=65_536)
    def entity_id(self, qid: str) -> int:
        row = self._connection.execute(
            "SELECT dense_id FROM entities WHERE qid=?", (qid,)
        ).fetchone()
        if row is None:
            raise UnknownEntityError(f"unknown entity QID: {qid}")
        return int(row[0])

    @lru_cache(maxsize=65_536)
    def entity_qid(self, entity_id: int) -> str:
        row = self._connection.execute(
            "SELECT qid FROM entities WHERE dense_id=?", (entity_id,)
        ).fetchone()
        if row is None:
            raise UnknownEntityError(f"unknown dense entity ID: {entity_id}")
        return str(row[0])

    @lru_cache(maxsize=4_096)
    def relation_id(self, pid: str) -> int:
        row = self._connection.execute(
            "SELECT dense_id FROM relations WHERE pid=?", (pid,)
        ).fetchone()
        if row is None:
            raise UnknownRelationError(f"unknown relation PID: {pid}")
        return int(row[0])

    @lru_cache(maxsize=4_096)
    def relation_pid(self, relation_id: int) -> str:
        row = self._connection.execute(
            "SELECT pid FROM relations WHERE dense_id=?", (relation_id,)
        ).fetchone()
        if row is None:
            raise UnknownRelationError(f"unknown dense relation ID: {relation_id}")
        return str(row[0])

    def _payload(
        self,
        handle_name: str,
        relative_path: Path,
        offset: int,
        length: int,
    ) -> str:
        handle = getattr(self, handle_name)
        if handle is None:
            handle = (self.source_root / relative_path).open("rb")
            setattr(self, handle_name, handle)
        handle.seek(offset)
        payload = handle.read(length)
        if len(payload) != length:
            raise CorruptIndexError(
                f"raw payload at offset {offset} in {relative_path} is truncated"
            )
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError as error:
            raise CorruptIndexError(
                f"raw payload at offset {offset} in {relative_path} is invalid UTF-8"
            ) from error

    def passage(self, qid: str) -> str | None:
        row = self._connection.execute(
            "SELECT passage_offset, passage_length FROM entities WHERE qid=?", (qid,)
        ).fetchone()
        if row is None:
            raise UnknownEntityError(f"unknown entity QID: {qid}")
        if row[0] is None:
            return None
        return self._payload(
            "_passage_handle", PASSAGE_FILE, int(row[0]), int(row[1])
        )

    def require_passage(self, qid: str) -> str:
        passage = self.passage(qid)
        if passage is None:
            raise MissingDataError(f"known entity {qid} has no passage")
        return passage

    @staticmethod
    def _split_alias_payload(payload: str) -> tuple[str, ...]:
        return () if not payload else tuple(payload.split("\t"))

    def entity_aliases(self, qid: str) -> tuple[str, ...]:
        row = self._connection.execute(
            "SELECT alias_offset, alias_length FROM entities WHERE qid=?", (qid,)
        ).fetchone()
        if row is None:
            raise UnknownEntityError(f"unknown entity QID: {qid}")
        if row[0] is None:
            return ()
        payload = self._payload(
            "_entity_alias_handle", ENTITY_FILE, int(row[0]), int(row[1])
        )
        return self._split_alias_payload(payload)

    def relation_aliases(self, pid: str) -> tuple[str, ...]:
        row = self._connection.execute(
            "SELECT alias_offset, alias_length FROM relations WHERE pid=?", (pid,)
        ).fetchone()
        if row is None:
            raise UnknownRelationError(f"unknown relation PID: {pid}")
        if row[0] is None:
            return ()
        payload = self._payload(
            "_relation_alias_handle", RELATION_FILE, int(row[0]), int(row[1])
        )
        return self._split_alias_payload(payload)

    @staticmethod
    def _split_codes(splits: Collection[GraphSplit | str] | None) -> np.ndarray:
        if splits is None:
            raise ValueError("splits must be provided explicitly")
        normalized: set[int] = set()
        for split in splits:
            try:
                normalized.add(SPLIT_TO_CODE[GraphSplit(split)])
            except (ValueError, KeyError) as error:
                raise ValueError(f"unknown graph split: {split!r}") from error
        if not normalized:
            raise ValueError("splits must contain at least one graph split")
        return np.asarray(sorted(normalized), dtype=np.uint8)

    @staticmethod
    def _query_direction(direction: QueryDirection | str) -> QueryDirection:
        try:
            return QueryDirection(direction)
        except ValueError as error:
            raise ValueError(f"unknown query direction: {direction!r}") from error

    def _dense_relation_filter(
        self, relation_ids: Collection[int] | None
    ) -> np.ndarray | None:
        if relation_ids is None:
            return None
        normalized: set[int] = set()
        for relation_id in relation_ids:
            if relation_id < 0 or relation_id >= self.relation_count:
                raise UnknownRelationError(f"unknown dense relation ID: {relation_id}")
            normalized.add(int(relation_id))
        return np.asarray(sorted(normalized), dtype=np.uint16)

    def _one_direction(
        self,
        entity_id: int,
        prefix: str,
        split_codes: np.ndarray,
        relation_ids: np.ndarray | None,
        *,
        remove_self: bool,
    ) -> DenseEdgeBatch:
        offsets = self._arrays[f"{prefix}_offsets"]
        start, stop = int(offsets[entity_id]), int(offsets[entity_id + 1])
        neighbors = self._arrays[f"{prefix}_neighbors"][start:stop]
        relations = self._arrays[f"{prefix}_relations"][start:stop]
        edge_splits = self._arrays[f"{prefix}_splits"][start:stop]
        mask = np.isin(edge_splits, split_codes)
        if relation_ids is not None:
            mask &= np.isin(relations, relation_ids)
        if remove_self:
            mask &= neighbors != entity_id
        neighbors = np.asarray(neighbors[mask], dtype=np.uint32)
        relations = np.asarray(relations[mask], dtype=np.uint16)
        edge_splits = np.asarray(edge_splits[mask], dtype=np.uint8)
        entity_ids = np.full(neighbors.shape, entity_id, dtype=np.uint32)
        if prefix == "out":
            heads, tails = entity_ids, neighbors
            direction_code = EdgeDirection.OUTGOING
        else:
            heads, tails = neighbors, entity_ids
            direction_code = EdgeDirection.INCOMING
        directions = np.full(neighbors.shape, direction_code, dtype=np.uint8)
        directions[heads == tails] = EdgeDirection.SELF
        return DenseEdgeBatch(heads, relations, tails, edge_splits, directions)

    @staticmethod
    def _concatenate(batches: Sequence[DenseEdgeBatch]) -> DenseEdgeBatch:
        if len(batches) == 1:
            return batches[0]
        return DenseEdgeBatch(
            *(np.concatenate([getattr(batch, field) for batch in batches]) for field in (
                "head_ids",
                "relation_ids",
                "tail_ids",
                "split_codes",
                "direction_codes",
            ))
        )

    def edges_dense(
        self,
        entity_id: int,
        *,
        direction: QueryDirection | str,
        splits: Collection[GraphSplit | str] | None,
        relation_ids: Collection[int] | None = None,
    ) -> DenseEdgeBatch:
        """Return vectorized edges for one dense entity ID."""

        if entity_id < 0 or entity_id >= self.entity_count:
            raise UnknownEntityError(f"unknown dense entity ID: {entity_id}")
        normalized_direction = self._query_direction(direction)
        split_codes = self._split_codes(splits)
        relation_filter = self._dense_relation_filter(relation_ids)
        batches: list[DenseEdgeBatch] = []
        if normalized_direction in (QueryDirection.OUTGOING, QueryDirection.BOTH):
            batches.append(
                self._one_direction(
                    entity_id,
                    "out",
                    split_codes,
                    relation_filter,
                    remove_self=False,
                )
            )
        if normalized_direction in (QueryDirection.INCOMING, QueryDirection.BOTH):
            batches.append(
                self._one_direction(
                    entity_id,
                    "in",
                    split_codes,
                    relation_filter,
                    remove_self=normalized_direction is QueryDirection.BOTH,
                )
            )
        return self._concatenate(batches)

    def edges(
        self,
        qid: str,
        *,
        direction: QueryDirection | str,
        splits: Collection[GraphSplit | str] | None,
        relations: Collection[str] | None = None,
    ) -> tuple[Edge, ...]:
        """Return decoded QID/PID edge records for one entity."""

        relation_ids = (
            None if relations is None else [self.relation_id(pid) for pid in relations]
        )
        batch = self.edges_dense(
            self.entity_id(qid),
            direction=direction,
            splits=splits,
            relation_ids=relation_ids,
        )
        return tuple(
            Edge(
                head_qid=self.entity_qid(int(batch.head_ids[index])),
                relation_pid=self.relation_pid(int(batch.relation_ids[index])),
                tail_qid=self.entity_qid(int(batch.tail_ids[index])),
                split=CODE_TO_SPLIT[int(batch.split_codes[index])],
                direction=EdgeDirection(int(batch.direction_codes[index])),
            )
            for index in range(len(batch))
        )

    def iter_entities(self) -> Iterator[EntityMapping]:
        cursor = self._connection.execute(
            "SELECT dense_id, qid FROM entities ORDER BY dense_id"
        )
        for row in cursor:
            yield EntityMapping(int(row[0]), str(row[1]))

    def iter_relations(self) -> Iterator[RelationMapping]:
        cursor = self._connection.execute(
            "SELECT dense_id, pid FROM relations ORDER BY dense_id"
        )
        for row in cursor:
            yield RelationMapping(int(row[0]), str(row[1]))

    def iter_passages(self) -> Iterator[tuple[str, str]]:
        cursor = self._connection.execute(
            "SELECT qid FROM entities WHERE passage_offset IS NOT NULL ORDER BY dense_id"
        )
        for row in cursor:
            qid = str(row[0])
            yield qid, self.require_passage(qid)

    def iter_entity_aliases(self) -> Iterator[tuple[str, tuple[str, ...]]]:
        for mapping in self.iter_entities():
            yield mapping.qid, self.entity_aliases(mapping.qid)

    def iter_relation_aliases(self) -> Iterator[tuple[str, tuple[str, ...]]]:
        for mapping in self.iter_relations():
            yield mapping.pid, self.relation_aliases(mapping.pid)

    def iter_edges(
        self,
        *,
        splits: Collection[GraphSplit | str] | None,
        relations: Collection[str] | None = None,
    ) -> Iterator[Edge]:
        """Iterate every stored triple once in dense-head adjacency order."""

        for mapping in self.iter_entities():
            yield from self.edges(
                mapping.qid,
                direction=QueryDirection.OUTGOING,
                splits=splits,
                relations=relations,
            )

    def stats(self) -> dict[str, object]:
        """Return the immutable counts recorded at build time."""

        counts = self.manifest.get("counts")
        assert isinstance(counts, dict)
        return dict(counts)
