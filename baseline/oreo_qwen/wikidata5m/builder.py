"""Build the persistent Wikidata5M hybrid index."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .errors import IndexBuildError, MalformedSourceError

SCHEMA_VERSION = 1
ENTITY_FILE = Path("wikidata5m_alias/wikidata5m_entity.txt")
RELATION_FILE = Path("wikidata5m_alias/wikidata5m_relation.txt")
PASSAGE_FILE = Path("wikidata5m_text.txt")
SPLIT_FILES = {
    "train": Path("wikidata5m_transductive/wikidata5m_transductive_train.txt"),
    "valid": Path("wikidata5m_transductive/wikidata5m_transductive_valid.txt"),
    "test": Path("wikidata5m_transductive/wikidata5m_transductive_test.txt"),
}
SPLIT_CODES = {"train": 0, "valid": 1, "test": 2}
ARRAY_DTYPES = {
    "offsets": np.dtype("uint64"),
    "neighbors": np.dtype("uint32"),
    "relations": np.dtype("uint16"),
    "splits": np.dtype("uint8"),
}
_ENTITY_RE = re.compile(rb"Q([1-9][0-9]*)\Z")
_RELATION_RE = re.compile(rb"P([1-9][0-9]*)\Z")
_GRAPH_CHUNK_SIZE = 250_000
_SQL_BATCH_SIZE = 50_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_without_newline(line: bytes) -> bytes:
    if line.endswith(b"\n"):
        line = line[:-1]
    if line.endswith(b"\r"):
        line = line[:-1]
    return line


def _numeric_id(raw: bytes, prefix: str, path: Path, line_number: int) -> int:
    matcher = _ENTITY_RE if prefix == "Q" else _RELATION_RE
    match = matcher.fullmatch(raw)
    if match is None:
        shown = raw[:80].decode("utf-8", errors="replace")
        raise MalformedSourceError(
            f"{path}:{line_number}: expected a canonical {prefix} identifier, got {shown!r}"
        )
    value = int(match.group(1))
    if value > np.iinfo(np.int64).max:
        raise MalformedSourceError(
            f"{path}:{line_number}: identifier exceeds SQLite's signed 64-bit range"
        )
    return value


def _stable_file_info(path: Path, digest: str) -> dict[str, int | str]:
    stat = path.stat()
    return {
        "path": str(path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest,
    }


def _assert_unchanged(path: Path, before: os.stat_result) -> None:
    after = path.stat()
    if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
        raise IndexBuildError(f"source file changed while it was being indexed: {path}")


def _create_staging_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        CREATE TABLE entities (
            qnum INTEGER PRIMARY KEY,
            passage_offset INTEGER,
            passage_length INTEGER,
            alias_offset INTEGER,
            alias_length INTEGER
        );
        CREATE TABLE relations (
            pnum INTEGER PRIMARY KEY,
            alias_offset INTEGER,
            alias_length INTEGER
        );
        CREATE TABLE passages (
            qnum INTEGER PRIMARY KEY,
            passage_offset INTEGER NOT NULL,
            passage_length INTEGER NOT NULL
        );
        """
    )
    return connection


def _scan_alias_file(
    connection: sqlite3.Connection,
    path: Path,
    *,
    entity: bool,
) -> tuple[dict[str, int | str], int, int]:
    before = path.stat()
    digest = hashlib.sha256()
    row_count = 0
    alias_count = 0
    table = "entities" if entity else "relations"
    number_column = "qnum" if entity else "pnum"
    prefix = "Q" if entity else "P"
    pending: list[tuple[int, int, int]] = []
    offset = 0

    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            digest.update(line)
            content = _content_without_newline(line)
            raw_identifier, separator, payload = content.partition(b"\t")
            if not separator:
                raise MalformedSourceError(
                    f"{path}:{line_number}: expected an identifier and tab-separated aliases"
                )
            number = _numeric_id(raw_identifier, prefix, path, line_number)
            try:
                payload.decode("utf-8")
            except UnicodeDecodeError as error:
                raise MalformedSourceError(
                    f"{path}:{line_number}: aliases are not valid UTF-8"
                ) from error
            payload_offset = offset + len(raw_identifier) + 1
            pending.append((number, payload_offset, len(payload)))
            row_count += 1
            alias_count += 0 if not payload else payload.count(b"\t") + 1
            offset += len(line)

            if len(pending) >= _SQL_BATCH_SIZE:
                try:
                    connection.executemany(
                        f"INSERT INTO {table} ({number_column}, alias_offset, alias_length) "
                        "VALUES (?, ?, ?)",
                        pending,
                    )
                except sqlite3.IntegrityError as error:
                    raise MalformedSourceError(
                        f"{path}: duplicate {prefix} identifier in alias file"
                    ) from error
                connection.commit()
                pending.clear()

    if pending:
        try:
            connection.executemany(
                f"INSERT INTO {table} ({number_column}, alias_offset, alias_length) "
                "VALUES (?, ?, ?)",
                pending,
            )
        except sqlite3.IntegrityError as error:
            raise MalformedSourceError(
                f"{path}: duplicate {prefix} identifier in alias file"
            ) from error
        connection.commit()

    _assert_unchanged(path, before)
    return _stable_file_info(path, digest.hexdigest()), row_count, alias_count


def _scan_passages(
    connection: sqlite3.Connection, path: Path
) -> tuple[dict[str, int | str], int]:
    before = path.stat()
    digest = hashlib.sha256()
    row_count = 0
    pending: list[tuple[int, int, int]] = []
    offset = 0

    def flush(rows: list[tuple[int, int, int]]) -> None:
        try:
            connection.executemany(
                "INSERT INTO passages VALUES (?, ?, ?)", rows
            )
        except sqlite3.IntegrityError as error:
            raise MalformedSourceError(f"{path}: duplicate QID passage row") from error
        connection.commit()

    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            digest.update(line)
            content = _content_without_newline(line)
            raw_identifier, separator, payload = content.partition(b"\t")
            if not separator:
                raise MalformedSourceError(
                    f"{path}:{line_number}: expected a QID and tab-separated passage"
                )
            qnum = _numeric_id(raw_identifier, "Q", path, line_number)
            try:
                payload.decode("utf-8")
            except UnicodeDecodeError as error:
                raise MalformedSourceError(
                    f"{path}:{line_number}: passage is not valid UTF-8"
                ) from error
            pending.append((qnum, offset + len(raw_identifier) + 1, len(payload)))
            row_count += 1
            offset += len(line)
            if len(pending) >= _SQL_BATCH_SIZE:
                flush(pending)
                pending.clear()
    if pending:
        flush(pending)

    connection.execute(
        "INSERT OR IGNORE INTO entities (qnum) SELECT qnum FROM passages"
    )
    connection.commit()

    _assert_unchanged(path, before)
    return _stable_file_info(path, digest.hexdigest()), row_count


def _iter_graph_chunks(
    path: Path,
    *,
    digest: hashlib._Hash | None = None,
    chunk_size: int = _GRAPH_CHUNK_SIZE,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    before = path.stat()
    heads: list[int] = []
    relations: list[int] = []
    tails: list[int] = []
    with path.open("rb") as handle:
        for line_number, line in enumerate(handle, 1):
            if digest is not None:
                digest.update(line)
            content = _content_without_newline(line)
            fields = content.split(b"\t")
            if len(fields) != 3:
                raise MalformedSourceError(
                    f"{path}:{line_number}: expected exactly three tab-separated fields"
                )
            heads.append(_numeric_id(fields[0], "Q", path, line_number))
            relations.append(_numeric_id(fields[1], "P", path, line_number))
            tails.append(_numeric_id(fields[2], "Q", path, line_number))
            if len(heads) >= chunk_size:
                yield (
                    np.asarray(heads, dtype=np.uint64),
                    np.asarray(relations, dtype=np.uint64),
                    np.asarray(tails, dtype=np.uint64),
                )
                heads.clear()
                relations.clear()
                tails.clear()
    if heads:
        yield (
            np.asarray(heads, dtype=np.uint64),
            np.asarray(relations, dtype=np.uint64),
            np.asarray(tails, dtype=np.uint64),
        )
    _assert_unchanged(path, before)


def _discover_graph_ids(
    connection: sqlite3.Connection, path: Path
) -> tuple[dict[str, int | str], int]:
    digest = hashlib.sha256()
    edge_count = 0
    for heads, relations, tails in _iter_graph_chunks(path, digest=digest):
        entity_numbers = np.unique(np.concatenate((heads, tails)))
        relation_numbers = np.unique(relations)
        connection.executemany(
            "INSERT OR IGNORE INTO entities (qnum) VALUES (?)",
            ((int(value),) for value in entity_numbers),
        )
        connection.executemany(
            "INSERT OR IGNORE INTO relations (pnum) VALUES (?)",
            ((int(value),) for value in relation_numbers),
        )
        connection.commit()
        edge_count += int(heads.shape[0])
    return _stable_file_info(path, digest.hexdigest()), edge_count


def _create_metadata_database(
    staging_connection: sqlite3.Connection,
    path: Path,
    work_dir: Path,
) -> tuple[np.memmap, np.memmap, int, int]:
    entity_count = int(
        staging_connection.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
    )
    relation_count = int(
        staging_connection.execute("SELECT COUNT(*) FROM relations").fetchone()[0]
    )
    if entity_count > np.iinfo(np.uint32).max + 1:
        raise IndexBuildError("entity count exceeds uint32 dense-ID capacity")
    if relation_count > np.iinfo(np.uint16).max + 1:
        raise IndexBuildError("relation count exceeds uint16 dense-ID capacity")

    q_numbers = np.lib.format.open_memmap(
        work_dir / "q_numbers.npy", mode="w+", dtype=np.uint64, shape=(entity_count,)
    )
    p_numbers = np.lib.format.open_memmap(
        work_dir / "p_numbers.npy", mode="w+", dtype=np.uint64, shape=(relation_count,)
    )
    final = sqlite3.connect(path)
    final.executescript(
        """
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE entities (
            dense_id INTEGER PRIMARY KEY,
            qid TEXT NOT NULL UNIQUE,
            passage_offset INTEGER,
            passage_length INTEGER,
            alias_offset INTEGER,
            alias_length INTEGER
        );
        CREATE TABLE relations (
            dense_id INTEGER PRIMARY KEY,
            pid TEXT NOT NULL UNIQUE,
            alias_offset INTEGER,
            alias_length INTEGER
        );
        CREATE TABLE index_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )

    cursor = staging_connection.execute(
        """SELECT e.qnum, p.passage_offset, p.passage_length,
                  e.alias_offset, e.alias_length
           FROM entities AS e
           LEFT JOIN passages AS p ON p.qnum = e.qnum
           ORDER BY e.qnum"""
    )
    dense_id = 0
    while rows := cursor.fetchmany(_SQL_BATCH_SIZE):
        output_rows = []
        for qnum, passage_offset, passage_length, alias_offset, alias_length in rows:
            q_numbers[dense_id] = qnum
            output_rows.append(
                (
                    dense_id,
                    f"Q{qnum}",
                    passage_offset,
                    passage_length,
                    alias_offset,
                    alias_length,
                )
            )
            dense_id += 1
        final.executemany("INSERT INTO entities VALUES (?, ?, ?, ?, ?, ?)", output_rows)

    cursor = staging_connection.execute(
        "SELECT pnum, alias_offset, alias_length FROM relations ORDER BY pnum"
    )
    dense_id = 0
    while rows := cursor.fetchmany(_SQL_BATCH_SIZE):
        output_rows = []
        for pnum, alias_offset, alias_length in rows:
            p_numbers[dense_id] = pnum
            output_rows.append((dense_id, f"P{pnum}", alias_offset, alias_length))
            dense_id += 1
        final.executemany("INSERT INTO relations VALUES (?, ?, ?, ?)", output_rows)

    final.executemany(
        "INSERT INTO index_metadata VALUES (?, ?)",
        (("schema_version", str(SCHEMA_VERSION)), ("complete", "1")),
    )
    final.commit()
    result = final.execute("PRAGMA integrity_check").fetchone()
    final.close()
    if result is None or result[0] != "ok":
        raise IndexBuildError(f"new metadata database failed integrity check: {result}")
    q_numbers.flush()
    p_numbers.flush()
    return q_numbers, p_numbers, entity_count, relation_count


def _map_numbers(numbers: np.ndarray, sorted_numbers: np.ndarray, kind: str) -> np.ndarray:
    positions = np.searchsorted(sorted_numbers, numbers)
    if np.any(positions >= sorted_numbers.shape[0]) or np.any(
        sorted_numbers[np.minimum(positions, sorted_numbers.shape[0] - 1)] != numbers
    ):
        raise IndexBuildError(f"internal error: graph references an undiscovered {kind}")
    return positions


def _write_offsets(path: Path, degrees: np.ndarray) -> np.memmap:
    offsets = np.lib.format.open_memmap(
        path, mode="w+", dtype=ARRAY_DTYPES["offsets"], shape=(degrees.shape[0] + 1,)
    )
    offsets[0] = 0
    np.cumsum(degrees, dtype=np.uint64, out=offsets[1:])
    offsets.flush()
    return offsets


def _placement_positions(ids: np.ndarray, cursors: np.ndarray) -> np.ndarray:
    """Return stable CSR insertion positions for a chunk with repeated IDs."""

    if ids.size == 0:
        return np.empty(0, dtype=np.uint64)
    order = np.argsort(ids, kind="stable")
    sorted_ids = ids[order]
    starts = np.empty(ids.shape[0], dtype=bool)
    starts[0] = True
    starts[1:] = sorted_ids[1:] != sorted_ids[:-1]
    group_starts = np.flatnonzero(starts)
    group_lengths = np.diff(np.append(group_starts, ids.shape[0]))
    ranks_sorted = np.arange(ids.shape[0], dtype=np.uint64) - np.repeat(
        group_starts.astype(np.uint64), group_lengths
    )
    ranks = np.empty(ids.shape[0], dtype=np.uint64)
    ranks[order] = ranks_sorted
    positions = cursors[ids] + ranks
    np.add.at(cursors, ids, 1)
    return positions


def _build_graph_arrays(
    source_paths: dict[str, Path],
    graph_dir: Path,
    q_numbers: np.ndarray,
    p_numbers: np.ndarray,
    entity_count: int,
    edge_count: int,
) -> None:
    out_degrees = np.zeros(entity_count, dtype=np.uint64)
    in_degrees = np.zeros(entity_count, dtype=np.uint64)
    for split in ("train", "valid", "test"):
        for heads, _, tails in _iter_graph_chunks(source_paths[split]):
            head_ids = _map_numbers(heads, q_numbers, "entity")
            tail_ids = _map_numbers(tails, q_numbers, "entity")
            np.add.at(out_degrees, head_ids, 1)
            np.add.at(in_degrees, tail_ids, 1)

    graph_dir.mkdir(parents=True, exist_ok=True)
    out_offsets = _write_offsets(graph_dir / "out_offsets.npy", out_degrees)
    in_offsets = _write_offsets(graph_dir / "in_offsets.npy", in_degrees)
    if int(out_offsets[-1]) != edge_count or int(in_offsets[-1]) != edge_count:
        raise IndexBuildError("edge counts changed between discovery and degree passes")

    arrays: dict[str, np.memmap] = {}
    for prefix in ("out", "in"):
        arrays[f"{prefix}_neighbors"] = np.lib.format.open_memmap(
            graph_dir / f"{prefix}_neighbors.npy",
            mode="w+",
            dtype=ARRAY_DTYPES["neighbors"],
            shape=(edge_count,),
        )
        arrays[f"{prefix}_relations"] = np.lib.format.open_memmap(
            graph_dir / f"{prefix}_relations.npy",
            mode="w+",
            dtype=ARRAY_DTYPES["relations"],
            shape=(edge_count,),
        )
        arrays[f"{prefix}_splits"] = np.lib.format.open_memmap(
            graph_dir / f"{prefix}_splits.npy",
            mode="w+",
            dtype=ARRAY_DTYPES["splits"],
            shape=(edge_count,),
        )

    out_cursors = np.asarray(out_offsets[:-1]).copy()
    in_cursors = np.asarray(in_offsets[:-1]).copy()
    for split in ("train", "valid", "test"):
        split_code = SPLIT_CODES[split]
        for heads, relations, tails in _iter_graph_chunks(source_paths[split]):
            head_ids = _map_numbers(heads, q_numbers, "entity")
            relation_ids = _map_numbers(relations, p_numbers, "relation")
            tail_ids = _map_numbers(tails, q_numbers, "entity")
            out_positions = _placement_positions(head_ids, out_cursors)
            in_positions = _placement_positions(tail_ids, in_cursors)
            arrays["out_neighbors"][out_positions] = tail_ids
            arrays["out_relations"][out_positions] = relation_ids
            arrays["out_splits"][out_positions] = split_code
            arrays["in_neighbors"][in_positions] = head_ids
            arrays["in_relations"][in_positions] = relation_ids
            arrays["in_splits"][in_positions] = split_code

    if not np.array_equal(out_cursors, np.asarray(out_offsets[1:])):
        raise IndexBuildError("outgoing CSR fill did not reach all expected offsets")
    if not np.array_equal(in_cursors, np.asarray(in_offsets[1:])):
        raise IndexBuildError("incoming CSR fill did not reach all expected offsets")
    for array in arrays.values():
        array.flush()


def _artifact_info(path: Path, root: Path) -> dict[str, object]:
    info: dict[str, object] = {
        "path": str(path.relative_to(root)),
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }
    if path.suffix == ".npy":
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        info["dtype"] = array.dtype.name
        info["shape"] = list(array.shape)
    return info


def _publish(staging_dir: Path, index_dir: Path, *, force: bool) -> None:
    manifest_path = index_dir / "manifest.json"
    if manifest_path.exists() and not force:
        raise IndexBuildError(
            f"index already exists at {index_dir}; pass force=True to replace it"
        )
    if force and manifest_path.exists():
        manifest_path.unlink()
    final_graph = index_dir / "graph"
    final_graph.mkdir(parents=True, exist_ok=True)
    os.replace(staging_dir / "metadata.sqlite", index_dir / "metadata.sqlite")
    for source in sorted((staging_dir / "graph").glob("*.npy")):
        os.replace(source, final_graph / source.name)
    os.replace(staging_dir / "manifest.json", manifest_path)


def build_index(
    source_root: str | Path,
    output_dir: str | Path | None = None,
    *,
    force: bool = False,
) -> Path:
    """Build and publish a Wikidata5M index.

    Args:
        source_root: Directory containing the unmodified Wikidata5M files.
        output_dir: Artifact directory. Defaults to ``source_root/indexed``.
        force: Replace a complete existing index after the new build validates.
    """

    source_root = Path(source_root).expanduser().resolve()
    output = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else source_root / "indexed"
    )
    required = [ENTITY_FILE, RELATION_FILE, PASSAGE_FILE, *SPLIT_FILES.values()]
    missing = [
        str(source_root / relative)
        for relative in required
        if not (source_root / relative).is_file()
    ]
    if missing:
        raise IndexBuildError("missing Wikidata5M source files: " + ", ".join(missing))

    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists() and not force:
        raise IndexBuildError(
            f"index already exists at {output}; pass force=True to replace it"
        )
    staging_dir = Path(tempfile.mkdtemp(prefix=".build-", dir=output))
    staging_database = staging_dir / "staging.sqlite"
    connection: sqlite3.Connection | None = None
    try:
        connection = _create_staging_database(staging_database)
        source_info: dict[str, dict[str, int | str]] = {}
        counts: dict[str, object] = {}

        info, entity_alias_rows, entity_aliases = _scan_alias_file(
            connection, source_root / ENTITY_FILE, entity=True
        )
        source_info["entity_aliases"] = info
        counts["entity_alias_rows"] = entity_alias_rows
        counts["entity_aliases"] = entity_aliases

        info, relation_alias_rows, relation_aliases = _scan_alias_file(
            connection, source_root / RELATION_FILE, entity=False
        )
        source_info["relation_aliases"] = info
        counts["relation_alias_rows"] = relation_alias_rows
        counts["relation_aliases"] = relation_aliases

        info, passage_count = _scan_passages(connection, source_root / PASSAGE_FILE)
        source_info["passages"] = info
        counts["passages"] = passage_count

        edge_counts: dict[str, int] = {}
        graph_paths: dict[str, Path] = {}
        for split, relative in SPLIT_FILES.items():
            path = source_root / relative
            graph_paths[split] = path
            info, count = _discover_graph_ids(connection, path)
            source_info[f"graph_{split}"] = info
            edge_counts[split] = count
        counts["edges_by_split"] = edge_counts
        edge_count = sum(edge_counts.values())
        counts["edges"] = edge_count

        q_numbers, p_numbers, entity_count, relation_count = _create_metadata_database(
            connection, staging_dir / "metadata.sqlite", staging_dir
        )
        counts["entities"] = entity_count
        counts["relations"] = relation_count
        connection.close()
        connection = None
        staging_database.unlink()

        _build_graph_arrays(
            graph_paths,
            staging_dir / "graph",
            q_numbers,
            p_numbers,
            entity_count,
            edge_count,
        )
        del q_numbers, p_numbers
        (staging_dir / "q_numbers.npy").unlink()
        (staging_dir / "p_numbers.npy").unlink()

        artifact_paths = [staging_dir / "metadata.sqlite"] + sorted(
            (staging_dir / "graph").glob("*.npy")
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_root": str(source_root),
            "sources": source_info,
            "counts": counts,
            "split_codes": SPLIT_CODES,
            "artifacts": {
                str(path.relative_to(staging_dir)): _artifact_info(path, staging_dir)
                for path in artifact_paths
            },
        }
        manifest_path = staging_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        # Import lazily to avoid a builder/store import cycle.
        from .store import validate_index

        validate_index(staging_dir, source_root=source_root, full=False)
        _publish(staging_dir, output, force=force)
        return output
    except Exception:
        if connection is not None:
            connection.close()
        raise
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
