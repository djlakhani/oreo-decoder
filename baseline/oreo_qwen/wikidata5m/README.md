# Wikidata5M Hybrid Datastore

This package provides read-only, indexed access to the local Wikidata5M
passages, aliases, and transductive graph. It is intended to support dataset
preprocessing without repeatedly scanning multi-gigabyte text files.

The implementation code lives here. Generated artifacts live outside the Git
repository under:

```text
datasets/Wikidata5M/indexed/
```

The current index has already been built and validated. Do not rebuild it
unless a raw Wikidata5M source file changes.

## Storage overview

The datastore combines three forms of storage:

1. The original Wikidata5M text files remain the canonical passage and alias
   payloads.
2. `metadata.sqlite` maps QIDs/PIDs to compact dense IDs and records byte
   offsets into the passage and alias files.
3. Memory-mapped NumPy arrays store incoming and outgoing graph adjacency in
   compressed sparse row (CSR) form.

`manifest.json` ties these components together. It records counts, source file
metadata, array shapes and dtypes, and SHA-256 checksums used for stale or
corrupt index detection.

### Generated layout

```text
datasets/Wikidata5M/indexed/
├── manifest.json
├── metadata.sqlite
└── graph/
    ├── out_offsets.npy
    ├── out_neighbors.npy
    ├── out_relations.npy
    ├── out_splits.npy
    ├── in_offsets.npy
    ├── in_neighbors.npy
    ├── in_relations.npy
    └── in_splits.npy
```

## Dense IDs and SQLite metadata

Entities and relations are assigned deterministic dense integer IDs in numeric
QID/PID order. SQLite provides both directions of each mapping:

```python
entity_id = store.entity_id("Q5")
qid = store.entity_qid(entity_id)

relation_id = store.relation_id("P31")
pid = store.relation_pid(relation_id)
```

The `entities` table contains:

```text
dense_id
qid
passage_offset
passage_length
alias_offset
alias_length
```

The `relations` table contains:

```text
dense_id
pid
alias_offset
alias_length
```

Offsets and lengths are measured in **bytes**, not Unicode characters. They
point to the payload after the `QID<TAB>` or `PID<TAB>` prefix and exclude the
line ending. The public API seeks and decodes these payloads automatically, so
preprocessing code should not normally open the raw files directly.

A graph-only entity can be known to the index but have no passage or alias row.
For such an entity, `passage()` returns `None` and `entity_aliases()` returns an
empty tuple. Unknown identifiers raise typed errors.

## Graph representation

For a triple:

```text
head --relation--> tail
```

the outgoing arrays group it under `head`, while the incoming arrays group it
under `tail`. Every source triple therefore appears once in each orientation.

For dense entity ID `i`, its outgoing range is:

```python
start = out_offsets[i]
stop = out_offsets[i + 1]
```

Within that half-open range `[start, stop)`:

```text
out_neighbors[j] = tail entity ID
out_relations[j] = relation ID
out_splits[j] = train/valid/test code
```

Its incoming range is found the same way:

```python
start = in_offsets[i]
stop = in_offsets[i + 1]
```

Within that range:

```text
in_neighbors[j] = head entity ID
in_relations[j] = relation ID
in_splits[j] = train/valid/test code
```

The offset arrays have `entity_count + 1` values. Values inside an offset array
are edge-array positions, not entity IDs. Adjacent entities share a boundary
without overlapping because Python/NumPy slices exclude the stop position:

```text
entity i:     [offsets[i],     offsets[i + 1])
entity i + 1: [offsets[i + 1], offsets[i + 2])
```

The dtypes are deliberately compact:

| Array | Dtype | Meaning |
|---|---|---|
| `*_offsets.npy` | `uint64` | Boundaries in the flattened edge arrays |
| `*_neighbors.npy` | `uint32` | Dense entity IDs |
| `*_relations.npy` | `uint16` | Dense relation IDs |
| `*_splits.npy` | `uint8` | `train=0`, `valid=1`, `test=2` |

Files are loaded with `np.load(..., mmap_mode="r")`. They remain persistent on
disk, and the operating system reads only the pages needed for a query.

Edges are grouped by entity. Within one entity's group, source order is stable:
train rows first, then valid rows, then test rows, preserving line order within
each split. Exact duplicate source triples are preserved.

## Python quick start

From `oreo-decoder/baseline`, ensure the package environment exists:

```bash
uv sync
```

Then open the store with a context manager:

```python
from pathlib import Path

from oreo_qwen.wikidata5m import Wikidata5MStore

index = Path("../../datasets/Wikidata5M/indexed")

with Wikidata5MStore(index) as store:
    print(store.entity_count)
    print(store.relation_count)
```

Opening the store performs quick validation of source file statistics, SQLite,
and array schemas. Closing it releases SQLite and raw-file handles.

### Retrieve passages and aliases

```python
with Wikidata5MStore(index) as store:
    passage = store.passage("Q5")
    entity_aliases = store.entity_aliases("Q5")
    relation_aliases = store.relation_aliases("P31")
```

Use `require_passage()` when a missing passage should be an error:

```python
passage = store.require_passage("Q5")
```

### Find entities pointing to target X

Incoming edges for `X` are triples of the form:

```text
other_entity --relation--> X
```

The decoded API is convenient for inspection:

```python
with Wikidata5MStore(index) as store:
    edges = store.edges(
        "Q5",
        direction="incoming",
        splits=["train"],
    )

    for edge in edges:
        print(edge.head_qid, edge.relation_pid, edge.tail_qid)
```

Use the dense API for large-scale preprocessing:

```python
import numpy as np

with Wikidata5MStore(index) as store:
    target_id = store.entity_id("Q5")
    batch = store.edges_dense(
        target_id,
        direction="incoming",
        splits=["train"],
    )

    # For incoming edges, these are the entities pointing to Q5.
    source_ids = batch.head_ids
    unique_source_ids = np.unique(source_ids)
```

`DenseEdgeBatch` contains aligned arrays:

```text
head_ids
relation_ids
tail_ids
split_codes
direction_codes
```

For an incoming query, `tail_ids` contains the queried target ID. For an
outgoing query, `head_ids` contains the queried source ID.

### Relation filtering

The decoded API accepts PIDs:

```python
edges = store.edges(
    "Q5",
    direction="incoming",
    splits=["train"],
    relations=["P31"],
)
```

The dense API accepts dense relation IDs:

```python
p31_id = store.relation_id("P31")
batch = store.edges_dense(
    store.entity_id("Q5"),
    direction="incoming",
    splits=["train"],
    relation_ids=[p31_id],
)
```

### Split safety

Every graph query requires an explicit nonempty split selection. This is
intentional: it prevents preprocessing code from silently including validation
or test edges.

Training preprocessing should normally use:

```python
splits=["train"]
```

Use all graph edges only when that is explicitly intended:

```python
splits=["train", "valid", "test"]
```

### Deterministic iteration

The store supports deterministic iterators:

```python
store.iter_entities()
store.iter_relations()
store.iter_passages()
store.iter_entity_aliases()
store.iter_relation_aliases()
store.iter_edges(splits=["train"])
```

`iter_edges()` returns decoded Python objects and is intended for inspection or
moderate workloads. Prefer entity-wise `edges_dense()` calls for high-volume
preprocessing.

## CLI

Run these commands from `oreo-decoder/baseline`.

Show index counts:

```bash
uv run wikidata5m-index stats ../../datasets/Wikidata5M/indexed
```

Inspect an entity and its train-split outgoing edges:

```bash
uv run wikidata5m-index inspect ../../datasets/Wikidata5M/indexed \
  --qid Q29387131 \
  --edges \
  --direction outgoing \
  --split train
```

Run quick validation:

```bash
uv run wikidata5m-index validate ../../datasets/Wikidata5M/indexed
```

Run full source/artifact checksum and graph-value validation:

```bash
uv run wikidata5m-index validate ../../datasets/Wikidata5M/indexed --full
```

## Rebuilding

The index is already ready for use. Rebuild only when a raw passage, alias, or
triple file changes:

```bash
uv run wikidata5m-index build ../../datasets/Wikidata5M --force
```

The builder writes to a hidden staging directory inside `indexed/`, validates
the completed artifacts, moves them into place, and publishes `manifest.json`
last. A partial build therefore cannot be mistaken for a complete index.

## Current boundary

This package supplies graph candidates, passages, and aliases. It does not yet
perform entity mention matching, ambiguity resolution, span selection,
corruption, local-graph construction, or training-example serialization. Those
belong to the next preprocessing layer and should consume this datastore rather
than reading the source triple files directly.
