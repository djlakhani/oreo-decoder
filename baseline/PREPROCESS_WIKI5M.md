# Wikidata5M Passage Preprocessing Plan

## Status and Review Gate

This document specifies how one indexed Wikidata5M passage becomes one
structured Stage-1 reconstruction example. It is a review document, not an
implementation. No preprocessing code or generated corpus should be created
until this contract is approved.

The datastore described in `oreo_qwen/wikidata5m/README.md` is the only source
for passage, alias, ID-mapping, and graph queries. Preprocessing must not scan
the raw triple files directly.

This document intentionally stops at an untokenized, full-passage example.
Tokenizer-budgeted cropping, Qwen chat-template rendering, tensor construction,
and batch collation are later stages.

## Goal

For the passage associated with entity `X`, construct one deterministic
reconstruction example that teaches three behaviors:

1. Recover every textual occurrence of `X` from visible entities with directed
   train edges into `X`.
2. When possible, recover up to two additional entity targets from their own
   visible one-hop incoming supporters.
3. Recover up to two deliberately easy context-only spans without KG support.

Every supervised reconstruction answer is the exact surface text removed from
the passage. Aliases are used for mention discovery and evaluation, never as
alternative language-model targets.

## Non-Goals

This phase does not decide or implement:

- hard, soft, or learned entity-state initialization;
- model freezing, bidirectional-layer behavior, KIL placement, or post-KIL
  layers;
- tokenizer-budgeted cropping or maximum model sequence length;
- disjoint-union graph batching;
- tensor construction for `OreoQwenBatch`;
- training loops, loss weights, or checkpointing;
- benchmark QA preprocessing;
- a global alias-to-QID index or unrestricted corpus-wide entity linker.

Those decisions must not leak into the passage annotation schema.

## Terminology

| Term | Meaning |
|---|---|
| Passage entity `X` | The QID at the start of a `wikidata5m_text.txt` row |
| Primary star | Directed train edges `candidate → X` |
| Nested star | Directed train edges `supporter → candidate` for a matched primary-star entity |
| Resolved mention | A non-overlapping passage span assigned to exactly one graph-guided QID |
| Mention group | Every resolved occurrence assigned to the same QID |
| Primary target | The passage entity `X` |
| Auxiliary target | A matched primary-star entity with a visible nested-star supporter |
| Context-only blank | A repeated textual span selected without an entity or graph target |
| Visible supporter | A resolved, unmasked entity with a directed edge into a selected target |
| Marker block | `<|s_ent|> <|rel|> <|t_ent|>` inserted after a visible entity mention |

## Fixed Defaults

The first implementation uses the following versioned defaults:

| Setting | Value |
|---|---:|
| Preprocessing schema | `wiki5m_reconstruction_v1` |
| Graph split | `train` only |
| Maximum incoming degree considered | `1,000` |
| Maximum aliases considered per QID | `64` |
| Maximum auxiliary target QIDs | `2` |
| Required visible supporters for `X` | `2` |
| Required visible supporters per auxiliary | `1` |
| Maximum context-only blanks | `2` |
| Context n-gram length | `1–4` word tokens |
| Maximum sentinel occurrences | `16` |
| Static examples per eligible passage | `1` |
| Output shard input range | `10,000` passage dense IDs |

Changing one of these values creates a different preprocessing configuration
and therefore a different output-manifest fingerprint.

## End-to-End Passage Flow

```text
load passage entity X and full passage
    ↓
match aliases of X
    ↓
query X's incoming train star
    ↓
match aliases of incoming candidates
    ↓
for each matched candidate, inspect its incoming train star
    ↓
match nested supporters for auxiliary-target eligibility
    ↓
resolve ambiguous and overlapping spans
    ↓
select X, up to two auxiliaries, and up to two context blanks
    ↓
validate support, overlap, and 16-sentinel invariants
    ↓
assign sentinels in passage order
    ↓
record marker insertions after every visible resolved occurrence
    ↓
emit one untokenized record plus evaluation metadata
```

## Stage 1 — Load the Passage

Iterate datastore entities in deterministic dense-ID order. A passage entity is
eligible for inspection when `store.passage(qid)` returns text.

The source record is:

```python
PassageSource(
    passage_qid: str,
    passage_entity_id: int,
    passage_text: str,
)
```

The generated JSONL record references the passage by QID and dense ID; it does
not duplicate `passage_text`. Later stages retrieve the text from the datastore
whose manifest hash is pinned by the preprocessing manifest.

All mention spans use Python Unicode character offsets over the decoded passage,
not datastore byte offsets. Every stored span must satisfy:

```python
passage_text[start_char:end_char] == surface
```

## Stage 2 — Candidate Graph Expansion

### Primary entity

The aliases of `X` are candidates for primary-target mentions. The passage is
rejected if no occurrence can be resolved to `X`; there would otherwise be no
primary text to reconstruct.

### Primary incoming star

Query:

```python
store.edges_dense(
    store.entity_id(X),
    direction="incoming",
    splits=["train"],
)
```

Every returned triple has the form:

```text
candidate --relation--> X
```

If `X` has more than 1,000 incoming train edges, reject the passage with
`hub_primary`. The actual index contains only 1,523 entities above this
threshold, approximately 0.031% of all indexed entities, while they account for
more than half of incoming-edge traversal work.

Exact duplicate triples remain present in the datastore but are deduplicated by
`(head_entity_id, relation_id, tail_entity_id)` inside one preprocessing record.
Multiple distinct relations between the same entity pair are preserved.

### Nested incoming stars

Only primary-star entities with at least one resolved passage mention are
eligible for nested expansion. For each such entity `C`, query directed train
edges:

```text
nested_supporter --relation--> C
```

If `C` has more than 1,000 incoming train edges, retain `C` as a primary-star
mention but mark it `nested_hub`; it cannot become an auxiliary target.

Expansion stops here. The preprocessor never recursively expands supporters of
nested supporters.

## Stage 3 — Alias Normalization and Matching

Candidate discovery remains graph-guided. The matcher only considers aliases
belonging to:

- `X`;
- entities in the primary incoming star; and
- entities in a nested star being evaluated.

There is no global reverse-alias lookup in v1.

### Alias preparation

For each candidate QID:

1. Read aliases in source order with `store.entity_aliases(qid)`.
2. Normalize each alias with Unicode NFKC.
3. Collapse consecutive Unicode whitespace to one ASCII space and strip the
   ends.
4. Case-fold for lookup while preserving the original alias.
5. Deduplicate aliases by normalized form, preserving the first occurrence.
6. Retain at most the first 64 eligible normalized aliases.

An alias is ineligible when it:

- is empty after normalization;
- contains no Unicode letter or number;
- is composed only of punctuation;
- is a single-character alias;
- is a case-folded stop alias such as an article, conjunction, pronoun, or
  common relation word from the versioned stop-alias set.

The primary entity `X` is exempt from the name-like capitalization test, but
not from empty, punctuation-only, or one-character rejection. Other candidate
aliases must be name-like:

- a single word of at least three characters that is title-cased or uppercase;
  or
- a multiword phrase with at least one title-cased or uppercase content word.

The exact stop-alias set is checked into the implementation and its SHA-256 is
recorded in the output manifest.

### Boundary matching

Matching is case-insensitive over an NFKC-normalized comparison view, but stored
spans refer to the untouched passage. A match is valid only when it is bounded
on both sides by either the passage boundary or a non-alphanumeric Unicode
character. This prevents aliases such as `York` from matching `Yorkshire`.

The implementation must maintain a mapping from normalized-view offsets back to
original Unicode character offsets. It must never apply normalized offsets
directly to the original passage.

### Span candidates and ambiguity

All candidate matches are retained until resolution. A raw span candidate
records:

```python
RawAliasMatch(
    start_char: int,
    end_char: int,
    surface: str,
    normalized_alias: str,
    candidate_qids: tuple[str, ...],
    graph_depth: int,  # 0 for X, 1 for primary star, 2 for nested star
)
```

Resolution rules are deterministic:

1. Merge matches with the same original `(start_char, end_char)` and union their
   candidate QIDs.
2. If `X` is a candidate for the span, resolve it to `X`.
3. Otherwise, resolve only spans with exactly one candidate QID in the current
   graph-guided candidate universe.
4. Leave every remaining multi-QID span unresolved.
5. Resolve overlap greedily by this priority:
   - subject `X` before non-subject entities;
   - longer character span before shorter span;
   - shallower graph depth before deeper graph depth;
   - lower start offset;
   - lower numeric QID as the final tie-break.
6. Store selected mentions ordered by `(start_char, end_char, numeric_qid)`.

Unresolved and overlap-rejected matches contribute aggregate diagnostic counts
but are not written as model mentions.

### Resolved mention schema

```python
ResolvedMention(
    mention_id: int,
    qid: str,
    entity_id: int,
    start_char: int,
    end_char: int,
    surface: str,
    matched_normalized_alias: str,
    graph_depth: int,
)
```

Mention IDs are assigned in passage order after overlap resolution. Every
occurrence of the same QID forms one mention group.

## Stage 4 — Build the Passage Graph

The untokenized local graph contains the union of discovered directed train
stars, not an unrestricted neighborhood:

- resolved primary-star edges `candidate → X`; and
- resolved nested-star edges `nested_supporter → candidate`.

Nodes are every endpoint of a retained edge plus `X`. Node order is deterministic
by global dense entity ID. Edges are ordered by:

```text
(source local node ID, destination local node ID, relation dense ID)
```

The schema is:

```python
LocalNode(
    local_node_id: int,
    qid: str,
    entity_id: int,
)

LocalEdge(
    source_local_node_id: int,
    destination_local_node_id: int,
    relation_id: int,
    relation_pid: str,
    split: "train",
    depth: int,  # 1 for primary star, 2 for nested star
)
```

Only edges whose source entity has a resolved mention are retained. A target
node may have all of its occurrences masked and therefore need not have a
visible marker block.

## Stage 5 — Select KG Reconstruction Targets

### Primary target

`X` is always the primary target. Every resolved occurrence of `X` is selected
for masking.

After auxiliary-target selection, `X` must retain at least two distinct visible
primary-star supporter QIDs. A supporter counts as visible only when at least one
of its resolved occurrences remains unmasked.

If fewer than two such supporters exist, reject the passage with
`insufficient_primary_support`.

### Auxiliary candidates

A primary-star entity `C` is eligible as an auxiliary target when:

- `C` has at least one resolved occurrence;
- `C` is not a nested hub;
- at least one distinct resolved nested supporter has an edge into `C`;
- masking every occurrence of `C` leaves at least one nested supporter visible;
- adding `C` still leaves two distinct visible supporters for `X`; and
- the resulting masked-occurrence count does not exceed 16.

### Deterministic ranking

Eligible auxiliary QIDs are ordered by the SHA-256 digest of:

```text
wiki5m_reconstruction_v1\0<PASSAGE_QID>\0<CANDIDATE_QID>
```

Digests are compared lexicographically, with numeric QID as a final tie-break.
Select candidates greedily in that order while rechecking every support and
sentinel invariant. Stop after two auxiliaries.

This produces deterministic pseudorandom diversity without Python's
process-randomized `hash()`.

### Target groups

```python
TargetGroup(
    target_group_id: int,
    kind: str,  # "primary_kg" or "auxiliary_kg"
    qid: str,
    entity_id: int,
    occurrence_mention_ids: tuple[int, ...],
    visible_supporter_qids: tuple[str, ...],
    support_edge_indices: tuple[int, ...],
)
```

The primary group is ID 0. Auxiliary groups follow deterministic candidate
ranking. Every occurrence of a selected QID belongs to its target group; none
may remain visible.

## Stage 6 — Select Context-Only Blanks

Context-only blanks are easy textual controls and have no QID or graph target.
They are selected only after KG target groups are fixed.

### Candidate extraction

Tokenize the original passage with a versioned Unicode word regex that returns
character spans. Enumerate contiguous one- through four-token n-grams that:

- remain inside one sentence-like segment;
- contain at least one Unicode letter;
- are not solely stopwords;
- contain at least three alphanumeric characters in total;
- do not overlap any resolved entity mention;
- do not overlap any KG target occurrence; and
- have at least two non-overlapping occurrences with the same NFKC,
  whitespace-collapsed, case-folded form.

### Selection

Rank normalized candidate forms using SHA-256 of:

```text
wiki5m_reconstruction_v1\0<PASSAGE_QID>\0context\0<NORMALIZED_FORM>
```

For each ranked form, rank its occurrences with the same digest plus the
occurrence start offset. Select one occurrence to mask only when at least one
other occurrence remains visible. Do not choose overlapping context blanks.

Select at most two context blanks and never exceed the remaining 16-sentinel
budget. It is valid to select zero when no candidate passes the rules.

Context-only selections are not added to the local graph and carry no gold QID.

## Stage 7 — Assign Sentinels and Marker Insertions

### Sentinel assignment

Combine all KG-target occurrences and selected context-only spans. Sort by:

```text
(start_char, end_char, mask kind priority, numeric QID if present)
```

KG masks precede context-only masks only as an impossible-overlap tie-break.
Assign `<|kg_mask_0|>` through `<|kg_mask_15|>` in this passage order.

Each masked occurrence records its exact original surface:

```python
MaskedOccurrence(
    sentinel_index: int,
    sentinel_token: str,
    start_char: int,
    end_char: int,
    exact_text: str,
    kind: str,  # primary_kg, auxiliary_kg, or context_only
    target_group_id: int | None,
    gold_qid: str | None,
    gold_entity_id: int | None,
)
```

### Marker contract

Every resolved entity occurrence that is not masked receives exactly one marker
block immediately after its unchanged surface text:

```text
entity surface <|s_ent|> <|rel|> <|t_ent|>
```

Examples:

```text
Ilan Rao <|s_ent|> <|rel|> <|t_ent|>
Northbridge University <|s_ent|> <|rel|> <|t_ent|>
```

Rules:

- The entity surface comes before all three markers.
- Marker order is exactly `s_ent`, `rel`, `t_ent`.
- Markers are added after every visible resolved occurrence, including repeated
  occurrences of one QID.
- Masked entity occurrences contain only their sentinel; no KG marker appears
  before, after, or around a blank.
- Context-only blanks contain only their sentinel.
- Insertions and replacements are applied right-to-left over original character
  offsets so earlier offsets remain valid.

The structured record stores insertion locations rather than making the marked
passage the canonical source. A rendering helper reconstructs the marked and
corrupted passage deterministically for inspection and later tokenization.

## Stage 8 — Exact Reconstruction Target

Sentinel answers are ordered by sentinel index and always use the exact removed
surface:

```text
<|kg_mask_0|> exact span 0
<|kg_mask_1|> exact span 1
...
```

Aliases are never substituted into the training suffix. If two occurrences of
one QID use different aliases, each sentinel reconstructs its own original
surface.

The future instruct rendering is:

```text
SYSTEM
Reconstruct each masked span. Return only each sentinel followed by its
original text, in sentinel order.

USER
<marked and corrupted passage>

ASSISTANT
<|kg_reconstruct|> <|kg_mask_0|> exact span 0
<|kg_mask_1|> exact span 1 ... EOS
```

After Qwen chat-template tokenization:

- all labels through and including `<|kg_reconstruct|>` are `-100`;
- every suffix sentinel, exact span token, and EOS token is supervised; and
- the reconstruction suffix must begin with `<|kg_mask_0|>`.

The chat template and token positions are not materialized in this phase.

## Untokenized JSONL Record

Each accepted line has this logical schema. Field names and types are public
within preprocessing schema v1.

```json
{
  "schema_version": "wiki5m_reconstruction_v1",
  "example_id": "wiki5m:Q900",
  "passage": {
    "qid": "Q900",
    "entity_id": 899,
    "text_sha256": "...",
    "character_count": 241
  },
  "mentions": [
    {
      "mention_id": 0,
      "qid": "Q101",
      "entity_id": 100,
      "start_char": 31,
      "end_char": 40,
      "surface": "Mira Chen",
      "matched_normalized_alias": "mira chen",
      "graph_depth": 1,
      "masked": true
    }
  ],
  "local_graph": {
    "nodes": [],
    "edges": []
  },
  "target_groups": [],
  "masked_occurrences": [],
  "marker_mention_ids": [],
  "evaluation": {
    "targets": [],
    "local_candidate_qids": [],
    "datastore_manifest_sha256": "..."
  }
}
```

Additional schema rules:

- `example_id` is uniquely derived from the passage QID because v1 creates one
  example per passage.
- `text_sha256` hashes the exact UTF-8 passage payload and detects stale
  character offsets.
- `marker_mention_ids` lists every visible mention in passage order.
- `mentions` includes selected resolved occurrences whether visible or masked.
- `evaluation.targets` repeats only the minimal information needed by the
  evaluator: sentinel index, exact text, gold QID if any, and mask kind.
- The record does not contain accepted alias lists or model tensors.

## Worked Example

This example uses illustrative IDs and text; it is not asserted to be a real
Wikidata passage.

### Source passage

Passage entity:

```text
Q900 = Aurora Institute
```

Passage:

```text
Aurora Institute was founded in 1998. Mira Chen joined Aurora Institute after
working with Ilan Rao. Northbridge University and Solace Foundation later
funded Aurora Institute. Its annual lecture became the city's best-known
lecture.
```

### Relevant train graph

```text
Q101 Mira Chen              --P108--> Q900 Aurora Institute
Q102 Northbridge University --P749--> Q900 Aurora Institute
Q103 Solace Foundation      --P463--> Q900 Aurora Institute
Q104 Ilan Rao               --P26---> Q101 Mira Chen
```

The primary incoming star contains `Q101`, `Q102`, and `Q103`. `Q101` is
eligible as an auxiliary target because its nested incoming star contains the
visible `Q104`.

### Resolved alias mentions

| QID | Surface | Role before selection |
|---|---|---|
| `Q900` | `Aurora Institute` ×3 | Primary target occurrences |
| `Q101` | `Mira Chen` | Primary supporter and auxiliary candidate |
| `Q102` | `Northbridge University` | Primary supporter |
| `Q103` | `Solace Foundation` | Primary supporter |
| `Q104` | `Ilan Rao` | Nested supporter of `Q101` |

All aliases resolve uniquely and do not overlap.

### Selected targets and support invariants

Targets:

```text
primary:   Q900 Aurora Institute
auxiliary: Q101 Mira Chen
```

After masking `Q101`, `Q900` retains two visible supporters:

```text
Q102 → Q900
Q103 → Q900
```

The auxiliary retains one visible supporter:

```text
Q104 → Q101
```

The repeated word `lecture` is selected as one context-only blank: its final
occurrence is masked while the earlier occurrence remains visible.

### Sentinel assignment

| Sentinel | Exact answer | Kind | Gold QID |
|---|---|---|---|
| `<|kg_mask_0|>` | `Aurora Institute` | `primary_kg` | `Q900` |
| `<|kg_mask_1|>` | `Mira Chen` | `auxiliary_kg` | `Q101` |
| `<|kg_mask_2|>` | `Aurora Institute` | `primary_kg` | `Q900` |
| `<|kg_mask_3|>` | `Aurora Institute` | `primary_kg` | `Q900` |
| `<|kg_mask_4|>` | `lecture` | `context_only` | `null` |

### Marked and corrupted passage

```text
<|kg_mask_0|> was founded in 1998. <|kg_mask_1|> joined <|kg_mask_2|> after
working with Ilan Rao <|s_ent|> <|rel|> <|t_ent|>. Northbridge University
<|s_ent|> <|rel|> <|t_ent|> and Solace Foundation <|s_ent|> <|rel|> <|t_ent|>
later funded <|kg_mask_3|>. Its annual lecture became the city's best-known
<|kg_mask_4|>.
```

There are no marker tokens around the entity blanks. Every remaining resolved
entity occurrence is followed by the complete marker block.

### Reconstruction suffix

```text
<|kg_mask_0|> Aurora Institute
<|kg_mask_1|> Mira Chen
<|kg_mask_2|> Aurora Institute
<|kg_mask_3|> Aurora Institute
<|kg_mask_4|> lecture
EOS
```

### Evaluation metadata

```json
{
  "targets": [
    {"sentinel_index": 0, "exact_text": "Aurora Institute", "gold_qid": "Q900", "kind": "primary_kg"},
    {"sentinel_index": 1, "exact_text": "Mira Chen", "gold_qid": "Q101", "kind": "auxiliary_kg"},
    {"sentinel_index": 2, "exact_text": "Aurora Institute", "gold_qid": "Q900", "kind": "primary_kg"},
    {"sentinel_index": 3, "exact_text": "Aurora Institute", "gold_qid": "Q900", "kind": "primary_kg"},
    {"sentinel_index": 4, "exact_text": "lecture", "gold_qid": null, "kind": "context_only"}
  ],
  "local_candidate_qids": ["Q101", "Q102", "Q103", "Q104", "Q900"]
}
```

## Evaluation Boundary

Training uses exact surface text only. Alias-aware scoring is a separate loop
and does not change labels or `OreoQwenBatch`.

The evaluator will report:

- exact surface accuracy;
- normalized surface accuracy;
- alias-aware entity accuracy for `primary_kg` and `auxiliary_kg` masks;
- sentinel/format validity; and
- each metric partitioned by mask kind.

For alias-aware accuracy, retrieve aliases for `gold_qid` from the datastore
pinned by `datastore_manifest_sha256`. Normalize predictions and aliases with
the same versioned normalization function. An alternative alias receives credit
only when it maps uniquely to the gold QID among `local_candidate_qids`; an
ambiguous alias receives no alias-aware credit unless it exactly matches the
removed surface.

No alias list is sent to the model or duplicated into every JSONL record.

## Output Dataset Layout

The future implementation writes only under:

```text
datasets/Wikidata5M/preprocessed/wiki5m_reconstruction_v1/
├── manifest.json
└── shards/
    ├── entities-00000000-00009999.jsonl.gz
    ├── entities-00010000-00019999.jsonl.gz
    └── ...
```

Each shard owns a fixed half-open range of passage dense IDs, regardless of how
many examples are accepted. Fixed input ranges make parallel processing,
resumption, and completeness checks deterministic.

Gzip output must use `mtime=0` and stable JSON serialization:

- UTF-8;
- one JSON object per line;
- sorted object keys;
- compact separators;
- one trailing newline per record; and
- records ordered by passage dense ID.

Write each shard to a temporary file inside the output directory, validate it,
then atomically rename it. A rerun skips a shard only when its recorded checksum
and configuration fingerprint match. Publish `manifest.json` last.

## Output Manifest and Statistics

The manifest records:

- preprocessing schema and implementation version;
- source datastore manifest path and SHA-256;
- complete preprocessing configuration;
- stop-alias and normalization versions;
- shard dense-ID ranges, record counts, byte sizes, and SHA-256 checksums;
- total inspected, accepted, and rejected passages;
- counts of resolved mentions, local nodes/edges, targets, and sentinels;
- histograms for primary degree, nested degree, mentions, auxiliaries, context
  blanks, local nodes/edges, and sentinel count; and
- rejection and nonfatal diagnostic counters.

Each rejected passage receives one primary rejection reason according to this
precedence:

1. `missing_passage`
2. `hub_primary`
3. `no_subject_mention`
4. `subject_sentinel_overflow`
5. `insufficient_primary_support`
6. `overlap_invariant_failed`
7. `record_validation_failed`

Nonfatal counters include:

```text
nested_hub
ambiguous_alias_span
overlap_rejected_span
ineligible_alias
auxiliary_rejected_support
auxiliary_rejected_sentinel_budget
no_context_candidate
```

For auditability, the manifest may contain at most 100 example QIDs per rejection
or diagnostic category; it must not write an unbounded rejection log by default.

## Planned Code Boundaries

The later implementation should add a preprocessing subpackage without changing
the read-only datastore API:

```text
oreo_qwen/wikidata5m/preprocessing/
├── schema.py       # frozen record/config dataclasses and validation
├── aliases.py      # normalization, matching, ambiguity/overlap resolution
├── selection.py    # target, supporter, context-span, and sentinel selection
├── render.py       # marker insertion and exact reconstruction rendering
├── pipeline.py     # one-passage orchestration and sharded corpus builder
└── cli.py          # build, inspect, stats, and validate commands
```

Public Python entry points:

```python
preprocess_passage(store, passage_qid, config) -> PreprocessResult
build_preprocessed_corpus(store, output_dir, config) -> Path
render_example(record, passage_text) -> RenderedExample
validate_preprocessed_record(record, passage_text, store) -> None
```

Planned CLI:

```text
wikidata5m-preprocess build SOURCE_ROOT [--output ...] [--start-id ...] [--stop-id ...]
wikidata5m-preprocess inspect INDEX_DIR --qid Q...
wikidata5m-preprocess stats OUTPUT_DIR
wikidata5m-preprocess validate OUTPUT_DIR [--full]
```

`inspect` renders alias candidates, resolved mentions, stars, target decisions,
markers, corrupted text, suffix, and rejection reasons without writing corpus
artifacts. It is the required review tool before a full build.

## Planned Validation and Tests

### Alias matching

- NFKC and case-folded matching preserves original character offsets.
- Multibyte Unicode surfaces round-trip through stored character spans.
- Token boundaries reject substring matches.
- Subject-QID preference resolves local ambiguity.
- Non-subject ambiguous aliases remain unresolved.
- Longest-first overlap resolution is deterministic.
- Alias deduplication and the 64-alias cap are stable.

### Graph expansion

- Only directed train edges are used.
- Primary and nested stars preserve direction and distinct relations.
- Exact duplicate triples are removed locally.
- A primary degree of 1,001 produces `hub_primary`.
- A nested degree of 1,001 produces `nested_hub` without rejecting the passage.
- Expansion never proceeds beyond the nested star.

### Target selection

- Every subject occurrence is masked.
- Every selected auxiliary occurrence is masked.
- The primary retains two distinct visible supporters.
- Each auxiliary retains one distinct visible supporter.
- Selecting one auxiliary cannot invalidate a previously selected target.
- Auxiliary ranking is identical across processes and Python versions.
- Target selection respects the 16-sentinel budget.

### Context-only blanks

- Only one occurrence of a repeated span is masked.
- At least one matching occurrence remains visible.
- Context spans never overlap resolved entity mentions or each other.
- Zero context blanks is valid.
- Selection is deterministic.

### Rendering and records

- Visible surfaces are followed by exactly `s_ent`, `rel`, `t_ent`.
- Masked entity and context spans contain no KG markers.
- Right-to-left edits preserve all original offsets.
- Sentinels follow passage order and are unique.
- Exact reconstruction text matches the removed source slice.
- Record validation fails when the datastore passage hash changes.

### Sharding

- Fixed dense-ID ranges cover the requested range exactly once.
- Gzip bytes and shard checksums are reproducible.
- Interrupted shard writes are not published.
- Valid shards resume without rewriting.
- Manifest publication occurs only after all requested shards validate.

### Integration

- A small hand-authored datastore fixture exercises the complete accepted path
  and every rejection reason.
- A read-only opt-in smoke test processes a bounded real Wikidata5M dense-ID
  range and writes only to a temporary directory.
- No default test builds or modifies the full preprocessed corpus.

## Future `INPUT_CONTRACT.md` Revisions

After this preprocessing contract is approved, the tensor-level input document
must be revised separately to:

- show `entity surface <|s_ent|> <|rel|> <|t_ent|>` marker order;
- state that every visible resolved occurrence is marked;
- state that masked occurrences have no surrounding KG markers;
- use the instruct chat serialization with `<|kg_reconstruct|>` at the start of
  the assistant response;
- preserve exact-surface, suffix-only causal labels;
- describe Stage-1 evaluation metadata outside `model.forward`;
- remove the assumption that hyperlink annotations already exist; and
- point to this document for passage matching, target selection, and untokenized
  record semantics.

These documentation changes do not add aliases or evaluation fields to
`OreoQwenBatch`.

## Approval Boundary

Approval of this document authorizes implementation of the single-passage
preprocessor, inspection CLI, schema validation, deterministic shard writer,
and their tests. It does not authorize a full-corpus preprocessing run, model
architecture changes, batching, or training.
