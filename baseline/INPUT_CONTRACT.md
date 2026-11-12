# OreoLM–Qwen Input Contract

## Architecture Summary

The first prototype uses one Knowledge Interaction Layer (KIL), placed after
completed Qwen layer 34 in the 36-layer Qwen3-4B backbone:

```text
Qwen layers 1–34
    → prompt-only REL/T_ENT contextualization
    → one differentiable graph walk over an externally linked local graph
    → frozen entity embedding → trainable bottleneck → T_ENT
    → strict preceding-T_ENT broadcast
    → Qwen layers 35–36
```

The model owns no entity or relation embedding table and performs no entity
linking. A frozen embedding encoder runs outside `OreoQwen3ForCausalLM`; the
batch supplies already resolved embedding tensors. The native Qwen backbone,
vocabulary embeddings, final norm, and LM head are frozen. Only Oreo adapters,
interaction layers, the relation temperature, and two marker-token deltas are
trainable.

Notation:

| Symbol | Meaning |
|---|---|
| `B` | Language examples |
| `L` | Padded token length |
| `M` | Marked graph mentions |
| `V` | Nodes in the batch's disjoint-union local graph |
| `E` | Edges in that graph |
| `R` | Global relation count |
| `D` | Frozen knowledge embedding width |
| `T` | KIL steps; `T = 1` by default |

## Language Tensors

| Name | Shape | Type | Required | Meaning |
|---|---:|---|---|---|
| `input_ids` | `[B, L]` | `torch.long` | Unless `inputs_embeds` is used | Complete prompt or teacher-forced sequence |
| `attention_mask` | `[B, L]` | bool/integer | Recommended | Active-token mask |
| `labels` | `[B, L]` | `torch.long` | Training only | Supervised IDs or `-100` |
| `assistant_content_start_positions` | `[B]` | `torch.long` | Optional | First assistant-content token |

When labels are supplied without explicit assistant starts, the LM wrapper uses
the first label other than `-100`. This boundary prevents pre-KIL marker queries
from reading teacher-forced answers. During prompt-only generation, labels and
starts are omitted and every active input token is prompt memory.

## Markers

Only two additional special tokens are registered:

```text
<|rel|>
<|t_ent|>
```

Every visible linked mention is serialized as:

```text
entity surface <|rel|> <|t_ent|>
```

The source marker is unnecessary because preprocessing supplies the linked
initial local node. `REL` is the contextual query used to select a relation;
`T_ENT` receives the retrieved entity value.

The complete native Qwen embedding table remains frozen. A trainable
`[2, hidden_size]` delta is added only at `REL` and `T_ENT` input positions, so
training the two new markers cannot update ordinary vocabulary rows or the tied
LM head.

| Name | Shape | Type | Meaning |
|---|---:|---|---|
| `kg_mention_batch` | `[M]` | `torch.long` | Batch row owning each mention |
| `kg_rel_positions` | `[M]` | `torch.long` | `REL` token positions |
| `kg_tgt_positions` | `[M]` | `torch.long` | `T_ENT` token positions |

Positions must be active, point to the configured marker ID, and occur before
assistant content. `(batch, T_ENT position)` pairs must be unique.

## Frozen Knowledge Embeddings

| Name | Shape | Type | Meaning |
|---|---:|---|---|
| `kg_node_embeddings` | `[V, D]` | floating | Frozen embeddings for only the local nodes in this batch |
| `kg_relation_embeddings` | `[R, D]` | floating | Frozen relation bank |

Both tensors must be finite and on the language-input device. They are ordinary
inputs, not parameters or persistent model buffers. The reasoner normalizes
them before use. The default provenance stored in configuration is
`Qwen/Qwen3-Embedding-0.6B`, with `D = 384` and a 512-wide trainable bottleneck.

`FrozenKnowledgeEmbedder` is an external helper. It batches cache misses under
inference mode, keeps a bounded detached CPU hot cache, optionally persists
vectors in SQLite, and returns only the requested rows. The encoder must not be
placed in an optimizer or in the Oreo checkpoint.
Relation embeddings may be computed once because Wikidata5M has only 828
relations. Entity embeddings should be resolved on demand or precomputed only
for the union of entities used by the experiment.

Recommended entity text includes label, short description, types, and aliases.
Recommended relation text includes label, description, and direction semantics.

## Disjoint-Union Graph

| Name | Shape | Type | Meaning |
|---|---:|---|---|
| `kg_node_batch` | `[V]` | `torch.long` | Example owning each local node |
| `kg_edge_src` | `[E]` | `torch.long` | Flattened local source index |
| `kg_edge_dst` | `[E]` | `torch.long` | Flattened local destination index |
| `kg_edge_type` | `[E]` | `torch.long` | Row in `kg_relation_embeddings` |

Edges may not cross example components. When collating graphs, both endpoint
arrays receive the cumulative local-node offset; edge types remain relation-bank
indices. Empty edge tensors are valid and trigger the walk's retain-state
fallback.

Dataset entity IDs/QIDs remain preprocessing and embedding-cache metadata. They
are not passed to the model as `kg_node_ids`.

## Required External Initialization

Exactly one initialization form is required:

| Name | Shape | Type | Meaning |
|---|---:|---|---|
| `kg_initial_node_indices` | `[M]` | `torch.long` | Externally linked local node per mention |
| `kg_initial_entity_probs` | `[M, V]` | floating | External distribution over local nodes |

Hard indices must belong to the mention's example. Soft rows must be finite,
nonnegative, sum to one, and assign no probability outside that component.
There is no learned fallback and no entity-linking loss.

## Relation Supervision

| Name | Shape | Type | Required | Meaning |
|---|---:|---|---|---|
| `kg_relation_targets` | `[T, M, R]` | floating | Optional | Gold/weak relation distributions |
| `kg_relation_target_mask` | `[T, M]` | bool | With targets | Supervised step/mention pairs |

Selected rows must be probability distributions. The relation-query bottleneck
produces a normalized vector in the frozen relation space. Cosine scores use a
trainable temperature before softmax. Soft-target cross entropy is equivalent
to minimizing `KL(target || predicted)` up to the target entropy.

## KIL Computation

For one step:

```text
REL hidden state
    → RMSNorm → 512 bottleneck → D-dimensional normalized query
    → cosine scores against frozen relation embeddings
    → relation probabilities
    → fixed inverse-degree graph walk
    → weighted sum of frozen node embeddings
    → RMSNorm → 512 bottleneck → Qwen hidden width
    → residual fusion into T_ENT → RMSNorm
```

The graph walk remains the fixed global inverse-out-degree walk. If all outgoing
edge scores are zero, the previous entity distribution is retained.

## Interaction Layers

The pre-KIL layer gathers `REL` and `T_ENT` as queries. They attend
bidirectionally to active prompt tokens, but only the two marker rows are
updated. Assistant content is excluded during teacher forcing.

The post-KIL layer exposes only strictly preceding `T_ENT` rows as memory. A
query with no preceding `T_ENT` receives an exact zero branch and is unchanged.
Both interaction layers use Qwen's grouped-query head geometry, branch
pre-normalization, residual connections, and zero-initialized scalar gates.

## Losses and Outputs

The training objective is:

```text
total_loss = lm_loss + relation_loss_weight * relation_loss
```

There is no entity-linking objective. Without relation targets, answer or
reconstruction loss still trains the relation adapter through the soft graph
walk and trains the entity-value adapter through `T_ENT` fusion.

`OreoQwen3ForCausalLM` freezes native Qwen parameters during construction and
again after vocabulary resizing. Optimizers must consume
`model.oreo_parameters()` (or equivalently filter on `requires_grad`) rather
than all model parameters. Frozen decoder operations remain in the autograd
graph where necessary to propagate gradients to Oreo parameters, but their own
parameter gradients stay absent.

The causal-LM output adds:

```text
lm_loss
relation_loss
entity_states       # optional initial and post-walk distributions
relation_probs      # optional relation distribution per KIL
```

## Cached Generation

Pre-KIL contextualization and graph traversal run once during prompt prefill.
Post-KIL attention maintains its own KV cache and continues on generated tokens
so they can read prompt `T_ENT` states. KG graph and marker tensors remain in
generation kwargs, while frozen embeddings are treated as read-only inputs.

The initial supported generation mode remains deterministic greedy decoding;
beam-specific graph expansion is deferred.

## Dataset and Semantic Collation

`Wikidata5MReconstructionDataset` streams the v3 gzipped JSONL shards and
joins each record to its canonical passage and entity aliases through the
immutable Wiki5M index. It does not retain an open datastore handle, and it
assigns whole shards across distributed ranks and DataLoader workers.

`OreoSemanticCollator` is CPU-only. It:

1. renders blanks and Oreo markers;
2. applies the tokenizer's instruct chat template with thinking disabled by
   default;
3. right-pads the batch and supervises only assistant answer tokens;
4. finds the tokenized `REL` and `T_ENT` positions;
5. combines local graphs into a disjoint union and offsets node indices;
6. binds every visible marker mention to its externally linked local node;
7. derives one-step relation targets from the preprocessing support edges; and
8. emits stable entity embedding requests in flattened node order.

It returns `SemanticOreoBatch`, not `OreoQwenBatch`, because embedding inference
must not run in DataLoader workers. The main process resolves the requests and
materializes the model batch:

```python
semantic = next(iter(loader))

node_vectors = embedder.resolve(
    semantic.kg.node_embedding_keys,
    semantic.kg.node_embedding_texts,
    output_device=train_device,
)
model_batch = semantic.materialize(
    node_embeddings=node_vectors,
    relation_embeddings=relation_bank,
    device=train_device,
)
outputs = model(**model_batch.as_model_kwargs())
```

The full relation bank can be initialized once from
`dataset.relation_embedding_requests()`. Entity request text currently uses the
first Wiki5M alias as its label and remaining aliases as additional names. The
request key includes a text-format version so a future formatting change does
not silently reuse stale cached vectors.

`max_length` never causes token-level truncation. The strict collator rejects an
overlength record; the production training configuration drops that complete
record and continues. Blind token truncation could detach marker positions,
source nodes, graph supervision, and answers from one another; a future
record-aware cropping policy must update all of those structures together.
