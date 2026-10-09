# OreoLM–Qwen Input Contract

## Purpose and Notation

This document specifies the public tensor interface between preprocessing,
batching, the KG reasoner, and the Qwen language model. The contract is designed
so a batch can be passed as keyword arguments to `OreoQwen3ForCausalLM`.

Notation:

| Symbol | Meaning |
|---|---|
| `B` | Number of language-model examples in the batch |
| `L` | Padded token-sequence length |
| `M` | Number of marked KG mentions/queries across the batch |
| `V` | Number of nodes across the batch's disjoint-union local graph |
| `E` | Number of edges across the disjoint-union local graph |
| `R` | Number of relation types in relation memory |
| `T` | Number of KIL steps, equal to the number of KIL boundaries |

The local graph uses one flattened node-index space `[0, V)`. Dataset-level
entity IDs are used only to look up rows in the global entity embedding table.

## Language-Model Tensors

| Name | Shape | Type | Required | Meaning |
|---|---:|---|---|---|
| `input_ids` | `[B, L]` | `torch.long` | Yes unless `inputs_embeds` is used | Tokenizer IDs for the complete teacher-forced sequence |
| `attention_mask` | `[B, L]` | bool or integer | Recommended | `1` for real tokens and `0` for padding |
| `labels` | `[B, L]` | `torch.long` | Training only | Token IDs at supervised positions and `-100` elsewhere |

The model uses Hugging Face's standard shifted causal loss: the logit at position
`i - 1` predicts `labels[:, i]`. Prompt, delimiter, and padding positions use
`-100`. The answer/reconstruction suffix and its terminating EOS token are
supervised.

When both `input_ids` and `inputs_embeds` are supplied, validation fails. When
`attention_mask` is omitted, the usual Qwen all-visible non-padding default is
used.

## Mention and Token-Position Tensors

| Name | Shape | Type | Required with KG | Meaning |
|---|---:|---|---|---|
| `kg_mention_batch` | `[M]` | `torch.long` | Yes | Batch-example index owning each mention |
| `kg_src_positions` | `[M]` | `torch.long` | Yes | Position of each `<|s_ent|>` token |
| `kg_rel_positions` | `[M]` | `torch.long` | Yes | Position of each `<|rel|>` token |
| `kg_tgt_positions` | `[M]` | `torch.long` | Yes | Position of each `<|t_ent|>` token receiving fused knowledge |

For mention `m`, the hidden states are addressed by the pairs:

```text
source query:   hidden_states[kg_mention_batch[m], kg_src_positions[m]]
relation query: hidden_states[kg_mention_batch[m], kg_rel_positions[m]]
fusion target:  hidden_states[kg_mention_batch[m], kg_tgt_positions[m]]
```

Positions must be in `[0, L)`, point to non-padding tokens, and contain their
configured marker token IDs when `input_ids` are available. Target `(batch,
position)` pairs must be unique so two mentions do not perform ambiguous writes
to the same hidden-state row.

The target position is explicit. The implementation must not infer it as
`kg_rel_positions + 1`, even if the standard serialization places the two tokens
next to one another.

## Disjoint-Union Local Graph

| Name | Shape | Type | Required with KG | Meaning |
|---|---:|---|---|---|
| `kg_node_ids` | `[V]` | `torch.long` | Yes | Dense global entity-memory row for each local node |
| `kg_node_batch` | `[V]` | `torch.long` | Yes | Batch-example index owning each local node |
| `kg_edge_src` | `[E]` | `torch.long` | Yes | Flattened local source-node index |
| `kg_edge_dst` | `[E]` | `torch.long` | Yes | Flattened local destination-node index |
| `kg_edge_type` | `[E]` | `torch.long` | Yes | Dense relation-memory row for each edge |

`kg_node_batch[v]` identifies which example owns flattened node `v`. It is what
allows several differently sized graphs to share one batch without padding them
to `[B, V_max, V_max]`.

The graph must satisfy these invariants:

- `kg_node_ids` is in `[0, num_entities)`.
- `kg_node_batch` and `kg_mention_batch` are in `[0, B)`.
- Edge endpoints are in `[0, V)`.
- Relation IDs are in `[0, num_relations)`.
- Every edge's source and destination have the same `kg_node_batch` value.
- Every mention has at least one node belonging to its batch example.
- Empty edge tensors are permitted; the walk then uses its all-zero-score
  fallback.

All edges use the flattened node space. When batching example graphs, the
collator must add the example's cumulative node offset to both endpoint arrays.

## Entity-State Initialization

There are three supported initialization modes, in priority order. Supplying
both hard and soft initialization is an error rather than an implicit
precedence rule.

### Hard linked entities

| Name | Shape | Type | Meaning |
|---|---:|---|---|
| `kg_initial_node_indices` | `[M]` | `torch.long` | One flattened local node index per mention |

Each selected node must belong to the same batch example as the corresponding
mention. The reasoner converts these indices to one-hot rows in `[M, V]`.

### External soft state

| Name | Shape | Type | Meaning |
|---|---:|---|---|
| `kg_initial_entity_probs` | `[M, V]` | floating point | Probability distribution over flattened local nodes |

Rows must be finite, nonnegative, sum to one within tolerance, and contain zero
mass on nodes owned by other examples. An upstream top-k linker can be supported
by scattering its candidates into this dense local representation.

### Learned entity predictor

If neither initialization tensor is supplied, the reasoner projects each source
marker hidden state and scores the raw embeddings of nodes owned by that
mention's example. Nodes in other examples are masked before softmax.

## Auxiliary Supervision

Entity initialization and entity-link supervision are deliberately separate.
An externally initialized state may still supervise the learned entity
predictor.

| Name | Shape | Type | Required | Meaning |
|---|---:|---|---|---|
| `kg_entity_link_targets` | `[M]` | `torch.long` | Optional | Correct flattened local node for each mention |
| `kg_entity_link_mask` | `[M]` | bool | With entity targets | Mentions included in entity-link loss |
| `kg_relation_targets` | `[T, M, R]` | floating point | Optional | Per-step target relation distributions |
| `kg_relation_target_mask` | `[T, M]` | bool | With relation targets | Step/mention pairs included in relation loss |

An entity target must belong to the same graph component as its mention.
Relation-target rows selected by the mask must be finite, nonnegative, and sum
to one. This supports one-hot labels and uniform distributions over multiple
valid relations.

The relation tensor's first dimension must equal the configured number of KIL
steps. A missing target tensor or an all-false loss mask produces a
differentiable scalar zero for that auxiliary loss.

## Complete Two-Example QA Batch

Suppose both questions require two hops:

```text
Example 0:
Where was <|s_ent|> Albert Einstein <|rel|> <|t_ent|>
born, and in which country is that city? <|kg_answer|> Germany EOS

Example 1:
Where was <|s_ent|> Marie Curie <|rel|> <|t_ent|>
born, and in which country is that city? <|kg_answer|> Poland EOS
```

The local graphs are:

```text
Einstein --P19/place of birth--> Ulm --P17/country--> Germany
Curie    --P19/place of birth--> Warsaw --P17/country--> Poland
```

For illustration, the dense entity-memory rows are arbitrary small numbers:

| Flat node | Entity-memory ID | Label | `kg_node_batch` |
|---:|---:|---|---:|
| 0 | 101 | Einstein | 0 |
| 1 | 205 | Ulm | 0 |
| 2 | 310 | Germany | 0 |
| 3 | 402 | Curie | 1 |
| 4 | 510 | Warsaw | 1 |
| 5 | 611 | Poland | 1 |

The graph tensors are:

```python
kg_node_ids = torch.tensor([101, 205, 310, 402, 510, 611])
kg_node_batch = torch.tensor([0, 0, 0, 1, 1, 1])

# Relation 0 is P19; relation 1 is P17.
kg_edge_src = torch.tensor([0, 1, 3, 4])
kg_edge_dst = torch.tensor([1, 2, 4, 5])
kg_edge_type = torch.tensor([0, 1, 0, 1])
```

Assume the illustrative tokenization gives the following marker positions in
both sequences:

```text
position:  0     1    2          3       4        5       6
token:     Where was  <|s_ent|>  Albert  Einstein <|rel|> <|t_ent|>
```

Then:

```python
kg_mention_batch = torch.tensor([0, 1])
kg_src_positions = torch.tensor([2, 2])
kg_rel_positions = torch.tensor([5, 5])
kg_tgt_positions = torch.tensor([6, 6])

kg_initial_node_indices = torch.tensor([0, 3])

kg_entity_link_targets = torch.tensor([0, 3])
kg_entity_link_mask = torch.tensor([True, True])

kg_relation_targets = torch.tensor([
    [[1.0, 0.0], [1.0, 0.0]],  # KIL 0: P19
    [[0.0, 1.0], [0.0, 1.0]],  # KIL 1: P17
])
kg_relation_target_mask = torch.tensor([
    [True, True],
    [True, True],
])
```

The equivalent soft initialization is:

```python
kg_initial_entity_probs = torch.tensor([
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
])
```

It is supplied instead of, not alongside, `kg_initial_node_indices`.

`input_ids`, `attention_mask`, and `labels` have shape `[2, L]`. For example 0,
all labels through `<|kg_answer|>` are `-100`; the token or tokens encoding
`Germany` and the final EOS are supervised. Example 1 treats `Poland` the same
way. Exact positions are derived after tokenization rather than hard-coded from
the human-readable string.

## Two-Stage Training Contract

There are exactly two supported training stages:

1. Salient-span reconstruction pretraining over the span-corrupted Wikipedia
   corpus.
2. Benchmark-specific adaptive QA fine-tuning over datasets such as
   NaturalQuestions.

The two stages use the same model forward interface and graph tensors. They
differ only in textual serialization, label masking, availability of auxiliary
targets, and checkpoint initialization.

## Serialization and Special Tokens

Register these additional special tokens:

```text
<|s_ent|>
<|rel|>
<|t_ent|>
<|kg_reconstruct|>
<|kg_answer|>
<|kg_mask_0|> ... <|kg_mask_15|>
```

The tokenizer's existing EOS and padding conventions are retained. After token
registration, the model's tied input embedding and LM-head vocabulary is
resized, and the resulting token IDs are persisted in `OreoQwen3Config`.

### Stage 1: salient-span reconstruction pretraining

```text
<marked passage containing unique sentinels> <|kg_reconstruct|>
<|kg_mask_0|> <span 0> <|kg_mask_1|> <span 1> ... EOS
```

Everything from the first passage token through `<|kg_reconstruct|>` is prompt
context and has label `-100`. Every token after the delimiter, beginning with
the first `<|kg_mask_0|>`, is supervised. This includes each suffix sentinel,
every reconstructed span token, and EOS.

For example:

```text
<|s_ent|> Albert Einstein <|rel|> <|t_ent|> was born in
<|kg_mask_0|>. <|kg_mask_1|> is located in Germany.
<|kg_reconstruct|> <|kg_mask_0|> Ulm <|kg_mask_1|> Ulm EOS
```

The two occurrences of `Ulm` receive different sentinels even though they link
to the same entity. The existing Wikipedia text, character-level hyperlink
spans, entity IDs, local graphs, and weak relation paths can be reused, but the
passage must be retokenized with the Qwen tokenizer and all token positions and
labels must be rebuilt.

Core KG tensors are required in this stage. Entity-link and relation-path targets
are supplied when the preprocessing pipeline can derive them.

### Stage 2: benchmark adaptive QA fine-tuning

```text
<marked question> <|kg_answer|> <answer> EOS
```

The complete marked question and `<|kg_answer|>` have label `-100`. Every answer
token and EOS is supervised. Questions must be entity-linked, marked with the
source/relation/target tokens, and paired with a local graph before they enter
the Oreo-Qwen collator.

Benchmark QA data does not need gold relation paths. Entity-link and relation
supervision tensors are optional during adaptive fine-tuning; when absent, their
losses are zero and answer loss trains the KIL path end to end.

The raw-Qwen control is independently fine-tuned on the same benchmark examples.
For the primary controlled comparison, it receives the same marked token text
and `<|kg_answer|>` suffix but no `kg_*` tensors. Its labels and optimization
budget match Oreo-Qwen. An additional plain-question control can be produced as
an ablation without changing this two-stage contract.

## Optionality

- A completely graph-free batch omits every `kg_*` argument. The model skips all
  KIL work and auxiliary losses. This is used by the raw-Qwen control, not by
  Oreo-Qwen's two KG-conditioned training stages.
- If any core KG tensor is provided, all mention, position, node, and edge core
  tensors must be provided, even when `E == 0`.
- Auxiliary supervision is optional at evaluation and generation time.
- `labels` are optional outside training.

## Forward Output Contract

The causal-LM wrapper will return the normal Qwen causal-LM fields plus:

```text
loss                 # weighted total loss, when any objective is active
lm_loss              # shifted causal-language-model loss or None
entity_link_loss     # scalar; differentiable zero when unsupervised
relation_loss        # scalar; differentiable zero when unsupervised
entity_states        # optional tuple of [M, V], one per KIL
relation_probs       # optional tuple of [M, R], one per KIL
```

Diagnostics are returned only when enabled, to avoid retaining large
intermediate tensors during normal training.

## Inference and Benchmark Evaluation Contract

Benchmark evaluation uses the same fine-tuned `OreoQwen3ForCausalLM` class as
training. The distinction is the inputs and execution mode, not a separate model
architecture.

For each Oreo-Qwen evaluation example:

1. Entity-link and mark the question exactly as during adaptive fine-tuning.
2. Build the local graph and core KG tensors.
3. Tokenize only the prompt through `<|kg_answer|>`; do not append a gold answer.
4. Omit `labels`, entity-link targets, and relation targets.
5. Call `model.eval()` and `model.generate(...)` under inference mode.
6. Slice off the input prompt IDs, stop at EOS, and decode only newly generated
   answer tokens.
7. Normalize and score the decoded answer with the benchmark's official metric
   against every accepted gold answer/alias.

The generation call initially uses deterministic greedy decoding:

```text
do_sample = False
use_cache = True
max_new_tokens = configured per benchmark
eos_token_id = tokenizer EOS
pad_token_id = tokenizer padding ID
```

KILs run once during prompt prefill. The model's generation adapter carries KG
arguments into that prefill call and removes or ignores them during cached
single-token decoding. Generated sequences returned by Hugging Face include the
prompt, so answer extraction begins at the padded input width returned to
`generate`, not at a hard-coded textual position.

The raw-Qwen control uses its separately fine-tuned checkpoint, the same textual
prompt through `<|kg_answer|>`, the same decoding configuration, and the same
answer extraction and metric code. It omits all KG tensors and uses the ordinary
Qwen generation path.

Evaluation records also carry non-model metadata outside `model.forward`, at
minimum:

```text
example_id
gold_answers       # all accepted answers/aliases
raw_question
```

Prediction artifacts contain the example ID, rendered prompt, decoded prediction,
gold-answer list, and optional Oreo reasoning diagnostics. Teacher-forced QA loss
may be reported as a diagnostic on a validation split, but benchmark answer
quality is measured from generated answers without exposing the gold suffix.
