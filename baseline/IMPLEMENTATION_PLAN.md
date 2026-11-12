# OreoLM–Qwen3-4B Baseline Implementation Plan

## Purpose

This directory will contain a self-contained baseline that integrates OreoLM-style
Knowledge Interaction Layers (KILs) into `Qwen/Qwen3-4B`.

The baseline is intentionally review-gated. Each phase ends with a hard stop so
that the implementation can be inspected, discussed, and revised before the next
phase begins. Passing tests does not automatically authorize work on the next
phase.

All source code, tests, documentation, examples, and baseline-specific dependency
configuration must remain inside `oreo-decoder/baseline/`.

## Agreed Architecture

The model hierarchy will be:

```text
OreoQwen3ForCausalLM
├── OreoQwen3Model
│   ├── Qwen token embeddings
│   ├── Qwen decoder layers
│   ├── pre-KIL marker-context attention
│   ├── KIL/KG reasoner
│   ├── post-KIL T_ENT broadcast attention
│   └── final Qwen RMSNorm
└── tied language-model head
```

### `OreoQwen3Config`

`OreoQwen3Config` will extend `Qwen3Config`. It is the serializable model
blueprint and will store:

- Standard Qwen architecture fields.
- Configurable KIL insertion boundaries.
- Frozen knowledge-embedding provenance and width.
- Adapter bottleneck width and relation-loss coefficient.
- OreoLM special-token IDs.
- Diagnostic-output settings.

It validates configuration but performs no forward computation. The default
36-layer Qwen3-4B prototype has one KIL after completed decoder layer 34,
leaving the final two decoder blocks to integrate broadcast knowledge.
Explicit in-range boundaries remain configurable for later ablations.

### `OreoQwen3Model`

`OreoQwen3Model` is the backbone without the vocabulary projection. It will:

1. Embed input tokens.
2. Run the native Qwen decoder-layer loop.
3. At every configured boundary, contextualize only the `REL` and `T_ENT` states
   with prompt-only bidirectional attention.
4. Consume frozen local-node and relation embeddings supplied by the batch and
   initialize from an externally linked node or distribution.
5. Invoke the KIL and thread the entity state explicitly between KILs.
6. Broadcast each post-KIL `T_ENT` state only to tokens strictly after it.
7. Apply Qwen's final RMSNorm.
8. Return hidden states, cache data, auxiliary losses, and optional KG
   diagnostics.

The implementation will preserve Qwen decoder-layer modules, RoPE, attention
masks, cache behavior, and base parameter names. If KG inputs are absent, the
KIL path is skipped and the backbone behaves like ordinary Qwen.

### `OreoQwen3ForCausalLM`

`OreoQwen3ForCausalLM` is the training and generation wrapper. It will own an
`OreoQwen3Model`, apply the tied language-model head, compute shifted causal-LM
loss, combine the auxiliary losses, and implement the Hugging Face pretrained
model and generation interfaces.

The default training objective is:

```text
total_loss = lm_loss
           + relation_loss_weight * relation_loss
```

The relation coefficient defaults to `1.0`. The frozen embedding encoder,
resolved knowledge vectors, and complete native Qwen backbone are never
optimized. Only the bottleneck adapters, interaction layers, relation
temperature, and two marker-token deltas remain trainable.

During cached greedy generation, pre-KIL contextualization and KG reasoning run
during prompt prefill. They are skipped on subsequent one-token decoding calls.
Each post-KIL layer keeps its own cached key/value states and continues to run so
new tokens can read preceding prompt `T_ENT` states. Beam-search graph expansion
is deferred.

## Trainable Interaction Layers

Every KIL is wrapped by two separately parameterized attention adapters. They
reuse Qwen's grouped-query head layout (`num_attention_heads`,
`num_key_value_heads`, and `head_dim`), Q/K RMSNorm, dropout setting, and linear
projection shapes. They are attention adapters rather than full Qwen decoder
blocks: neither adds a self-attention causal path over ordinary tokens nor an
unconditional MLP.

The pre-KIL adapter gathers the `REL` and `T_ENT` states for each
mention as queries. Those queries may attend to every active prompt token in the
same example, regardless of whether it occurs before or after the marker. Only
the two marker rows are written back. During teacher forcing, assistant
content is excluded using `assistant_content_start_positions`, so a marker
cannot read the target answer. The causal Qwen states used as prompt memory also
cannot have encoded a later answer.

The post-KIL adapter uses all sequence states as queries but exposes only
`T_ENT` rows as keys and values. Its mask is strict: a query at position `q` may
read a target marker at position `t` only when `t < q`. It never attends to an
ordinary token through this branch. The update is residual and uses branch-only
pre-normalization; there is no post-residual normalization. A query with no
eligible target marker receives a forced zero branch output, so its
representation remains bit-for-bit unchanged.

Both adapters use a scalar `tanh` residual gate initialized to zero. This keeps
a newly converted native-Qwen checkpoint behaviorally unchanged before
fine-tuning and lets training open the new paths gradually. It also means the
gate learns on the first optimization step while its attention projections
begin receiving task gradients once the gate moves away from zero.

## KG Reasoning Semantics

The reasoner will contain:

- A trainable relation-query bottleneck for each KIL step.
- A sparse graph walker implemented with native PyTorch indexing and scatter
  operations.
- A trainable frozen-entity-to-Qwen bottleneck and fusion module per step.

The frozen Qwen3-Embedding-0.6B encoder lives outside the language model. An
on-demand CPU cache batches unseen entity text, while all 828 relation vectors
may be precomputed. Neither the encoder nor cached vectors enter the optimizer
or Oreo checkpoint.

The frozen local entity embeddings are aggregated using the walked entity
distribution before the step-specific value projection is applied. The projected value is
added to the target-marker hidden state and passed through Qwen RMSNorm. The
per-batch entity state is passed explicitly through the forward call; it must not
be stored as mutable module state.

### Fixed global inverse-degree walk

The baseline preserves the original implementation's fixed inverse-out-degree
approximation. It does not add a learned relation-importance weight.

For mention `m` and edge `e = (s_e, r_e, t_e)`:

```text
edge_score[m, e]
    = previous_state[m, s_e]
    * relation_probability[m, r_e]
    * (1 / out_degree(s_e))
```

Scores are L1-normalized globally across all edges for each mention, then
scatter-added into their destination nodes:

```text
normalized_edge_score[m, e]
    = edge_score[m, e] / sum_over_edges(edge_score[m, e])

next_state[m, v]
    = sum(normalized_edge_score[m, e] for e where t_e == v)
```

This is deliberately not a per-source transition normalization. Probability
mass on a dead-end node is discarded whenever some nonzero outgoing edge score
exists elsewhere, and the remaining edge mass is globally renormalized. If all
edge scores for a mention are zero, the previous state is retained as a
numerical-safety fallback.

## Two-Stage Training Lifecycle

There are exactly two training stages for the Oreo-Qwen model:

```text
Qwen3-4B
    -> Stage 1: Wikipedia salient-span reconstruction pretraining
    -> saved Oreo-Qwen pretraining checkpoint
    -> Stage 2: benchmark-specific adaptive QA fine-tuning
    -> benchmark evaluation checkpoint
```

### Stage 1 — Salient-span reconstruction pretraining

The initial training corpus consists of the span-corrupted Wikipedia data used
by OreoLM. Reconstruction samples contain marked mentions, a local graph, and
one unique textual blank identifier for each selected span occurrence:

```text
user: <marked and corrupted passage containing [BLANK_i] identifiers>
assistant: [BLANK_0] first span [BLANK_1] second span ... EOS
```

All system, user, and assistant-header labels are `-100`. Every assistant
content token beginning with `[BLANK_0]`, including blank identifiers,
reconstructed span tokens, and EOS, contributes to causal-LM loss.

The preprocessing configuration initially limits examples to sixteen textual
blank identifiers. If the same entity is masked in multiple locations, each
occurrence receives a distinct identifier so its exact surface form can be
reconstructed.

This stage also uses weak relation-path supervision when available. Entity
linking is performed by preprocessing and is not a model objective.

### Stage 2 — Benchmark adaptive QA fine-tuning

For each benchmark, such as NaturalQuestions, entity mentions in the question
are linked and marked, a local KG is supplied, and the answer is serialized as
assistant content:

```text
user: Where was Albert Einstein <|rel|> <|t_ent|>
      born, and in which country is that city?
assistant: Germany EOS
```

All system, user, and assistant-header labels are `-100`; only answer tokens and
EOS are supervised. Relation targets are optional in this stage
because QA benchmarks generally do not provide gold reasoning paths. Even
without auxiliary targets, answer loss remains differentiable through the KILs.

The raw-Qwen control starts from the same Qwen3-4B base checkpoint and undergoes
the same benchmark adaptive fine-tuning, but it does not receive a graph or run
KILs. The primary controlled comparison uses the same marked textual prompt and
chat template for both models so the presence of text and role tokens is held
constant. Both models use the same train/dev/test split, answer targets,
optimization budget, decoding settings, and evaluator. A separate plain-question
raw-Qwen result may be reported as an additional ablation, but it is not a third
training stage.

## Review-Gated Phases

### Phase 0 — Documentation

- Save this implementation plan.
- Save the full tensor contract, invariants, example batches, and serialization
  rules in `INPUT_CONTRACT.md`.
- Add no Python implementation.
- Stop for review.

### Phase 1 — Contract Foundation

- Implement the typed tensor contract and its validation rules.
- Implement the special-token definitions and tokenizer registration helper.
- Use only small, hand-authored tensors inside unit tests to exercise the
  contract; do not add synthetic dataset or sample-builder APIs.
- Test tensor shapes and dtypes, graph ownership, component isolation, marker
  positions, initialization modes, auxiliary-target masks, and invalid inputs.
- Leave SSM corruption, QA serialization, collators, and all real dataset
  construction to Phase 4.
- Prompt-only inference batches rely on Qwen's assistant generation prompt. The
  collator records assistant-content start positions for teacher-forced label
  validation, without introducing a task delimiter token.
- Stop for review and revisions.

### Phase 2 — KG Reasoner

- Implement external frozen-embedding inputs, required linked initialization,
  relation and entity bottlenecks, the fixed inverse-degree walker, frozen-value
  aggregation, and residual fusion.
- Test hand-computed transitions, global normalization, zero-score fallback,
  component isolation, loss masks, and gradients.
- Stop for review and revisions.

### Phase 3 — Qwen Architecture

- Implement the custom configuration, backbone, causal-LM wrapper, model output
  objects, configurable KIL placement, checkpoint loading, save/load, and cached
  greedy generation.
- Test the one-KIL configuration using a tiny Qwen3 model.
- Run forward/backward tests and verify gradients pass through frozen Qwen to
  the KIL adapters, interaction layers, and marker-token deltas, while native
  Qwen parameters and frozen knowledge data receive no gradients.
- Provide an opt-in smoke test for the official Qwen3-4B checkpoint.
- Stop for review and revisions.

### Phase 4 — Batching and Preprocessing

- Convert the existing Wikipedia span-corruption data into the Stage 1 contract.
- Add benchmark adapters that entity-link and mark questions, construct local
  graphs, and produce Stage 2 answer suffixes.
- Build disjoint-union graph batching and edge-index offsets.
- Build marker-position extraction, suffix-only labels, linked initial nodes,
  frozen embedding resolution, and relation targets.
- Add end-to-end dataset-to-loss tests.
- Stop for review before any full training run.

### Phase 5 — Training and Benchmark Evaluation

- Run Stage 1 reconstruction pretraining and save a resumable checkpoint that
  includes frozen Qwen weights, KIL adapters, tokenizer, configuration, the
  Oreo-only optimizer, scheduler, and progress state.
- Adaptively fine-tune the Oreo-Qwen checkpoint and a separately initialized raw
  Qwen control on each benchmark's training split.
- Implement prompt-only generation, answer extraction, benchmark-specific answer
  normalization/metrics, and reproducible prediction artifacts.
- Verify that no gold answer, labels, or auxiliary targets enter generation.
- Stop for review before reporting final comparisons.

## Acceptance Criteria and Deferred Work

- All work remains inside `oreo-decoder/baseline/`.
- Native PyTorch scatter/index operations are used instead of `torch_scatter`.
- `pytest` is used for automated tests.
- A dense entity table is used for the baseline; tiny hand-authored graphs are
  used only as deterministic unit-test fixtures.
- Save/load round trips preserve outputs in evaluation mode.
- The architecture supports standard no-graph Qwen calls.
- Greedy cached generation is supported initially.
- Fine-tuned raw-Qwen and Oreo-Qwen checkpoints are evaluated through the same
  generation and metric pipeline.
- Full-Wikidata memory sharding, distributed training, and beam search are
  deferred from the baseline.
- Direct tests cover prompt/answer isolation, bidirectional prompt access,
  strict preceding-`T_ENT` masking, exact no-memory identity, grouped-query head
  dimensions, gradients, and cached-decoding equivalence.
