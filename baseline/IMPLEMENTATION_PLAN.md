# OreoLM–Qwen3-0.6B Baseline Implementation Plan

## Purpose

This directory will contain a self-contained baseline that integrates OreoLM-style
Knowledge Interaction Layers (KILs) into `Qwen/Qwen3-0.6B`.

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
│   ├── KIL/KG reasoner
│   └── final Qwen RMSNorm
└── tied language-model head
```

### `OreoQwen3Config`

`OreoQwen3Config` will extend `Qwen3Config`. It is the serializable model
blueprint and will store:

- Standard Qwen architecture fields.
- Configurable KIL insertion boundaries.
- Entity and relation vocabulary sizes.
- Entity and relation embedding dimensions.
- Entity-linking and relation-loss coefficients.
- OreoLM special-token IDs.
- Diagnostic-output settings.

It validates configuration but performs no forward computation. The default
Qwen3-0.6B layout has two KILs after completed decoder layers 9 and 19, producing
groups of 9, 10, and 9 Qwen layers. Other strictly increasing, in-range
boundaries will be supported.

### `OreoQwen3Model`

`OreoQwen3Model` is the backbone without the vocabulary projection. It will:

1. Embed input tokens.
2. Prepare local entity embeddings and initialize the mention-level entity
   state once before the first KIL.
3. Run the native Qwen decoder-layer loop.
4. Invoke a KIL after each configured layer boundary.
5. Thread the entity state explicitly from one KIL to the next.
6. Apply Qwen's final RMSNorm.
7. Return hidden states, cache data, auxiliary losses, and optional KG
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
           + entity_link_loss
           + relation_loss
```

The three coefficients are configurable and default to `1.0`. Full fine-tuning
of Qwen, KIL modules, KG memories, and new token embeddings is the initial
training mode.

During cached greedy generation, KILs run during prompt prefill. They are skipped
on subsequent one-token cached decoding calls because the marker positions occur
in the prompt and their effects are already represented in downstream KV caches.
Beam-search graph expansion is deferred.

## KG Reasoning Semantics

The reasoner will contain:

- A dense raw entity embedding table.
- A shared relation-key memory.
- An entity predictor for learned initialization when no external entity state
  is supplied.
- A separate relation-prediction head for each KIL step.
- A sparse graph walker implemented with native PyTorch indexing and scatter
  operations.
- A separate entity-value projection and fusion module for each KIL step.

The raw entity embeddings are aggregated using the walked entity distribution
before the step-specific value projection is applied. The projected value is
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
Qwen3-0.6B
    -> Stage 1: Wikipedia salient-span reconstruction pretraining
    -> saved Oreo-Qwen pretraining checkpoint
    -> Stage 2: benchmark-specific adaptive QA fine-tuning
    -> benchmark evaluation checkpoint
```

### Stage 1 — Salient-span reconstruction pretraining

The initial training corpus consists of the span-corrupted Wikipedia data used
by OreoLM. Reconstruction samples contain marked mentions, a local graph, a
reconstruction delimiter, and one unique sentinel for each selected span
occurrence:

```text
<marked and corrupted passage> <|kg_reconstruct|>
<|kg_mask_0|> first span <|kg_mask_1|> second span ... EOS
```

All labels from the beginning of the passage through `<|kg_reconstruct|>` are
`-100`. Every suffix token beginning with `<|kg_mask_0|>`, including sentinels,
reconstructed span tokens, and EOS, contributes to causal-LM loss.

Sixteen sentinels, `<|kg_mask_0|>` through `<|kg_mask_15|>`, are reserved. If the
same entity is masked in multiple locations, each occurrence receives a distinct
sentinel so its exact surface form can be reconstructed.

This stage also uses the entity-linking and weak relation-path objectives when
their targets are available. It does not require QA examples and does not train
`<|kg_answer|>` behavior.

### Stage 2 — Benchmark adaptive QA fine-tuning

For each benchmark, such as NaturalQuestions, entity mentions in the question
are linked and marked, a local KG is supplied, and the answer is appended after
the answer delimiter:

```text
Where was <|s_ent|> Albert Einstein <|rel|> <|t_ent|>
born, and in which country is that city? <|kg_answer|> Germany EOS
```

All labels through `<|kg_answer|>` are `-100`; only answer tokens and EOS are
supervised. Entity-link and relation targets are optional in this stage because
QA benchmarks generally do not provide gold reasoning paths. Even without
auxiliary targets, answer loss remains differentiable through the KILs.

The raw-Qwen control starts from the same Qwen3-0.6B base checkpoint and undergoes
the same benchmark adaptive fine-tuning, but it does not receive a graph or run
KILs. The primary controlled comparison uses the same marked textual prompt and
`<|kg_answer|>` delimiter for both models so the presence of text and special
tokens is held constant. Both models use the same train/dev/test split, answer
targets, optimization budget, decoding settings, and evaluator. A separate
plain-question raw-Qwen result may be reported as an additional ablation, but it
is not a third training stage.

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
- Stop for review and revisions.

### Phase 2 — KG Reasoner

- Implement entity and relation memories, entity initialization, per-step
  relation predictors, the fixed inverse-degree walker, raw-entity aggregation,
  and residual fusion.
- Test hand-computed transitions, global normalization, zero-score fallback,
  component isolation, loss masks, and gradients.
- Stop for review and revisions.

### Phase 3 — Qwen Architecture

- Implement the custom configuration, backbone, causal-LM wrapper, model output
  objects, configurable KIL placement, checkpoint loading, save/load, and cached
  greedy generation.
- Test one- and two-KIL configurations using a tiny Qwen3 model.
- Run forward/backward tests and verify gradients reach Qwen, KIL modules, KG
  memories, and new token embeddings.
- Provide an opt-in smoke test for the official Qwen3-0.6B checkpoint.
- Stop for review and revisions.

### Phase 4 — Batching and Preprocessing

- Convert the existing Wikipedia span-corruption data into the Stage 1 contract.
- Add benchmark adapters that entity-link and mark questions, construct local
  graphs, and produce Stage 2 answer suffixes.
- Build disjoint-union graph batching and edge-index offsets.
- Build marker-position extraction, suffix-only labels, relation targets, and
  entity-link targets.
- Add end-to-end dataset-to-loss tests.
- Stop for review before any full training run.

### Phase 5 — Training and Benchmark Evaluation

- Run Stage 1 reconstruction pretraining and save a resumable checkpoint that
  includes Qwen weights, KILs, KG memories, tokenizer, configuration, optimizer,
  scheduler, and progress state.
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
- The decoder causal-attention limitation is intentionally out of scope for this
  baseline.
