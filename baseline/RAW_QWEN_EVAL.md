# Raw Qwen Reconstruction Evaluation

This evaluation measures how well unmodified `Qwen/Qwen3-0.6B` reconstructs
the Wikidata5M blanks before Oreo/KG components are introduced. Qwen sees only
the corrupted passage and textual blank identifiers such as `[BLANK_0]`. It
never sees QIDs, aliases, graph edges, dense IDs, or Oreo special tokens.

## Evaluation protocol

- Thinking is disabled through Qwen's chat template.
- Generation is greedy (`do_sample=False`).
- The prompt contains one fixed synthetic formatting example.
- The required response grammar is one line per blank, in order:

  ```text
  [BLANK_0] missing text
  [BLANK_1] other missing text
  ```

- Input passages are never silently truncated. Records exceeding
  `--max-input-tokens` receive the status `input_too_long`.
- Smoke, pilot, and main samples use the same fixed seed. Passage dense IDs are
  randomly permuted and preprocessing continues until the requested number of
  accepted examples is reached.

The 32-example smoke run checks the complete workflow. The 200-example pilot
checks formatting, token lengths, runtime, and memory. The main script evaluates
5,000 examples.

## Gold identities and alias evaluation

Every materialized target stores the following evaluation-only data:

- `blank_index` and `blank_token`
- exact text removed from the passage
- normalized exact text
- mask kind (`primary_kg`, `auxiliary_kg`, or `context_only`)
- gold QID and dense entity ID for KG masks
- normalized aliases that are unambiguous among the example's local candidate
  entities

Context-only blanks have no QID. These fields live in the input shard and are
copied into the prediction record. They are not interpolated into either model
prompt. Consequently, a saved response can always be rescored later against the
correct entity and aliases without reopening the datastore.

The scorer reports strict-format validity, recoverable-format validity, exact
surface accuracy, normalized accuracy, and alias-aware accuracy. Alias-aware
matching is deterministic string matching; no LLM judge is required. Both
per-target and complete-response metrics are retained, grouped by mask kind.

## Artifacts

An input directory contains:

```text
manifest.json
sample_manifest.json
shards/inputs-00000.jsonl.gz
```

`sample_manifest.json` records the sample order, passage QID, and source-record
hash. Each gzipped JSONL input record contains the exact prompt plus its hidden
evaluation sidecar. Checksums and record ranges are recorded in `manifest.json`.

A run directory contains:

```text
manifest.json
environment.json
final_metrics.json
rescored_metrics.json
partial_metrics.json
predictions/predictions-00000.jsonl.gz
```

Prediction records include the raw decoded generation (with special tokens),
the clean decoded generation, the full prompt, token counts, target sidecar,
and provisional score. `raw-qwen-eval score` independently recomputes the
authoritative metrics from saved generations and target sidecars.

Preparation and inference are resumable. Input preparation checkpoints accepted
records locally. Inference publishes each prediction shard atomically and skips
already verified shards when rerun with the same configuration.

## AWS storage layout

The scripts use the instance-store NVMe mount for all working data:

```text
/opt/dlami/nvme/kgrealm/
├── cache/uv/
├── tools/
├── venv/
├── datasets/Wikidata5M/
├── eval-inputs/
└── eval-runs/
```

This disk is ephemeral. S3 is the durable system of record:

```text
s3://kgrealm-oreo-data-589498925064-us-east-1-an/kgrealm-oreo-data/
├── datasets/Wikidata5M/
├── eval-inputs/
└── eval-runs/
```

The Hugging Face model cache stays at its existing instance path. The root disk
contains only the repository and existing AMI/model files; the project virtual
environment, package cache, temporary files, Wikidata, and evaluation artifacts
are redirected to NVMe.

## One-time setup

First upload the complete local Wikidata5M directory from the development
machine. It must include the canonical source files and `indexed/`:

```bash
cd oreo-decoder/baseline
./scripts/aws/00_upload_wikidata_to_s3.sh \
  /Users/diyalakhani/Documents/Diya/ucla/thesis/KGReaLM/datasets/Wikidata5M
```

On EC2, clone or update the repository, then create the local environment file:

```bash
cd /home/ubuntu/oreo-decoder/baseline/scripts/aws
cp aws_env.example.sh aws_env.sh
```

Edit `aws_env.sh` so `OREO_DECODER_REPO` is the actual clone path and
`HF_HUB_CACHE` is the already populated Hugging Face hub cache. Do not commit
this machine-specific file. Then run:

```bash
./01_setup_instance.sh
./02_download_and_validate_wikidata.sh
```

`01_setup_instance.sh` installs `uv` under NVMe only when it is absent, creates
the locked environment there, verifies CUDA, and confirms the cached Qwen model
can be opened offline. `02_download_and_validate_wikidata.sh` downloads the
dataset/index from S3 and performs full index validation against its sources.

## Running evaluations

Run these from `baseline/scripts/aws/` in order:

```bash
./10_smoke_eval.sh
./15_pilot_eval.sh
./20_main_eval_5000.sh
```

Each script prepares and validates its deterministic input sample, runs offline
non-thinking Qwen inference, independently rescores predictions, and syncs the
input and run directories to S3. Review smoke outputs before launching the
pilot, and review pilot token lengths and failures before launching the main
run.

Useful manual commands from `baseline/` are:

```bash
uv run raw-qwen-eval validate /path/to/eval-input
uv run raw-qwen-eval inspect /path/to/eval-input --count 2
uv run raw-qwen-eval inspect /path/to/eval-input --count 1 --show-gold
uv run raw-qwen-eval score /path/to/eval-run
```

Without `--show-gold`, inspection deliberately hides exact answers, QIDs, and
aliases. Generated artifacts and `scripts/aws/aws_env.sh` must remain untracked.
