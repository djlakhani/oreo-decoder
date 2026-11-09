# Qwen3-4B Raw Reconstruction Evaluation

This run uses the same prompts, deterministic sample seed, non-thinking chat
template, greedy decoding, scoring, and S3 bucket as the Qwen3-0.6B baseline.
It reuses the existing `eval-inputs` shards from S3 whenever available, so the
two checkpoints see byte-identical evaluation examples.

The checkpoint is [`Qwen/Qwen3-4B`](https://huggingface.co/Qwen/Qwen3-4B), a
4.0B-parameter BF16 causal model. The raw evaluation explicitly applies
`enable_thinking=False`, as it did for Qwen3-0.6B.

## Storage on a fresh instance

The scripts keep large writable data off the root volume:

```text
/opt/dlami/nvme/kgrealm/
├── cache/huggingface/    # Qwen3-4B download
├── cache/uv/
├── datasets/Wikidata5M/
├── eval-inputs/
├── eval-runs/
├── tools/
└── venv/
```

The 100 GB root volume therefore holds the AMI and repository but not the model,
project environment, dataset, or generated results. The scripts require
`/opt/dlami/nvme` to be a mounted filesystem; confirm this instance exposes the
same instance-store volume before setup:

```bash
df -hT /opt/dlami/nvme
```

S3 remains the durable store:

```text
s3://kgrealm-oreo-data-589498925064-us-east-1-an/kgrealm-oreo-data/
├── datasets/Wikidata5M/
├── eval-inputs/
└── eval-runs/
```

## Fresh-instance setup

Clone the repository, create the machine-local environment file, and leave the
default Hugging Face paths pointing to NVMe:

```bash
cd /home/ubuntu/oreo-decoder/baseline/scripts/aws
cp aws_env.example.sh aws_env.sh
```

Verify at least `OREO_DECODER_REPO`, `KGREALM_SCRATCH`, and `KGREALM_S3_URI` in
`aws_env.sh`. No pre-existing Hugging Face cache is required.

Run setup inside `tmux` because downloading the checkpoint and dependencies can
take time:

```bash
tmux new -s qwen4b-setup
cd /home/ubuntu/oreo-decoder/baseline/scripts/aws
./01_setup_instance_qwen3_4b.sh
```

This installs `uv` if needed, creates the locked project environment on NVMe,
downloads the complete public `Qwen/Qwen3-4B` snapshot into `HF_HUB_CACHE`, and
verifies that its config and tokenizer resolve in offline mode. A rerun is safe
and downloads only missing files.

Download and validate Wikidata5M next:

```bash
./02_download_and_validate_wikidata.sh
```

The evaluation scripts normally download the already-generated input samples
from S3. Wikidata5M is still retained as a fallback if an input sample is absent
and must be deterministically rebuilt.

## Evaluation sequence

Run the 32-example smoke test first:

```bash
tmux new -s qwen4b-smoke
cd /home/ubuntu/oreo-decoder/baseline/scripts/aws
./10_smoke_eval_qwen3_4b.sh
```

The default batch size is two. If the new instance has less GPU memory than the
previous L4 instance or the smoke run reports CUDA out-of-memory, retry with:

```bash
QWEN3_4B_BATCH_SIZE=1 ./10_smoke_eval_qwen3_4b.sh
```

After reviewing the smoke report, run:

```bash
./15_pilot_eval_qwen3_4b.sh
./20_main_eval_5000_qwen3_4b.sh
```

The output directories and S3 prefixes are distinct from Qwen3-0.6B:

```text
eval-runs/qwen3-4b-native-smoke32-v1/
eval-runs/qwen3-4b-native-pilot200-v1/
eval-runs/qwen3-4b-native-main5000-v1/
```

Each run saves raw generations, prediction sidecars, provisional and rescored
metrics, environment/model commit metadata, checksums, and `review.html`. The
input folders retain their existing model-independent names:

```text
eval-inputs/raw-qwen-native-smoke32-v1/
eval-inputs/raw-qwen-native-pilot200-v1/
eval-inputs/raw-qwen-native-main5000-v1/
```

The scripts use offline inference after setup, preventing an evaluation from
silently switching model snapshots. Completed artifacts are synchronized to the
same S3 bucket at the end of each run.

During inference, a terminal progress bar updates after approximately every ten
examples. This is display-only and does not add checkpoint I/O. Recovery remains shard-based:
completed prediction shards and `partial_metrics.json` survive an interruption,
while the currently running shard is recomputed after restart.
