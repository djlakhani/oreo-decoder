# RunPod Oreo-Qwen Training

The scripts target one RTX 6000 Ada (48 GB), 62 GB system RAM, a 40 GB
container disk, and a persistent volume mounted at `/workspace`. All model,
package, data, embedding-cache, and checkpoint files are redirected to the
persistent volume.

Run in order:

```bash
cp scripts/runpod/runpod_env.example.sh scripts/runpod/runpod_env.sh
bash scripts/runpod/00_setup.sh
bash scripts/runpod/01_download_data.sh
bash scripts/runpod/10_smoke_train.sh
bash scripts/runpod/02_prepare_corpus.sh
bash scripts/runpod/20_train.sh
```

The S3 commands never use `--delete`. The smoke run performs one complete
forward/backward/optimizer step, verifies that frozen Qwen weights did not
change, writes an Oreo-only resumable checkpoint, and uploads it when
`UPLOAD_TRAINING_RUN=1`. If the full corpus is not built yet, the smoke script
first creates a bounded 100,000-entity preprocessing corpus so architecture
validation does not have to wait for the full CPU preprocessing pass.

For anonymous public S3 reads, set `S3_NO_SIGN_REQUEST=1` and also set
`UPLOAD_TRAINING_RUN=0` and `UPLOAD_PREPROCESSED=0`. Uploads require AWS
credentials supplied as RunPod secrets.

Full-training defaults are conservative: BF16, batch size 1, gradient
accumulation 8, gradient checkpointing, 2,048 tokens, and one epoch. Override
shell variables such as `MAX_LENGTH`, `GRADIENT_ACCUMULATION_STEPS`,
`LEARNING_RATE`, `NUM_WORKERS`, and `SAVE_STEPS` after the smoke run establishes
the actual memory headroom. Corpus construction defaults to eight independent
processes (`PREPROCESS_WORKERS=8`) and publishes a resumable manifest plus live
throughput and ETA after every 10,000-entity shard.
Only the latest three local checkpoints are retained by default
(`SAVE_TOTAL_LIMIT=3`) so Adam state does not fill the 100 GB volume. This
pruning is local only; the scripts never delete S3 objects.
If the smoke run is close to 48 GB, set `EMBEDDING_DEVICE=cpu` for full training
so the frozen 0.6B embedding encoder does not remain on the training GPU.

