"""Single-GPU reconstruction training for the frozen-backbone Oreo-Qwen model."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from .data import (
    OreoSemanticCollator,
    SemanticOreoBatch,
    Wikidata5MReconstructionDataset,
)
from .embedding import FrozenKnowledgeEmbedder
from .modeling import OreoQwen3ForCausalLM, register_oreo_tokens_and_resize


@dataclass(frozen=True, slots=True)
class TrainingConfiguration:
    model_name: str
    embedding_model_name: str
    corpus_dir: str
    index_dir: str
    output_dir: str
    embedding_cache: str
    kil_boundary: int
    max_length: int
    batch_size: int
    gradient_accumulation_steps: int
    epochs: int
    max_steps: int | None
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float
    num_workers: int
    embedding_batch_size: int
    save_steps: int
    save_total_limit: int
    seed: int
    dtype: str
    embedding_device: str
    gradient_checkpointing: bool
    local_files_only: bool
    smoke_one_batch: bool


@dataclass(slots=True)
class TrainingState:
    global_step: int = 0
    epoch: int = 0
    batches_consumed_in_epoch: int = 0
    examples_seen: int = 0
    dropped_batches: int = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oreo-qwen-train")
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--embedding-cache", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B")
    parser.add_argument(
        "--embedding-model-name", default="Qwen/Qwen3-Embedding-0.6B"
    )
    parser.add_argument("--kil-boundary", type=int, default=34)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument(
        "--embedding-device", default="cuda", help="cuda, cpu, or an explicit device"
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume-from")
    parser.add_argument("--smoke-one-batch", action="store_true")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    positive = (
        "kil_boundary",
        "max_length",
        "batch_size",
        "gradient_accumulation_steps",
        "epochs",
        "learning_rate",
        "max_grad_norm",
        "embedding_batch_size",
        "save_steps",
        "save_total_limit",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("max-steps must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers must be nonnegative")
    if args.weight_decay < 0:
        raise ValueError("weight-decay must be nonnegative")
    if not 0 <= args.warmup_ratio < 1:
        raise ValueError("warmup-ratio must lie in [0, 1)")


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _configuration(args: argparse.Namespace) -> TrainingConfiguration:
    max_steps = 1 if args.smoke_one_batch else args.max_steps
    accumulation = 1 if args.smoke_one_batch else args.gradient_accumulation_steps
    workers = 0 if args.smoke_one_batch else args.num_workers
    return TrainingConfiguration(
        model_name=args.model_name,
        embedding_model_name=args.embedding_model_name,
        corpus_dir=str(Path(args.corpus).expanduser().resolve()),
        index_dir=str(Path(args.index).expanduser().resolve()),
        output_dir=str(Path(args.output).expanduser().resolve()),
        embedding_cache=str(Path(args.embedding_cache).expanduser().resolve()),
        kil_boundary=args.kil_boundary,
        max_length=args.max_length,
        batch_size=args.batch_size,
        gradient_accumulation_steps=accumulation,
        epochs=args.epochs,
        max_steps=max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=args.max_grad_norm,
        num_workers=workers,
        embedding_batch_size=args.embedding_batch_size,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        seed=args.seed,
        dtype=args.dtype,
        embedding_device=args.embedding_device,
        gradient_checkpointing=args.gradient_checkpointing,
        local_files_only=args.local_files_only,
        smoke_one_batch=args.smoke_one_batch,
    )


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _parameter_counts(model: OreoQwen3ForCausalLM) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.oreo_parameters())
    return total, trainable


def _assert_parameter_policy(model: OreoQwen3ForCausalLM) -> None:
    trainable_names = {name for name, _ in model.oreo_named_parameters()}
    if not trainable_names:
        raise RuntimeError("the model has no trainable Oreo parameters")
    allowed = (
        "model.reasoner.",
        "model.pre_kil_layers.",
        "model.post_kil_layers.",
    )
    invalid = sorted(
        name
        for name in trainable_names
        if name != "model.oreo_token_deltas" and not name.startswith(allowed)
    )
    if invalid:
        raise RuntimeError(f"native Qwen parameters are trainable: {invalid[:5]}")


def _assert_gradient_policy(model: OreoQwen3ForCausalLM) -> None:
    missing = [
        name
        for name, parameter in model.oreo_named_parameters()
        if parameter.grad is None
    ]
    # Zero-initialized residual gates intentionally delay projection gradients;
    # require at least one Oreo gradient rather than every Oreo gradient.
    if len(missing) == sum(1 for _ in model.oreo_named_parameters()):
        raise RuntimeError("no Oreo parameter received a gradient")
    leaked = [
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad and parameter.grad is not None
    ]
    if leaked:
        raise RuntimeError(f"frozen Qwen parameters received gradients: {leaked[:5]}")


def _optimizer(
    model: OreoQwen3ForCausalLM, config: TrainingConfiguration
) -> torch.optim.Optimizer:
    decay: list[Tensor] = []
    no_decay: list[Tensor] = []
    for name, parameter in model.oreo_named_parameters():
        if (
            parameter.ndim < 2
            or name.endswith(".bias")
            or name == "model.oreo_token_deltas"
        ):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    kwargs: dict[str, Any] = {
        "lr": config.learning_rate,
        "betas": (0.9, 0.95),
    }
    if torch.cuda.is_available():
        kwargs["fused"] = True
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        **kwargs,
    )


def _checkpoint_name(step: int) -> str:
    return f"checkpoint-{step:08d}"


def _prune_local_checkpoints(
    output_dir: Path, *, keep: Path, limit: int
) -> None:
    candidates = sorted(
        path
        for path in output_dir.glob("checkpoint-????????")
        if path.is_dir() and path.name.removeprefix("checkpoint-").isdigit()
    )
    removable = [path for path in candidates if path != keep]
    while len(candidates) > limit and removable:
        target = removable.pop(0)
        if not target.resolve().is_relative_to(output_dir.resolve()):
            raise RuntimeError("refusing to prune a checkpoint outside the run directory")
        shutil.rmtree(target)
        candidates.remove(target)
        print(f"pruned local checkpoint {target}", flush=True)


def save_training_checkpoint(
    *,
    output_dir: Path,
    model: OreoQwen3ForCausalLM,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    state: TrainingState,
    config: TrainingConfiguration,
) -> Path:
    destination = output_dir / _checkpoint_name(state.global_step)
    if destination.exists():
        return destination
    temporary = output_dir / f".{destination.name}.tmp"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    oreo_state = {
        name: parameter.detach().to(device="cpu").contiguous()
        for name, parameter in model.oreo_named_parameters()
    }
    save_file(oreo_state, temporary / "oreo_model.safetensors")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "state": asdict(state),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all(),
        },
        temporary / "trainer_state.pt",
    )
    model.config.save_pretrained(temporary)
    tokenizer.save_pretrained(temporary / "tokenizer")
    _atomic_json(temporary / "training_config.json", asdict(config))
    os.replace(temporary, destination)
    _atomic_json(
        output_dir / "latest_checkpoint.json",
        {"path": destination.name, "global_step": state.global_step},
    )
    _prune_local_checkpoints(
        output_dir, keep=destination, limit=config.save_total_limit
    )
    return destination


def load_training_checkpoint(
    *,
    checkpoint: Path,
    model: OreoQwen3ForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> TrainingState:
    expected = {name for name, _ in model.oreo_named_parameters()}
    oreo_state = load_file(checkpoint / "oreo_model.safetensors", device="cpu")
    if set(oreo_state) != expected:
        missing = sorted(expected - set(oreo_state))
        extra = sorted(set(oreo_state) - expected)
        raise ValueError(f"Oreo checkpoint keys differ; missing={missing}, extra={extra}")
    incompatible = model.load_state_dict(oreo_state, strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"unexpected Oreo checkpoint keys: {incompatible.unexpected_keys}")
    trainer = torch.load(
        checkpoint / "trainer_state.pt", map_location="cpu", weights_only=False
    )
    optimizer.load_state_dict(trainer["optimizer"])
    scheduler.load_state_dict(trainer["scheduler"])
    torch.set_rng_state(trainer["torch_rng_state"])
    torch.cuda.set_rng_state_all(trainer["cuda_rng_state"])
    return TrainingState(**trainer["state"])


def _latest_checkpoint(output_dir: Path) -> Path | None:
    path = output_dir / "latest_checkpoint.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = (output_dir / payload["path"]).resolve()
    if not checkpoint.is_relative_to(output_dir.resolve()) or not checkpoint.is_dir():
        raise ValueError("latest checkpoint pointer is invalid")
    return checkpoint


def _gpu_summary(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {"device": str(device)}
    properties = torch.cuda.get_device_properties(device)
    return {
        "device": str(device),
        "name": properties.name,
        "memory_gib": round(properties.total_memory / 2**30, 2),
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }


def train(config: TrainingConfiguration, *, resume_from: str | None = None) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("Oreo-Qwen training requires a CUDA GPU")
    device = torch.device("cuda:0")
    model_dtype = _dtype(config.dtype)
    if model_dtype is torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected GPU does not support bfloat16")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    _seed_everything(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = output_dir / "training_config.json"
    if run_config_path.exists():
        existing = json.loads(run_config_path.read_text(encoding="utf-8"))
        if existing != asdict(config):
            raise ValueError("output directory contains a different training configuration")
    else:
        _atomic_json(run_config_path, asdict(config))

    print(json.dumps(_gpu_summary(device), indent=2), flush=True)
    print(f"Loading tokenizer: {config.model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, local_files_only=config.local_files_only
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = Wikidata5MReconstructionDataset(
        config.corpus_dir,
        index_dir=config.index_dir,
        shuffle_shards=True,
        seed=config.seed,
    )
    print(
        f"Loading frozen Qwen backbone and Oreo modules: {config.model_name}",
        flush=True,
    )
    model = OreoQwen3ForCausalLM.from_qwen3_pretrained(
        config.model_name,
        oreo_config_overrides={
            "num_relations": dataset.relation_count,
            "kil_layer_boundaries": [config.kil_boundary],
            "knowledge_embedding_dim": 384,
            "adapter_bottleneck_dim": 512,
            "knowledge_embedding_model_name": config.embedding_model_name,
        },
        dtype=model_dtype,
        attn_implementation="sdpa",
        local_files_only=config.local_files_only,
    )
    register_oreo_tokens_and_resize(model, tokenizer)
    model.config.use_cache = False
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.to(device)
    # Keep the small trainable Oreo state and Adam moments in FP32 while the
    # frozen Qwen backbone and its activations use BF16 under autocast.
    for parameter in model.oreo_parameters():
        parameter.data = parameter.data.float()
    model.train()
    _assert_parameter_policy(model)
    total_parameters, trainable_parameters = _parameter_counts(model)
    print(
        f"parameters total={total_parameters:,} trainable_oreo={trainable_parameters:,} "
        f"({100 * trainable_parameters / total_parameters:.3f}%)",
        flush=True,
    )

    print(
        f"Loading frozen knowledge encoder on {config.embedding_device}: "
        f"{config.embedding_model_name}",
        flush=True,
    )
    embedder = FrozenKnowledgeEmbedder(
        config.embedding_model_name,
        output_dim=model.config.knowledge_embedding_dim,
        device=config.embedding_device,
        model_dtype=model_dtype,
        cache_path=config.embedding_cache,
        max_memory_items=100_000,
        local_files_only=config.local_files_only,
    )
    relation_requests = dataset.relation_embedding_requests()
    relation_bank = embedder.resolve(
        [item.key for item in relation_requests],
        [item.text for item in relation_requests],
        batch_size=config.embedding_batch_size,
        output_device=device,
        output_dtype=model_dtype,
    )
    print(
        f"relation_bank={tuple(relation_bank.shape)} "
        f"persistent_cache_items={embedder.persistent_cached_items:,}",
        flush=True,
    )

    collator = OreoSemanticCollator(
        tokenizer,
        num_relations=dataset.relation_count,
        num_kil_steps=model.config.num_kil_steps,
        max_length=config.max_length,
        pad_to_multiple_of=8,
        overflow_policy="drop",
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        collate_fn=collator,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2 if config.num_workers > 0 else None,
    )
    batches_per_epoch = math.ceil(len(dataset) / config.batch_size)
    estimated_updates = math.ceil(
        batches_per_epoch / config.gradient_accumulation_steps
    ) * config.epochs
    total_updates = (
        min(estimated_updates, config.max_steps)
        if config.max_steps is not None
        else estimated_updates
    )
    if total_updates <= 0:
        raise RuntimeError("the corpus and training settings produce no optimizer steps")
    warmup_steps = int(total_updates * config.warmup_ratio)
    optimizer = _optimizer(model, config)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_updates
    )

    checkpoint = (
        Path(resume_from).expanduser().resolve()
        if resume_from is not None
        else _latest_checkpoint(output_dir)
    )
    state = TrainingState()
    if checkpoint is not None:
        print(f"Resuming from {checkpoint}", flush=True)
        state = load_training_checkpoint(
            checkpoint=checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if state.global_step >= total_updates or (
            config.max_steps is not None and state.global_step >= config.max_steps
        ):
            print("Requested training is already complete.", flush=True)
            embedder.close()
            return checkpoint

    metrics_path = output_dir / "metrics.jsonl"
    progress = tqdm(
        total=total_updates,
        initial=state.global_step,
        desc="Oreo training",
        unit="step",
        dynamic_ncols=True,
    )
    optimizer.zero_grad(set_to_none=True)
    accumulated_batches = 0
    accumulated_loss = 0.0
    start_time = time.monotonic()
    frozen_probe = (
        model.model.layers[0].self_attn.q_proj.weight.view(-1)[:16].detach().clone()
    )
    stop = False
    try:
        for epoch in range(state.epoch, config.epochs):
            dataset.set_epoch(epoch)
            skip_batches = state.batches_consumed_in_epoch if epoch == state.epoch else 0
            for batch_index, semantic in enumerate(loader):
                if batch_index < skip_batches:
                    continue
                state.batches_consumed_in_epoch = batch_index + 1
                if semantic is None:
                    state.dropped_batches += 1
                    continue
                if not isinstance(semantic, SemanticOreoBatch):
                    raise TypeError("collator returned an unexpected batch type")
                node_embeddings = embedder.resolve(
                    semantic.kg.node_embedding_keys,
                    semantic.kg.node_embedding_texts,
                    batch_size=config.embedding_batch_size,
                    output_device=device,
                    output_dtype=model_dtype,
                )
                batch = semantic.materialize(
                    node_embeddings=node_embeddings,
                    relation_embeddings=relation_bank,
                    device=device,
                )
                with torch.autocast(
                    device_type="cuda",
                    dtype=model_dtype,
                    enabled=model_dtype is not torch.float32,
                ):
                    outputs = model(
                        **batch.as_model_kwargs(),
                        use_cache=False,
                    )
                    if outputs.loss is None:
                        raise RuntimeError("training forward returned no loss")
                    scaled_loss = outputs.loss / config.gradient_accumulation_steps
                scaled_loss.backward()
                accumulated_batches += 1
                accumulated_loss += float(outputs.loss.detach().float().item())
                state.examples_seen += len(semantic.example_ids)

                last_batch = batch_index + 1 >= batches_per_epoch
                should_step = (
                    accumulated_batches >= config.gradient_accumulation_steps
                    or last_batch
                )
                if not should_step:
                    continue
                if state.global_step == 0:
                    _assert_gradient_policy(model)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(model.oreo_parameters()), config.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                state.global_step += 1
                mean_loss = accumulated_loss / accumulated_batches
                accumulated_batches = 0
                accumulated_loss = 0.0
                elapsed = max(time.monotonic() - start_time, 1e-6)
                metrics = {
                    "step": state.global_step,
                    "epoch": epoch,
                    "loss": mean_loss,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "grad_norm": float(grad_norm.detach().float().item()),
                    "examples_seen": state.examples_seen,
                    "examples_per_second": state.examples_seen / elapsed,
                    "persistent_cache_items": embedder.persistent_cached_items,
                    "dropped_batches": state.dropped_batches,
                    "max_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(metrics, sort_keys=True) + "\n")
                progress.update(1)
                progress.set_postfix(
                    loss=f"{mean_loss:.4f}",
                    lr=f"{metrics['learning_rate']:.2e}",
                    cache=metrics["persistent_cache_items"],
                    vram=f"{metrics['max_cuda_memory_gib']:.1f}G",
                )
                if not torch.equal(
                    model.model.layers[0].self_attn.q_proj.weight.view(-1)[:16],
                    frozen_probe,
                ):
                    raise RuntimeError("a frozen Qwen parameter changed during training")
                if state.global_step % config.save_steps == 0:
                    saved = save_training_checkpoint(
                        output_dir=output_dir,
                        model=model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        state=state,
                        config=config,
                    )
                    progress.write(f"saved {saved}")
                if config.max_steps is not None and state.global_step >= config.max_steps:
                    stop = True
                    break
            if stop:
                break
            state.epoch = epoch + 1
            state.batches_consumed_in_epoch = 0
        if state.global_step == 0:
            raise RuntimeError("training produced no optimizer steps")
        saved = save_training_checkpoint(
            output_dir=output_dir,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            state=state,
            config=config,
        )
        _atomic_json(
            output_dir / "final_metrics.json",
            {
                "status": "complete",
                "global_step": state.global_step,
                "examples_seen": state.examples_seen,
                "dropped_batches": state.dropped_batches,
                "elapsed_seconds": time.monotonic() - start_time,
                "checkpoint": saved.name,
                "max_cuda_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
            },
        )
        return saved
    finally:
        progress.close()
        embedder.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_args(args)
        config = _configuration(args)
        checkpoint = train(config, resume_from=args.resume_from)
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"training complete: {checkpoint}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "TrainingConfiguration",
    "TrainingState",
    "build_parser",
    "load_training_checkpoint",
    "main",
    "save_training_checkpoint",
    "train",
]
