from __future__ import annotations

from pathlib import Path

import torch

from oreo_qwen import OreoQwen3Config, OreoQwen3ForCausalLM
from oreo_qwen.training import (
    TrainingConfiguration,
    TrainingState,
    build_parser,
    load_training_checkpoint,
    save_training_checkpoint,
)


class _Tokenizer:
    def save_pretrained(self, path: Path) -> None:
        path.mkdir(parents=True)
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")


def _model() -> OreoQwen3ForCausalLM:
    return OreoQwen3ForCausalLM(
        OreoQwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            num_relations=2,
            kil_layer_boundaries=[1],
            knowledge_embedding_dim=4,
            adapter_bottleneck_dim=8,
            tie_word_embeddings=True,
        )
    )


def _configuration(tmp_path: Path) -> TrainingConfiguration:
    return TrainingConfiguration(
        model_name="Qwen/Qwen3-4B",
        embedding_model_name="Qwen/Qwen3-Embedding-0.6B",
        corpus_dir=str(tmp_path / "corpus"),
        index_dir=str(tmp_path / "index"),
        output_dir=str(tmp_path / "output"),
        embedding_cache=str(tmp_path / "cache.sqlite3"),
        kil_boundary=34,
        max_length=1024,
        batch_size=1,
        gradient_accumulation_steps=1,
        epochs=1,
        max_steps=1,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.03,
        max_grad_norm=1.0,
        num_workers=0,
        embedding_batch_size=16,
        save_steps=1,
        save_total_limit=3,
        seed=17,
        dtype="bfloat16",
        embedding_device="cuda:0",
        gradient_checkpointing=True,
        local_files_only=True,
        smoke_one_batch=True,
    )


def test_training_cli_defaults_to_late_kil() -> None:
    args = build_parser().parse_args(
        [
            "--corpus",
            "corpus",
            "--index",
            "index",
            "--output",
            "output",
            "--embedding-cache",
            "cache.sqlite3",
        ]
    )
    assert args.kil_boundary == 34
    assert args.gradient_checkpointing is True
    assert args.dtype == "bfloat16"


def test_oreo_only_checkpoint_round_trip(tmp_path: Path) -> None:
    model = _model()
    optimizer = torch.optim.AdamW(list(model.oreo_parameters()), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    state = TrainingState(global_step=4, epoch=0, batches_consumed_in_epoch=7)
    original = {
        name: parameter.detach().clone()
        for name, parameter in model.oreo_named_parameters()
    }
    output = tmp_path / "run"
    output.mkdir()
    checkpoint = output
    for step in range(1, 5):
        checkpoint = save_training_checkpoint(
            output_dir=output,
            model=model,
            tokenizer=_Tokenizer(),
            optimizer=optimizer,
            scheduler=scheduler,
            state=TrainingState(
                global_step=step,
                epoch=0,
                batches_consumed_in_epoch=7 if step == 4 else step,
            ),
            config=_configuration(tmp_path),
        )
    assert [path.name for path in sorted(output.glob("checkpoint-*"))] == [
        "checkpoint-00000002",
        "checkpoint-00000003",
        "checkpoint-00000004",
    ]
    assert (checkpoint / "oreo_model.safetensors").is_file()
    assert not (checkpoint / "model.safetensors").exists()

    with torch.no_grad():
        next(model.oreo_parameters()).add_(1.0)
    restored = load_training_checkpoint(
        checkpoint=checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    assert restored == state
    for name, parameter in model.oreo_named_parameters():
        torch.testing.assert_close(parameter, original[name])
    assert all(
        parameter.grad is None
        for name, parameter in model.named_parameters()
        if not name.startswith(
            (
                "model.reasoner.",
                "model.pre_kil_layers.",
                "model.post_kil_layers.",
                "model.oreo_token_deltas",
            )
        )
    )
