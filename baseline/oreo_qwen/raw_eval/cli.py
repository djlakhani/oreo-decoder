"""CLI for preparing, running, validating, inspecting, and scoring raw Qwen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from ..wikidata5m import Wikidata5MError, Wikidata5MStore
from ..wikidata5m.preprocessing import PreprocessConfig
from .artifacts import iter_eval_examples, validate_eval_dataset
from .inference import run_inference
from .prepare import prepare_random_sample
from .report import rescore_run
from .schema import RawEvalError


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _preprocess_config(args: argparse.Namespace) -> PreprocessConfig:
    return PreprocessConfig(
        max_incoming_degree=args.max_incoming_degree,
        max_aliases_per_entity=args.max_aliases_per_entity,
        max_auxiliary_targets=args.max_auxiliary_targets,
        max_context_blanks=args.max_context_blanks,
    )


def _prepare(args: argparse.Namespace) -> None:
    source = Path(args.source).expanduser().resolve()
    with Wikidata5MStore(source / "indexed", source_root=source) as store:
        output = prepare_random_sample(
            store,
            args.output,
            sample_size=args.sample_size,
            seed=args.seed,
            shard_size=args.shard_size,
            preprocess_config=_preprocess_config(args),
            force=args.force,
            checkpoint_interval=args.checkpoint_interval,
        )
    print(output)


def _run(args: argparse.Namespace) -> None:
    output = run_inference(
        args.input,
        args.output,
        model_name=args.model,
        revision=args.revision,
        batch_size=args.batch_size,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        local_files_only=not args.allow_download,
        seed=args.seed,
    )
    print(output)


def _score(args: argparse.Namespace) -> None:
    _print(rescore_run(args.run, args.output))


def _validate(args: argparse.Namespace) -> None:
    manifest = validate_eval_dataset(args.input)
    _print({"status": "ok", "records": manifest["records"], "shards": len(manifest["shards"])})


def _inspect(args: argparse.Namespace) -> None:
    root = Path(args.input).expanduser().resolve()
    manifest = validate_eval_dataset(root)
    remaining = args.count
    for entry in manifest["shards"]:
        for example in iter_eval_examples(root / entry["path"]):
            payload = example.to_dict()
            if not args.show_gold:
                payload["targets"] = [
                    {
                        "blank_index": item["blank_index"],
                        "blank_token": item["blank_token"],
                        "kind": item["kind"],
                    }
                    for item in payload["targets"]
                ]
            _print(payload)
            remaining -= 1
            if remaining == 0:
                return


def _add_preprocess_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = PreprocessConfig()
    parser.add_argument("--max-incoming-degree", type=int, default=defaults.max_incoming_degree)
    parser.add_argument("--max-aliases-per-entity", type=int, default=defaults.max_aliases_per_entity)
    parser.add_argument("--max-auxiliary-targets", type=int, default=defaults.max_auxiliary_targets)
    parser.add_argument("--max-context-blanks", type=int, default=defaults.max_context_blanks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="raw-qwen-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare", help="sample accepted passages and build native raw-Qwen inputs"
    )
    prepare.add_argument("source", help="Wikidata5M source root containing indexed/")
    prepare.add_argument("output")
    prepare.add_argument("--sample-size", type=int, required=True)
    prepare.add_argument("--seed", type=int, default=20261104)
    prepare.add_argument("--shard-size", type=int, default=250)
    prepare.add_argument("--checkpoint-interval", type=int, default=1000)
    prepare.add_argument("--force", action="store_true")
    _add_preprocess_arguments(prepare)
    prepare.set_defaults(handler=_prepare)

    run = subparsers.add_parser("run", help="run non-thinking Qwen inference")
    run.add_argument("input")
    run.add_argument("output")
    run.add_argument("--model", default="Qwen/Qwen3-0.6B")
    run.add_argument("--revision")
    run.add_argument("--batch-size", type=int, default=4)
    run.add_argument("--max-input-tokens", type=int, default=8192)
    run.add_argument("--max-new-tokens", type=int, default=512)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--allow-download", action="store_true")
    run.set_defaults(handler=_run)

    score = subparsers.add_parser("score", help="authoritatively rescore saved predictions")
    score.add_argument("run")
    score.add_argument("--output")
    score.set_defaults(handler=_score)

    validate = subparsers.add_parser("validate", help="validate materialized eval inputs")
    validate.add_argument("input")
    validate.set_defaults(handler=_validate)

    inspect = subparsers.add_parser("inspect", help="print materialized prompts")
    inspect.add_argument("input")
    inspect.add_argument("--count", type=int, default=1)
    inspect.add_argument("--show-gold", action="store_true")
    inspect.set_defaults(handler=_inspect)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (RawEvalError, Wikidata5MError, OSError, ValueError, KeyError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
