"""Command-line interface for Wikidata5M reconstruction preprocessing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from ..errors import Wikidata5MError
from ..store import Wikidata5MStore
from .pipeline import (
    build_preprocessed_corpus,
    preprocess_passage,
    validate_preprocessed_corpus,
)
from .render import render_example
from .schema import PreprocessConfig


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _config(args: argparse.Namespace) -> PreprocessConfig:
    return PreprocessConfig(
        max_incoming_degree=args.max_incoming_degree,
        max_aliases_per_entity=args.max_aliases_per_entity,
        max_auxiliary_targets=args.max_auxiliary_targets,
        max_context_blanks=args.max_context_blanks,
        shard_entity_count=args.shard_entity_count,
    )


def _build(args: argparse.Namespace) -> None:
    source = Path(args.source).expanduser().resolve()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else source / "preprocessed" / "wiki5m_reconstruction_v2"
    )
    with Wikidata5MStore(source / "indexed", source_root=source) as store:
        result = build_preprocessed_corpus(
            store,
            output,
            _config(args),
            start_entity_id=args.start_entity_id,
            stop_entity_id=args.stop_entity_id,
            force=args.force,
        )
    print(result)


def _inspect(args: argparse.Namespace) -> None:
    index = Path(args.index).expanduser().resolve()
    with Wikidata5MStore(index, source_root=args.source) as store:
        result = preprocess_passage(store, args.qid, _config(args))
        payload: dict[str, Any] = {
            "accepted": result.accepted,
            "passage_qid": result.passage_qid,
            "passage_entity_id": result.passage_entity_id,
            "rejection_reason": result.rejection_reason,
            "diagnostics": result.diagnostics,
            "inspection": result.inspection,
        }
        if result.record is not None:
            passage = store.require_passage(args.qid)
            rendered = render_example(result.record, passage)
            payload["record"] = result.record.to_dict()
            payload["rendered"] = {
                "system_instruction": rendered.system_instruction,
                "corrupted_passage": rendered.corrupted_passage,
                "reconstruction_suffix": rendered.reconstruction_suffix,
            }
    _print(payload)


def _stats(args: argparse.Namespace) -> None:
    output = Path(args.output).expanduser().resolve()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    _print(
        {
            "schema_version": manifest["schema_version"],
            "entity_range": manifest["entity_range"],
            "records": manifest["records"],
            "rejections": manifest["rejections"],
            "statistics": manifest.get("statistics", {}),
            "shards": len(manifest["shards"]),
        }
    )


def _validate(args: argparse.Namespace) -> None:
    if args.full:
        if not args.source:
            raise ValueError("--source is required with --full")
        source = Path(args.source).expanduser().resolve()
        with Wikidata5MStore(source / "indexed", source_root=source) as store:
            manifest = validate_preprocessed_corpus(args.output, store=store, full=True)
    else:
        manifest = validate_preprocessed_corpus(args.output)
    _print({"status": "ok", "records": manifest["records"], "shards": len(manifest["shards"])})


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = PreprocessConfig()
    parser.add_argument("--max-incoming-degree", type=int, default=defaults.max_incoming_degree)
    parser.add_argument("--max-aliases-per-entity", type=int, default=defaults.max_aliases_per_entity)
    parser.add_argument("--max-auxiliary-targets", type=int, default=defaults.max_auxiliary_targets)
    parser.add_argument("--max-context-blanks", type=int, default=defaults.max_context_blanks)
    parser.add_argument("--shard-entity-count", type=int, default=defaults.shard_entity_count)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wikidata5m-preprocess")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build deterministic gzipped JSONL shards")
    build.add_argument("source", help="Wikidata5M source root containing indexed/")
    build.add_argument("--output")
    build.add_argument("--start-id", "--start-entity-id", dest="start_entity_id", type=int, default=0)
    build.add_argument("--stop-id", "--stop-entity-id", dest="stop_entity_id", type=int)
    build.add_argument("--force", action="store_true")
    _add_config_arguments(build)
    build.set_defaults(handler=_build)

    inspect = subparsers.add_parser("inspect", help="preprocess and render one indexed passage")
    inspect.add_argument("index", help="path to Wikidata5M indexed/")
    inspect.add_argument("--source", help="override source root")
    inspect.add_argument("--qid", required=True)
    _add_config_arguments(inspect)
    inspect.set_defaults(handler=_inspect)

    stats = subparsers.add_parser("stats", help="print corpus-manifest statistics")
    stats.add_argument("output")
    stats.set_defaults(handler=_stats)

    validate = subparsers.add_parser("validate", help="validate a preprocessing corpus")
    validate.add_argument("output")
    validate.add_argument("--source")
    validate.add_argument("--full", action="store_true")
    validate.set_defaults(handler=_validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except (Wikidata5MError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.exit(2, f"error: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
