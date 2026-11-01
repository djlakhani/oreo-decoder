"""Command-line interface for Wikidata5M index operations."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Sequence

import numpy as np

from .builder import build_index
from .errors import Wikidata5MError
from .store import Wikidata5MStore, validate_index
from .types import GraphSplit, QueryDirection


def _json_default(value: object) -> object:
    if isinstance(value, Enum):
        return value.name.lower()
    if isinstance(value, np.integer):
        return int(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _print_json(value: object) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, default=_json_default))


def _build_command(args: argparse.Namespace) -> int:
    output = build_index(args.source, args.output, force=args.force)
    print(output)
    return 0


def _validate_command(args: argparse.Namespace) -> int:
    manifest = validate_index(
        args.index, source_root=args.source, full=args.full
    )
    _print_json(
        {
            "index": str(Path(args.index).expanduser().resolve()),
            "mode": "full" if args.full else "quick",
            "schema_version": manifest["schema_version"],
            "counts": manifest["counts"],
            "status": "ok",
        }
    )
    return 0


def _stats_command(args: argparse.Namespace) -> int:
    with Wikidata5MStore(args.index, source_root=args.source) as store:
        _print_json(store.stats())
    return 0


def _inspect_command(args: argparse.Namespace) -> int:
    with Wikidata5MStore(args.index, source_root=args.source) as store:
        if args.qid is not None:
            output: dict[str, object] = {
                "qid": args.qid,
                "entity_id": store.entity_id(args.qid),
                "aliases": store.entity_aliases(args.qid),
                "passage": store.passage(args.qid),
            }
            if args.edges:
                if not args.split:
                    raise ValueError("inspect --edges requires at least one explicit --split")
                output["edges"] = [
                    asdict(edge)
                    for edge in store.edges(
                        args.qid,
                        direction=args.direction,
                        splits=args.split,
                        relations=args.relation,
                    )
                ]
        else:
            assert args.pid is not None
            output = {
                "pid": args.pid,
                "relation_id": store.relation_id(args.pid),
                "aliases": store.relation_aliases(args.pid),
            }
        _print_json(output)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wikidata5m-index",
        description="Build and query the persistent Wikidata5M hybrid index.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build and publish an index")
    build.add_argument("source", help="Wikidata5M source directory")
    build.add_argument(
        "--output", help="output directory (default: SOURCE/indexed)"
    )
    build.add_argument(
        "--force", action="store_true", help="replace an existing complete index"
    )
    build.set_defaults(handler=_build_command)

    validate = subparsers.add_parser("validate", help="validate an existing index")
    validate.add_argument("index", help="index directory")
    validate.add_argument("--source", help="override the manifest source directory")
    validate.add_argument(
        "--full",
        action="store_true",
        help="hash all files and scan all graph-array values",
    )
    validate.set_defaults(handler=_validate_command)

    stats = subparsers.add_parser("stats", help="print build-time dataset counts")
    stats.add_argument("index", help="index directory")
    stats.add_argument("--source", help="override the manifest source directory")
    stats.set_defaults(handler=_stats_command)

    inspect = subparsers.add_parser("inspect", help="inspect one QID or PID")
    inspect.add_argument("index", help="index directory")
    inspect.add_argument("--source", help="override the manifest source directory")
    identifier = inspect.add_mutually_exclusive_group(required=True)
    identifier.add_argument("--qid")
    identifier.add_argument("--pid")
    inspect.add_argument("--edges", action="store_true", help="include graph edges")
    inspect.add_argument(
        "--direction",
        choices=[direction.value for direction in QueryDirection],
        default=QueryDirection.BOTH.value,
    )
    inspect.add_argument(
        "--split",
        action="append",
        choices=[split.value for split in GraphSplit],
        help="edge split; repeat to select more than one",
    )
    inspect.add_argument(
        "--relation", action="append", help="PID edge filter; repeat as needed"
    )
    inspect.set_defaults(handler=_inspect_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError, Wikidata5MError) as error:
        parser.error(str(error))
    return 2
