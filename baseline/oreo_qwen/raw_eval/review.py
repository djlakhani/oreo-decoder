"""Human-readable HTML review for completed raw-Qwen evaluation runs."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import file_sha256, load_manifest
from .inference import iter_prediction_records
from .schema import EvalArtifactError
from .scoring import rescore_prediction_record

_TASK_START = "TASK\n\nPassage:\n"
_TASK_END = "\n\nRequired blank identifiers:"


def _artifact_directory(root: Path, kind: str) -> Path:
    for name in (f"eval-{kind}", f"eval_{kind}"):
        candidate = root / name
        if candidate.is_dir():
            return candidate
    raise EvalArtifactError(
        f"evaluation root has neither eval-{kind}/ nor eval_{kind}/: {root}"
    )


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _rate(value: object) -> str:
    if value is None:
        return "—"
    return f"{float(value):.2%}"


def _extract_passage(prompt_text: str) -> str:
    start = prompt_text.rfind(_TASK_START)
    if start < 0:
        raise EvalArtifactError("prediction prompt has no recognizable TASK passage")
    start += len(_TASK_START)
    stop = prompt_text.find(_TASK_END, start)
    if stop < 0:
        raise EvalArtifactError("prediction prompt has no blank-identifier delimiter")
    return prompt_text[start:stop]


def _find_eval_root(explicit: str | Path | None) -> Path:
    if explicit is not None:
        root = Path(explicit).expanduser().resolve()
        if not root.is_dir():
            raise EvalArtifactError(f"evaluation root does not exist: {root}")
        return root
    current = Path.cwd().resolve()
    for base in (current, *current.parents):
        candidate = base / "aws-eval"
        if (candidate / "eval-runs").is_dir() or (candidate / "eval_runs").is_dir():
            return candidate
    raise EvalArtifactError(
        "could not find an aws-eval directory; pass --eval-root /path/to/aws-eval"
    )


def resolve_metrics_path(
    source: str | Path,
    *,
    eval_root: str | Path | None = None,
) -> tuple[Path, Path | None]:
    """Resolve a metrics file, run directory, or run folder name."""

    direct = Path(source).expanduser()
    if direct.exists():
        resolved = direct.resolve()
        metrics = resolved / "rescored_metrics.json" if resolved.is_dir() else resolved
        return metrics, None
    root = _find_eval_root(eval_root)
    run = _artifact_directory(root, "runs") / str(source)
    if not run.is_dir():
        raise EvalArtifactError(f"evaluation run folder does not exist: {run}")
    return run / "rescored_metrics.json", root


def _validate_input_pair(metrics_path: Path, input_dir: Path) -> None:
    if not input_dir.is_dir():
        raise EvalArtifactError(f"evaluation input folder does not exist: {input_dir}")
    run_manifest = load_manifest(metrics_path.parent)
    expected = run_manifest.get("configuration", {}).get("input_manifest_sha256")
    if not isinstance(expected, str):
        raise EvalArtifactError("run manifest has no input-manifest checksum")
    input_manifest = input_dir / "manifest.json"
    if not input_manifest.is_file():
        raise EvalArtifactError(f"evaluation input manifest is missing: {input_manifest}")
    if file_sha256(input_manifest) != expected:
        raise EvalArtifactError(
            "the selected eval-input folder does not match the selected eval-run"
        )


def _combined_kg_metrics(by_kind: Mapping[str, Any]) -> dict[str, Any]:
    groups = [
        by_kind[kind]
        for kind in ("primary_kg", "auxiliary_kg")
        if kind in by_kind
    ]
    targets = sum(int(group.get("targets", 0)) for group in groups)
    alias_targets = sum(int(group.get("alias_targets", 0)) for group in groups)
    exact = sum(int(group.get("exact", 0)) for group in groups)
    normalized = sum(int(group.get("normalized", 0)) for group in groups)
    alias_aware = sum(int(group.get("alias_aware", 0)) for group in groups)
    alias_substring_aware = sum(
        int(group.get("alias_substring_aware", 0)) for group in groups
    )
    return {
        "targets": targets,
        "exact": exact,
        "exact_rate": None if targets == 0 else exact / targets,
        "normalized": normalized,
        "normalized_rate": None if targets == 0 else normalized / targets,
        "alias_targets": alias_targets,
        "alias_aware": alias_aware,
        "alias_aware_rate": (
            None if alias_targets == 0 else alias_aware / alias_targets
        ),
        "alias_substring_aware": alias_substring_aware,
        "alias_substring_aware_rate": (
            None if alias_targets == 0 else alias_substring_aware / alias_targets
        ),
    }


def _summary_table(metrics: Mapping[str, Any]) -> str:
    by_kind = dict(metrics.get("targets_by_kind", {}))
    ordered: list[tuple[str, Mapping[str, Any]]] = [
        ("All KG blanks", _combined_kg_metrics(by_kind))
    ]
    labels = {
        "primary_kg": "Primary KG",
        "auxiliary_kg": "Auxiliary KG",
        "context_only": "Context-only",
    }
    ordered.extend(
        (labels.get(kind, kind), by_kind[kind])
        for kind in ("primary_kg", "auxiliary_kg", "context_only")
        if kind in by_kind
    )
    rows = []
    for label, group in ordered:
        alias = _rate(group.get("alias_aware_rate"))
        substring = _rate(group.get("alias_substring_aware_rate"))
        rows.append(
            "<tr>"
            f"<th>{_escape(label)}</th>"
            f"<td>{int(group.get('targets', 0)):,}</td>"
            f"<td>{_rate(group.get('exact_rate'))}</td>"
            f"<td>{_rate(group.get('normalized_rate'))}</td>"
            f"<td>{alias}</td>"
            f"<td>{substring}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Blank group</th><th>Count</th>"
        "<th>Exact</th><th>Normalized</th><th>Alias-aware</th>"
        "<th>Alias substring</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _response_summary(metrics: Mapping[str, Any]) -> str:
    response = metrics.get("response", {})
    rows = [
        ("Examples", f"{int(response.get('examples', 0)):,}"),
        ("Generated", f"{int(response.get('generated', 0)):,}"),
        ("Strict format valid", _rate(response.get("strict_format_valid_rate"))),
        ("Recoverable format", _rate(response.get("recovery_format_valid_rate"))),
        ("Complete exact response", _rate(response.get("complete_exact_rate"))),
        (
            "Complete alias-aware response",
            _rate(response.get("complete_alias_aware_rate")),
        ),
        (
            "Complete alias-substring response",
            _rate(response.get("complete_alias_substring_aware_rate")),
        ),
    ]
    return "<dl>" + "".join(
        f"<div><dt>{_escape(label)}</dt><dd>{_escape(value)}</dd></div>"
        for label, value in rows
    ) + "</dl>"


def _answer_list(target: Mapping[str, Any]) -> str:
    exact = str(target["exact_text"])
    aliases = sorted(
        {
            str(alias)
            for alias in target.get("accepted_normalized_aliases", [])
            if str(alias) != str(target.get("normalized_exact_text", ""))
        }
    )
    items = [f"<li><strong>Exact:</strong> {_escape(exact)}</li>"]
    if aliases:
        items.extend(
            f"<li><strong>Alias:</strong> {_escape(alias)}</li>" for alias in aliases
        )
    return "<ul class='answers'>" + "".join(items) + "</ul>"


def _result_badges(target_score: Mapping[str, Any]) -> str:
    values = [
        ("exact", target_score.get("exact")),
        ("normalized", target_score.get("normalized")),
    ]
    if target_score.get("alias_aware") is not None:
        values.append(("alias", target_score.get("alias_aware")))
        values.append(
            ("alias substring", target_score.get("alias_substring_aware"))
        )
    return " ".join(
        f"<span class='badge {'pass' if passed else 'fail'}'>"
        f"{_escape(label)}: {'✓' if passed else '✗'}</span>"
        for label, passed in values
    )


def _record_card(record: Mapping[str, Any], ordinal: int) -> str:
    status = str(record.get("status", "unknown"))
    passage = _extract_passage(str(record.get("prompt_text", "")))
    targets = list(record.get("targets", []))
    score = rescore_prediction_record(record)
    if score is None:
        strict_valid = False
        recovery_valid = False
        target_scores: list[Mapping[str, Any]] = [{} for _ in targets]
    else:
        strict_valid = bool(score["strict_parse"]["valid"])
        recovery_valid = bool(score["recovery_parse"]["valid"])
        target_scores = list(
            score["strict_targets"] if strict_valid else score["recovery_targets"]
        )
    rows = []
    for target, target_score in zip(targets, target_scores, strict=True):
        prediction = target_score.get("prediction")
        rows.append(
            "<tr>"
            f"<td><code>{_escape(target['blank_token'])}</code></td>"
            f"<td>{_escape(target['kind'])}</td>"
            f"<td>{_escape(target.get('gold_qid') or '—')}</td>"
            f"<td>{_answer_list(target)}</td>"
            f"<td>{_escape(prediction if prediction is not None else '—')}</td>"
            f"<td>{_result_badges(target_score)}</td>"
            "</tr>"
        )
    generated = record.get("generated_text")
    response = "—" if generated is None else str(generated)
    return f"""
    <section class="example">
      <h2>#{ordinal} · {_escape(record.get('passage_qid', 'unknown QID'))}</h2>
      <p class="meta">
        <strong>Example:</strong> {_escape(record.get('example_id', '—'))} ·
        <strong>Status:</strong> {_escape(status)} ·
        <strong>Strict format:</strong> {'valid' if strict_valid else 'invalid'} ·
        <strong>Recovery parse:</strong> {'valid' if recovery_valid else 'invalid'}
      </p>
      <h3>Passage with blanks</h3>
      <pre class="passage">{_escape(passage)}</pre>
      <h3>Model response</h3>
      <pre>{_escape(response)}</pre>
      <h3>Blank-by-blank evaluation</h3>
      <table class="targets">
        <thead><tr><th>Blank</th><th>Kind</th><th>Gold QID</th>
        <th>Accepted answers</th><th>Model answer</th><th>Result</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </section>
    """


def generate_review(
    rescored_metrics_path: str | Path,
    output_path: str | Path | None = None,
    *,
    limit: int | None = None,
) -> Path:
    """Generate an HTML report using a metrics file and its sibling run artifacts."""

    metrics_path = Path(rescored_metrics_path).expanduser().resolve()
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvalArtifactError(f"cannot read rescored metrics {metrics_path}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("metrics"), dict):
        raise EvalArtifactError("expected a rescored_metrics.json object with a metrics field")
    run = metrics_path.parent
    manifest = load_manifest(run)
    if manifest.get("complete") is not True:
        raise EvalArtifactError("review requires a complete evaluation run")
    expected_manifest_hash = payload.get("run_manifest_sha256")
    if expected_manifest_hash != file_sha256(run / "manifest.json"):
        raise EvalArtifactError("rescored metrics do not match the sibling run manifest")

    records: list[dict[str, Any]] = []
    for entry in manifest.get("shards", []):
        path = run / str(entry["path"])
        if not path.is_file() or path.stat().st_size != entry.get("size"):
            raise EvalArtifactError(f"prediction shard is missing or truncated: {path}")
        if file_sha256(path) != entry.get("sha256"):
            raise EvalArtifactError(f"prediction shard checksum failed: {path}")
        for record in iter_prediction_records(path):
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
        if limit is not None and len(records) >= limit:
            break
    if not records:
        raise EvalArtifactError("run contains no prediction records to review")

    metrics = payload["metrics"]
    cards = "".join(
        _record_card(record, index)
        for index, record in enumerate(records, start=1)
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Raw Qwen evaluation review</title>
<style>
:root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }}
body {{ max-width: 1500px; margin: 0 auto; padding: 2rem; line-height: 1.45; }}
h1, h2, h3 {{ line-height: 1.2; }}
.summary, .example {{ border: 1px solid #8886; border-radius: 12px; padding: 1.25rem; margin: 1.5rem 0; }}
.meta {{ color: #777; }}
table {{ width: 100%; border-collapse: collapse; }}
th, td {{ border: 1px solid #8886; padding: .55rem; text-align: left; vertical-align: top; }}
thead th {{ background: #8882; }}
pre {{ white-space: pre-wrap; overflow-wrap: anywhere; padding: 1rem; border-radius: 8px; background: #8882; }}
.passage {{ font-family: ui-serif, Georgia, serif; font-size: 1rem; }}
dl {{ display: flex; flex-wrap: wrap; gap: .75rem 2rem; }}
dl div {{ min-width: 12rem; }}
dt {{ font-weight: 650; }} dd {{ margin: .2rem 0 0; }}
.answers {{ margin: 0; padding-left: 1.25rem; }}
.badge {{ display: inline-block; padding: .15rem .4rem; margin: .1rem; border-radius: 5px; white-space: nowrap; }}
.pass {{ background: #16803c33; color: #16803c; }}
.fail {{ background: #c5282833; color: #c52828; }}
code {{ white-space: nowrap; }}
</style>
</head>
<body>
<h1>Raw Qwen reconstruction evaluation</h1>
<p>Authoritative metrics from <code>{_escape(metrics_path.name)}</code>. Showing {_escape(len(records))} prediction record(s). Alias-substring accuracy is a lenient diagnostic: a non-empty normalized prediction passes when it occurs inside an accepted alias.</p>
<section class="summary">
<h2>Accuracy by blank type</h2>
{_summary_table(metrics)}
<h2>Response-level results</h2>
{_response_summary(metrics)}
</section>
{cards}
</body>
</html>
"""
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else run / "review.html"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")
    return destination


def cli_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a readable HTML report for a completed raw-Qwen run."
    )
    parser.add_argument(
        "source",
        help="rescored metrics path, run directory, or folder name under eval-runs",
    )
    parser.add_argument(
        "--eval-root",
        help="directory containing eval-runs/ and eval-inputs/ (auto-detects aws-eval)",
    )
    parser.add_argument(
        "--input-name",
        help="folder name under eval-inputs; verifies that it matches the run",
    )
    parser.add_argument("--output", help="HTML output path (default: RUN/review.html)")
    parser.add_argument("--limit", type=int, help="show only the first N passages")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    try:
        metrics_path, discovered_root = resolve_metrics_path(
            args.source, eval_root=args.eval_root
        )
        if args.input_name is not None:
            root = discovered_root or _find_eval_root(args.eval_root)
            inputs = _artifact_directory(root, "inputs")
            _validate_input_pair(metrics_path, inputs / args.input_name)
        path = generate_review(metrics_path, args.output, limit=args.limit)
    except (EvalArtifactError, OSError, KeyError, TypeError, ValueError) as error:
        parser.exit(2, f"error: {error}\n")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_main())
