"""Paper-facing table generation for EGRI benchmark artifacts."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from egri.benchmarks.dataset_catalog import BenchmarkSuiteManifest
from egri.benchmarks.dataset_catalog import load_benchmark_suite_manifest


class PaperTableError(ValueError):
    """Raised when paper-facing benchmark tables cannot be generated safely."""


@dataclass(frozen=True, slots=True)
class PaperTableResult:
    """Output handles for generated paper-facing experiment tables."""

    output_dir: Path
    row_count: int
    output_paths: tuple[Path, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable table-generation summary."""
        return {
            "output_dir": str(self.output_dir),
            "row_count": self.row_count,
            "output_paths": [str(path) for path in self.output_paths],
        }


_SUMMARY_METRIC_FIELDS = (
    "mean_answer_exact_match",
    "mean_answer_contains_match",
    "mean_citation_precision",
    "mean_citation_recall",
    "mean_citation_f1",
    "mean_unsupported_citation_rate",
    "abstention_rate",
)

_TABLE_FIELDNAMES = (
    "run_mode",
    "artifact_dir",
    "dataset_path",
    "example_count",
    "policy_name",
    "condition_name",
    "recursion_enabled",
    "traceguard_enabled",
    "memory_enabled",
    "memory_mode",
    "deterministic",
    "live_model_calls",
    "total_tokens",
    "total_cost_usd",
    "traceguard_reject_rate",
    "mean_traceguard_repair_count",
    *_SUMMARY_METRIC_FIELDS,
)


def build_paper_experiment_tables(
    *,
    output_dir: str | Path,
    offline_run_dirs: Sequence[str | Path] | None = None,
    live_run_dirs: Sequence[str | Path] | None = None,
    ablation_run_dirs: Sequence[str | Path] | None = None,
    suite_manifest_path: str | Path | None = None,
) -> PaperTableResult:
    """Build Markdown, CSV, and LaTeX-ready tables from benchmark artifacts.

    The generated rows intentionally preserve run-mode flags so deterministic
    offline fixtures cannot be misread as live model results in the paper.
    """
    offline_dirs = _resolve_dirs(offline_run_dirs or ())
    live_dirs = _resolve_dirs(live_run_dirs or ())
    ablation_dirs = _resolve_dirs(ablation_run_dirs or ())
    if not offline_dirs and not live_dirs and not ablation_dirs:
        msg = "at least one benchmark artifact directory must be provided"
        raise PaperTableError(msg)

    manifest = (
        load_benchmark_suite_manifest(suite_manifest_path)
        if suite_manifest_path is not None
        else None
    )
    rows: list[dict[str, Any]] = []
    for run_dir in offline_dirs:
        rows.extend(_rows_from_standard_run(run_dir, expected_mode="offline"))
    for run_dir in live_dirs:
        rows.extend(_rows_from_standard_run(run_dir, expected_mode="live"))
    for run_dir in ablation_dirs:
        rows.extend(_rows_from_ablation_run(run_dir))

    resolved_output_dir = Path(output_dir).resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = resolved_output_dir / "experiment_summary.csv"
    markdown_path = resolved_output_dir / "experiment_tables.md"
    latex_path = resolved_output_dir / "experiment_tables.tex"
    metadata_path = resolved_output_dir / "table_metadata.json"

    _write_csv(csv_path, rows)
    markdown_path.write_text(_render_markdown(rows, manifest), encoding="utf-8")
    latex_path.write_text(_render_latex(rows), encoding="utf-8")
    metadata_path.write_text(
        json.dumps(
            _metadata(rows, manifest, offline_dirs, live_dirs, ablation_dirs),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return PaperTableResult(
        output_dir=resolved_output_dir,
        row_count=len(rows),
        output_paths=(csv_path, markdown_path, latex_path, metadata_path),
    )


def _resolve_dirs(paths: Sequence[str | Path]) -> tuple[Path, ...]:
    resolved = tuple(Path(path).resolve() for path in paths)
    for path in resolved:
        if not path.is_dir():
            msg = f"benchmark artifact directory does not exist: {path}"
            raise PaperTableError(msg)
    return resolved


def _rows_from_standard_run(
    run_dir: Path, *, expected_mode: str
) -> tuple[dict[str, Any], ...]:
    metadata = _read_json_object(run_dir / "run_metadata.json")
    summaries = _read_json_list(run_dir / "per_policy_summary.json")
    if expected_mode == "offline":
        _validate_offline_metadata(metadata, run_dir)
    elif expected_mode == "live":
        _validate_live_metadata(metadata, run_dir)
    else:
        msg = f"unsupported standard run mode: {expected_mode}"
        raise PaperTableError(msg)

    usage_by_policy = (
        _usage_by_policy(run_dir / "per_example_results.jsonl")
        if expected_mode == "live"
        else {}
    )
    if expected_mode == "live":
        _validate_live_usage_consistency(metadata, summaries, usage_by_policy, run_dir)
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        policy_name = _required_str(summary, "policy_name", run_dir)
        usage = usage_by_policy.get(policy_name, {})
        rows.append(
            _normalize_row(
                {
                    "run_mode": expected_mode,
                    "artifact_dir": str(run_dir),
                    "dataset_path": metadata.get("dataset_path", ""),
                    "example_count": summary.get(
                        "example_count", metadata.get("example_count", "")
                    ),
                    "policy_name": policy_name,
                    "condition_name": "",
                    "recursion_enabled": "",
                    "traceguard_enabled": "",
                    "memory_enabled": "",
                    "memory_mode": "",
                    "deterministic": expected_mode == "offline",
                    "live_model_calls": usage.get(
                        "live_model_calls", metadata.get("live_model_calls", 0)
                    ),
                    "total_tokens": usage.get(
                        "total_tokens", metadata.get("total_tokens", "")
                    ),
                    "total_cost_usd": usage.get(
                        "total_cost_usd", metadata.get("total_cost_usd", "")
                    ),
                    "traceguard_reject_rate": usage.get("traceguard_reject_rate", ""),
                    "mean_traceguard_repair_count": usage.get(
                        "mean_traceguard_repair_count", ""
                    ),
                    **_summary_metrics(summary, run_dir),
                }
            )
        )
    return tuple(rows)


def _rows_from_ablation_run(run_dir: Path) -> tuple[dict[str, Any], ...]:
    metadata = _read_json_object(run_dir / "ablation_metadata.json")
    summaries = _read_json_list(run_dir / "ablation_summary.json")
    if metadata.get("deterministic") is not True:
        msg = f"ablation run must be deterministic: {run_dir}"
        raise PaperTableError(msg)
    if metadata.get("live_model_calls") != 0:
        msg = f"ablation run must not report live_model_calls: {run_dir}"
        raise PaperTableError(msg)
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        condition_name = _required_str(summary, "condition_name", run_dir)
        live_model_calls = summary.get("live_model_calls")
        if (
            isinstance(live_model_calls, bool)
            or not isinstance(live_model_calls, int)
            or live_model_calls != 0
        ):
            msg = f"ablation summary must report live_model_calls=0: {run_dir}"
            raise PaperTableError(msg)
        rows.append(
            _normalize_row(
                {
                    "run_mode": "ablation",
                    "artifact_dir": str(run_dir),
                    "dataset_path": metadata.get("dataset_path", ""),
                    "example_count": summary.get(
                        "example_count", metadata.get("example_count", "")
                    ),
                    "policy_name": summary.get("policy_name", condition_name),
                    "condition_name": condition_name,
                    "recursion_enabled": summary.get("recursion_enabled", ""),
                    "traceguard_enabled": summary.get("traceguard_enabled", ""),
                    "memory_enabled": summary.get("memory_enabled", ""),
                    "memory_mode": summary.get("memory_mode", ""),
                    "deterministic": True,
                    "live_model_calls": live_model_calls,
                    "total_tokens": "",
                    "total_cost_usd": "",
                    "traceguard_reject_rate": "",
                    "mean_traceguard_repair_count": "",
                    **_summary_metrics(summary, run_dir),
                }
            )
        )
    return tuple(rows)


def _validate_offline_metadata(metadata: dict[str, Any], run_dir: Path) -> None:
    if metadata.get("live_model_calls") != 0:
        msg = f"offline run must report live_model_calls=0: {run_dir}"
        raise PaperTableError(msg)


def _validate_live_metadata(metadata: dict[str, Any], run_dir: Path) -> None:
    if metadata.get("run_mode") != "live":
        msg = f"live run metadata must include run_mode='live': {run_dir}"
        raise PaperTableError(msg)
    if metadata.get("deterministic") is not False:
        msg = f"live run metadata must include deterministic=false: {run_dir}"
        raise PaperTableError(msg)
    calls = metadata.get("live_model_calls")
    if isinstance(calls, bool) or not isinstance(calls, int) or calls <= 0:
        msg = f"live run metadata requires positive live_model_calls: {run_dir}"
        raise PaperTableError(msg)
    for field in ("total_tokens", "total_cost_usd"):
        _validate_finite_non_negative(metadata.get(field), field, run_dir)


def _validate_live_usage_consistency(
    metadata: dict[str, Any],
    summaries: list[dict[str, Any]],
    usage_by_policy: dict[str, dict[str, Any]],
    run_dir: Path,
) -> None:
    summary_names = [
        _required_str(summary, "policy_name", run_dir) for summary in summaries
    ]
    if len(set(summary_names)) != len(summary_names):
        msg = f"duplicate live summary policy names are not allowed: {run_dir}"
        raise PaperTableError(msg)
    summary_policy_names = set(summary_names)
    if summary_policy_names != set(usage_by_policy):
        msg = (
            "live summary policies must exactly match per-example result policies: "
            f"{run_dir}"
        )
        raise PaperTableError(msg)
    metadata_policy_names = metadata.get("policy_names")
    if (
        not isinstance(metadata_policy_names, list)
        or any(not isinstance(name, str) or not name for name in metadata_policy_names)
        or len(set(metadata_policy_names)) != len(metadata_policy_names)
        or set(metadata_policy_names) != summary_policy_names
    ):
        msg = f"live metadata policy_names must match summary policies: {run_dir}"
        raise PaperTableError(msg)
    for summary in summaries:
        policy_name = _required_str(summary, "policy_name", run_dir)
        example_count = summary.get("example_count")
        if (
            isinstance(example_count, bool)
            or not isinstance(example_count, int)
            or example_count <= 0
        ):
            msg = f"live summary example_count must be a positive integer: {run_dir}"
            raise PaperTableError(msg)
        if usage_by_policy[policy_name]["example_count"] != example_count:
            msg = f"live result row count must match summary example_count: {run_dir}"
            raise PaperTableError(msg)

    live_model_calls = sum(
        int(usage["live_model_calls"]) for usage in usage_by_policy.values()
    )
    total_tokens = sum(int(usage["total_tokens"]) for usage in usage_by_policy.values())
    total_cost_usd = sum(
        float(usage["total_cost_usd"]) for usage in usage_by_policy.values()
    )
    if live_model_calls != metadata["live_model_calls"]:
        msg = f"live result aggregate must match metadata live_model_calls: {run_dir}"
        raise PaperTableError(msg)
    if total_tokens != metadata["total_tokens"]:
        msg = f"live result aggregate must match metadata total_tokens: {run_dir}"
        raise PaperTableError(msg)
    if abs(total_cost_usd - float(metadata["total_cost_usd"])) > 1e-12:
        msg = f"live result aggregate must match metadata total_cost_usd: {run_dir}"
        raise PaperTableError(msg)


def _summary_metrics(summary: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for field in _SUMMARY_METRIC_FIELDS:
        value = summary.get(field)
        _validate_finite_non_negative(value, field, run_dir)
        metrics[field] = value
    return metrics


def _usage_by_policy(results_path: Path) -> dict[str, dict[str, Any]]:
    rows = _read_jsonl(results_path)
    usage: dict[str, dict[str, Any]] = {}
    for row in rows:
        policy_name = _required_str(row, "policy_name", results_path)
        metadata = row.get("metadata")
        if not isinstance(metadata, dict):
            msg = f"live result row metadata must be an object: {results_path}"
            raise PaperTableError(msg)
        for field in ("live_model_calls", "total_tokens"):
            value = metadata.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                msg = f"live result metadata requires non-negative {field}: {results_path}"
                raise PaperTableError(msg)
        for field in ("cost_usd",):
            _validate_finite_non_negative(metadata.get(field), field, results_path)
        if not isinstance(metadata.get("traceguard_rejected"), bool):
            msg = f"live result metadata requires traceguard_rejected: {results_path}"
            raise PaperTableError(msg)
        repair_count = metadata.get("traceguard_repair_count")
        if (
            isinstance(repair_count, bool)
            or not isinstance(repair_count, int)
            or repair_count < 0
        ):
            msg = (
                f"live result metadata requires traceguard_repair_count: {results_path}"
            )
            raise PaperTableError(msg)

        aggregate = usage.setdefault(
            policy_name,
            {
                "example_count": 0,
                "live_model_calls": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "traceguard_reject_count": 0,
                "traceguard_repair_count": 0,
            },
        )
        aggregate["example_count"] += 1
        aggregate["live_model_calls"] += metadata["live_model_calls"]
        aggregate["total_tokens"] += metadata["total_tokens"]
        aggregate["total_cost_usd"] += float(metadata["cost_usd"])
        aggregate["traceguard_reject_count"] += int(metadata["traceguard_rejected"])
        aggregate["traceguard_repair_count"] += metadata["traceguard_repair_count"]

    return {
        policy_name: {
            "example_count": aggregate["example_count"],
            "live_model_calls": aggregate["live_model_calls"],
            "total_tokens": aggregate["total_tokens"],
            "total_cost_usd": aggregate["total_cost_usd"],
            "traceguard_reject_rate": aggregate["traceguard_reject_count"]
            / aggregate["example_count"],
            "mean_traceguard_repair_count": aggregate["traceguard_repair_count"]
            / aggregate["example_count"],
        }
        for policy_name, aggregate in usage.items()
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    data = _read_json(path)
    if not isinstance(data, dict):
        msg = f"expected JSON object: {path}"
        raise PaperTableError(msg)
    return data


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    data = _read_json(path)
    if not isinstance(data, list) or any(not isinstance(row, dict) for row in data):
        msg = f"expected JSON list of objects: {path}"
        raise PaperTableError(msg)
    return data


def _read_json(path: Path) -> Any:
    if not path.is_file():
        msg = f"required benchmark artifact is missing: {path}"
        raise PaperTableError(msg)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"invalid JSON artifact: {path}"
        raise PaperTableError(msg) from exc


def _read_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        msg = f"required benchmark artifact is missing: {path}"
        raise PaperTableError(msg)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            msg = f"invalid JSONL row {line_number} in {path}"
            raise PaperTableError(msg) from exc
        if not isinstance(row, dict):
            msg = f"expected JSON object on row {line_number} in {path}"
            raise PaperTableError(msg)
        rows.append(row)
    if not rows:
        msg = f"JSONL artifact must not be empty: {path}"
        raise PaperTableError(msg)
    return tuple(rows)


def _required_str(row: dict[str, Any], field: str, source: Path) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        msg = f"{source}: required field {field} must be a non-empty string"
        raise PaperTableError(msg)
    return value


def _validate_finite_non_negative(value: Any, field: str, source: Path) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not isfinite(float(value))
        or float(value) < 0.0
    ):
        msg = f"{source}: field {field} must be a finite non-negative number"
        raise PaperTableError(msg)


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {field: row.get(field, "") for field in _TABLE_FIELDNAMES}


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_TABLE_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _metadata(
    rows: Sequence[dict[str, Any]],
    manifest: BenchmarkSuiteManifest | None,
    offline_dirs: Sequence[Path],
    live_dirs: Sequence[Path],
    ablation_dirs: Sequence[Path],
) -> dict[str, Any]:
    return {
        "suite_name": manifest.name if manifest is not None else "",
        "suite_version": manifest.version if manifest is not None else "",
        "dataset_counts": _dataset_counts(manifest),
        "run_counts": {
            "offline": len(offline_dirs),
            "live": len(live_dirs),
            "ablation": len(ablation_dirs),
        },
        "row_count": len(rows),
        "contains_live_results": any(row["run_mode"] == "live" for row in rows),
        "contains_deterministic_only_results": any(
            row["deterministic"] is True for row in rows
        ),
        "warning": "deterministic/offline fixture rows are not live model results",
    }


def _dataset_counts(manifest: BenchmarkSuiteManifest | None) -> dict[str, int]:
    if manifest is None:
        return {}
    counts: dict[str, int] = {}
    for source in manifest.sources:
        counts[source.dataset] = counts.get(source.dataset, 0) + source.example_count
    return dict(sorted(counts.items()))


def _render_markdown(
    rows: Sequence[dict[str, Any]], manifest: BenchmarkSuiteManifest | None
) -> str:
    lines = [
        "# EGRI benchmark tables",
        "",
        "> Warning: deterministic/offline fixture rows are not live model results.",
        "",
        "## Dataset counts",
        "",
        "| dataset | examples |",
        "| --- | ---: |",
    ]
    counts = _dataset_counts(manifest)
    if counts:
        lines.extend(f"| {dataset} | {count} |" for dataset, count in counts.items())
    else:
        lines.append("| unspecified |  |")
    lines.extend(
        [
            "",
            "## Policy results",
            "",
            "| mode | policy/condition | n | answer EM | citation F1 | unsupported cite rate | deterministic | live calls | tokens | cost USD | TraceGuard reject rate | repair count |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        label = row["condition_name"] or row["policy_name"]
        lines.append(
            "| {run_mode} | {label} | {example_count} | {answer_em} | {citation_f1} | {unsupported} | {deterministic} | {calls} | {tokens} | {cost} | {reject} | {repair} |".format(
                run_mode=row["run_mode"],
                label=label,
                example_count=row["example_count"],
                answer_em=_fmt(row["mean_answer_exact_match"]),
                citation_f1=_fmt(row["mean_citation_f1"]),
                unsupported=_fmt(row["mean_unsupported_citation_rate"]),
                deterministic=row["deterministic"],
                calls=row["live_model_calls"],
                tokens=row["total_tokens"],
                cost=_fmt(row["total_cost_usd"]),
                reject=_fmt(row["traceguard_reject_rate"]),
                repair=_fmt(row["mean_traceguard_repair_count"]),
            )
        )
    return "\n".join(lines) + "\n"


def _render_latex(rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "% Auto-generated from benchmark artifacts; deterministic rows are not live model results.",
        "\\begin{tabular}{llrrrr}",
        "Mode & Policy/condition & N & Answer EM & Citation F1 & Live calls \\\\ ",
        "\\hline",
    ]
    for row in rows:
        label = _escape_latex(str(row["condition_name"] or row["policy_name"]))
        lines.append(
            f"{_escape_latex(str(row['run_mode']))} & {label} & {row['example_count']} & {_fmt(row['mean_answer_exact_match'])} & {_fmt(row['mean_citation_f1'])} & {row['live_model_calls']} \\\\"
        )
    lines.append("\\end{tabular}")
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    if value == "" or value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _escape_latex(value: str) -> str:
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in value)
