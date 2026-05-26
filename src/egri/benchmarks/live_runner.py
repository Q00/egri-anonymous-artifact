"""Live-capable benchmark runner for injected model-client policies."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Any

from egri.benchmarks.evaluator import BenchmarkEvaluation
from egri.benchmarks.evaluator import PolicyAggregateSummary
from egri.benchmarks.evaluator import aggregate_evaluations
from egri.benchmarks.evaluator import evaluate_policy_result
from egri.benchmarks.jsonl_loader import load_jsonl_examples
from egri.benchmarks.policies import BenchmarkPolicy
from egri.benchmarks.policies import PolicyResult


class LiveBenchmarkRunnerError(ValueError):
    """Raised when a live-capable benchmark run violates its contract."""


@dataclass(frozen=True, slots=True)
class LiveBenchmarkRunResult:
    """In-memory handles and output paths for one live-capable benchmark run."""

    dataset_path: Path
    output_dir: Path
    example_count: int
    policy_names: tuple[str, ...]
    policy_results: tuple[PolicyResult, ...]
    evaluations: tuple[BenchmarkEvaluation, ...]
    summaries: tuple[PolicyAggregateSummary, ...]
    output_paths: tuple[Path, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable run summary."""
        return {
            "dataset_path": str(self.dataset_path),
            "output_dir": str(self.output_dir),
            "example_count": self.example_count,
            "policy_names": list(self.policy_names),
            "policy_result_count": len(self.policy_results),
            "evaluation_count": len(self.evaluations),
            "summary_count": len(self.summaries),
            "output_paths": [str(path) for path in self.output_paths],
        }


def run_live_benchmark(
    dataset_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    policies: Sequence[BenchmarkPolicy] | None = None,
) -> LiveBenchmarkRunResult:
    """Run live-capable benchmark policies and persist evaluator outputs."""
    if output_dir is None:
        msg = "output_dir must be provided for live benchmark runs"
        raise LiveBenchmarkRunnerError(msg)
    selected_policies = tuple(policies or ())
    if not selected_policies:
        msg = "live benchmark runner requires at least one policy"
        raise LiveBenchmarkRunnerError(msg)
    for policy in selected_policies:
        if not isinstance(policy.name, str) or not policy.name:
            msg = "all benchmark policies must expose a non-empty name"
            raise LiveBenchmarkRunnerError(msg)

    examples = load_jsonl_examples(dataset_path)
    resolved_dataset_path = Path(dataset_path).resolve()
    resolved_output_dir = Path(output_dir).resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    policy_results: list[PolicyResult] = []
    evaluations: list[BenchmarkEvaluation] = []
    for example in examples:
        for policy in selected_policies:
            policy_result = policy.run(example)
            _validate_live_policy_result(policy_result)
            policy_results.append(policy_result)
            evaluations.append(evaluate_policy_result(example, policy_result))

    summaries = _aggregate_evaluations_stable(evaluations, selected_policies)
    output_paths = _write_live_outputs(
        dataset_path=resolved_dataset_path,
        output_dir=resolved_output_dir,
        example_count=len(examples),
        policy_names=tuple(policy.name for policy in selected_policies),
        policy_results=tuple(policy_results),
        evaluations=tuple(evaluations),
        summaries=summaries,
        examples_by_index=tuple(example.example_id for example in examples),
    )
    return LiveBenchmarkRunResult(
        dataset_path=resolved_dataset_path,
        output_dir=resolved_output_dir,
        example_count=len(examples),
        policy_names=tuple(policy.name for policy in selected_policies),
        policy_results=tuple(policy_results),
        evaluations=tuple(evaluations),
        summaries=summaries,
        output_paths=output_paths,
    )


def _validate_live_policy_result(policy_result: PolicyResult) -> None:
    metadata = dict(policy_result.metadata or {})
    live_model_calls = metadata.get("live_model_calls")
    if (
        isinstance(live_model_calls, bool)
        or not isinstance(live_model_calls, int)
        or live_model_calls <= 0
    ):
        msg = "live benchmark runner requires live_model_calls > 0"
        raise LiveBenchmarkRunnerError(msg)
    if metadata.get("deterministic") is not False:
        msg = "live benchmark runner requires deterministic=False policy outputs"
        raise LiveBenchmarkRunnerError(msg)
    prompt_tokens = metadata.get("prompt_tokens")
    completion_tokens = metadata.get("completion_tokens")
    total_tokens = metadata.get("total_tokens")
    for field_name, value in (
        ("prompt_tokens", prompt_tokens),
        ("completion_tokens", completion_tokens),
        ("total_tokens", total_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            msg = f"live policy metadata requires non-negative {field_name}"
            raise LiveBenchmarkRunnerError(msg)
    if total_tokens != prompt_tokens + completion_tokens:
        msg = "live policy metadata total_tokens must equal prompt_tokens + completion_tokens"
        raise LiveBenchmarkRunnerError(msg)
    for field_name in ("latency_ms", "cost_usd"):
        value = metadata.get(field_name)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not isfinite(float(value))
            or float(value) < 0.0
        ):
            msg = f"live policy metadata requires non-negative {field_name}"
            raise LiveBenchmarkRunnerError(msg)
    if not isinstance(metadata.get("traceguard_rejected"), bool):
        msg = "live policy metadata requires traceguard_rejected boolean"
        raise LiveBenchmarkRunnerError(msg)
    repair_count = metadata.get("traceguard_repair_count")
    if (
        isinstance(repair_count, bool)
        or not isinstance(repair_count, int)
        or repair_count < 0
    ):
        msg = "live policy metadata requires non-negative traceguard_repair_count"
        raise LiveBenchmarkRunnerError(msg)
    evidence_handles = metadata.get("evidence_handles")
    if not isinstance(evidence_handles, list) or not evidence_handles:
        msg = "live policy metadata requires evidence_handles list"
        raise LiveBenchmarkRunnerError(msg)
    if any(
        not isinstance(handle, str) or not handle.startswith("child:")
        for handle in evidence_handles
    ):
        msg = "live policy metadata evidence_handles must be child: handles"
        raise LiveBenchmarkRunnerError(msg)
    if len(set(evidence_handles)) != len(evidence_handles):
        msg = "live policy metadata evidence_handles must not contain duplicate values"
        raise LiveBenchmarkRunnerError(msg)
    cited_handle_targets = {
        f"child:{chunk_id}" for chunk_id in policy_result.cited_chunk_ids
    }
    handle_targets = set(evidence_handles)
    if cited_handle_targets != handle_targets:
        msg = "live policy metadata evidence_handles must match cited chunks"
        raise LiveBenchmarkRunnerError(msg)


def _aggregate_evaluations_stable(
    evaluations: list[BenchmarkEvaluation],
    policies: tuple[BenchmarkPolicy, ...],
) -> tuple[PolicyAggregateSummary, ...]:
    summaries_by_name = {
        summary.policy_name: summary for summary in aggregate_evaluations(evaluations)
    }
    return tuple(summaries_by_name[policy.name] for policy in policies)


def _write_live_outputs(
    *,
    dataset_path: Path,
    output_dir: Path,
    example_count: int,
    policy_names: tuple[str, ...],
    policy_results: tuple[PolicyResult, ...],
    evaluations: tuple[BenchmarkEvaluation, ...],
    summaries: tuple[PolicyAggregateSummary, ...],
    examples_by_index: tuple[str, ...],
) -> tuple[Path, ...]:
    per_example_results_path = output_dir / "per_example_results.jsonl"
    per_example_evaluations_path = output_dir / "per_example_evaluations.jsonl"
    per_policy_summary_json_path = output_dir / "per_policy_summary.json"
    per_policy_summary_csv_path = output_dir / "per_policy_summary.csv"
    metadata_path = output_dir / "run_metadata.json"

    _write_jsonl(
        per_example_results_path,
        _policy_result_rows(policy_results, examples_by_index, policy_names),
    )
    _write_jsonl(
        per_example_evaluations_path,
        (evaluation.to_dict() for evaluation in evaluations),
    )
    per_policy_summary_json_path.write_text(
        json.dumps(
            [summary.to_dict() for summary in summaries],
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_summary_csv(per_policy_summary_csv_path, summaries)
    metadata_path.write_text(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "example_count": example_count,
                "policy_names": list(policy_names),
                "run_mode": "live",
                "deterministic": False,
                "live_model_calls": _sum_int_metadata(
                    policy_results, "live_model_calls"
                ),
                "total_tokens": _sum_int_metadata(policy_results, "total_tokens"),
                "total_cost_usd": _sum_float_metadata(policy_results, "cost_usd"),
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return (
        per_example_results_path,
        per_example_evaluations_path,
        per_policy_summary_json_path,
        per_policy_summary_csv_path,
        metadata_path,
    )


def _policy_result_rows(
    policy_results: tuple[PolicyResult, ...],
    example_ids: tuple[str, ...],
    policy_names: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for index, policy_result in enumerate(policy_results):
        example_id = example_ids[index // len(policy_names)]
        row = policy_result.to_dict()
        rows.append({"example_id": example_id, **row})
    return tuple(rows)


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")


def _write_summary_csv(
    path: Path, summaries: tuple[PolicyAggregateSummary, ...]
) -> None:
    fieldnames = (
        "policy_name",
        "example_count",
        "mean_answer_exact_match",
        "mean_answer_contains_match",
        "mean_citation_precision",
        "mean_citation_recall",
        "mean_citation_f1",
        "mean_unsupported_citation_rate",
        "abstention_rate",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary.to_dict())


def _sum_int_metadata(policy_results: tuple[PolicyResult, ...], field_name: str) -> int:
    return sum(
        int(dict(result.metadata or {})[field_name]) for result in policy_results
    )


def _sum_float_metadata(
    policy_results: tuple[PolicyResult, ...], field_name: str
) -> float:
    return sum(
        float(dict(result.metadata or {})[field_name]) for result in policy_results
    )
