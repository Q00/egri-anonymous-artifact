"""Offline benchmark runner for deterministic EGRI experiments."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from egri.benchmarks.evaluator import BenchmarkEvaluation
from egri.benchmarks.evaluator import PolicyAggregateSummary
from egri.benchmarks.evaluator import aggregate_evaluations
from egri.benchmarks.evaluator import evaluate_policy_result
from egri.benchmarks.jsonl_loader import load_jsonl_examples
from egri.benchmarks.policies import BenchmarkPolicy
from egri.benchmarks.policies import PolicyResult
from egri.benchmarks.policies import iter_baseline_policies


class BenchmarkRunnerError(ValueError):
    """Raised when the offline benchmark runner cannot complete safely."""


@dataclass(frozen=True, slots=True)
class BenchmarkRunResult:
    """In-memory handles and output paths for one offline benchmark run."""

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


def run_offline_benchmark(
    dataset_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    policies: Sequence[BenchmarkPolicy] | None = None,
) -> BenchmarkRunResult:
    """Run deterministic policies over a JSONL benchmark file and persist outputs."""
    if output_dir is None:
        msg = "output_dir must be provided for offline benchmark runs"
        raise BenchmarkRunnerError(msg)

    selected_policies = (
        tuple(policies) if policies is not None else iter_baseline_policies()
    )
    if not selected_policies:
        msg = "offline benchmark runner requires at least one policy"
        raise BenchmarkRunnerError(msg)
    for policy in selected_policies:
        if not isinstance(policy.name, str) or not policy.name:
            msg = "all benchmark policies must expose a non-empty name"
            raise BenchmarkRunnerError(msg)

    examples = load_jsonl_examples(dataset_path)
    resolved_dataset_path = Path(dataset_path).resolve()
    resolved_output_dir = Path(output_dir).resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    policy_results: list[PolicyResult] = []
    evaluations: list[BenchmarkEvaluation] = []
    for example in examples:
        for policy in selected_policies:
            policy_result = policy.run(example)
            _validate_offline_policy_result(policy_result)
            policy_results.append(policy_result)
            evaluations.append(evaluate_policy_result(example, policy_result))

    summaries = _aggregate_evaluations_stable(evaluations, selected_policies)
    output_paths = _write_outputs(
        dataset_path=resolved_dataset_path,
        output_dir=resolved_output_dir,
        example_count=len(examples),
        policy_names=tuple(policy.name for policy in selected_policies),
        policy_results=tuple(policy_results),
        evaluations=tuple(evaluations),
        summaries=summaries,
        examples_by_index=tuple(example.example_id for example in examples),
    )

    return BenchmarkRunResult(
        dataset_path=resolved_dataset_path,
        output_dir=resolved_output_dir,
        example_count=len(examples),
        policy_names=tuple(policy.name for policy in selected_policies),
        policy_results=tuple(policy_results),
        evaluations=tuple(evaluations),
        summaries=summaries,
        output_paths=output_paths,
    )


def _validate_offline_policy_result(policy_result: PolicyResult) -> None:
    metadata = dict(policy_result.metadata or {})
    if metadata.get("live_model_calls", 0) != 0:
        msg = "offline benchmark runner rejects policy outputs with live_model_calls"
        raise BenchmarkRunnerError(msg)
    if metadata.get("deterministic") is not True:
        msg = "offline benchmark runner requires deterministic policy outputs"
        raise BenchmarkRunnerError(msg)


def _aggregate_evaluations_stable(
    evaluations: list[BenchmarkEvaluation],
    policies: tuple[BenchmarkPolicy, ...],
) -> tuple[PolicyAggregateSummary, ...]:
    summaries_by_name = {
        summary.policy_name: summary for summary in aggregate_evaluations(evaluations)
    }
    return tuple(summaries_by_name[policy.name] for policy in policies)


def _write_outputs(
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
            [summary.to_dict() for summary in summaries], indent=2, sort_keys=True
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
                "live_model_calls": 0,
            },
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
        row = {"example_id": example_id, **row}
        rows.append(row)
    return tuple(rows)


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


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
