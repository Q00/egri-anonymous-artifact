"""Offline ablation matrix runner for deterministic benchmark experiments."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from egri.benchmarks.policies import BenchmarkPolicy
from egri.benchmarks.policies import PolicyResult
from egri.benchmarks.policies import get_baseline_policy
from egri.benchmarks.runner import BenchmarkRunResult
from egri.benchmarks.runner import run_offline_benchmark
from egri.benchmarks.schema import BenchmarkExample


class AblationMatrixError(ValueError):
    """Raised when the offline ablation matrix cannot be produced safely."""


@dataclass(frozen=True, slots=True)
class AblationCondition:
    """One named ablation condition backed by a deterministic offline policy."""

    name: str
    base_policy_name: str
    recursion_enabled: bool
    traceguard_enabled: bool
    memory_enabled: bool
    memory_mode: str = "off"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            msg = "ablation condition name must be a non-empty string"
            raise AblationMatrixError(msg)
        if not isinstance(self.base_policy_name, str) or not self.base_policy_name:
            msg = f"{self.name}: base_policy_name must be a non-empty string"
            raise AblationMatrixError(msg)
        for field_name in (
            "recursion_enabled",
            "traceguard_enabled",
            "memory_enabled",
        ):
            if not isinstance(getattr(self, field_name), bool):
                msg = f"{self.name}: {field_name} must be boolean"
                raise AblationMatrixError(msg)
        if not isinstance(self.memory_mode, str) or not self.memory_mode:
            msg = f"{self.name}: memory_mode must be a non-empty string"
            raise AblationMatrixError(msg)
        if self.memory_enabled and self.memory_mode == "off":
            msg = f"{self.name}: memory-enabled ablations must name a memory_mode"
            raise AblationMatrixError(msg)
        if not self.memory_enabled and self.memory_mode != "off":
            msg = f"{self.name}: memory-disabled ablations must use memory_mode='off'"
            raise AblationMatrixError(msg)
        if self.recursion_enabled and not self.base_policy_name.startswith(
            "recursive-"
        ):
            msg = f"{self.name}: base policy is not recursive"
            raise AblationMatrixError(msg)
        if not self.recursion_enabled and self.base_policy_name.startswith(
            "recursive-"
        ):
            msg = f"{self.name}: base policy is recursive"
            raise AblationMatrixError(msg)

    def to_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable metadata for this ablation condition."""
        return {
            "condition_name": self.name,
            "base_policy_name": self.base_policy_name,
            "recursion_enabled": self.recursion_enabled,
            "traceguard_enabled": self.traceguard_enabled,
            "memory_enabled": self.memory_enabled,
            "memory_mode": self.memory_mode,
            "deterministic": True,
            "live_model_calls": 0,
        }


@dataclass(frozen=True, slots=True)
class AblationMatrixResult:
    """In-memory handles and output paths for one offline ablation matrix run."""

    dataset_path: Path
    output_dir: Path
    example_count: int
    condition_names: tuple[str, ...]
    benchmark_result: BenchmarkRunResult
    summaries: tuple[dict[str, Any], ...]
    output_paths: tuple[Path, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable run summary."""
        return {
            "dataset_path": str(self.dataset_path),
            "output_dir": str(self.output_dir),
            "example_count": self.example_count,
            "condition_names": list(self.condition_names),
            "summary_count": len(self.summaries),
            "output_paths": [str(path) for path in self.output_paths],
        }


@dataclass(frozen=True, slots=True)
class _AblationPolicy(BenchmarkPolicy):
    condition: AblationCondition
    base_policy: BenchmarkPolicy

    @property
    def name(self) -> str:
        return self.condition.name

    def run(self, example: BenchmarkExample) -> PolicyResult:
        result = self.base_policy.run(example)
        metadata = {
            **dict(result.metadata or {}),
            **self.condition.to_metadata(),
            "base_policy_result_name": result.policy_name,
        }
        return PolicyResult(
            policy_name=self.condition.name,
            answer=result.answer,
            cited_chunk_ids=result.cited_chunk_ids,
            metadata=metadata,
        )


def iter_offline_ablation_conditions() -> tuple[AblationCondition, ...]:
    """Return ablation conditions in stable paper-table order.

    These deterministic stand-ins isolate the experiment-table plumbing before
    live adapters are introduced. Memory-on currently marks the offline memory
    stub metadata without changing answers or citations.
    """
    return (
        AblationCondition(
            name="flat-no-gate-no-memory",
            base_policy_name="vanilla-single",
            recursion_enabled=False,
            traceguard_enabled=False,
            memory_enabled=False,
        ),
        AblationCondition(
            name="flat-gate-no-memory",
            base_policy_name="evidence-gated-nonrecursive",
            recursion_enabled=False,
            traceguard_enabled=True,
            memory_enabled=False,
        ),
        AblationCondition(
            name="recursive-no-gate-no-memory",
            base_policy_name="recursive-ungated",
            recursion_enabled=True,
            traceguard_enabled=False,
            memory_enabled=False,
        ),
        AblationCondition(
            name="recursive-gate-no-memory",
            base_policy_name="recursive-traceguard-stub",
            recursion_enabled=True,
            traceguard_enabled=True,
            memory_enabled=False,
        ),
        AblationCondition(
            name="recursive-gate-memory",
            base_policy_name="recursive-traceguard-stub",
            recursion_enabled=True,
            traceguard_enabled=True,
            memory_enabled=True,
            memory_mode="offline-deterministic-stub",
        ),
    )


def run_offline_ablation_matrix(
    dataset_path: str | Path,
    *,
    output_dir: str | Path | None = None,
    conditions: Sequence[AblationCondition] | None = None,
) -> AblationMatrixResult:
    """Run deterministic ablation conditions and persist paper-table artifacts."""
    if output_dir is None:
        msg = "output_dir must be provided for offline ablation matrix runs"
        raise AblationMatrixError(msg)

    selected_conditions = (
        tuple(conditions)
        if conditions is not None
        else iter_offline_ablation_conditions()
    )
    _validate_conditions(selected_conditions)

    resolved_output_dir = Path(output_dir).resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_output_dir = resolved_output_dir / "benchmark_run"
    policies = tuple(
        _policy_for_condition(condition) for condition in selected_conditions
    )
    benchmark_result = run_offline_benchmark(
        dataset_path,
        output_dir=benchmark_output_dir,
        policies=policies,
    )
    summaries = _ablation_summary_rows(selected_conditions, benchmark_result)
    output_paths = _write_ablation_outputs(
        dataset_path=benchmark_result.dataset_path,
        output_dir=resolved_output_dir,
        example_count=benchmark_result.example_count,
        condition_names=tuple(condition.name for condition in selected_conditions),
        summaries=summaries,
    )
    return AblationMatrixResult(
        dataset_path=benchmark_result.dataset_path,
        output_dir=resolved_output_dir,
        example_count=benchmark_result.example_count,
        condition_names=tuple(condition.name for condition in selected_conditions),
        benchmark_result=benchmark_result,
        summaries=summaries,
        output_paths=output_paths,
    )


def _validate_conditions(conditions: tuple[AblationCondition, ...]) -> None:
    if not conditions:
        msg = "offline ablation matrix requires at least one condition"
        raise AblationMatrixError(msg)
    if any(not isinstance(condition, AblationCondition) for condition in conditions):
        msg = "conditions must be AblationCondition objects"
        raise AblationMatrixError(msg)
    names = [condition.name for condition in conditions]
    if len(set(names)) != len(names):
        msg = "offline ablation matrix rejects duplicate condition names"
        raise AblationMatrixError(msg)


def _policy_for_condition(condition: AblationCondition) -> BenchmarkPolicy:
    try:
        base_policy = get_baseline_policy(condition.base_policy_name)
    except KeyError as exc:
        msg = f"{condition.name}: unknown base policy {condition.base_policy_name!r}"
        raise AblationMatrixError(msg) from exc
    return _AblationPolicy(condition=condition, base_policy=base_policy)


def _ablation_summary_rows(
    conditions: tuple[AblationCondition, ...],
    benchmark_result: BenchmarkRunResult,
) -> tuple[dict[str, Any], ...]:
    summaries_by_policy = {
        summary.policy_name: summary for summary in benchmark_result.summaries
    }
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        summary = summaries_by_policy[condition.name]
        rows.append({**condition.to_metadata(), **summary.to_dict()})
    return tuple(rows)


def _write_ablation_outputs(
    *,
    dataset_path: Path,
    output_dir: Path,
    example_count: int,
    condition_names: tuple[str, ...],
    summaries: tuple[dict[str, Any], ...],
) -> tuple[Path, ...]:
    summary_json_path = output_dir / "ablation_summary.json"
    summary_csv_path = output_dir / "ablation_summary.csv"
    metadata_path = output_dir / "ablation_metadata.json"

    summary_json_path.write_text(
        json.dumps(list(summaries), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_ablation_csv(summary_csv_path, summaries)
    metadata_path.write_text(
        json.dumps(
            {
                "dataset_path": str(dataset_path),
                "example_count": example_count,
                "condition_names": list(condition_names),
                "live_model_calls": 0,
                "deterministic": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return (summary_json_path, summary_csv_path, metadata_path)


def _write_ablation_csv(path: Path, summaries: tuple[dict[str, Any], ...]) -> None:
    fieldnames = (
        "condition_name",
        "base_policy_name",
        "policy_name",
        "recursion_enabled",
        "traceguard_enabled",
        "memory_enabled",
        "memory_mode",
        "deterministic",
        "live_model_calls",
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
        for row in summaries:
            writer.writerow(row)
