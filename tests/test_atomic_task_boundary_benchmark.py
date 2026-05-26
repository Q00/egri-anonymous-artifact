from __future__ import annotations

import csv
import json

import pytest

from egri.benchmarks import AtomicTaskBoundaryError
from egri.benchmarks import AtomicTaskBoundaryResult
from egri.benchmarks import iter_atomic_task_boundary_conditions
from egri.benchmarks import run_atomic_task_boundary_benchmark


def test_atomic_task_boundary_conditions_define_stable_paper_order() -> None:
    conditions = iter_atomic_task_boundary_conditions()

    assert [condition.name for condition in conditions] == [
        "broad-unchecked-parent",
        "one-level-atomic-traceguard",
        "dfs-refinement-traceguard",
        "dfs-traceguard-structured-claim-guard",
    ]
    assert [condition.traversal for condition in conditions] == [
        "single-shot",
        "one-level",
        "dfs",
        "dfs",
    ]
    assert [condition.requires_atomic_leaf for condition in conditions] == [
        False,
        True,
        True,
        True,
    ]
    assert [condition.traceguard_enabled for condition in conditions] == [
        False,
        True,
        True,
        True,
    ]
    assert [condition.structured_claim_guard_enabled for condition in conditions] == [
        False,
        False,
        False,
        True,
    ]


def test_atomic_task_boundary_benchmark_writes_json_csv_metadata(tmp_path) -> None:
    output_dir = tmp_path / "atomic-boundary"

    result = run_atomic_task_boundary_benchmark(output_dir=output_dir)

    assert isinstance(result, AtomicTaskBoundaryResult)
    assert result.output_dir == output_dir.resolve()
    assert result.fixture_count == 12
    assert result.condition_names == tuple(
        condition.name for condition in iter_atomic_task_boundary_conditions()
    )
    assert {path.name for path in result.output_paths} == {
        "atomic_task_boundary_rows.json",
        "atomic_task_boundary_summary.csv",
        "atomic_task_boundary_metadata.json",
    }

    rows = json.loads((output_dir / "atomic_task_boundary_rows.json").read_text())
    with (output_dir / "atomic_task_boundary_summary.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    metadata = json.loads((output_dir / "atomic_task_boundary_metadata.json").read_text())

    assert len(rows) == 48
    assert [row["condition_name"] for row in csv_rows] == list(result.condition_names)
    assert metadata == {
        "benchmark_name": "atomic-task-boundary",
        "fixture_count": 12,
        "condition_names": list(result.condition_names),
        "deterministic": True,
        "live_model_calls": 0,
        "network_calls": 0,
        "claim": (
            "Child model calls are not assumed deterministic; recursion refines "
            "tasks until leaf outputs have deterministic acceptance predicates."
        ),
    }


def test_atomic_task_boundary_summary_shows_atomic_leaf_checkability(tmp_path) -> None:
    result = run_atomic_task_boundary_benchmark(output_dir=tmp_path / "atomic-boundary")
    rows_by_name = {row["condition_name"]: row for row in result.summary_rows}

    broad = rows_by_name["broad-unchecked-parent"]
    atomic = rows_by_name["one-level-atomic-traceguard"]
    dfs = rows_by_name["dfs-refinement-traceguard"]
    structured = rows_by_name["dfs-traceguard-structured-claim-guard"]

    assert broad["leaf_atomic_rate"] == 0.0
    assert broad["deterministic_acceptance_predicate_rate"] == 0.0
    assert broad["unsupported_commit_rate"] == 1.0
    assert broad["semantic_miss_rejection_rate"] == 0.0

    assert atomic["leaf_atomic_rate"] == 1.0
    assert atomic["deterministic_acceptance_predicate_rate"] == 1.0
    assert atomic["unsupported_commit_rate"] == 0.0
    assert atomic["semantic_miss_commit_rate"] == 1.0
    assert atomic["mean_max_depth"] == 1.0

    assert dfs["leaf_atomic_rate"] == 1.0
    assert dfs["deterministic_acceptance_predicate_rate"] == 1.0
    assert dfs["unsupported_commit_rate"] == 0.0
    assert dfs["semantic_miss_commit_rate"] == 1.0
    assert dfs["mean_max_depth"] == 2.0
    assert dfs["mean_leaf_count"] > atomic["mean_leaf_count"]

    assert structured["leaf_atomic_rate"] == 1.0
    assert structured["unsupported_commit_rate"] == 0.0
    assert structured["semantic_miss_attempt_rate"] == 1.0
    assert structured["semantic_miss_rejection_rate"] == 1.0
    assert structured["semantic_miss_commit_rate"] == 0.0
    assert structured["mean_leaf_count"] == dfs["mean_leaf_count"]


def test_atomic_task_boundary_structured_guard_uses_typed_claim_terms(tmp_path) -> None:
    result = run_atomic_task_boundary_benchmark(output_dir=tmp_path / "atomic-boundary")
    semantic_rows = [
        row
        for row in result.rows
        if row["condition_name"] == "dfs-traceguard-structured-claim-guard"
    ]

    assert semantic_rows
    assert all(row["structured_claim_guard_enabled"] for row in semantic_rows)
    assert all(row["semantic_miss_attempted"] for row in semantic_rows)
    assert all(row["traceguard_accepted_semantic_miss"] for row in semantic_rows)
    assert all(row["semantic_miss_rejected_before_commit"] for row in semantic_rows)
    assert all(not row["semantic_miss_committed"] for row in semantic_rows)
    assert all(row["structured_missing_terms"] for row in semantic_rows)
    assert semantic_rows[0]["semantic_miss_claim_terms"] == {
        "behavior": "admin_delete_denied",
        "result": "passed",
    }
    assert "behavior" in semantic_rows[0]["structured_missing_terms"]


def test_atomic_task_boundary_benchmark_rejects_missing_output_dir() -> None:
    with pytest.raises(AtomicTaskBoundaryError, match="output_dir"):
        run_atomic_task_boundary_benchmark()
