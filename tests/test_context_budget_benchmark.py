from __future__ import annotations

import csv
import json

import pytest

from egri.benchmarks import ContextBudgetError
from egri.benchmarks import ContextBudgetResult
from egri.benchmarks import iter_context_budget_conditions
from egri.benchmarks import run_context_budget_benchmark


def test_context_budget_conditions_define_paper_order() -> None:
    conditions = iter_context_budget_conditions()

    assert [condition.name for condition in conditions] == [
        "single-shot-context-window",
        "flat-topk-retrieval",
        "recursive-evidence-cache",
        "recursive-traceguard-dfs",
    ]
    assert [condition.recursive for condition in conditions] == [False, False, True, True]
    assert [condition.traceguard_enabled for condition in conditions] == [False, False, False, True]
    assert [condition.visible_leaf_facts for condition in conditions] == [3, 5, 8, 8]


def test_context_budget_benchmark_writes_json_csv_metadata(tmp_path) -> None:
    output_dir = tmp_path / "context-budget"

    result = run_context_budget_benchmark(output_dir=output_dir)

    assert isinstance(result, ContextBudgetResult)
    assert result.output_dir == output_dir.resolve()
    assert result.fixture_count == 10
    assert result.condition_names == tuple(
        condition.name for condition in iter_context_budget_conditions()
    )
    assert {path.name for path in result.output_paths} == {
        "context_budget_rows.json",
        "context_budget_summary.csv",
        "context_budget_metadata.json",
    }

    rows = json.loads((output_dir / "context_budget_rows.json").read_text())
    with (output_dir / "context_budget_summary.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    metadata = json.loads((output_dir / "context_budget_metadata.json").read_text())

    assert len(rows) == 40
    assert [row["condition_name"] for row in csv_rows] == list(result.condition_names)
    assert metadata == {
        "benchmark_name": "context-budget-pressure",
        "fixture_count": 10,
        "condition_names": list(result.condition_names),
        "deterministic": True,
        "live_model_calls": 0,
        "network_calls": 0,
        "claim": (
            "Recursive evidence caching preserves inspectable supported claims under a "
            "fixed context budget; TraceGuard prevents pressure-induced unsupported commits."
        ),
    }


def test_context_budget_summary_shows_recursion_under_pressure(tmp_path) -> None:
    result = run_context_budget_benchmark(output_dir=tmp_path / "context-budget")
    rows_by_name = {row["condition_name"]: row for row in result.summary_rows}

    single = rows_by_name["single-shot-context-window"]
    topk = rows_by_name["flat-topk-retrieval"]
    cache = rows_by_name["recursive-evidence-cache"]
    guarded = rows_by_name["recursive-traceguard-dfs"]

    assert single["mean_supported_claim_recall"] == 0.375
    assert topk["mean_supported_claim_recall"] == 0.625
    assert cache["mean_supported_claim_recall"] == 1.0
    assert guarded["mean_supported_claim_recall"] == 1.0

    assert single["unsupported_commit_rate"] == 1.0
    assert topk["unsupported_commit_rate"] == 1.0
    assert cache["unsupported_commit_rate"] == 1.0
    assert guarded["unsupported_commit_rate"] == 0.0
    assert guarded["unsupported_rejection_rate"] == 1.0
    assert guarded["mean_evidence_bandwidth_multiplier"] > topk[
        "mean_evidence_bandwidth_multiplier"
    ]


def test_context_budget_benchmark_rejects_missing_output_dir() -> None:
    with pytest.raises(ContextBudgetError, match="output_dir"):
        run_context_budget_benchmark()
