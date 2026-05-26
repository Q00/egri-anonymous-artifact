from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from egri.benchmarks import AblationCondition
from egri.benchmarks import AblationMatrixError
from egri.benchmarks import AblationMatrixResult
from egri.benchmarks import iter_offline_ablation_conditions
from egri.benchmarks import run_offline_ablation_matrix


_SAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "data/benchmark_samples/qasper_mini.jsonl"
)


def test_offline_ablation_conditions_have_stable_paper_table_order() -> None:
    conditions = iter_offline_ablation_conditions()

    assert [condition.name for condition in conditions] == [
        "flat-no-gate-no-memory",
        "flat-gate-no-memory",
        "recursive-no-gate-no-memory",
        "recursive-gate-no-memory",
        "recursive-gate-memory",
    ]
    assert [condition.base_policy_name for condition in conditions] == [
        "vanilla-single",
        "evidence-gated-nonrecursive",
        "recursive-ungated",
        "recursive-traceguard-stub",
        "recursive-traceguard-stub",
    ]
    assert [condition.recursion_enabled for condition in conditions] == [
        False,
        False,
        True,
        True,
        True,
    ]
    assert [condition.traceguard_enabled for condition in conditions] == [
        False,
        True,
        False,
        True,
        True,
    ]
    assert [condition.memory_enabled for condition in conditions] == [
        False,
        False,
        False,
        False,
        True,
    ]


def test_offline_ablation_matrix_runs_fixture_and_writes_json_csv_tables(
    tmp_path,
) -> None:
    output_dir = tmp_path / "ablation-output"

    result = run_offline_ablation_matrix(_SAMPLE_PATH, output_dir=output_dir)

    assert isinstance(result, AblationMatrixResult)
    assert result.dataset_path == _SAMPLE_PATH.resolve()
    assert result.output_dir == output_dir.resolve()
    assert result.example_count == 3
    assert result.condition_names == tuple(
        condition.name for condition in iter_offline_ablation_conditions()
    )
    assert len(result.benchmark_result.policy_results) == 15
    assert len(result.summaries) == 5
    assert [summary["condition_name"] for summary in result.summaries] == list(
        result.condition_names
    )
    assert {path.name for path in result.output_paths} == {
        "ablation_summary.json",
        "ablation_summary.csv",
        "ablation_metadata.json",
    }

    json_rows = json.loads((output_dir / "ablation_summary.json").read_text())
    with (output_dir / "ablation_summary.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    metadata = json.loads((output_dir / "ablation_metadata.json").read_text())

    assert json_rows == list(result.summaries)
    assert [row["condition_name"] for row in csv_rows] == list(result.condition_names)

    per_example_rows = [
        json.loads(line)
        for line in (output_dir / "benchmark_run/per_example_results.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(per_example_rows) == 15
    assert {row["metadata"]["condition_name"] for row in per_example_rows} == set(
        result.condition_names
    )
    assert all(row["metadata"]["deterministic"] is True for row in per_example_rows)
    assert all(row["metadata"]["live_model_calls"] == 0 for row in per_example_rows)

    assert metadata == {
        "dataset_path": str(_SAMPLE_PATH.resolve()),
        "example_count": 3,
        "condition_names": list(result.condition_names),
        "live_model_calls": 0,
        "deterministic": True,
    }


def test_offline_ablation_summary_propagates_metrics_and_metadata(tmp_path) -> None:
    result = run_offline_ablation_matrix(_SAMPLE_PATH, output_dir=tmp_path / "ablation")

    rows_by_name = {row["condition_name"]: row for row in result.summaries}
    flat = rows_by_name["flat-no-gate-no-memory"]
    gated = rows_by_name["recursive-gate-no-memory"]
    memory = rows_by_name["recursive-gate-memory"]

    assert flat["recursion_enabled"] is False
    assert flat["traceguard_enabled"] is False
    assert flat["memory_enabled"] is False
    assert flat["deterministic"] is True
    assert flat["live_model_calls"] == 0
    assert flat["mean_unsupported_citation_rate"] > 0.0

    assert gated["recursion_enabled"] is True
    assert gated["traceguard_enabled"] is True
    assert gated["memory_enabled"] is False
    assert gated["mean_answer_exact_match"] == 1.0
    assert gated["mean_citation_f1"] == 1.0
    assert gated["mean_unsupported_citation_rate"] == 0.0

    assert memory["memory_enabled"] is True
    assert memory["memory_mode"] == "offline-deterministic-stub"
    assert memory["base_policy_name"] == "recursive-traceguard-stub"


def test_ablation_matrix_rejects_missing_output_dir() -> None:
    with pytest.raises(AblationMatrixError, match="output_dir"):
        run_offline_ablation_matrix(_SAMPLE_PATH)


def test_ablation_matrix_rejects_duplicate_condition_names(tmp_path) -> None:
    condition = iter_offline_ablation_conditions()[0]

    with pytest.raises(AblationMatrixError, match="duplicate"):
        run_offline_ablation_matrix(
            _SAMPLE_PATH,
            output_dir=tmp_path / "duplicate",
            conditions=[condition, condition],
        )


def test_ablation_condition_rejects_bool_metadata_mismatch() -> None:
    with pytest.raises(AblationMatrixError, match="base policy"):
        AblationCondition(
            name="invalid-recursive-stand-in",
            base_policy_name="vanilla-single",
            recursion_enabled=True,
            traceguard_enabled=False,
            memory_enabled=False,
        )
