from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from egri.benchmarks import BenchmarkExample
from egri.benchmarks import BenchmarkPolicy
from egri.benchmarks import BenchmarkRunResult
from egri.benchmarks import BenchmarkRunnerError
from egri.benchmarks import PolicyResult
from egri.benchmarks import iter_baseline_policies
from egri.benchmarks import run_offline_benchmark


_SAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "data/benchmark_samples/qasper_mini.jsonl"
)


def test_offline_runner_executes_all_baselines_on_committed_fixture(tmp_path) -> None:
    output_dir = tmp_path / "benchmark-output"

    result = run_offline_benchmark(_SAMPLE_PATH, output_dir=output_dir)

    expected_policy_names = tuple(policy.name for policy in iter_baseline_policies())
    assert isinstance(result, BenchmarkRunResult)
    assert result.dataset_path == _SAMPLE_PATH.resolve()
    assert result.output_dir == output_dir.resolve()
    assert result.example_count == 3
    assert result.policy_names == expected_policy_names
    assert len(result.policy_results) == 15
    assert len(result.evaluations) == 15
    assert [summary.policy_name for summary in result.summaries] == list(
        expected_policy_names
    )
    assert {summary.example_count for summary in result.summaries} == {3}
    assert all(
        policy_result.metadata["live_model_calls"] == 0
        for policy_result in result.policy_results
    )


def test_offline_runner_writes_jsonl_json_and_csv_outputs(tmp_path) -> None:
    output_dir = tmp_path / "benchmark-output"

    result = run_offline_benchmark(_SAMPLE_PATH, output_dir=output_dir)

    expected_files = {
        "per_example_results.jsonl",
        "per_example_evaluations.jsonl",
        "per_policy_summary.json",
        "per_policy_summary.csv",
        "run_metadata.json",
    }
    assert {path.name for path in result.output_paths} == expected_files
    assert expected_files <= {path.name for path in output_dir.iterdir()}

    policy_result_rows = [
        json.loads(line)
        for line in (output_dir / "per_example_results.jsonl").read_text().splitlines()
    ]
    evaluation_rows = [
        json.loads(line)
        for line in (output_dir / "per_example_evaluations.jsonl")
        .read_text()
        .splitlines()
    ]
    summary_rows = json.loads((output_dir / "per_policy_summary.json").read_text())
    metadata = json.loads((output_dir / "run_metadata.json").read_text())

    assert len(policy_result_rows) == 15
    assert len(evaluation_rows) == 15
    assert len(summary_rows) == 5
    assert metadata == {
        "dataset_path": str(_SAMPLE_PATH.resolve()),
        "example_count": 3,
        "policy_names": [policy.name for policy in iter_baseline_policies()],
        "live_model_calls": 0,
    }
    first_result = policy_result_rows[0]
    assert set(first_result) == {
        "example_id",
        "policy_name",
        "answer",
        "cited_chunk_ids",
        "metadata",
    }
    assert first_result["example_id"] == "qasper-mini-001"
    assert first_result["policy_name"] == "vanilla-single"
    assert first_result["metadata"]["deterministic"] is True

    with (output_dir / "per_policy_summary.csv").open(newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert [row["policy_name"] for row in csv_rows] == [
        policy.name for policy in iter_baseline_policies()
    ]
    assert all(row["example_count"] == "3" for row in csv_rows)


def test_offline_runner_stable_policy_order_for_custom_subset(tmp_path) -> None:
    policies = (
        iter_baseline_policies()[3],
        iter_baseline_policies()[0],
    )

    result = run_offline_benchmark(
        _SAMPLE_PATH,
        output_dir=tmp_path / "custom-output",
        policies=policies,
    )

    assert result.policy_names == ("recursive-ungated", "vanilla-single")
    assert [summary.policy_name for summary in result.summaries] == [
        "recursive-ungated",
        "vanilla-single",
    ]
    assert [evaluation.policy_name for evaluation in result.evaluations[:2]] == [
        "recursive-ungated",
        "vanilla-single",
    ]


def test_offline_runner_rejects_empty_policy_list(tmp_path) -> None:
    with pytest.raises(BenchmarkRunnerError, match="at least one policy"):
        run_offline_benchmark(_SAMPLE_PATH, output_dir=tmp_path / "empty", policies=[])


def test_offline_runner_rejects_policy_outputs_with_live_model_calls(tmp_path) -> None:
    class LivePolicy(BenchmarkPolicy):
        name = "live-policy"

        def run(self, example: BenchmarkExample) -> PolicyResult:
            return PolicyResult(
                policy_name=self.name,
                answer=example.gold_answers[0],
                cited_chunk_ids=example.gold_evidence_chunk_ids,
                metadata={"deterministic": False, "live_model_calls": 1},
            )

    with pytest.raises(BenchmarkRunnerError, match="live_model_calls"):
        run_offline_benchmark(
            _SAMPLE_PATH,
            output_dir=tmp_path / "live-output",
            policies=[LivePolicy()],
        )


def test_offline_runner_requires_explicit_output_dir() -> None:
    with pytest.raises(BenchmarkRunnerError, match="output_dir"):
        run_offline_benchmark(_SAMPLE_PATH)
