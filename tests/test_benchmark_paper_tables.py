from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from egri.benchmarks import BenchmarkExample
from egri.benchmarks import LiveModelResponse
from egri.benchmarks import LiveRecursiveTraceGuardPolicy
from egri.benchmarks import ModelClient
from egri.benchmarks import PaperTableError
from egri.benchmarks import PaperTableResult
from egri.benchmarks import build_paper_experiment_tables
from egri.benchmarks import run_live_benchmark
from egri.benchmarks import run_offline_ablation_matrix
from egri.benchmarks import run_offline_benchmark


_SAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "data/benchmark_samples/qasper_mini.jsonl"
)
_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1]
    / "data/benchmark_suites/real_multidataset_mini.json"
)


@dataclass
class FakeModelClient(ModelClient):
    responses: tuple[LiveModelResponse, ...]
    calls: int = 0

    def complete(self, prompt: str, *, example: BenchmarkExample) -> LiveModelResponse:
        assert example.example_id in prompt
        response = self.responses[self.calls]
        self.calls += 1
        return response


def _fake_responses() -> tuple[LiveModelResponse, ...]:
    return (
        LiveModelResponse(
            answer="TraceGuard enforces fresh child evidence handles before state mutation.",
            cited_chunk_ids=("qasper-mini-001:abstract",),
            prompt_tokens=17,
            completion_tokens=13,
            latency_ms=42.5,
            cost_usd=0.0012,
            traceguard_rejected=False,
            traceguard_repair_count=0,
            evidence_handles=("child:qasper-mini-001:abstract",),
        ),
        LiveModelResponse(
            answer="It treats memory as an operational prior, not factual evidence.",
            cited_chunk_ids=("qasper-mini-002:memory", "qasper-mini-002:evidence"),
            prompt_tokens=19,
            completion_tokens=11,
            latency_ms=51.0,
            cost_usd=0.0015,
            traceguard_rejected=False,
            traceguard_repair_count=1,
            evidence_handles=(
                "child:qasper-mini-002:memory",
                "child:qasper-mini-002:evidence",
            ),
        ),
        LiveModelResponse(
            answer="TraceGuard verifies provenance admissibility, not semantic truth.",
            cited_chunk_ids=("qasper-mini-003:limitation",),
            prompt_tokens=23,
            completion_tokens=12,
            latency_ms=44.0,
            cost_usd=0.0014,
            traceguard_rejected=True,
            traceguard_repair_count=2,
            evidence_handles=("child:qasper-mini-003:limitation",),
        ),
    )


def test_paper_table_generator_combines_offline_live_and_ablation_outputs(
    tmp_path,
) -> None:
    offline = run_offline_benchmark(_SAMPLE_PATH, output_dir=tmp_path / "offline-run")
    live_client = FakeModelClient(responses=_fake_responses())
    live = run_live_benchmark(
        _SAMPLE_PATH,
        output_dir=tmp_path / "live-run",
        policies=[
            LiveRecursiveTraceGuardPolicy(
                model_client=live_client, name="live-recursive-traceguard"
            )
        ],
    )
    ablation = run_offline_ablation_matrix(
        _SAMPLE_PATH, output_dir=tmp_path / "ablation-run"
    )

    result = build_paper_experiment_tables(
        output_dir=tmp_path / "paper-tables",
        offline_run_dirs=[offline.output_dir],
        live_run_dirs=[live.output_dir],
        ablation_run_dirs=[ablation.output_dir],
        suite_manifest_path=_MANIFEST_PATH,
    )

    assert isinstance(result, PaperTableResult)
    assert result.output_dir == (tmp_path / "paper-tables").resolve()
    assert result.row_count == 11
    assert {path.name for path in result.output_paths} == {
        "experiment_summary.csv",
        "experiment_tables.md",
        "experiment_tables.tex",
        "table_metadata.json",
    }

    metadata = json.loads((result.output_dir / "table_metadata.json").read_text())
    assert metadata["suite_name"] == "real-multidataset-mini"
    assert metadata["dataset_counts"] == {
        "2wikimultihopqa": 2,
        "asqa": 2,
        "hotpotqa": 2,
        "qasper": 2,
    }
    assert metadata["run_counts"] == {
        "offline": 1,
        "live": 1,
        "ablation": 1,
    }
    assert metadata["row_count"] == 11
    assert metadata["contains_live_results"] is True
    assert metadata["contains_deterministic_only_results"] is True

    with (result.output_dir / "experiment_summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 11
    assert {row["run_mode"] for row in rows} == {"offline", "live", "ablation"}
    live_row = next(row for row in rows if row["run_mode"] == "live")
    assert live_row["policy_name"] == "live-recursive-traceguard"
    assert live_row["deterministic"] == "False"
    assert live_row["live_model_calls"] == "3"
    assert live_row["total_tokens"] == "95"
    assert float(live_row["total_cost_usd"]) == pytest.approx(0.0041)
    assert float(live_row["traceguard_reject_rate"]) == pytest.approx(1 / 3)
    assert float(live_row["mean_traceguard_repair_count"]) == pytest.approx(1.0)

    ablation_row = next(
        row for row in rows if row["policy_name"] == "recursive-gate-memory"
    )
    assert ablation_row["run_mode"] == "ablation"
    assert ablation_row["memory_enabled"] == "True"
    assert ablation_row["memory_mode"] == "offline-deterministic-stub"
    assert ablation_row["deterministic"] == "True"
    assert ablation_row["live_model_calls"] == "0"

    markdown = (result.output_dir / "experiment_tables.md").read_text()
    assert "## Dataset counts" in markdown
    assert "| qasper | 2 |" in markdown
    assert "## Policy results" in markdown
    assert "live-recursive-traceguard" in markdown
    assert "deterministic/offline fixture rows are not live model results" in markdown

    latex = (result.output_dir / "experiment_tables.tex").read_text()
    assert "\\begin{tabular}" in latex
    assert "live-recursive-traceguard" in latex


def test_paper_table_generator_rejects_missing_inputs(tmp_path) -> None:
    with pytest.raises(PaperTableError, match="at least one"):
        build_paper_experiment_tables(output_dir=tmp_path / "empty")


def test_paper_table_generator_rejects_live_run_without_usage_metadata(
    tmp_path,
) -> None:
    live_dir = tmp_path / "broken-live"
    live_dir.mkdir()
    (live_dir / "per_policy_summary.json").write_text(
        json.dumps(
            [
                {
                    "policy_name": "broken-live",
                    "example_count": 1,
                    "mean_answer_exact_match": 1.0,
                    "mean_answer_contains_match": 1.0,
                    "mean_citation_precision": 1.0,
                    "mean_citation_recall": 1.0,
                    "mean_citation_f1": 1.0,
                    "mean_unsupported_citation_rate": 0.0,
                    "abstention_rate": 0.0,
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (live_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "dataset_path": str(_SAMPLE_PATH),
                "example_count": 1,
                "policy_names": ["broken-live"],
                "run_mode": "live",
                "deterministic": False,
                "live_model_calls": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (live_dir / "per_example_results.jsonl").write_text(
        json.dumps(
            {
                "example_id": "x",
                "policy_name": "broken-live",
                "answer": "x",
                "cited_chunk_ids": ["chunk"],
                "metadata": {"live_model_calls": 1, "deterministic": False},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(PaperTableError, match="total_tokens"):
        build_paper_experiment_tables(
            output_dir=tmp_path / "paper-tables",
            live_run_dirs=[live_dir],
        )


def test_paper_table_generator_rejects_live_summary_without_result_rows(
    tmp_path,
) -> None:
    live_dir = tmp_path / "missing-live-policy"
    live_dir.mkdir()
    (live_dir / "per_policy_summary.json").write_text(
        json.dumps(
            [
                {
                    "policy_name": "missing-policy",
                    "example_count": 1,
                    "mean_answer_exact_match": 1.0,
                    "mean_answer_contains_match": 1.0,
                    "mean_citation_precision": 1.0,
                    "mean_citation_recall": 1.0,
                    "mean_citation_f1": 1.0,
                    "mean_unsupported_citation_rate": 0.0,
                    "abstention_rate": 0.0,
                }
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (live_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "dataset_path": str(_SAMPLE_PATH),
                "example_count": 1,
                "policy_names": ["missing-policy"],
                "run_mode": "live",
                "deterministic": False,
                "live_model_calls": 1,
                "total_tokens": 3,
                "total_cost_usd": 0.1,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (live_dir / "per_example_results.jsonl").write_text(
        json.dumps(
            {
                "example_id": "x",
                "policy_name": "unexpected-policy",
                "answer": "x",
                "cited_chunk_ids": ["chunk"],
                "metadata": {
                    "live_model_calls": 1,
                    "deterministic": False,
                    "total_tokens": 3,
                    "cost_usd": 0.1,
                    "traceguard_rejected": False,
                    "traceguard_repair_count": 0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(PaperTableError, match="live summary policies"):
        build_paper_experiment_tables(
            output_dir=tmp_path / "paper-tables",
            live_run_dirs=[live_dir],
        )


def test_paper_table_generator_rejects_duplicate_live_policy_names(tmp_path) -> None:
    client = FakeModelClient(responses=_fake_responses())
    live = run_live_benchmark(
        _SAMPLE_PATH,
        output_dir=tmp_path / "live-run",
        policies=[
            LiveRecursiveTraceGuardPolicy(
                model_client=client, name="live-recursive-traceguard"
            )
        ],
    )
    summary_path = live.output_dir / "per_policy_summary.json"
    rows = json.loads(summary_path.read_text())
    rows.append(dict(rows[0]))
    summary_path.write_text(json.dumps(rows) + "\n", encoding="utf-8")

    with pytest.raises(PaperTableError, match="duplicate live summary policy"):
        build_paper_experiment_tables(
            output_dir=tmp_path / "paper-tables",
            live_run_dirs=[live.output_dir],
        )


def test_paper_table_generator_rejects_live_aggregate_metadata_mismatch(
    tmp_path,
) -> None:
    client = FakeModelClient(responses=_fake_responses())
    live = run_live_benchmark(
        _SAMPLE_PATH,
        output_dir=tmp_path / "live-run",
        policies=[
            LiveRecursiveTraceGuardPolicy(
                model_client=client, name="live-recursive-traceguard"
            )
        ],
    )
    metadata_path = live.output_dir / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["total_tokens"] = 999
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")

    with pytest.raises(PaperTableError, match="metadata total_tokens"):
        build_paper_experiment_tables(
            output_dir=tmp_path / "paper-tables",
            live_run_dirs=[live.output_dir],
        )


def test_paper_table_generator_rejects_ablation_summary_with_live_calls(
    tmp_path,
) -> None:
    ablation = run_offline_ablation_matrix(
        _SAMPLE_PATH, output_dir=tmp_path / "ablation-run"
    )
    summary_path = ablation.output_dir / "ablation_summary.json"
    rows = json.loads(summary_path.read_text())
    rows[0]["live_model_calls"] = 1
    summary_path.write_text(json.dumps(rows) + "\n", encoding="utf-8")

    with pytest.raises(PaperTableError, match="ablation summary"):
        build_paper_experiment_tables(
            output_dir=tmp_path / "paper-tables",
            ablation_run_dirs=[ablation.output_dir],
        )
