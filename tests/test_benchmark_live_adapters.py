from __future__ import annotations

import json
from dataclasses import dataclass
from math import inf
from math import nan
from pathlib import Path

import pytest

from egri.benchmarks import BenchmarkExample
from egri.benchmarks import BenchmarkRunnerError
from egri.benchmarks import LiveBenchmarkRunnerError
from egri.benchmarks import LiveModelResponse
from egri.benchmarks import LiveRecursiveTraceGuardPolicy
from egri.benchmarks import ModelClient
from egri.benchmarks import PolicyResult
from egri.benchmarks import run_live_benchmark
from egri.benchmarks import run_offline_benchmark


_SAMPLE_PATH = (
    Path(__file__).resolve().parents[1] / "data/benchmark_samples/qasper_mini.jsonl"
)


@dataclass
class FakeModelClient(ModelClient):
    responses: tuple[LiveModelResponse, ...]
    calls: int = 0

    def complete(self, prompt: str, *, example: BenchmarkExample) -> LiveModelResponse:
        assert "TraceGuard" in prompt
        assert example.example_id in prompt
        response = self.responses[self.calls]
        self.calls += 1
        return response


def _fake_response(example_index: int = 0) -> LiveModelResponse:
    responses = (
        LiveModelResponse(
            answer="TraceGuard enforces that parent synthesis cites fresh child evidence handles before state mutation.",
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
            answer="It treats memory as an operational prior, not as factual evidence for parent claims.",
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
            answer="TraceGuard acceptance does not prove semantic truth; it only verifies provenance admissibility over evidence handles.",
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
    return responses[example_index]


def test_live_model_response_validates_usage_and_evidence_metadata() -> None:
    response = _fake_response()

    assert response.total_tokens == 30
    assert response.to_metadata() == {
        "prompt_tokens": 17,
        "completion_tokens": 13,
        "total_tokens": 30,
        "latency_ms": 42.5,
        "cost_usd": 0.0012,
        "traceguard_rejected": False,
        "traceguard_repair_count": 0,
        "evidence_handles": ["child:qasper-mini-001:abstract"],
    }


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"prompt_tokens": -1}, "prompt_tokens"),
        ({"completion_tokens": True}, "completion_tokens"),
        ({"latency_ms": -0.1}, "latency_ms"),
        ({"latency_ms": nan}, "latency_ms"),
        ({"latency_ms": inf}, "latency_ms"),
        ({"cost_usd": -0.1}, "cost_usd"),
        ({"cost_usd": nan}, "cost_usd"),
        ({"cost_usd": inf}, "cost_usd"),
        ({"traceguard_repair_count": -1}, "traceguard_repair_count"),
        ({"evidence_handles": ()}, "evidence_handles"),
        (
            {"evidence_handles": ("memory:qasper-mini-001:abstract",)},
            "evidence_handles",
        ),
        (
            {"evidence_handles": ("child:chunk-1", "child:chunk-1")},
            "evidence_handles",
        ),
    ],
)
def test_live_model_response_rejects_invalid_usage_metadata(kwargs, match) -> None:
    values = {
        "answer": "answer",
        "cited_chunk_ids": ("chunk-1",),
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "latency_ms": 1.0,
        "cost_usd": 0.0,
        "traceguard_rejected": False,
        "traceguard_repair_count": 0,
        "evidence_handles": ("child:chunk-1",),
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=match):
        LiveModelResponse(**values)


def test_live_policy_metadata_marks_live_nondeterministic_traceguard_output() -> None:
    from egri.benchmarks import load_jsonl_examples

    example = load_jsonl_examples(_SAMPLE_PATH)[0]
    client = FakeModelClient(responses=(_fake_response(),))
    policy = LiveRecursiveTraceGuardPolicy(
        model_client=client, name="live-recursive-traceguard"
    )

    result = policy.run(example)

    assert client.calls == 1
    assert result.policy_name == "live-recursive-traceguard"
    assert result.cited_chunk_ids == ("qasper-mini-001:abstract",)
    assert result.metadata["deterministic"] is False
    assert result.metadata["live_model_calls"] == 1
    assert result.metadata["recursive"] is True
    assert result.metadata["traceguard"] == "live-provenance-gate"
    assert result.metadata["prompt_tokens"] == 17
    assert result.metadata["completion_tokens"] == 13
    assert result.metadata["total_tokens"] == 30
    assert result.metadata["latency_ms"] == 42.5
    assert result.metadata["cost_usd"] == 0.0012
    assert result.metadata["traceguard_rejected"] is False
    assert result.metadata["traceguard_repair_count"] == 0
    assert result.metadata["evidence_handles"] == ["child:qasper-mini-001:abstract"]


def test_offline_runner_rejects_live_policy_outputs(tmp_path) -> None:
    client = FakeModelClient(responses=(_fake_response(),))
    policy = LiveRecursiveTraceGuardPolicy(
        model_client=client, name="live-recursive-traceguard"
    )

    with pytest.raises(BenchmarkRunnerError, match="live_model_calls"):
        run_offline_benchmark(
            _SAMPLE_PATH,
            output_dir=tmp_path / "offline-rejects-live",
            policies=[policy],
        )


def test_live_runner_accepts_fake_live_policy_and_writes_usage_metadata(
    tmp_path,
) -> None:
    client = FakeModelClient(
        responses=(_fake_response(0), _fake_response(1), _fake_response(2))
    )
    policy = LiveRecursiveTraceGuardPolicy(
        model_client=client, name="live-recursive-traceguard"
    )

    result = run_live_benchmark(
        _SAMPLE_PATH,
        output_dir=tmp_path / "live-output",
        policies=[policy],
    )

    assert client.calls == 3
    assert result.policy_names == ("live-recursive-traceguard",)
    assert len(result.policy_results) == 3
    assert len(result.evaluations) == 3
    assert result.summaries[0].policy_name == "live-recursive-traceguard"
    assert {path.name for path in result.output_paths} == {
        "per_example_results.jsonl",
        "per_example_evaluations.jsonl",
        "per_policy_summary.json",
        "per_policy_summary.csv",
        "run_metadata.json",
    }

    metadata = json.loads((result.output_dir / "run_metadata.json").read_text())
    rows = [
        json.loads(line)
        for line in (result.output_dir / "per_example_results.jsonl")
        .read_text()
        .splitlines()
    ]

    assert metadata["run_mode"] == "live"
    assert metadata["live_model_calls"] == 3
    assert metadata["deterministic"] is False
    assert metadata["total_tokens"] == 95
    assert metadata["total_cost_usd"] == pytest.approx(0.0041)
    assert rows[2]["metadata"]["traceguard_rejected"] is True
    assert rows[2]["metadata"]["traceguard_repair_count"] == 2


def test_live_runner_rejects_deterministic_offline_policy_outputs(tmp_path) -> None:
    from egri.benchmarks import iter_baseline_policies

    with pytest.raises(LiveBenchmarkRunnerError, match="live_model_calls"):
        run_live_benchmark(
            _SAMPLE_PATH,
            output_dir=tmp_path / "live-rejects-offline",
            policies=[iter_baseline_policies()[0]],
        )


def test_live_runner_rejects_inconsistent_token_totals_from_custom_policy(
    tmp_path,
) -> None:
    class BadTokenPolicy:
        name = "bad-token-policy"

        def run(self, example: BenchmarkExample):
            return PolicyResult(
                policy_name=self.name,
                answer=example.gold_answers[0],
                cited_chunk_ids=example.gold_evidence_chunk_ids,
                metadata={
                    "deterministic": False,
                    "live_model_calls": 1,
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 99,
                    "latency_ms": 1.0,
                    "cost_usd": 0.0,
                    "traceguard_rejected": False,
                    "traceguard_repair_count": 0,
                    "evidence_handles": [f"child:{example.gold_evidence_chunk_ids[0]}"],
                },
            )

    with pytest.raises(LiveBenchmarkRunnerError, match="total_tokens"):
        run_live_benchmark(
            _SAMPLE_PATH,
            output_dir=tmp_path / "bad-tokens",
            policies=[BadTokenPolicy()],
        )


def test_live_runner_rejects_evidence_handles_unrelated_to_citations(tmp_path) -> None:
    class BadEvidencePolicy:
        name = "bad-evidence-policy"

        def run(self, example: BenchmarkExample):
            return PolicyResult(
                policy_name=self.name,
                answer=example.gold_answers[0],
                cited_chunk_ids=example.gold_evidence_chunk_ids,
                metadata={
                    "deterministic": False,
                    "live_model_calls": 1,
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5,
                    "latency_ms": 1.0,
                    "cost_usd": 0.0,
                    "traceguard_rejected": False,
                    "traceguard_repair_count": 0,
                    "evidence_handles": ["child:unrelated-chunk"],
                },
            )

    with pytest.raises(LiveBenchmarkRunnerError, match="evidence_handles"):
        run_live_benchmark(
            _SAMPLE_PATH,
            output_dir=tmp_path / "bad-evidence",
            policies=[BadEvidencePolicy()],
        )


def test_live_runner_rejects_duplicate_evidence_handles_from_custom_policy(
    tmp_path,
) -> None:
    class DuplicateEvidencePolicy:
        name = "duplicate-evidence-policy"

        def run(self, example: BenchmarkExample):
            chunk_id = example.gold_evidence_chunk_ids[0]
            return PolicyResult(
                policy_name=self.name,
                answer=example.gold_answers[0],
                cited_chunk_ids=(chunk_id,),
                metadata={
                    "deterministic": False,
                    "live_model_calls": 1,
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5,
                    "latency_ms": 1.0,
                    "cost_usd": 0.0,
                    "traceguard_rejected": False,
                    "traceguard_repair_count": 0,
                    "evidence_handles": [f"child:{chunk_id}", f"child:{chunk_id}"],
                },
            )

    with pytest.raises(LiveBenchmarkRunnerError, match="evidence_handles"):
        run_live_benchmark(
            _SAMPLE_PATH,
            output_dir=tmp_path / "duplicate-evidence",
            policies=[DuplicateEvidencePolicy()],
        )


def test_live_runner_rejects_nonfinite_latency_or_cost_from_custom_policy(
    tmp_path,
) -> None:
    class NonfiniteUsagePolicy:
        name = "nonfinite-usage-policy"

        def run(self, example: BenchmarkExample):
            chunk_id = example.gold_evidence_chunk_ids[0]
            return PolicyResult(
                policy_name=self.name,
                answer=example.gold_answers[0],
                cited_chunk_ids=(chunk_id,),
                metadata={
                    "deterministic": False,
                    "live_model_calls": 1,
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5,
                    "latency_ms": inf,
                    "cost_usd": nan,
                    "traceguard_rejected": False,
                    "traceguard_repair_count": 0,
                    "evidence_handles": [f"child:{chunk_id}"],
                },
            )

    with pytest.raises(LiveBenchmarkRunnerError, match="latency_ms"):
        run_live_benchmark(
            _SAMPLE_PATH,
            output_dir=tmp_path / "nonfinite-usage",
            policies=[NonfiniteUsagePolicy()],
        )
