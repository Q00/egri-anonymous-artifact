from __future__ import annotations

import pytest

from egri.benchmarks import BenchmarkExample
from egri.benchmarks import ContextChunk
from egri.benchmarks import PolicyOutputValidationError
from egri.benchmarks import PolicyResult
from egri.benchmarks import get_baseline_policy
from egri.benchmarks import iter_baseline_policies
from egri.benchmarks import normalize_cited_chunk_ids
from egri.benchmarks.policies import BenchmarkPolicy


def _example() -> BenchmarkExample:
    return BenchmarkExample(
        example_id="toy-001",
        dataset="toy",
        question="What boundary does TraceGuard enforce?",
        context_chunks=(
            ContextChunk(
                chunk_id="toy-001:gold",
                title="Gold",
                text="TraceGuard enforces provenance admissibility using fresh evidence handles.",
            ),
            ContextChunk(
                chunk_id="toy-001:distractor",
                title="Distractor",
                text="Durable memory can guide routing, but it is not factual evidence.",
            ),
        ),
        gold_answers=(
            "TraceGuard enforces provenance admissibility using fresh evidence handles.",
        ),
        gold_evidence_chunk_ids=("toy-001:gold",),
    )


def test_policy_result_requires_normalized_answer_and_citations() -> None:
    result = PolicyResult(
        policy_name="vanilla-single",
        answer="TraceGuard enforces provenance admissibility.",
        cited_chunk_ids=("toy-001:gold",),
        metadata={"mode": "single"},
    )

    assert result.to_dict() == {
        "policy_name": "vanilla-single",
        "answer": "TraceGuard enforces provenance admissibility.",
        "cited_chunk_ids": ["toy-001:gold"],
        "metadata": {"mode": "single"},
    }

    with pytest.raises(PolicyOutputValidationError, match="policy_name"):
        PolicyResult(policy_name="", answer="answer", cited_chunk_ids=("toy-001:gold",))
    with pytest.raises(PolicyOutputValidationError, match="answer"):
        PolicyResult(policy_name="x", answer="", cited_chunk_ids=("toy-001:gold",))
    with pytest.raises(PolicyOutputValidationError, match="tuple"):
        PolicyResult(policy_name="x", answer="answer", cited_chunk_ids=["toy-001:gold"])  # type: ignore[arg-type]
    with pytest.raises(PolicyOutputValidationError, match="non-empty strings"):
        PolicyResult(policy_name="x", answer="answer", cited_chunk_ids=("",))
    with pytest.raises(PolicyOutputValidationError, match="duplicate"):
        PolicyResult(
            policy_name="x",
            answer="answer",
            cited_chunk_ids=("toy-001:gold", "toy-001:gold"),
        )


def test_normalize_cited_chunk_ids_deduplicates_and_validates_against_example() -> None:
    example = _example()

    assert normalize_cited_chunk_ids(
        ["toy-001:gold", "toy-001:gold", "toy-001:distractor"],
        example,
    ) == ("toy-001:gold", "toy-001:distractor")

    with pytest.raises(PolicyOutputValidationError, match="unknown cited_chunk_ids"):
        normalize_cited_chunk_ids(["toy-001:missing"], example)
    with pytest.raises(PolicyOutputValidationError, match="non-empty string"):
        normalize_cited_chunk_ids(["toy-001:gold", ""], example)


def test_all_baseline_policies_share_interface_and_return_valid_citations() -> None:
    example = _example()

    policies = tuple(iter_baseline_policies())
    assert [policy.name for policy in policies] == [
        "vanilla-single",
        "chunk-map-reduce",
        "evidence-gated-nonrecursive",
        "recursive-ungated",
        "recursive-traceguard-stub",
    ]
    assert all(isinstance(policy, BenchmarkPolicy) for policy in policies)

    for policy in policies:
        result = policy.run(example)
        assert result.policy_name == policy.name
        assert result.answer
        assert result.cited_chunk_ids
        assert set(result.cited_chunk_ids) <= set(example.context_chunk_ids)
        assert result.metadata["deterministic"] is True
        assert result.metadata["live_model_calls"] == 0


def test_get_baseline_policy_returns_named_policy_and_rejects_unknown_name() -> None:
    assert get_baseline_policy("vanilla-single").name == "vanilla-single"
    assert (
        get_baseline_policy("recursive-traceguard-stub").name
        == "recursive-traceguard-stub"
    )

    with pytest.raises(KeyError, match="unknown baseline policy"):
        get_baseline_policy("live-gpt")


def test_evidence_gated_policy_cites_only_gold_evidence() -> None:
    example = _example()

    result = get_baseline_policy("evidence-gated-nonrecursive").run(example)

    assert result.answer == example.gold_answers[0]
    assert result.cited_chunk_ids == example.gold_evidence_chunk_ids
    assert result.metadata["uses_gold_evidence"] is True
    assert result.metadata["recursive"] is False


def test_recursive_traceguard_stub_records_provenance_gate_without_live_calls() -> None:
    example = _example()

    result = get_baseline_policy("recursive-traceguard-stub").run(example)

    assert result.answer == example.gold_answers[0]
    assert result.cited_chunk_ids == example.gold_evidence_chunk_ids
    assert result.metadata["recursive"] is True
    assert result.metadata["traceguard"] == "stubbed-provenance-gate"
    assert result.metadata["live_model_calls"] == 0
