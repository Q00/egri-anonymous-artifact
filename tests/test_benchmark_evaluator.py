from __future__ import annotations

import pytest

from egri.benchmarks import BenchmarkExample
from egri.benchmarks import ContextChunk
from egri.benchmarks import PolicyResult
from egri.benchmarks import aggregate_evaluations
from egri.benchmarks import evaluate_policy_result
from egri.benchmarks.evaluator import BenchmarkEvaluation
from egri.benchmarks.evaluator import EvaluationValidationError


def _example() -> BenchmarkExample:
    return BenchmarkExample(
        example_id="eval-001",
        dataset="toy",
        question="What does TraceGuard validate?",
        context_chunks=(
            ContextChunk(
                chunk_id="eval-001:gold-a",
                text="TraceGuard validates parent claims against fresh child evidence handles.",
            ),
            ContextChunk(
                chunk_id="eval-001:gold-b",
                text="Memory is only an operational prior, not admissible factual evidence.",
            ),
            ContextChunk(
                chunk_id="eval-001:distractor",
                text="This unrelated chunk discusses runtime logging only.",
            ),
        ),
        gold_answers=(
            "TraceGuard validates parent claims against fresh child evidence handles.",
            "TraceGuard checks provenance admissibility before parent synthesis is accepted.",
        ),
        gold_evidence_chunk_ids=("eval-001:gold-a", "eval-001:gold-b"),
    )


def _result(answer: str, cited_chunk_ids: tuple[str, ...]) -> PolicyResult:
    return PolicyResult(
        policy_name="test-policy",
        answer=answer,
        cited_chunk_ids=cited_chunk_ids,
    )


def test_perfect_match_scores_answer_and_citations() -> None:
    example = _example()
    result = _result(example.gold_answers[0], example.gold_evidence_chunk_ids)

    evaluation = evaluate_policy_result(example, result)

    assert evaluation.example_id == example.example_id
    assert evaluation.policy_name == "test-policy"
    assert evaluation.answer_exact_match == 1.0
    assert evaluation.answer_contains_match == 1.0
    assert evaluation.citation_precision == 1.0
    assert evaluation.citation_recall == 1.0
    assert evaluation.unsupported_citation_rate == 0.0
    assert evaluation.abstained is False
    assert evaluation.to_dict()["citation_f1"] == 1.0


def test_missing_evidence_lowers_recall_but_not_precision() -> None:
    example = _example()
    result = _result(example.gold_answers[0], ("eval-001:gold-a",))

    evaluation = evaluate_policy_result(example, result)

    assert evaluation.citation_precision == 1.0
    assert evaluation.citation_recall == 0.5
    assert evaluation.citation_f1 == pytest.approx(2 / 3)
    assert evaluation.unsupported_citation_rate == 0.0


def test_extra_unsupported_citation_lowers_precision_and_flags_rate() -> None:
    example = _example()
    result = _result(
        example.gold_answers[0],
        ("eval-001:gold-a", "eval-001:gold-b", "eval-001:distractor"),
    )

    evaluation = evaluate_policy_result(example, result)

    assert evaluation.citation_precision == pytest.approx(2 / 3)
    assert evaluation.citation_recall == 1.0
    assert evaluation.unsupported_citation_rate == pytest.approx(1 / 3)


def test_wrong_answer_with_right_citation_separates_answer_from_evidence_metrics() -> (
    None
):
    example = _example()
    result = _result(
        "TraceGuard proves semantic truth for all claims.",
        example.gold_evidence_chunk_ids,
    )

    evaluation = evaluate_policy_result(example, result)

    assert evaluation.answer_exact_match == 0.0
    assert evaluation.answer_contains_match == 0.0
    assert evaluation.citation_precision == 1.0
    assert evaluation.citation_recall == 1.0


def test_abstention_flag_for_empty_or_refusal_style_answers() -> None:
    example = _example()
    result = _result("I don't know", ("eval-001:gold-a",))

    evaluation = evaluate_policy_result(example, result)

    assert evaluation.abstained is True
    assert evaluation.answer_exact_match == 0.0
    assert evaluation.answer_contains_match == 0.0


def test_evaluator_rejects_outputs_with_unknown_citations() -> None:
    example = _example()
    invalid_result = object.__new__(PolicyResult)
    object.__setattr__(invalid_result, "policy_name", "bad-policy")
    object.__setattr__(invalid_result, "answer", "TraceGuard validates parent claims.")
    object.__setattr__(invalid_result, "cited_chunk_ids", ("eval-001:missing",))
    object.__setattr__(invalid_result, "metadata", {})

    with pytest.raises(EvaluationValidationError, match="unknown cited_chunk_ids"):
        evaluate_policy_result(example, invalid_result)


def test_evaluator_rejects_empty_or_whitespace_only_answers_from_bypassed_result() -> (
    None
):
    example = _example()
    invalid_result = object.__new__(PolicyResult)
    object.__setattr__(invalid_result, "policy_name", "bad-policy")
    object.__setattr__(invalid_result, "answer", "   ")
    object.__setattr__(invalid_result, "cited_chunk_ids", ("eval-001:gold-a",))
    object.__setattr__(invalid_result, "metadata", {})

    with pytest.raises(
        EvaluationValidationError, match="answer must be a non-empty string"
    ):
        evaluate_policy_result(example, invalid_result)


def test_evaluator_rejects_duplicate_citations_from_bypassed_result() -> None:
    example = _example()
    invalid_result = object.__new__(PolicyResult)
    object.__setattr__(invalid_result, "policy_name", "bad-policy")
    object.__setattr__(invalid_result, "answer", "TraceGuard validates parent claims.")
    object.__setattr__(
        invalid_result, "cited_chunk_ids", ("eval-001:gold-a", "eval-001:gold-a")
    )
    object.__setattr__(invalid_result, "metadata", {})

    with pytest.raises(EvaluationValidationError, match="duplicate"):
        evaluate_policy_result(example, invalid_result)


def test_aggregate_evaluations_groups_per_policy_and_averages_metrics() -> None:
    example = _example()
    perfect = evaluate_policy_result(
        example, _result(example.gold_answers[0], example.gold_evidence_chunk_ids)
    )
    missing = evaluate_policy_result(
        example, _result(example.gold_answers[0], ("eval-001:gold-a",))
    )
    other_policy = BenchmarkEvaluation(
        example_id="eval-002",
        policy_name="other-policy",
        answer_exact_match=0.0,
        answer_contains_match=0.0,
        citation_precision=0.0,
        citation_recall=0.0,
        citation_f1=0.0,
        unsupported_citation_rate=1.0,
        abstained=True,
    )

    summaries = aggregate_evaluations([perfect, missing, other_policy])

    by_name = {summary.policy_name: summary for summary in summaries}
    assert by_name["test-policy"].example_count == 2
    assert by_name["test-policy"].mean_answer_exact_match == 1.0
    assert by_name["test-policy"].mean_answer_contains_match == 1.0
    assert by_name["test-policy"].mean_citation_precision == 1.0
    assert by_name["test-policy"].mean_citation_recall == 0.75
    assert by_name["test-policy"].mean_citation_f1 == pytest.approx((1.0 + (2 / 3)) / 2)
    assert by_name["test-policy"].mean_unsupported_citation_rate == 0.0
    assert by_name["test-policy"].abstention_rate == 0.0
    assert by_name["other-policy"].example_count == 1
    assert by_name["other-policy"].abstention_rate == 1.0
    assert by_name["test-policy"].to_dict()["example_count"] == 2


def test_aggregate_rejects_empty_input() -> None:
    with pytest.raises(EvaluationValidationError, match="at least one"):
        aggregate_evaluations([])


def test_evaluation_metric_validation_rejects_bool_values() -> None:
    with pytest.raises(EvaluationValidationError, match="answer_exact_match"):
        BenchmarkEvaluation(
            example_id="eval-bool",
            policy_name="test-policy",
            answer_exact_match=True,  # type: ignore[arg-type]
            answer_contains_match=1.0,
            citation_precision=1.0,
            citation_recall=1.0,
            citation_f1=1.0,
            unsupported_citation_rate=0.0,
            abstained=False,
        )
