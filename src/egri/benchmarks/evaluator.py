"""Deterministic benchmark evaluation metrics."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from egri.benchmarks.policies import PolicyResult
from egri.benchmarks.schema import BenchmarkExample


class EvaluationValidationError(ValueError):
    """Raised when an evaluation input violates the metric contract."""


@dataclass(frozen=True, slots=True)
class BenchmarkEvaluation:
    """Per-example deterministic evaluation of one policy output."""

    example_id: str
    policy_name: str
    answer_exact_match: float
    answer_contains_match: float
    citation_precision: float
    citation_recall: float
    citation_f1: float
    unsupported_citation_rate: float
    abstained: bool

    def __post_init__(self) -> None:
        if not isinstance(self.example_id, str) or not self.example_id:
            msg = "example_id must be a non-empty string"
            raise EvaluationValidationError(msg)
        if not isinstance(self.policy_name, str) or not self.policy_name:
            msg = "policy_name must be a non-empty string"
            raise EvaluationValidationError(msg)
        for field_name in (
            "answer_exact_match",
            "answer_contains_match",
            "citation_precision",
            "citation_recall",
            "citation_f1",
            "unsupported_citation_rate",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not 0.0 <= float(value) <= 1.0
            ):
                msg = f"{field_name} must be a numeric value between 0 and 1"
                raise EvaluationValidationError(msg)
        if not isinstance(self.abstained, bool):
            msg = "abstained must be a boolean"
            raise EvaluationValidationError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "example_id": self.example_id,
            "policy_name": self.policy_name,
            "answer_exact_match": self.answer_exact_match,
            "answer_contains_match": self.answer_contains_match,
            "citation_precision": self.citation_precision,
            "citation_recall": self.citation_recall,
            "citation_f1": self.citation_f1,
            "unsupported_citation_rate": self.unsupported_citation_rate,
            "abstained": self.abstained,
        }


@dataclass(frozen=True, slots=True)
class PolicyAggregateSummary:
    """Mean metrics for one policy over multiple benchmark examples."""

    policy_name: str
    example_count: int
    mean_answer_exact_match: float
    mean_answer_contains_match: float
    mean_citation_precision: float
    mean_citation_recall: float
    mean_citation_f1: float
    mean_unsupported_citation_rate: float
    abstention_rate: float

    def __post_init__(self) -> None:
        if not isinstance(self.policy_name, str) or not self.policy_name:
            msg = "policy_name must be a non-empty string"
            raise EvaluationValidationError(msg)
        if not isinstance(self.example_count, int) or self.example_count <= 0:
            msg = "example_count must be a positive integer"
            raise EvaluationValidationError(msg)
        for field_name in (
            "mean_answer_exact_match",
            "mean_answer_contains_match",
            "mean_citation_precision",
            "mean_citation_recall",
            "mean_citation_f1",
            "mean_unsupported_citation_rate",
            "abstention_rate",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not 0.0 <= float(value) <= 1.0
            ):
                msg = f"{field_name} must be a numeric value between 0 and 1"
                raise EvaluationValidationError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "policy_name": self.policy_name,
            "example_count": self.example_count,
            "mean_answer_exact_match": self.mean_answer_exact_match,
            "mean_answer_contains_match": self.mean_answer_contains_match,
            "mean_citation_precision": self.mean_citation_precision,
            "mean_citation_recall": self.mean_citation_recall,
            "mean_citation_f1": self.mean_citation_f1,
            "mean_unsupported_citation_rate": self.mean_unsupported_citation_rate,
            "abstention_rate": self.abstention_rate,
        }


def evaluate_policy_result(
    example: BenchmarkExample,
    result: PolicyResult,
) -> BenchmarkEvaluation:
    """Score one normalized policy output against one benchmark example."""
    cited_chunk_ids = _validate_result_citations(example, result)
    gold_ids = set(example.gold_evidence_chunk_ids)
    cited_ids = set(cited_chunk_ids)
    supported_ids = cited_ids & gold_ids
    unsupported_ids = cited_ids - gold_ids

    citation_precision = len(supported_ids) / len(cited_ids)
    citation_recall = len(supported_ids) / len(gold_ids)
    citation_f1 = _f1(citation_precision, citation_recall)
    unsupported_citation_rate = len(unsupported_ids) / len(cited_ids)
    answer = result.answer.strip()
    abstained = _is_abstention(answer)
    exact_match = (
        0.0 if abstained else _answer_exact_match(answer, example.gold_answers)
    )
    contains_match = (
        0.0 if abstained else _answer_contains_match(answer, example.gold_answers)
    )

    return BenchmarkEvaluation(
        example_id=example.example_id,
        policy_name=result.policy_name,
        answer_exact_match=exact_match,
        answer_contains_match=contains_match,
        citation_precision=citation_precision,
        citation_recall=citation_recall,
        citation_f1=citation_f1,
        unsupported_citation_rate=unsupported_citation_rate,
        abstained=abstained,
    )


def aggregate_evaluations(
    evaluations: Iterable[BenchmarkEvaluation],
) -> tuple[PolicyAggregateSummary, ...]:
    """Aggregate per-example evaluations into per-policy mean summaries."""
    grouped: dict[str, list[BenchmarkEvaluation]] = defaultdict(list)
    for evaluation in evaluations:
        if not isinstance(evaluation, BenchmarkEvaluation):
            msg = "aggregate_evaluations expects BenchmarkEvaluation objects"
            raise EvaluationValidationError(msg)
        grouped[evaluation.policy_name].append(evaluation)

    if not grouped:
        msg = "aggregate_evaluations requires at least one evaluation"
        raise EvaluationValidationError(msg)

    summaries: list[PolicyAggregateSummary] = []
    for policy_name in sorted(grouped):
        group = grouped[policy_name]
        summaries.append(
            PolicyAggregateSummary(
                policy_name=policy_name,
                example_count=len(group),
                mean_answer_exact_match=_mean(e.answer_exact_match for e in group),
                mean_answer_contains_match=_mean(
                    e.answer_contains_match for e in group
                ),
                mean_citation_precision=_mean(e.citation_precision for e in group),
                mean_citation_recall=_mean(e.citation_recall for e in group),
                mean_citation_f1=_mean(e.citation_f1 for e in group),
                mean_unsupported_citation_rate=_mean(
                    e.unsupported_citation_rate for e in group
                ),
                abstention_rate=_mean(1.0 if e.abstained else 0.0 for e in group),
            )
        )
    return tuple(summaries)


def _validate_result_citations(
    example: BenchmarkExample,
    result: PolicyResult,
) -> tuple[str, ...]:
    if not isinstance(result.policy_name, str) or not result.policy_name:
        msg = "policy_name must be a non-empty string"
        raise EvaluationValidationError(msg)
    if not isinstance(result.answer, str) or not result.answer.strip():
        msg = f"{result.policy_name}: answer must be a non-empty string"
        raise EvaluationValidationError(msg)
    if not isinstance(result.cited_chunk_ids, tuple) or not result.cited_chunk_ids:
        msg = f"{result.policy_name}: cited_chunk_ids must be a non-empty tuple"
        raise EvaluationValidationError(msg)
    if any(
        not isinstance(chunk_id, str) or not chunk_id
        for chunk_id in result.cited_chunk_ids
    ):
        msg = f"{result.policy_name}: cited_chunk_ids entries must be non-empty strings"
        raise EvaluationValidationError(msg)
    if len(set(result.cited_chunk_ids)) != len(result.cited_chunk_ids):
        msg = f"{result.policy_name}: cited_chunk_ids must not contain duplicate values"
        raise EvaluationValidationError(msg)
    known = set(example.context_chunk_ids)
    missing = sorted(set(result.cited_chunk_ids) - known)
    if missing:
        msg = f"{example.example_id}: unknown cited_chunk_ids: {', '.join(missing)}"
        raise EvaluationValidationError(msg)
    return result.cited_chunk_ids


def _normalize_answer(text: str) -> str:
    return " ".join(text.casefold().strip().split())


def _answer_exact_match(answer: str, gold_answers: tuple[str, ...]) -> float:
    normalized = _normalize_answer(answer)
    return (
        1.0
        if any(normalized == _normalize_answer(gold) for gold in gold_answers)
        else 0.0
    )


def _answer_contains_match(answer: str, gold_answers: tuple[str, ...]) -> float:
    normalized = _normalize_answer(answer)
    if any(normalized == _normalize_answer(gold) for gold in gold_answers):
        return 1.0
    return (
        1.0
        if any(_normalize_answer(gold) in normalized for gold in gold_answers)
        else 0.0
    )


def _is_abstention(answer: str) -> bool:
    normalized = _normalize_answer(answer)
    abstentions = {
        "i don't know",
        "i do not know",
        "unknown",
        "cannot answer",
        "unanswerable",
        "no answer",
        "not enough information",
    }
    return normalized in abstentions


def _f1(precision: float, recall: float) -> float:
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _mean(values: Iterable[float]) -> float:
    collected = tuple(values)
    return sum(collected) / len(collected)
