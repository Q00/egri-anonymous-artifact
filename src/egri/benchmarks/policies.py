"""Common policy interface and deterministic baselines for benchmark runs."""

from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from egri.benchmarks.schema import BenchmarkExample


class PolicyOutputValidationError(ValueError):
    """Raised when a policy output violates the normalized policy contract."""


@dataclass(frozen=True, slots=True)
class PolicyResult:
    """Normalized answer and citation output from a benchmark policy."""

    policy_name: str
    answer: str
    cited_chunk_ids: tuple[str, ...]
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.policy_name, str) or not self.policy_name:
            msg = "policy_name must be a non-empty string"
            raise PolicyOutputValidationError(msg)
        if not isinstance(self.answer, str) or not self.answer:
            msg = f"{self.policy_name}: answer must be a non-empty string"
            raise PolicyOutputValidationError(msg)
        if not isinstance(self.cited_chunk_ids, tuple) or not self.cited_chunk_ids:
            msg = f"{self.policy_name}: cited_chunk_ids must be a non-empty tuple"
            raise PolicyOutputValidationError(msg)
        if any(
            not isinstance(chunk_id, str) or not chunk_id
            for chunk_id in self.cited_chunk_ids
        ):
            msg = (
                f"{self.policy_name}: cited_chunk_ids entries must be non-empty strings"
            )
            raise PolicyOutputValidationError(msg)
        if len(set(self.cited_chunk_ids)) != len(self.cited_chunk_ids):
            msg = (
                f"{self.policy_name}: cited_chunk_ids must not contain duplicate values"
            )
            raise PolicyOutputValidationError(msg)
        if self.metadata is not None and not isinstance(self.metadata, Mapping):
            msg = f"{self.policy_name}: metadata must be a mapping when present"
            raise PolicyOutputValidationError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "policy_name": self.policy_name,
            "answer": self.answer,
            "cited_chunk_ids": list(self.cited_chunk_ids),
            "metadata": dict(self.metadata or {}),
        }


class BenchmarkPolicy(ABC):
    """Policy contract for benchmark baselines and future live model adapters."""

    name: str

    @abstractmethod
    def run(self, example: BenchmarkExample) -> PolicyResult:
        """Answer a benchmark example with normalized cited chunk IDs."""


def normalize_cited_chunk_ids(
    cited_chunk_ids: Iterable[str],
    example: BenchmarkExample,
) -> tuple[str, ...]:
    """Deduplicate cited chunks in order and reject IDs outside the example."""
    normalized: list[str] = []
    seen: set[str] = set()
    for chunk_id in cited_chunk_ids:
        if not isinstance(chunk_id, str) or not chunk_id:
            msg = "cited_chunk_ids entries must be non-empty string values"
            raise PolicyOutputValidationError(msg)
        if chunk_id not in seen:
            normalized.append(chunk_id)
            seen.add(chunk_id)

    known = set(example.context_chunk_ids)
    missing = sorted(set(normalized) - known)
    if missing:
        msg = f"{example.example_id}: unknown cited_chunk_ids: {', '.join(missing)}"
        raise PolicyOutputValidationError(msg)
    if not normalized:
        msg = f"{example.example_id}: policy output must cite at least one chunk"
        raise PolicyOutputValidationError(msg)
    return tuple(normalized)


@dataclass(frozen=True, slots=True)
class _DeterministicBaselinePolicy(BenchmarkPolicy):
    name: str
    mode: str
    recursive: bool
    cite_gold_only: bool
    traceguard: str | None = None

    def run(self, example: BenchmarkExample) -> PolicyResult:
        if self.cite_gold_only:
            cited_chunk_ids = normalize_cited_chunk_ids(
                example.gold_evidence_chunk_ids, example
            )
            answer = example.gold_answers[0]
            uses_gold_evidence = True
        else:
            cited_chunk_ids = normalize_cited_chunk_ids(
                example.context_chunk_ids, example
            )
            answer = _extract_answer_from_chunks(example, cited_chunk_ids)
            uses_gold_evidence = False

        metadata: dict[str, Any] = {
            "deterministic": True,
            "live_model_calls": 0,
            "mode": self.mode,
            "recursive": self.recursive,
            "uses_gold_evidence": uses_gold_evidence,
        }
        if self.traceguard is not None:
            metadata["traceguard"] = self.traceguard
        return PolicyResult(
            policy_name=self.name,
            answer=answer,
            cited_chunk_ids=cited_chunk_ids,
            metadata=metadata,
        )


def _extract_answer_from_chunks(
    example: BenchmarkExample,
    cited_chunk_ids: tuple[str, ...],
) -> str:
    chunks_by_id = {chunk.chunk_id: chunk for chunk in example.context_chunks}
    for chunk_id in cited_chunk_ids:
        text = chunks_by_id[chunk_id].text.strip()
        if text:
            return text
    return example.gold_answers[0]


def iter_baseline_policies() -> tuple[BenchmarkPolicy, ...]:
    """Return deterministic baselines in stable experiment-table order.

    The evidence-gated and TraceGuard-stub policies are oracle/stub baselines:
    they intentionally use gold evidence and gold answers to isolate evaluator
    wiring before live model adapters are introduced.
    """
    return (
        _DeterministicBaselinePolicy(
            name="vanilla-single",
            mode="single-context",
            recursive=False,
            cite_gold_only=False,
        ),
        _DeterministicBaselinePolicy(
            name="chunk-map-reduce",
            mode="chunk-map-reduce",
            recursive=False,
            cite_gold_only=False,
        ),
        _DeterministicBaselinePolicy(
            name="evidence-gated-nonrecursive",
            mode="evidence-gated",
            recursive=False,
            cite_gold_only=True,
        ),
        _DeterministicBaselinePolicy(
            name="recursive-ungated",
            mode="recursive-ungated",
            recursive=True,
            cite_gold_only=False,
        ),
        _DeterministicBaselinePolicy(
            name="recursive-traceguard-stub",
            mode="recursive-traceguard-stub",
            recursive=True,
            cite_gold_only=True,
            traceguard="stubbed-provenance-gate",
        ),
    )


def get_baseline_policy(name: str) -> BenchmarkPolicy:
    """Return a deterministic baseline by name."""
    policies = {policy.name: policy for policy in iter_baseline_policies()}
    try:
        return policies[name]
    except KeyError as exc:
        available = ", ".join(sorted(policies))
        msg = f"unknown baseline policy {name!r}; available: {available}"
        raise KeyError(msg) from exc
