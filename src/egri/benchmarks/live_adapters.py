"""Live model adapter contracts for benchmark policies.

The classes in this module define dependency-injected live policy adapters. They
perform no network or API calls themselves; callers must provide a ModelClient
implementation, which lets tests use fake clients and production code inject a
real client explicitly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from typing import Any
from typing import Protocol

from egri.benchmarks.policies import BenchmarkPolicy
from egri.benchmarks.policies import PolicyOutputValidationError
from egri.benchmarks.policies import PolicyResult
from egri.benchmarks.policies import normalize_cited_chunk_ids
from egri.benchmarks.schema import BenchmarkExample


class ModelClient(Protocol):
    """Protocol for injected model clients used by live benchmark adapters."""

    def complete(
        self, prompt: str, *, example: BenchmarkExample
    ) -> "LiveModelResponse":
        """Return one model completion for a benchmark example."""


@dataclass(frozen=True, slots=True)
class LiveModelResponse:
    """Structured response from an injected live model client."""

    answer: str
    cited_chunk_ids: tuple[str, ...]
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: float
    traceguard_rejected: bool
    traceguard_repair_count: int
    evidence_handles: tuple[str, ...]
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.answer, str) or not self.answer.strip():
            msg = "answer must be a non-empty string"
            raise PolicyOutputValidationError(msg)
        if not isinstance(self.cited_chunk_ids, tuple) or not self.cited_chunk_ids:
            msg = "cited_chunk_ids must be a non-empty tuple"
            raise PolicyOutputValidationError(msg)
        if any(
            not isinstance(chunk_id, str) or not chunk_id
            for chunk_id in self.cited_chunk_ids
        ):
            msg = "cited_chunk_ids entries must be non-empty strings"
            raise PolicyOutputValidationError(msg)
        if len(set(self.cited_chunk_ids)) != len(self.cited_chunk_ids):
            msg = "cited_chunk_ids must not contain duplicate values"
            raise PolicyOutputValidationError(msg)
        _validate_nonnegative_int("prompt_tokens", self.prompt_tokens)
        _validate_nonnegative_int("completion_tokens", self.completion_tokens)
        _validate_nonnegative_float("latency_ms", self.latency_ms)
        _validate_nonnegative_float("cost_usd", self.cost_usd)
        if not isinstance(self.traceguard_rejected, bool):
            msg = "traceguard_rejected must be boolean"
            raise PolicyOutputValidationError(msg)
        _validate_nonnegative_int(
            "traceguard_repair_count", self.traceguard_repair_count
        )
        if not isinstance(self.evidence_handles, tuple) or not self.evidence_handles:
            msg = "evidence_handles must be a non-empty tuple"
            raise PolicyOutputValidationError(msg)
        if any(
            not isinstance(handle, str) or not handle.startswith("child:")
            for handle in self.evidence_handles
        ):
            msg = "evidence_handles entries must be non-empty child: handles"
            raise PolicyOutputValidationError(msg)
        if len(set(self.evidence_handles)) != len(self.evidence_handles):
            msg = "evidence_handles must not contain duplicate values"
            raise PolicyOutputValidationError(msg)
        if self.metadata is not None and not isinstance(self.metadata, Mapping):
            msg = "metadata must be a mapping when present"
            raise PolicyOutputValidationError(msg)

    @property
    def total_tokens(self) -> int:
        """Return prompt plus completion tokens."""
        return self.prompt_tokens + self.completion_tokens

    def to_metadata(self) -> dict[str, Any]:
        """Return usage, cost, latency, and TraceGuard metadata."""
        metadata = dict(self.metadata or {})
        metadata.update(
            {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "latency_ms": self.latency_ms,
                "cost_usd": self.cost_usd,
                "traceguard_rejected": self.traceguard_rejected,
                "traceguard_repair_count": self.traceguard_repair_count,
                "evidence_handles": list(self.evidence_handles),
            }
        )
        return metadata


@dataclass(frozen=True, slots=True)
class LiveRecursiveTraceGuardPolicy(BenchmarkPolicy):
    """Live recursive TraceGuard policy backed by an injected model client."""

    model_client: ModelClient
    name: str = "live-recursive-traceguard"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            msg = "live policy name must be a non-empty string"
            raise PolicyOutputValidationError(msg)
        if not hasattr(self.model_client, "complete"):
            msg = "model_client must provide complete(prompt, *, example)"
            raise PolicyOutputValidationError(msg)

    def run(self, example: BenchmarkExample) -> PolicyResult:
        """Run the injected model client and normalize its evidence citations."""
        response = self.model_client.complete(_build_prompt(example), example=example)
        cited_chunk_ids = normalize_cited_chunk_ids(response.cited_chunk_ids, example)
        _validate_evidence_handles_reference_citations(
            response.evidence_handles, cited_chunk_ids
        )
        metadata = {
            "deterministic": False,
            "live_model_calls": 1,
            "mode": "live-recursive-traceguard",
            "recursive": True,
            "traceguard": "live-provenance-gate",
            **response.to_metadata(),
        }
        return PolicyResult(
            policy_name=self.name,
            answer=response.answer,
            cited_chunk_ids=cited_chunk_ids,
            metadata=metadata,
        )


def _validate_evidence_handles_reference_citations(
    evidence_handles: tuple[str, ...], cited_chunk_ids: tuple[str, ...]
) -> None:
    cited_handle_targets = {f"child:{chunk_id}" for chunk_id in cited_chunk_ids}
    handle_targets = set(evidence_handles)
    if not cited_handle_targets <= handle_targets:
        msg = "evidence_handles must include a child: handle for every cited chunk"
        raise PolicyOutputValidationError(msg)
    if not handle_targets <= cited_handle_targets:
        msg = "evidence_handles must not reference uncited chunks"
        raise PolicyOutputValidationError(msg)


def _build_prompt(example: BenchmarkExample) -> str:
    chunks = "\n".join(
        f"[{chunk.chunk_id}] {chunk.title or ''}\n{chunk.text}"
        for chunk in example.context_chunks
    )
    return (
        "You are the EGRI live recursive TraceGuard benchmark policy.\n"
        "Answer the question using only cited fresh child evidence handles.\n"
        f"Example ID: {example.example_id}\n"
        f"Question: {example.question}\n"
        f"Context chunks:\n{chunks}\n"
    )


def _validate_nonnegative_int(field_name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"{field_name} must be a non-negative integer"
        raise PolicyOutputValidationError(msg)


def _validate_nonnegative_float(field_name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not isfinite(float(value))
        or float(value) < 0.0
    ):
        msg = f"{field_name} must be a non-negative numeric value"
        raise PolicyOutputValidationError(msg)
