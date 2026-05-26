"""Dataset-neutral benchmark schema for EGRI experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from egri.traceguard import TraceGuardEvidence


class BenchmarkValidationError(ValueError):
    """Raised when a benchmark example violates the normalized schema."""


@dataclass(frozen=True, slots=True)
class ContextChunk:
    """One addressable context span that can support answer claims."""

    chunk_id: str
    text: str
    title: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.chunk_id, str) or not self.chunk_id:
            msg = "context chunk_id must be a non-empty string"
            raise BenchmarkValidationError(msg)
        if not isinstance(self.text, str) or not self.text:
            msg = f"context chunk {self.chunk_id} text must be a non-empty string"
            raise BenchmarkValidationError(msg)
        if self.title is not None and (
            not isinstance(self.title, str) or not self.title
        ):
            msg = f"context chunk {self.chunk_id} title must be a non-empty string when present"
            raise BenchmarkValidationError(msg)
        if self.metadata is not None and not isinstance(self.metadata, Mapping):
            msg = (
                f"context chunk {self.chunk_id} metadata must be a mapping when present"
            )
            raise BenchmarkValidationError(msg)

    def to_traceguard_evidence(
        self,
        *,
        fact_id: str,
        child_call_id: str | None = None,
    ) -> TraceGuardEvidence:
        """Convert this chunk into a TraceGuard evidence handle."""
        if not fact_id:
            msg = "fact_id must be a non-empty string"
            raise BenchmarkValidationError(msg)
        return TraceGuardEvidence(
            fact_id=fact_id,
            chunk_id=self.chunk_id,
            text=self.text,
            child_call_id=child_call_id,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        result: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "text": self.text,
        }
        if self.title is not None:
            result["title"] = self.title
        if self.metadata:
            result["metadata"] = dict(self.metadata)
        return result


@dataclass(frozen=True, slots=True)
class BenchmarkExample:
    """A normalized evidence-grounded QA/citation benchmark example."""

    example_id: str
    dataset: str
    question: str
    context_chunks: tuple[ContextChunk, ...]
    gold_answers: tuple[str, ...]
    gold_evidence_chunk_ids: tuple[str, ...]
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.example_id, str) or not self.example_id:
            msg = "example_id must be a non-empty string"
            raise BenchmarkValidationError(msg)
        if not isinstance(self.dataset, str) or not self.dataset:
            msg = f"{self.example_id}: dataset must be a non-empty string"
            raise BenchmarkValidationError(msg)
        if not isinstance(self.question, str) or not self.question:
            msg = f"{self.example_id}: question must be a non-empty string"
            raise BenchmarkValidationError(msg)
        if not isinstance(self.context_chunks, tuple) or not self.context_chunks:
            msg = f"{self.example_id}: context_chunks must be a non-empty tuple"
            raise BenchmarkValidationError(msg)
        if any(not isinstance(chunk, ContextChunk) for chunk in self.context_chunks):
            msg = f"{self.example_id}: context_chunks entries must be ContextChunk objects"
            raise BenchmarkValidationError(msg)
        if not isinstance(self.gold_answers, tuple) or not self.gold_answers:
            msg = f"{self.example_id}: gold_answers must be a non-empty tuple"
            raise BenchmarkValidationError(msg)
        if any(
            not isinstance(answer, str) or not answer for answer in self.gold_answers
        ):
            msg = f"{self.example_id}: gold_answers entries must be non-empty strings"
            raise BenchmarkValidationError(msg)
        if (
            not isinstance(self.gold_evidence_chunk_ids, tuple)
            or not self.gold_evidence_chunk_ids
        ):
            msg = (
                f"{self.example_id}: gold_evidence_chunk_ids must be a non-empty tuple"
            )
            raise BenchmarkValidationError(msg)
        if any(
            not isinstance(chunk_id, str) or not chunk_id
            for chunk_id in self.gold_evidence_chunk_ids
        ):
            msg = f"{self.example_id}: gold_evidence_chunk_ids entries must be non-empty strings"
            raise BenchmarkValidationError(msg)
        if self.metadata is not None and not isinstance(self.metadata, Mapping):
            msg = f"{self.example_id}: metadata must be a mapping when present"
            raise BenchmarkValidationError(msg)

        seen: set[str] = set()
        duplicates: list[str] = []
        for chunk in self.context_chunks:
            if chunk.chunk_id in seen:
                duplicates.append(chunk.chunk_id)
            seen.add(chunk.chunk_id)
        if duplicates:
            msg = (
                f"{self.example_id}: duplicate context chunk_id values: "
                f"{', '.join(sorted(set(duplicates)))}"
            )
            raise BenchmarkValidationError(msg)

        missing = sorted(set(self.gold_evidence_chunk_ids) - seen)
        if missing:
            msg = (
                f"{self.example_id}: unknown gold_evidence_chunk_ids: "
                f"{', '.join(missing)}"
            )
            raise BenchmarkValidationError(msg)

    @property
    def context_chunk_ids(self) -> tuple[str, ...]:
        """Return context chunk identifiers in prompt order."""
        return tuple(chunk.chunk_id for chunk in self.context_chunks)

    @property
    def evidence_chunks(self) -> tuple[ContextChunk, ...]:
        """Return gold evidence chunks in gold evidence order."""
        chunks_by_id = {chunk.chunk_id: chunk for chunk in self.context_chunks}
        return tuple(
            chunks_by_id[chunk_id] for chunk_id in self.gold_evidence_chunk_ids
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "example_id": self.example_id,
            "dataset": self.dataset,
            "question": self.question,
            "context_chunks": [chunk.to_dict() for chunk in self.context_chunks],
            "gold_answers": list(self.gold_answers),
            "gold_evidence_chunk_ids": list(self.gold_evidence_chunk_ids),
            "metadata": dict(self.metadata or {}),
        }
