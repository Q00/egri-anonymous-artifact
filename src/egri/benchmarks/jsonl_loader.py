"""JSONL loading for normalized benchmark examples."""

from __future__ import annotations

import json
from collections.abc import Iterable
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from egri.benchmarks.schema import BenchmarkExample
from egri.benchmarks.schema import BenchmarkValidationError
from egri.benchmarks.schema import ContextChunk


def load_jsonl_examples(
    path: str | Path,
    *,
    base_dir: str | Path | None = None,
) -> list[BenchmarkExample]:
    """Load and validate normalized benchmark examples from JSONL.

    Args:
        path: Local JSONL file path. Callers that pass user-controlled paths
            should also pass ``base_dir``.
        base_dir: Optional trusted benchmark-data root. When provided, ``path``
            must resolve inside this directory.
    """
    source = _resolve_source_path(path, base_dir=base_dir)
    if not source.exists():
        msg = f"benchmark JSONL file does not exist: {source}"
        raise FileNotFoundError(msg)
    if source.suffix != ".jsonl":
        msg = f"benchmark file must use .jsonl suffix: {source}"
        raise BenchmarkValidationError(msg)

    examples: list[BenchmarkExample] = []
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            raw = json.loads(stripped)
        except json.JSONDecodeError as exc:
            msg = f"{source}:{line_number}: invalid JSON: {exc.msg}"
            raise BenchmarkValidationError(msg) from exc
        if not isinstance(raw, Mapping):
            msg = f"{source}:{line_number}: each JSONL record must be an object"
            raise BenchmarkValidationError(msg)
        try:
            examples.append(_example_from_mapping(raw))
        except BenchmarkValidationError as exc:
            msg = f"{source}:{line_number}: {exc}"
            raise BenchmarkValidationError(msg) from exc

    if not examples:
        msg = f"benchmark JSONL file contains no examples: {source}"
        raise BenchmarkValidationError(msg)
    return examples


def _resolve_source_path(path: str | Path, *, base_dir: str | Path | None) -> Path:
    source = Path(path).expanduser().resolve()
    if base_dir is None:
        return source

    trusted_root = Path(base_dir).expanduser().resolve()
    try:
        source.relative_to(trusted_root)
    except ValueError as exc:
        msg = f"benchmark path is outside trusted benchmark base_dir: {source}"
        raise BenchmarkValidationError(msg) from exc
    return source


def _example_from_mapping(raw: Mapping[str, Any]) -> BenchmarkExample:
    example_id = _required_str(raw, "example_id", record_id="<unknown>")
    return BenchmarkExample(
        example_id=example_id,
        dataset=_required_str(raw, "dataset", record_id=example_id),
        question=_required_str(raw, "question", record_id=example_id),
        context_chunks=tuple(
            _context_chunk_from_mapping(item, example_id=example_id)
            for item in _required_list(raw, "context_chunks", record_id=example_id)
        ),
        gold_answers=tuple(
            _str_items(
                _required_list(raw, "gold_answers", record_id=example_id),
                field="gold_answers",
                record_id=example_id,
            )
        ),
        gold_evidence_chunk_ids=tuple(
            _str_items(
                _required_list(raw, "gold_evidence_chunk_ids", record_id=example_id),
                field="gold_evidence_chunk_ids",
                record_id=example_id,
            )
        ),
        metadata=_optional_mapping(raw, "metadata", record_id=example_id),
    )


def _context_chunk_from_mapping(raw: Any, *, example_id: str) -> ContextChunk:
    if not isinstance(raw, Mapping):
        msg = f"{example_id}: context_chunks entries must be objects"
        raise BenchmarkValidationError(msg)
    return ContextChunk(
        chunk_id=_required_str(raw, "chunk_id", record_id=example_id),
        text=_required_str(raw, "text", record_id=example_id),
        title=_optional_str(raw, "title", record_id=example_id),
        metadata=_optional_mapping(raw, "metadata", record_id=example_id),
    )


def _required_str(raw: Mapping[str, Any], field: str, *, record_id: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value:
        msg = f"{record_id}: {field} must be a non-empty string"
        raise BenchmarkValidationError(msg)
    return value


def _optional_str(raw: Mapping[str, Any], field: str, *, record_id: str) -> str | None:
    value = raw.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        msg = f"{record_id}: {field} must be a non-empty string when present"
        raise BenchmarkValidationError(msg)
    return value


def _required_list(raw: Mapping[str, Any], field: str, *, record_id: str) -> list[Any]:
    value = raw.get(field)
    if not isinstance(value, list) or not value:
        msg = f"{record_id}: {field} must be a non-empty list"
        raise BenchmarkValidationError(msg)
    return value


def _str_items(items: Iterable[Any], *, field: str, record_id: str) -> list[str]:
    values: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, str) or not item:
            msg = f"{record_id}: {field}[{index}] must be a non-empty string"
            raise BenchmarkValidationError(msg)
        values.append(item)
    return values


def _optional_mapping(
    raw: Mapping[str, Any],
    field: str,
    *,
    record_id: str,
) -> Mapping[str, Any] | None:
    value = raw.get(field)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        msg = f"{record_id}: {field} must be an object when present"
        raise BenchmarkValidationError(msg)
    return dict(value)
