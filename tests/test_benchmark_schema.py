from __future__ import annotations

import json
from pathlib import Path

import pytest

from egri.benchmarks.jsonl_loader import load_jsonl_examples
from egri.benchmarks.schema import BenchmarkValidationError
from egri.benchmarks.schema import ContextChunk


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def _valid_record() -> dict[str, object]:
    return {
        "example_id": "qasper-mini-001",
        "dataset": "qasper_mini",
        "question": "What does TraceGuard require before parent synthesis commits?",
        "context_chunks": [
            {
                "chunk_id": "qasper-mini-001:abstract",
                "text": "TraceGuard requires parent claims to cite accepted child evidence handles.",
                "title": "Abstract",
                "metadata": {"section": "abstract"},
            },
            {
                "chunk_id": "qasper-mini-001:limitations",
                "text": "The gate checks provenance handles rather than semantic truth.",
            },
        ],
        "gold_answers": ["TraceGuard requires accepted child evidence handles."],
        "gold_evidence_chunk_ids": ["qasper-mini-001:abstract"],
        "metadata": {"paper_id": "egri-mini"},
    }


def test_context_chunk_to_traceguard_evidence_uses_stable_fact_id() -> None:
    chunk = ContextChunk(
        chunk_id="paper:1-3",
        text="EGRI separates memory priors from admissible evidence.",
    )

    evidence = chunk.to_traceguard_evidence(
        fact_id="FACT-001", child_call_id="child-0001"
    )

    assert evidence.fact_id == "FACT-001"
    assert evidence.chunk_id == "paper:1-3"
    assert (
        evidence.text == "EGRI separates memory priors from admissible evidence."
    )
    assert evidence.child_call_id == "child-0001"


def test_load_jsonl_examples_returns_validated_benchmark_examples(tmp_path) -> None:
    path = tmp_path / "qasper_mini.jsonl"
    _write_jsonl(path, [_valid_record()])

    examples = load_jsonl_examples(path)

    assert len(examples) == 1
    example = examples[0]
    assert example.example_id == "qasper-mini-001"
    assert example.dataset == "qasper_mini"
    assert example.gold_answers == (
        "TraceGuard requires accepted child evidence handles.",
    )
    assert example.gold_evidence_chunk_ids == ("qasper-mini-001:abstract",)
    assert example.context_chunk_ids == (
        "qasper-mini-001:abstract",
        "qasper-mini-001:limitations",
    )
    assert example.evidence_chunks[0].text.startswith("TraceGuard requires")


def test_loader_rejects_gold_evidence_ids_missing_from_context(tmp_path) -> None:
    record = _valid_record()
    record["gold_evidence_chunk_ids"] = ["qasper-mini-001:missing"]
    path = tmp_path / "bad_missing_evidence.jsonl"
    _write_jsonl(path, [record])

    with pytest.raises(
        BenchmarkValidationError, match="unknown gold_evidence_chunk_ids"
    ):
        load_jsonl_examples(path)


def test_loader_rejects_duplicate_chunk_ids(tmp_path) -> None:
    record = _valid_record()
    record["context_chunks"] = [
        {"chunk_id": "duplicate", "text": "first"},
        {"chunk_id": "duplicate", "text": "second"},
    ]
    record["gold_evidence_chunk_ids"] = ["duplicate"]
    path = tmp_path / "bad_duplicate_chunks.jsonl"
    _write_jsonl(path, [record])

    with pytest.raises(BenchmarkValidationError, match="duplicate context chunk_id"):
        load_jsonl_examples(path)


def test_loader_rejects_invalid_json_and_non_object_records(tmp_path) -> None:
    invalid_json = tmp_path / "invalid.jsonl"
    invalid_json.write_text("{not-json}\n")
    non_object = tmp_path / "non_object.jsonl"
    non_object.write_text(json.dumps(["not", "an", "object"]) + "\n")

    with pytest.raises(BenchmarkValidationError, match="invalid JSON"):
        load_jsonl_examples(invalid_json)
    with pytest.raises(BenchmarkValidationError, match="record must be an object"):
        load_jsonl_examples(non_object)


def test_loader_rejects_missing_required_fields_and_empty_files(tmp_path) -> None:
    missing_question = _valid_record()
    missing_question.pop("question")
    missing_path = tmp_path / "missing_question.jsonl"
    _write_jsonl(missing_path, [missing_question])
    empty_path = tmp_path / "empty.jsonl"
    empty_path.write_text("\n")

    with pytest.raises(
        BenchmarkValidationError, match="question must be a non-empty string"
    ):
        load_jsonl_examples(missing_path)
    with pytest.raises(BenchmarkValidationError, match="contains no examples"):
        load_jsonl_examples(empty_path)


def test_loader_rejects_wrong_list_item_types_and_malformed_chunks(tmp_path) -> None:
    bad_answer = _valid_record()
    bad_answer["gold_answers"] = [123]
    bad_chunk = _valid_record()
    bad_chunk["context_chunks"] = ["not-a-chunk"]
    bad_answer_path = tmp_path / "bad_answer.jsonl"
    bad_chunk_path = tmp_path / "bad_chunk.jsonl"
    _write_jsonl(bad_answer_path, [bad_answer])
    _write_jsonl(bad_chunk_path, [bad_chunk])

    with pytest.raises(BenchmarkValidationError, match=r"gold_answers\[0\]"):
        load_jsonl_examples(bad_answer_path)
    with pytest.raises(
        BenchmarkValidationError, match="context_chunks entries must be objects"
    ):
        load_jsonl_examples(bad_chunk_path)


def test_direct_schema_construction_validates_field_types() -> None:
    with pytest.raises(
        BenchmarkValidationError, match="chunk_id must be a non-empty string"
    ):
        ContextChunk(chunk_id=123, text="valid")  # type: ignore[arg-type]
    with pytest.raises(
        BenchmarkValidationError, match="text must be a non-empty string"
    ):
        ContextChunk(chunk_id="chunk", text=["invalid"])  # type: ignore[arg-type]


def test_loader_can_constrain_paths_to_a_trusted_base_dir(tmp_path) -> None:
    base_dir = tmp_path / "data"
    base_dir.mkdir()
    inside = base_dir / "inside.jsonl"
    outside = tmp_path / "outside.jsonl"
    _write_jsonl(inside, [_valid_record()])
    _write_jsonl(outside, [_valid_record()])

    assert len(load_jsonl_examples(inside, base_dir=base_dir)) == 1
    with pytest.raises(
        BenchmarkValidationError, match="outside trusted benchmark base_dir"
    ):
        load_jsonl_examples(outside, base_dir=base_dir)


def test_committed_qasper_mini_sample_loads_offline() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sample_path = repo_root / "data/benchmark_samples/qasper_mini.jsonl"

    examples = load_jsonl_examples(sample_path)

    assert len(examples) == 3
    assert {example.dataset for example in examples} == {"qasper_mini"}
    for example in examples:
        assert example.gold_answers
        assert example.gold_evidence_chunk_ids
        assert all(
            chunk_id in example.context_chunk_ids
            for chunk_id in example.gold_evidence_chunk_ids
        )
