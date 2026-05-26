from __future__ import annotations

import json
from pathlib import Path

import pytest

from egri.benchmarks import BenchmarkDatasetCatalogError
from egri.benchmarks import BenchmarkDatasetSpec
from egri.benchmarks import BenchmarkManifestSource
from egri.benchmarks import BenchmarkSuiteManifest
from egri.benchmarks import iter_dataset_specs
from egri.benchmarks import load_benchmark_suite_manifest
from egri.benchmarks import load_manifest_examples
from egri.benchmarks import materialize_manifest_subset

_DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
_MANIFEST_PATH = _DATA_ROOT / "benchmark_suites" / "real_multidataset_mini.json"


def test_dataset_catalog_lists_real_benchmark_families_in_stable_order() -> None:
    specs = iter_dataset_specs()

    assert tuple(spec.name for spec in specs) == (
        "qasper",
        "hotpotqa",
        "2wikimultihopqa",
        "asqa",
    )
    assert all(isinstance(spec, BenchmarkDatasetSpec) for spec in specs)
    assert specs[0].task_type == "long-document evidence QA"
    assert specs[1].requires_multihop_reasoning is True
    assert specs[3].requires_abstractive_synthesis is True
    assert all(spec.license_name for spec in specs)


def test_committed_real_multidataset_manifest_loads_normalized_examples() -> None:
    manifest = load_benchmark_suite_manifest(_MANIFEST_PATH, base_dir=_DATA_ROOT)
    examples = load_manifest_examples(manifest)

    assert isinstance(manifest, BenchmarkSuiteManifest)
    assert manifest.name == "real-multidataset-mini"
    assert manifest.version == "2026-05-10"
    assert manifest.example_count == 8
    assert manifest.dataset_counts == {
        "qasper": 2,
        "hotpotqa": 2,
        "2wikimultihopqa": 2,
        "asqa": 2,
    }
    assert len(examples) == 8
    assert {example.dataset for example in examples} == set(manifest.dataset_counts)
    assert all(
        example.metadata["split"] in {"validation", "dev"} for example in examples
    )
    assert all(
        example.metadata["source_format"] == "normalized-jsonl" for example in examples
    )
    assert all(example.metadata["license_name"] for example in examples)
    assert all(example.gold_evidence_chunk_ids for example in examples)


def test_materialize_manifest_subset_writes_jsonl_and_metadata(tmp_path) -> None:
    manifest = load_benchmark_suite_manifest(_MANIFEST_PATH, base_dir=_DATA_ROOT)

    result = materialize_manifest_subset(manifest, output_dir=tmp_path / "subset")

    assert result.example_count == 8
    assert result.dataset_counts == manifest.dataset_counts
    assert result.examples_path.name == "real-multidataset-mini.jsonl"
    assert result.metadata_path.name == "real-multidataset-mini.metadata.json"
    rows = [
        json.loads(line)
        for line in result.examples_path.read_text(encoding="utf-8").splitlines()
    ]
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert len(rows) == 8
    assert metadata["suite_name"] == manifest.name
    assert metadata["dataset_counts"] == manifest.dataset_counts
    assert metadata["normalized_schema"] == "egri.benchmarks.BenchmarkExample"


def test_manifest_rejects_unknown_dataset_name(tmp_path) -> None:
    manifest_path = tmp_path / "bad.json"
    source_path = tmp_path / "examples.jsonl"
    source_path.write_text("{}\n", encoding="utf-8")
    manifest_path.write_text(
        json.dumps(
            {
                "name": "bad-suite",
                "version": "1",
                "description": "bad",
                "sources": [
                    {
                        "dataset": "unknownqa",
                        "split": "validation",
                        "path": "examples.jsonl",
                        "example_count": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkDatasetCatalogError, match="unknown dataset"):
        load_benchmark_suite_manifest(manifest_path, base_dir=tmp_path)


def test_manifest_rejects_paths_outside_base_dir(tmp_path) -> None:
    outside_path = tmp_path.parent / "outside.jsonl"
    outside_path.write_text("{}\n", encoding="utf-8")
    manifest_path = tmp_path / "bad-path.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "bad-path-suite",
                "version": "1",
                "description": "bad",
                "sources": [
                    {
                        "dataset": "qasper",
                        "split": "validation",
                        "path": str(outside_path),
                        "example_count": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(BenchmarkDatasetCatalogError, match="outside trusted"):
        load_benchmark_suite_manifest(manifest_path, base_dir=tmp_path)


def test_manifest_source_rejects_non_integer_example_counts() -> None:
    for invalid_count in (True, 1.5, "1", 0, -1):
        with pytest.raises(BenchmarkDatasetCatalogError, match="example_count"):
            BenchmarkManifestSource(
                dataset="qasper",
                split="validation",
                path=Path("examples.jsonl"),
                example_count=invalid_count,  # type: ignore[arg-type]
                license_name="CC BY 4.0",
            )


def test_manifest_example_count_must_match_loaded_records(tmp_path) -> None:
    source_path = tmp_path / "examples.jsonl"
    source_path.write_text(
        json.dumps(
            {
                "example_id": "qasper-count-1",
                "dataset": "qasper",
                "question": "What does the fixture test?",
                "context_chunks": [
                    {
                        "chunk_id": "qasper-count-1:a",
                        "text": "It tests manifest count validation.",
                    }
                ],
                "gold_answers": ["manifest count validation"],
                "gold_evidence_chunk_ids": ["qasper-count-1:a"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "count.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "count-suite",
                "version": "1",
                "description": "bad count",
                "sources": [
                    {
                        "dataset": "qasper",
                        "split": "validation",
                        "path": "examples.jsonl",
                        "example_count": 2,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest = load_benchmark_suite_manifest(manifest_path, base_dir=tmp_path)

    with pytest.raises(BenchmarkDatasetCatalogError, match="example_count"):
        load_manifest_examples(manifest)
