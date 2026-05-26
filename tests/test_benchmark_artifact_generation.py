from __future__ import annotations

import json
from pathlib import Path

import pytest

from egri.benchmarks import ExperimentArtifactGenerationError
from egri.benchmarks import ExperimentArtifactGenerationResult
from egri.benchmarks import generate_local_experiment_artifacts
from egri.benchmarks.experiment_artifacts import main


_REPO_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST_PATH = _REPO_ROOT / "data/benchmark_suites/real_multidataset_mini.json"


def test_generate_local_experiment_artifacts_materializes_runs_and_tables(tmp_path) -> None:
    result = generate_local_experiment_artifacts(
        repo_root=_REPO_ROOT,
        manifest_path=_MANIFEST_PATH,
        output_dir=tmp_path / "artifacts",
    )

    assert isinstance(result, ExperimentArtifactGenerationResult)
    assert result.output_dir == (tmp_path / "artifacts").resolve()
    assert result.materialized_examples_path.name == "real-multidataset-mini.jsonl"
    assert result.offline_run_dir == result.output_dir / "offline_benchmark"
    assert result.ablation_run_dir == result.output_dir / "offline_ablation"
    assert result.paper_tables_dir == result.output_dir / "paper_tables"
    assert result.table_result.row_count == 10
    assert set(result.output_paths) == {
        result.materialized_examples_path,
        result.materialized_metadata_path,
        result.metadata_path,
        result.paper_tables_dir / "experiment_summary.csv",
        result.paper_tables_dir / "experiment_tables.md",
        result.paper_tables_dir / "experiment_tables.tex",
        result.paper_tables_dir / "table_metadata.json",
    }

    top_metadata = json.loads(result.metadata_path.read_text())
    assert top_metadata["run_mode"] == "offline-deterministic-artifact-generation"
    assert top_metadata["deterministic"] is True
    assert top_metadata["live_model_calls"] == 0
    assert top_metadata["network_calls"] == 0
    assert top_metadata["warning"] == (
        "Generated outputs are deterministic fixture artifacts, not live model results."
    )
    assert top_metadata["suite"]["name"] == "real-multidataset-mini"
    assert top_metadata["materialized_subset"]["example_count"] == 8
    assert top_metadata["paper_tables"]["row_count"] == 10

    table_metadata = json.loads(
        (result.paper_tables_dir / "table_metadata.json").read_text()
    )
    assert table_metadata["run_counts"] == {"offline": 1, "live": 0, "ablation": 1}
    assert table_metadata["contains_live_results"] is False
    assert table_metadata["contains_deterministic_only_results"] is True

    materialized_lines = result.materialized_examples_path.read_text().splitlines()
    assert len(materialized_lines) == 8
    assert all(json.loads(line)["dataset"] for line in materialized_lines)


def test_generate_local_experiment_artifacts_rejects_output_inside_data_dir(tmp_path) -> None:
    with pytest.raises(ExperimentArtifactGenerationError, match="generated output"):
        generate_local_experiment_artifacts(
            repo_root=_REPO_ROOT,
            manifest_path=_MANIFEST_PATH,
            output_dir=_REPO_ROOT / "data/generated-artifacts",
        )


def test_generate_local_experiment_artifacts_requires_manifest_under_repo(tmp_path) -> None:
    outside_manifest = tmp_path / "outside.json"
    outside_manifest.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ExperimentArtifactGenerationError, match="manifest"):
        generate_local_experiment_artifacts(
            repo_root=_REPO_ROOT,
            manifest_path=outside_manifest,
            output_dir=tmp_path / "artifacts",
        )


def test_experiment_artifact_generation_cli_writes_outputs(tmp_path) -> None:
    exit_code = main(
        [
            "--repo-root",
            str(_REPO_ROOT),
            "--manifest",
            str(_MANIFEST_PATH),
            "--output-dir",
            str(tmp_path / "cli-artifacts"),
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "cli-artifacts/metadata.json").is_file()
    assert (tmp_path / "cli-artifacts/paper_tables/experiment_summary.csv").is_file()
