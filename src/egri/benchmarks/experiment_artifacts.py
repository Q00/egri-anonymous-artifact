"""Reproducible local artifact generation for paper-facing benchmark tables."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from egri.benchmarks.ablations import run_offline_ablation_matrix
from egri.benchmarks.dataset_catalog import materialize_manifest_subset
from egri.benchmarks.dataset_catalog import load_benchmark_suite_manifest
from egri.benchmarks.paper_tables import PaperTableResult
from egri.benchmarks.paper_tables import build_paper_experiment_tables
from egri.benchmarks.runner import run_offline_benchmark


class ExperimentArtifactGenerationError(ValueError):
    """Raised when local experiment artifacts cannot be generated safely."""


@dataclass(frozen=True, slots=True)
class ExperimentArtifactGenerationResult:
    """Paths produced by one deterministic local artifact-generation run."""

    output_dir: Path
    materialized_examples_path: Path
    materialized_metadata_path: Path
    offline_run_dir: Path
    ablation_run_dir: Path
    paper_tables_dir: Path
    metadata_path: Path
    table_result: PaperTableResult
    output_paths: tuple[Path, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable artifact-generation summary."""
        return {
            "output_dir": str(self.output_dir),
            "materialized_examples_path": str(self.materialized_examples_path),
            "materialized_metadata_path": str(self.materialized_metadata_path),
            "offline_run_dir": str(self.offline_run_dir),
            "ablation_run_dir": str(self.ablation_run_dir),
            "paper_tables_dir": str(self.paper_tables_dir),
            "metadata_path": str(self.metadata_path),
            "table_result": self.table_result.to_dict(),
            "output_paths": [str(path) for path in self.output_paths],
        }


def generate_local_experiment_artifacts(
    *,
    repo_root: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
) -> ExperimentArtifactGenerationResult:
    """Generate deterministic local benchmark artifacts for paper table drafting.

    This pipeline intentionally performs no network/API/model calls. It consumes
    committed normalized fixtures, materializes a merged benchmark subset, runs
    offline deterministic baselines and ablations, and then builds paper-facing
    CSV/Markdown/LaTeX-ready tables.
    """
    root = Path(repo_root).expanduser().resolve()
    if not root.is_dir():
        msg = f"repo_root must be an existing directory: {root}"
        raise ExperimentArtifactGenerationError(msg)

    manifest = _resolve_under_root(manifest_path, root, label="manifest")
    destination = Path(output_dir).expanduser().resolve()
    _validate_output_dir(destination, root)

    suite = load_benchmark_suite_manifest(manifest, base_dir=root / "data")
    materialized_dir = destination / "materialized_subset"
    offline_run_dir = destination / "offline_benchmark"
    ablation_run_dir = destination / "offline_ablation"
    paper_tables_dir = destination / "paper_tables"
    metadata_path = destination / "metadata.json"
    destination.mkdir(parents=True, exist_ok=True)

    materialized = materialize_manifest_subset(suite, output_dir=materialized_dir)
    offline_result = run_offline_benchmark(
        materialized.examples_path,
        output_dir=offline_run_dir,
    )
    ablation_result = run_offline_ablation_matrix(
        materialized.examples_path,
        output_dir=ablation_run_dir,
    )
    table_result = build_paper_experiment_tables(
        output_dir=paper_tables_dir,
        offline_run_dirs=[offline_result.output_dir],
        ablation_run_dirs=[ablation_result.output_dir],
        suite_manifest_path=manifest,
    )

    metadata = {
        "run_mode": "offline-deterministic-artifact-generation",
        "deterministic": True,
        "live_model_calls": 0,
        "network_calls": 0,
        "warning": "Generated outputs are deterministic fixture artifacts, not live model results.",
        "repo_root": str(root),
        "manifest_path": str(manifest),
        "suite": suite.to_dict(),
        "materialized_subset": materialized.to_dict(),
        "offline_benchmark": offline_result.to_dict(),
        "offline_ablation": ablation_result.to_dict(),
        "paper_tables": table_result.to_dict(),
    }
    metadata_path.write_text(
        json.dumps(metadata, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    output_paths = (
        materialized.examples_path,
        materialized.metadata_path,
        metadata_path,
        *table_result.output_paths,
    )
    return ExperimentArtifactGenerationResult(
        output_dir=destination,
        materialized_examples_path=materialized.examples_path,
        materialized_metadata_path=materialized.metadata_path,
        offline_run_dir=offline_result.output_dir,
        ablation_run_dir=ablation_result.output_dir,
        paper_tables_dir=table_result.output_dir,
        metadata_path=metadata_path,
        table_result=table_result,
        output_paths=output_paths,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for deterministic local artifact generation."""
    parser = argparse.ArgumentParser(
        description="Generate deterministic local benchmark artifacts for EGRI paper tables."
    )
    parser.add_argument("--repo-root", default=Path.cwd())
    parser.add_argument(
        "--manifest",
        default="data/benchmark_suites/real_multidataset_mini.json",
        help="Path to a committed benchmark suite manifest under repo root.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/experiments/real_multidataset_mini_offline",
        help="Generated artifact output directory; should be ignored by git.",
    )
    args = parser.parse_args(argv)
    result = generate_local_experiment_artifacts(
        repo_root=args.repo_root,
        manifest_path=_path_under_cli_root(args.manifest, args.repo_root),
        output_dir=_path_under_cli_root(args.output_dir, args.repo_root),
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


def _path_under_cli_root(path: str | Path, repo_root: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(repo_root).expanduser().resolve() / candidate


def _resolve_under_root(path: str | Path, root: Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        msg = f"{label} must be under repo_root: {resolved}"
        raise ExperimentArtifactGenerationError(msg) from exc
    if not resolved.is_file():
        msg = f"{label} must exist: {resolved}"
        raise ExperimentArtifactGenerationError(msg)
    return resolved


def _validate_output_dir(output_dir: Path, root: Path) -> None:
    try:
        relative = output_dir.relative_to(root)
    except ValueError:
        return
    if not relative.parts:
        msg = "generated output directory must not be the repository root"
        raise ExperimentArtifactGenerationError(msg)
    if relative.parts[0] != "artifacts":
        msg = "generated output directory must live under the ignored artifacts/ directory"
        raise ExperimentArtifactGenerationError(msg)
