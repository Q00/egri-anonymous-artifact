"""Benchmark harness primitives for real EGRI evaluations."""

from egri.benchmarks.ablations import AblationCondition
from egri.benchmarks.ablations import AblationMatrixError
from egri.benchmarks.ablations import AblationMatrixResult
from egri.benchmarks.ablations import iter_offline_ablation_conditions
from egri.benchmarks.ablations import run_offline_ablation_matrix
from egri.benchmarks.atomic_task_boundary import AtomicTaskBoundaryCondition
from egri.benchmarks.atomic_task_boundary import AtomicTaskBoundaryError
from egri.benchmarks.atomic_task_boundary import AtomicTaskBoundaryResult
from egri.benchmarks.atomic_task_boundary import iter_atomic_task_boundary_conditions
from egri.benchmarks.atomic_task_boundary import run_atomic_task_boundary_benchmark
from egri.benchmarks.context_budget import ContextBudgetCondition
from egri.benchmarks.context_budget import ContextBudgetError
from egri.benchmarks.context_budget import ContextBudgetResult
from egri.benchmarks.context_budget import iter_context_budget_conditions
from egri.benchmarks.context_budget import run_context_budget_benchmark
from egri.benchmarks.dataset_catalog import BenchmarkDatasetCatalogError
from egri.benchmarks.dataset_catalog import BenchmarkDatasetSpec
from egri.benchmarks.dataset_catalog import BenchmarkManifestSource
from egri.benchmarks.dataset_catalog import BenchmarkSuiteManifest
from egri.benchmarks.dataset_catalog import MaterializedBenchmarkSubset
from egri.benchmarks.dataset_catalog import get_dataset_spec
from egri.benchmarks.dataset_catalog import iter_dataset_specs
from egri.benchmarks.dataset_catalog import load_benchmark_suite_manifest
from egri.benchmarks.dataset_catalog import load_manifest_examples
from egri.benchmarks.dataset_catalog import materialize_manifest_subset
from egri.benchmarks.evaluator import BenchmarkEvaluation
from egri.benchmarks.evaluator import EvaluationValidationError
from egri.benchmarks.evaluator import PolicyAggregateSummary
from egri.benchmarks.evaluator import aggregate_evaluations
from egri.benchmarks.evaluator import evaluate_policy_result
from egri.benchmarks.experiment_artifacts import ExperimentArtifactGenerationError
from egri.benchmarks.experiment_artifacts import ExperimentArtifactGenerationResult
from egri.benchmarks.experiment_artifacts import generate_local_experiment_artifacts
from egri.benchmarks.jsonl_loader import load_jsonl_examples
from egri.benchmarks.live_adapters import LiveModelResponse
from egri.benchmarks.live_adapters import LiveRecursiveTraceGuardPolicy
from egri.benchmarks.live_adapters import ModelClient
from egri.benchmarks.live_runner import LiveBenchmarkRunnerError
from egri.benchmarks.live_runner import LiveBenchmarkRunResult
from egri.benchmarks.live_runner import run_live_benchmark
from egri.benchmarks.paper_tables import PaperTableError
from egri.benchmarks.paper_tables import PaperTableResult
from egri.benchmarks.paper_tables import build_paper_experiment_tables
from egri.benchmarks.policies import BenchmarkPolicy
from egri.benchmarks.policies import PolicyOutputValidationError
from egri.benchmarks.policies import PolicyResult
from egri.benchmarks.policies import get_baseline_policy
from egri.benchmarks.policies import iter_baseline_policies
from egri.benchmarks.policies import normalize_cited_chunk_ids
from egri.benchmarks.runner import BenchmarkRunnerError
from egri.benchmarks.runner import BenchmarkRunResult
from egri.benchmarks.runner import run_offline_benchmark
from egri.benchmarks.schema import BenchmarkExample
from egri.benchmarks.schema import BenchmarkValidationError
from egri.benchmarks.schema import ContextChunk

__all__ = [
    "AblationCondition",
    "AblationMatrixError",
    "AblationMatrixResult",
    "AtomicTaskBoundaryCondition",
    "AtomicTaskBoundaryError",
    "AtomicTaskBoundaryResult",
    "BenchmarkDatasetCatalogError",
    "BenchmarkDatasetSpec",
    "BenchmarkEvaluation",
    "BenchmarkExample",
    "BenchmarkManifestSource",
    "BenchmarkPolicy",
    "BenchmarkRunResult",
    "BenchmarkRunnerError",
    "BenchmarkSuiteManifest",
    "BenchmarkValidationError",
    "ContextChunk",
    "ContextBudgetCondition",
    "ContextBudgetError",
    "ContextBudgetResult",
    "EmnlpAdditionalExperimentError",
    "EmnlpAdditionalExperimentResult",
    "EvaluationValidationError",
    "ExperimentArtifactGenerationError",
    "ExperimentArtifactGenerationResult",
    "LiveBenchmarkRunnerError",
    "LiveBenchmarkRunResult",
    "LiveModelResponse",
    "LiveRecursiveTraceGuardPolicy",
    "MaterializedBenchmarkSubset",
    "ModelClient",
    "PaperTableError",
    "PaperTableResult",
    "PolicyAggregateSummary",
    "PolicyOutputValidationError",
    "PolicyResult",
    "aggregate_evaluations",
    "build_paper_experiment_tables",
    "evaluate_policy_result",
    "get_baseline_policy",
    "get_dataset_spec",
    "generate_emnlp_additional_experiments",
    "generate_local_experiment_artifacts",
    "iter_atomic_task_boundary_conditions",
    "iter_baseline_policies",
    "iter_context_budget_conditions",
    "iter_dataset_specs",
    "iter_offline_ablation_conditions",
    "load_benchmark_suite_manifest",
    "load_jsonl_examples",
    "load_manifest_examples",
    "materialize_manifest_subset",
    "normalize_cited_chunk_ids",
    "run_atomic_task_boundary_benchmark",
    "run_context_budget_benchmark",
    "run_live_benchmark",
    "run_offline_ablation_matrix",
    "run_offline_benchmark",
]


def __getattr__(name: str):  # pragma: no cover - exercised by import machinery
    """Lazily expose optional paper-facing generators without -m runpy warnings."""

    if name in {
        "EmnlpAdditionalExperimentError",
        "EmnlpAdditionalExperimentResult",
        "generate_emnlp_additional_experiments",
    }:
        from egri.benchmarks import emnlp_additional

        return getattr(emnlp_additional, name)
    raise AttributeError(name)
