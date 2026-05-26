"""Anonymous EGRI reproducibility package.

The deterministic artifact modules (TraceGuard, memory-policy benchmarks, and
context-budget benchmarks) run without live provider credentials. Optional live
integration helpers import the external runtime only when those paths are used.
"""

from pathlib import Path
import sys


def _activate_vendored_runtime_scaffold_branch() -> None:
    """Prefer the anonymized runtime-scaffold branch snapshot when present."""
    vendor_src = (
        Path(__file__).resolve().parent
        / "_vendor"
        / "runtime_scaffold_branch"
        / "src"
    )
    if vendor_src.exists():
        vendor = str(vendor_src)
        if vendor not in sys.path:
            sys.path.insert(0, vendor)


_activate_vendored_runtime_scaffold_branch()

from egri.memory import (  # noqa: F401
    LocalJsonMemoryBackend,
    MemoryBackend,
    MemoryObservation,
    MemoryPrior,
    NoopMemoryBackend,
    validate_memory_record,
)
from egri.traceguard import (  # noqa: F401
    TraceGuardClaim,
    TraceGuardEvidence,
    TraceGuardRejection,
    TraceGuardResult,
    build_manifest_from_fixture,
    extract_parent_claims,
    normalize_allowed_evidence_manifest,
    validate_parent_synthesis,
)

try:  # Optional live/runtime integration dependency.
    from egri.runtime_scaffold_traceguard import (  # noqa: F401
        RuntimeScaffoldTraceGuardGate,
        install_runtime_scaffold_cli_gate,
        validate_runtime_scaffold_result,
    )
except ModuleNotFoundError:  # pragma: no cover - optional live integration path.
    RuntimeScaffoldTraceGuardGate = None  # type: ignore[assignment]
    install_runtime_scaffold_cli_gate = None  # type: ignore[assignment]
    validate_runtime_scaffold_result = None  # type: ignore[assignment]

try:  # Optional live/runtime integration dependency.
    from ouroboros.orchestrator.hermes_runtime import HermesCliRuntime  # noqa: F401
    from ouroboros.rlm import (  # noqa: F401
        MAX_RLM_AC_TREE_DEPTH,
        MAX_RLM_AMBIGUITY_THRESHOLD,
        RLM_MVP_SRC_DOGFOOD_BENCHMARK_ID,
        RLMRunConfig,
        RLMRunResult,
        RLMSharedTruncationBenchmarkConfig,
        RLMSharedTruncationBenchmarkResult,
        RLMTraceStore,
        RLMVanillaTruncationBaselineConfig,
        run_rlm_benchmark,
        run_rlm_loop,
        run_shared_truncation_benchmark,
        run_vanilla_truncation_baseline,
    )
except ModuleNotFoundError:  # pragma: no cover - optional live integration path.
    HermesCliRuntime = None  # type: ignore[assignment]
    MAX_RLM_AC_TREE_DEPTH = None  # type: ignore[assignment]
    MAX_RLM_AMBIGUITY_THRESHOLD = None  # type: ignore[assignment]
    RLM_MVP_SRC_DOGFOOD_BENCHMARK_ID = None  # type: ignore[assignment]
    RLMRunConfig = None  # type: ignore[assignment]
    RLMRunResult = None  # type: ignore[assignment]
    RLMSharedTruncationBenchmarkConfig = None  # type: ignore[assignment]
    RLMSharedTruncationBenchmarkResult = None  # type: ignore[assignment]
    RLMTraceStore = None  # type: ignore[assignment]
    RLMVanillaTruncationBaselineConfig = None  # type: ignore[assignment]
    run_rlm_benchmark = None  # type: ignore[assignment]
    run_rlm_loop = None  # type: ignore[assignment]
    run_shared_truncation_benchmark = None  # type: ignore[assignment]
    run_vanilla_truncation_baseline = None  # type: ignore[assignment]

__all__ = [
    "HermesCliRuntime",
    "LocalJsonMemoryBackend",
    "MAX_RLM_AC_TREE_DEPTH",
    "MAX_RLM_AMBIGUITY_THRESHOLD",
    "MemoryBackend",
    "MemoryObservation",
    "MemoryPrior",
    "NoopMemoryBackend",
    "RuntimeScaffoldTraceGuardGate",
    "RLM_MVP_SRC_DOGFOOD_BENCHMARK_ID",
    "RLMRunConfig",
    "RLMRunResult",
    "RLMSharedTruncationBenchmarkConfig",
    "RLMSharedTruncationBenchmarkResult",
    "RLMTraceStore",
    "RLMVanillaTruncationBaselineConfig",
    "run_rlm_benchmark",
    "run_rlm_loop",
    "run_shared_truncation_benchmark",
    "run_vanilla_truncation_baseline",
    "TraceGuardClaim",
    "TraceGuardEvidence",
    "TraceGuardRejection",
    "TraceGuardResult",
    "build_manifest_from_fixture",
    "extract_parent_claims",
    "install_runtime_scaffold_cli_gate",
    "normalize_allowed_evidence_manifest",
    "validate_runtime_scaffold_result",
    "validate_memory_record",
    "validate_parent_synthesis",
]

__version__ = "0.1.0"
