# EGRI Anonymous Reproducibility Artifact

This archive contains the anonymized reproducibility material for **EGRI: Evidence-Gated Recursive Inference as an Agent Runtime Contract**.

The artifact is aligned to the submission manuscript's framing: EGRI is a runtime contract, not a model-quality benchmark. The checked-in material therefore emphasizes deterministic replay, contract-level verification, and optional live portability evidence. In the paper terminology, **EGRI** means the recursive substrate plus TG/SCG enforcement; **EGRI substrate w/o TG** means the same substrate with the enforcement stack disabled for ablation.

## What reviewers can validate without credentials

The required artifact path uses no live model calls and no network access after dependencies are installed. It validates:

- TraceGuard/TG acceptance of manifest-backed parent synthesis and rejection of unsupported, chunk-only, or memory-only support.
- Claim-aware scoring for retained facts, omitted facts, residual-gap caveats, evidence references, and truncation-boundary reporting.
- Deterministic gate-ablation controls on the same eight primary fixture classes used by the live portability matrix.
- Context-budget pressure results showing supported-fact recall and unsupported-commit behavior under a fixed parent budget.
- Guarded operational-memory/GOM behavior as schema and repair priors, not admissible factual evidence.
- Post-hoc detection versus pre-commit blocking and normalized overhead tables used for submission triage.

Optional live portability scripts are included for transparency, but they require external provider credentials and are **not** required to validate the deterministic anonymous artifact.

## Contents

- `src/egri/`: EGRI package code, TraceGuard/TG, GOM, replay helpers, and benchmark modules.
- `scripts/`: deterministic benchmark/replay entrypoints and optional live-provider scripts.
- `benchmarks/`, `data/`, `examples/`: checked-in fixtures and small normalized examples.
- `experiments/`: checked-in deterministic and contracts-only outputs used as paper-facing evidence.
- `tests/`: smoke and regression tests for deterministic components.
- `src/egri/_vendor/runtime_scaffold_branch/`: anonymized runtime-scaffold branch snapshot used only to reproduce the in-process parent-synthesis gate without an external checkout.
- `ARTIFACT_MANIFEST.md`: mapping from manuscript tables/claims to scripts and outputs.
- `REPRODUCIBILITY.md`: exact commands for required deterministic runs and optional live runs.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
```

Then run the required deterministic checks:

```bash
python3 scripts/run-traceguard-demo.py
python3 scripts/run-claim-aware-suite.py
python3 scripts/run-synthetic-omitted-fact-benchmark.py
python3 scripts/run-unsupported-claim-rate-benchmark.py
python3 scripts/run-memory-runtime-benefit-benchmark.py
python3 scripts/run-memory-contribution-benchmarks.py
python3 scripts/run-runtime-scaffold-inprocess-traceguard-experiment.py
python3 scripts/run-paper-runtime-scaffold-traceguard-gate.py
python -m egri.benchmarks.context_budget --output-dir experiments/context-budget-pressure
python -m egri.benchmarks.emnlp_additional --output-dir experiments/emnlp-additional --examples-per-dataset 20
```

If `uv` is available, the same commands can be run under `uv run` after `uv sync --extra dev`.

## Anonymous-review note

Repository URLs, authorship metadata, local paths, and account-specific live-provider configuration have been removed or replaced for anonymous review. Provider names used as runtime-family labels are retained only where they define the experimental adapter family. Public repository URLs and full provenance links can be restored after the review period.
