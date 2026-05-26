# Reproducibility Guide

This artifact separates deterministic contract-level reproduction from optional live-provider runs. The deterministic path is the review-critical path: it validates the manuscript's runtime-control claims without provider credentials.

## Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[dev]'
```

If `uv` is available, reviewers may alternatively run:

```bash
uv sync --extra dev
```

## Required deterministic runs

These commands are intended to run without provider credentials and without network access after package dependencies are installed.

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

Expected high-level behavior:

- TG accepts manifest-backed parent claims and rejects unsupported or chunk-only claims.
- Claim-aware scorer suites pass all controlled cases.
- EGRI prevents unsupported commits under the contract, while EGRI substrate w/o TG commits unsafe parent syntheses in the gate-ablation replay.
- GOM benchmarks show operational-memory priors reduce repair work while answer-memory contamination is rejected.
- The persisted `runtime-scaffold` paper run is accepted after handle repair, while bad child handles and memory-answer facts are rejected before parent state admission. The anonymized runtime-scaffold branch snapshot bundled under `src/egri/_vendor/` makes the wrapper-smoke result report `patch_installed=true` without an external checkout.
- Context-budget pressure outputs show recursive evidence caching increases supported-fact recall, and TG blocks unsupported parent commits.
- Post-hoc checker comparisons show detection after commit is different from pre-commit blocking.

## Checked-in outputs

The artifact includes a minimal checked-in `experiments/` directory with paper-facing deterministic outputs and contracts-only replay files. Reviewers may regenerate these files in place, or write to a separate output directory if they want to preserve the checked-in copies.

Important checked-in files:

- `experiments/traceguard-demo.*`
- `experiments/memory-runtime-benefit-benchmark.*`
- `experiments/live-portability-primary-memory-fixed.*`
- `experiments/live-portability-memory-primary-fixed.jsonl`
- `experiments/paper-key-sections-runtime-scaffold-demo.*`
- `experiments/runtime-scaffold-inprocess-traceguard-gate.*`
- `experiments/paper-runtime-scaffold-traceguard-gate.*`
- `benchmarks/vanilla-baseline-rlm-long-context-truncation-v1.json`
- `benchmarks/rlm-long-context-truncation-v1.json`
- `experiments/egri-without-tg-baseline/egri-without-traceguard-safe-control.*`
- `experiments/egri-without-tg-baseline/egri-without-traceguard-baseline.*`
- `experiments/egri-without-tg-baseline/live-portability-contracts-only-rerun.*`

## Optional live-provider runs

The live portability scripts exercise the same contract through external agent/runtime providers. They are included for transparency but are not necessary for anonymous artifact validation.

```bash
python3 scripts/run-live-portability-matrix.py --help
```

A live run requires credentials and locally configured provider CLIs. Account-specific command paths and credentials are intentionally omitted from this anonymous archive and replaced with portable placeholders in checked-in replay metadata.

## Output locations

Regenerated outputs should be written under `experiments/`. The artifact's `.gitignore` excludes virtual environments, caches, build products, and runtime scratch directories.
