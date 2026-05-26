# Artifact Manifest

This manifest maps the manuscript's evidence families to files in the anonymous artifact. It intentionally separates **live evidence**, **deterministic contract replay**, and **optional live-provider reruns** so reviewers can validate the central runtime-control claim without credentials.

Terminology follows the paper:

- **EGRI** = recursive substrate plus TG/SCG enforcement.
- **EGRI substrate w/o TG** = same substrate with TG/SCG disabled, used only as an ablation.
- **GOM** = guarded operational memory; policy/retry priors only, never admissible factual evidence.

## Paper-facing mapping

| Paper evidence family | Artifact files / commands | Evidence type | Live/network calls |
| --- | --- | --- | --- |
| TG enforcement demo | `scripts/run-traceguard-demo.py`; `experiments/traceguard-demo.*` | deterministic replay | no |
| Claim-aware scorer / omitted-fact safety | `scripts/run-claim-aware-suite.py`; `scripts/run-synthetic-omitted-fact-benchmark.py` | deterministic scorer tests | no |
| Vanilla live truncation comparison | checked-in outputs: `benchmarks/vanilla-baseline-rlm-long-context-truncation-v1.json`; `benchmarks/rlm-long-context-truncation-v1.json`; replay: `python -m egri.replay benchmarks/rlm-long-context-truncation-v1.json` | checked-in live output + deterministic replay | replay: no |
| Live portability primary matrix | checked-in contracts-only replay: `experiments/egri-without-tg-baseline/live-portability-contracts-only-rerun.*`; optional rerun: `scripts/run-live-portability-matrix.py` | checked-in replay + optional live | replay: no; optional live: yes |
| Gate-ablation controls on the same eight primary fixtures | `experiments/egri-without-tg-baseline/egri-without-traceguard-safe-control.*`; `experiments/egri-without-tg-baseline/egri-without-traceguard-baseline.*` | deterministic replay | no |
| Unsupported-claim-rate contract ablation | `scripts/run-unsupported-claim-rate-benchmark.py` | deterministic contract benchmark | no |
| Atomic task-boundary / structured claim guard | `tests/test_atomic_task_boundary_benchmark.py`; `scripts/run-live-atomic-task-boundary-pilot.py` for optional pilot | deterministic test + optional pilot | test: no; pilot: yes |
| Context-budget pressure | `python -m egri.benchmarks.context_budget --output-dir experiments/context-budget-pressure` | deterministic benchmark | no |
| Additional submission-triage artifacts | `python -m egri.benchmarks.emnlp_additional --output-dir experiments/emnlp-additional --examples-per-dataset 20` | deterministic benchmark | no |
| GOM / memory-policy controls | `scripts/run-memory-runtime-benefit-benchmark.py`; `scripts/run-memory-contribution-benchmarks.py`; `experiments/memory-runtime-benefit-benchmark.*` | deterministic benchmark | no |
| Memory-enabled live portability matrix | checked-in output: `experiments/live-portability-primary-memory-fixed.*`; memory store: `experiments/live-portability-memory-primary-fixed.jsonl`; optional rerun: `scripts/run-live-portability-matrix.py --memory-mode read-write` | checked-in live output + optional live rerun | output: no; optional rerun: yes |
| In-process `runtime-scaffold` TG gate | checked-in demo/input/output: `experiments/paper-key-sections-runtime-scaffold-demo.*`; `experiments/runtime-scaffold-inprocess-traceguard-gate.*`; `experiments/paper-runtime-scaffold-traceguard-gate.*`; rerun scripts: `scripts/run-runtime-scaffold-inprocess-traceguard-experiment.py`; `scripts/run-paper-runtime-scaffold-traceguard-gate.py` | deterministic replay / wrapper smoke over persisted run | no |
| Runtime-scaffold branch snapshot | `src/egri/_vendor/runtime_scaffold_branch/`; fixture copy: `tests/fixtures/rlm/` | anonymized vendored source for wrapper smoke | no |

## Required deterministic scripts

### TraceGuard enforcement demo

- Script: `scripts/run-traceguard-demo.py`
- Purpose: demonstrates acceptance of manifest-backed parent synthesis and rejection of omitted-fact or chunk-only evidence claims.
- Live model calls: no.
- Network calls: no.

### Claim-aware scorer suite

- Script: `scripts/run-claim-aware-suite.py`
- Purpose: validates controlled completion shapes for retained facts, omitted facts, residual-gap caveats, evidence references, and truncation-boundary reporting.
- Live model calls: no.
- Network calls: no.

### Synthetic omitted-fact benchmark

- Script: `scripts/run-synthetic-omitted-fact-benchmark.py`
- Purpose: generated scorer sanity checks over varying fact count, retained/omitted ratio, distractor density, and omitted-claim targets.
- Live model calls: no.
- Network calls: no.

### Unsupported-claim-rate contract ablation

- Script: `scripts/run-unsupported-claim-rate-benchmark.py`
- Purpose: compares loose single call, guarded single call, chunk-only map-reduce, leaky map-reduce, EGRI, and EGRI substrate w/o TG.
- Live model calls: no.
- Network calls: no.

### Memory-policy runtime benchmarks

- Scripts:
  - `scripts/run-memory-runtime-benefit-benchmark.py`
  - `scripts/run-memory-contribution-benchmarks.py`
- Purpose: tests operational-memory priors, answer-memory contamination rejection, layered memory ablation, and adaptive repair memory.
- Live model calls: no.
- Network calls: no.

### Memory-enabled live portability output

- Checked-in output: `experiments/live-portability-primary-memory-fixed.json`
- Human-readable summary: `experiments/live-portability-primary-memory-fixed.md`
- Memory store: `experiments/live-portability-memory-primary-fixed.jsonl`
- Purpose: records the memory-enabled 24-cell live portability run with read-write operational memory enabled.
- Live model calls for checked-in output inspection: no.
- Live model calls for optional rerun: yes.

### In-process runtime-scaffold TraceGuard gate

- Persisted demo input: `experiments/paper-key-sections-runtime-scaffold-demo.json`
- Scripts:
  - `scripts/run-runtime-scaffold-inprocess-traceguard-experiment.py`
  - `scripts/run-paper-runtime-scaffold-traceguard-gate.py`
- Checked-in outputs:
  - `experiments/runtime-scaffold-inprocess-traceguard-gate.*`
  - `experiments/paper-runtime-scaffold-traceguard-gate.*`
- Vendored source: `src/egri/_vendor/runtime_scaffold_branch/`
- Purpose: validates the persisted paper `runtime-scaffold` run through the project-local TraceGuard adapter and shows rejection of bad child handles and memory-answer contamination. The vendored branch snapshot supplies the parent-synthesis command hook used by the wrapper smoke, so the deterministic rerun should report `patch_installed=true`.
- Live model calls: no.
- Network calls: no.

### Context-budget pressure benchmark

- Command: `python -m egri.benchmarks.context_budget --output-dir experiments/context-budget-pressure`
- Purpose: regenerates the fixed-budget supported-recall and unsupported-commit results.
- Live model calls: no.
- Network calls: no.

### Additional submission-triage artifacts

- Command: `python -m egri.benchmarks.emnlp_additional --output-dir experiments/emnlp-additional --examples-per-dataset 20`
- Purpose: regenerates scaled normalized dataset-family fixtures, post-hoc versus pre-commit comparison, and normalized overhead outputs.
- Live model calls: no.
- Network calls: no.

## Optional live portability matrix

- Script: `scripts/run-live-portability-matrix.py`
- Purpose: reruns the live runtime-family matrix when provider credentials and local CLIs are available.
- Live model calls: yes.
- Network calls: yes.
- Required for anonymous deterministic validation: no.

The checked-in contracts-only replay records the same eight primary fixture classes and adapter-family metadata without requiring reviewers to run provider-backed calls.
