#!/usr/bin/env bash
# 03 — Side-by-side vanilla vs EGRI recursive path on the same truncation fixture.
# This optional live command invokes the project-local Ouroboros wrapper and
# persists JSON artifacts under .ouroboros/rlm/{benchmarks,baselines}/ for replay.
set -euo pipefail
exec runtime-scaffold --truncation-benchmark "$@"
