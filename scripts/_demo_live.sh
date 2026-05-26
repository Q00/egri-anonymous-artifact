#!/usr/bin/env bash
# Live-mode demo script. Spawns real Hermes via HermesCliRuntime.
set -euo pipefail

say() { printf '\n\033[1;36m$ %s\033[0m\n' "$1"; sleep 1.0; }

if ! command -v hermes >/dev/null 2>&1; then
  echo "error: hermes CLI not installed. See docs/hermes-setup.md" >&2
  exit 1
fi

if ! command -v runtime-scaffold >/dev/null 2>&1; then
  echo "error: runtime-scaffold CLI not installed. Run: pip install -e ." >&2
  exit 1
fi

say "# EGRI — live truncation benchmark (real Hermes, ~60s)"
sleep 0.5

say "hermes --version"
hermes --version
sleep 1.0

say "runtime-scaffold --truncation-benchmark"
runtime-scaffold --truncation-benchmark
sleep 2.0

say "# Corrected claim-aware scoring should report a tie on the committed fixture."
sleep 1.5
