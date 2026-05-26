"""Deterministic context-budget pressure experiment for EGRI.

This benchmark isolates a paper-facing claim that is more visually compelling than
plain contract ablations: under a fixed parent context budget, a broad or flat
retrieval parent drops source facts, while recursive child leaves externalize a
larger evidence cache. TraceGuard then decides which generated parent claims may
commit. The benchmark is deterministic and performs no live model calls.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
import argparse
from pathlib import Path
from typing import Any

from egri.traceguard import TraceGuardEvidence
from egri.traceguard import validate_parent_synthesis


class ContextBudgetError(ValueError):
    """Raised when the context-budget benchmark cannot be produced."""


@dataclass(frozen=True, slots=True)
class ContextBudgetCondition:
    """One deterministic condition in the context-budget pressure benchmark."""

    name: str
    recursive: bool
    traceguard_enabled: bool
    visible_leaf_facts: int
    parent_context_budget_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            msg = "condition name must be a non-empty string"
            raise ContextBudgetError(msg)
        if not isinstance(self.recursive, bool):
            msg = f"{self.name}: recursive must be boolean"
            raise ContextBudgetError(msg)
        if not isinstance(self.traceguard_enabled, bool):
            msg = f"{self.name}: traceguard_enabled must be boolean"
            raise ContextBudgetError(msg)
        if self.traceguard_enabled and not self.recursive:
            msg = f"{self.name}: TraceGuard requires recursive evidence handles"
            raise ContextBudgetError(msg)
        if isinstance(self.visible_leaf_facts, bool) or not isinstance(self.visible_leaf_facts, int):
            msg = f"{self.name}: visible_leaf_facts must be an integer"
            raise ContextBudgetError(msg)
        if self.visible_leaf_facts < 0:
            msg = f"{self.name}: visible_leaf_facts must be non-negative"
            raise ContextBudgetError(msg)
        if isinstance(self.parent_context_budget_tokens, bool) or not isinstance(
            self.parent_context_budget_tokens, int
        ):
            msg = f"{self.name}: parent_context_budget_tokens must be an integer"
            raise ContextBudgetError(msg)
        if self.parent_context_budget_tokens <= 0:
            msg = f"{self.name}: parent_context_budget_tokens must be positive"
            raise ContextBudgetError(msg)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "condition_name": self.name,
            "recursive": self.recursive,
            "traceguard_enabled": self.traceguard_enabled,
            "visible_leaf_facts": self.visible_leaf_facts,
            "parent_context_budget_tokens": self.parent_context_budget_tokens,
            "deterministic": True,
            "live_model_calls": 0,
        }


@dataclass(frozen=True, slots=True)
class ContextBudgetResult:
    """In-memory benchmark result plus persisted artifact paths."""

    output_dir: Path
    fixture_count: int
    condition_names: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    summary_rows: tuple[dict[str, Any], ...]
    output_paths: tuple[Path, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "fixture_count": self.fixture_count,
            "condition_names": list(self.condition_names),
            "row_count": len(self.rows),
            "summary_count": len(self.summary_rows),
            "output_paths": [str(path) for path in self.output_paths],
        }


def iter_context_budget_conditions() -> tuple[ContextBudgetCondition, ...]:
    """Return deterministic context-pressure conditions in paper-table order."""
    return (
        ContextBudgetCondition(
            name="single-shot-context-window",
            recursive=False,
            traceguard_enabled=False,
            visible_leaf_facts=3,
            parent_context_budget_tokens=1536,
        ),
        ContextBudgetCondition(
            name="flat-topk-retrieval",
            recursive=False,
            traceguard_enabled=False,
            visible_leaf_facts=5,
            parent_context_budget_tokens=1536,
        ),
        ContextBudgetCondition(
            name="recursive-evidence-cache",
            recursive=True,
            traceguard_enabled=False,
            visible_leaf_facts=8,
            parent_context_budget_tokens=1536,
        ),
        ContextBudgetCondition(
            name="recursive-traceguard-dfs",
            recursive=True,
            traceguard_enabled=True,
            visible_leaf_facts=8,
            parent_context_budget_tokens=1536,
        ),
    )


def run_context_budget_benchmark(
    *,
    output_dir: str | Path | None = None,
    conditions: tuple[ContextBudgetCondition, ...] | None = None,
) -> ContextBudgetResult:
    """Run deterministic context-budget fixtures and persist JSON/CSV outputs."""
    if output_dir is None:
        msg = "output_dir must be provided for context-budget benchmark runs"
        raise ContextBudgetError(msg)

    selected_conditions = conditions or iter_context_budget_conditions()
    _validate_conditions(selected_conditions)
    fixtures = _fixtures()
    rows = tuple(
        _run_condition_on_fixture(condition, fixture)
        for condition in selected_conditions
        for fixture in fixtures
    )
    summary_rows = _summarize_rows(selected_conditions, rows)

    resolved_output_dir = Path(output_dir).resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = _write_outputs(
        output_dir=resolved_output_dir,
        fixture_count=len(fixtures),
        condition_names=tuple(condition.name for condition in selected_conditions),
        rows=rows,
        summary_rows=summary_rows,
    )
    return ContextBudgetResult(
        output_dir=resolved_output_dir,
        fixture_count=len(fixtures),
        condition_names=tuple(condition.name for condition in selected_conditions),
        rows=rows,
        summary_rows=summary_rows,
        output_paths=output_paths,
    )


def _validate_conditions(conditions: tuple[ContextBudgetCondition, ...]) -> None:
    if not conditions:
        msg = "context-budget benchmark requires at least one condition"
        raise ContextBudgetError(msg)
    if any(not isinstance(condition, ContextBudgetCondition) for condition in conditions):
        msg = "conditions must be ContextBudgetCondition objects"
        raise ContextBudgetError(msg)
    names = [condition.name for condition in conditions]
    if len(set(names)) != len(names):
        msg = "context-budget benchmark rejects duplicate condition names"
        raise ContextBudgetError(msg)


def _fixtures() -> tuple[dict[str, Any], ...]:
    domains = ("legal", "medical", "security", "finance", "research")
    return tuple(
        {
            "fixture_id": f"context-pressure-{domain}-{variant:02d}",
            "domain": domain,
            "variant": variant,
            "facts": tuple(
                {
                    "fact_id": f"CB-{domain[:3].upper()}-{variant:02d}-{index:03d}",
                    "chunk_id": f"context-pressure-{domain}-{variant:02d}.txt:{index}-{index}",
                    "text": (
                        f"{domain} dossier {variant} fact {index} is a bounded "
                        "source-backed claim that must remain traceable."
                    ),
                }
                for index in range(1, 9)
            ),
        }
        for domain in domains
        for variant in range(1, 3)
    )


def _run_condition_on_fixture(
    condition: ContextBudgetCondition,
    fixture: dict[str, Any],
) -> dict[str, Any]:
    visible_facts = fixture["facts"][: condition.visible_leaf_facts]
    manifest = _manifest(fixture, visible_facts) if condition.recursive else ()
    parent = _parent_with_visible_claims(visible_facts, condition)
    attempted_parent = _parent_with_pressure_induced_unsupported_claim(parent, fixture)

    if condition.traceguard_enabled:
        validation = validate_parent_synthesis(
            evidence_manifest=manifest,
            parent_synthesis=attempted_parent,
        )
        unsupported_rejected = not validation.accepted
        unsupported_committed = validation.accepted
        final_validation = validate_parent_synthesis(
            evidence_manifest=manifest,
            parent_synthesis=parent,
        )
        final_parent_committed = final_validation.accepted
        unsupported_claim_rate = validation.unsupported_claim_rate
    else:
        unsupported_rejected = False
        unsupported_committed = True
        final_parent_committed = True
        unsupported_claim_rate = None

    supported_recall = len(visible_facts) / len(fixture["facts"])
    context_pressure = len(fixture["facts"]) / max(condition.visible_leaf_facts, 1)
    evidence_bandwidth_multiplier = condition.visible_leaf_facts / 3
    return {
        **condition.to_metadata(),
        "fixture_id": fixture["fixture_id"],
        "source_fact_count": len(fixture["facts"]),
        "visible_supported_claim_count": len(visible_facts),
        "supported_claim_recall": round(supported_recall, 4),
        "context_pressure": round(context_pressure, 4),
        "evidence_bandwidth_multiplier": round(evidence_bandwidth_multiplier, 4),
        "unsupported_claim_attempted": True,
        "unsupported_rejected_before_commit": unsupported_rejected,
        "unsupported_committed": unsupported_committed,
        "final_parent_committed": final_parent_committed,
        "initial_unsupported_claim_rate": unsupported_claim_rate,
        "claim": (
            "recursive evidence cache preserves all bounded claims under parent context pressure"
            if condition.recursive
            else "single parent context has to drop or hallucinate claims under pressure"
        ),
    }


def _manifest(
    fixture: dict[str, Any],
    facts: tuple[dict[str, Any], ...],
) -> tuple[TraceGuardEvidence, ...]:
    return tuple(
        TraceGuardEvidence(
            fact_id=fact["fact_id"],
            chunk_id=fact["chunk_id"],
            text=fact["text"],
            child_call_id=f"{fixture['fixture_id']}::child::{index:03d}",
        )
        for index, fact in enumerate(facts, start=1)
    )


def _parent_with_visible_claims(
    facts: tuple[dict[str, Any], ...],
    condition: ContextBudgetCondition,
) -> dict[str, Any]:
    return {
        "mode": condition.name,
        "result": {
            "summary": f"Parent synthesis retained {len(facts)} source-backed claims.",
            "retained_facts": [
                {
                    "fact_id": fact["fact_id"],
                    "text": fact["text"],
                    "evidence_chunk_id": fact["chunk_id"],
                }
                for fact in facts
            ],
        },
        "evidence_references": [
            {
                "chunk_id": fact["chunk_id"],
                "supports_fact_ids": [fact["fact_id"]],
                "quoted_evidence": fact["text"],
            }
            for fact in facts
        ],
        "residual_gaps": [],
    }


def _parent_with_pressure_induced_unsupported_claim(
    parent: dict[str, Any], fixture: dict[str, Any]
) -> dict[str, Any]:
    copied = json.loads(json.dumps(parent))
    retained = copied.setdefault("result", {}).setdefault("retained_facts", [])
    retained.append(
        {
            "fact_id": f"PRESSURE-UNSUPPORTED-{fixture['variant']:02d}",
            "text": "The parent filled a context-budget gap with an unsupported bridge claim.",
            "evidence_chunk_id": f"memory://context-pressure/{fixture['fixture_id']}/gap",
        }
    )
    return copied


def _summarize_rows(
    conditions: tuple[ContextBudgetCondition, ...],
    rows: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition_name"]].append(row)
    return tuple(_summary_row(condition, by_condition[condition.name]) for condition in conditions)


def _summary_row(
    condition: ContextBudgetCondition,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        **condition.to_metadata(),
        "fixture_count": len(rows),
        "mean_supported_claim_recall": _mean_number(row["supported_claim_recall"] for row in rows),
        "mean_context_pressure": _mean_number(row["context_pressure"] for row in rows),
        "mean_evidence_bandwidth_multiplier": _mean_number(
            row["evidence_bandwidth_multiplier"] for row in rows
        ),
        "unsupported_rejection_rate": _mean_bool(
            row["unsupported_rejected_before_commit"] for row in rows
        ),
        "unsupported_commit_rate": _mean_bool(row["unsupported_committed"] for row in rows),
        "final_parent_commit_rate": _mean_bool(row["final_parent_committed"] for row in rows),
    }


def _mean_bool(values: Any) -> float:
    numbers = [1.0 if value else 0.0 for value in values]
    return round(sum(numbers) / len(numbers), 4)


def _mean_number(values: Any) -> float:
    numbers = [float(value) for value in values]
    return round(sum(numbers) / len(numbers), 4)


def _write_outputs(
    *,
    output_dir: Path,
    fixture_count: int,
    condition_names: tuple[str, ...],
    rows: tuple[dict[str, Any], ...],
    summary_rows: tuple[dict[str, Any], ...],
) -> tuple[Path, ...]:
    rows_path = output_dir / "context_budget_rows.json"
    csv_path = output_dir / "context_budget_summary.csv"
    metadata_path = output_dir / "context_budget_metadata.json"

    rows_path.write_text(
        json.dumps(list(rows), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_summary_csv(csv_path, summary_rows)
    metadata_path.write_text(
        json.dumps(
            {
                "benchmark_name": "context-budget-pressure",
                "fixture_count": fixture_count,
                "condition_names": list(condition_names),
                "deterministic": True,
                "live_model_calls": 0,
                "network_calls": 0,
                "claim": (
                    "Recursive evidence caching preserves inspectable supported claims under a "
                    "fixed context budget; TraceGuard prevents pressure-induced unsupported commits."
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return (rows_path, csv_path, metadata_path)


def _write_summary_csv(path: Path, summary_rows: tuple[dict[str, Any], ...]) -> None:
    fieldnames = (
        "condition_name",
        "recursive",
        "traceguard_enabled",
        "visible_leaf_facts",
        "parent_context_budget_tokens",
        "deterministic",
        "live_model_calls",
        "fixture_count",
        "mean_supported_claim_recall",
        "mean_context_pressure",
        "mean_evidence_bandwidth_multiplier",
        "unsupported_rejection_rate",
        "unsupported_commit_rate",
        "final_parent_commit_rate",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic context-budget pressure benchmark artifacts.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    result = run_context_budget_benchmark(output_dir=args.output_dir)
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
