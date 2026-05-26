"""Deterministic experiment for atomic child-task acceptance boundaries.

This benchmark isolates a narrow systems claim for the paper: child model calls
are not deterministic, but recursive decomposition can make leaf task outputs
small enough to check with deterministic contracts. The rows are deterministic
fixture artifacts and do not call a model.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from egri.traceguard import TraceGuardEvidence
from egri.traceguard import validate_parent_synthesis


class AtomicTaskBoundaryError(ValueError):
    """Raised when the atomic task-boundary benchmark cannot be produced."""


@dataclass(frozen=True, slots=True)
class AtomicTaskBoundaryCondition:
    """One deterministic condition in the atomic task-boundary benchmark."""

    name: str
    traversal: str
    requires_atomic_leaf: bool
    traceguard_enabled: bool
    structured_claim_guard_enabled: bool
    max_depth: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            msg = "condition name must be a non-empty string"
            raise AtomicTaskBoundaryError(msg)
        if self.traversal not in {"single-shot", "one-level", "dfs"}:
            msg = f"{self.name}: unsupported traversal {self.traversal!r}"
            raise AtomicTaskBoundaryError(msg)
        if not isinstance(self.requires_atomic_leaf, bool):
            msg = f"{self.name}: requires_atomic_leaf must be boolean"
            raise AtomicTaskBoundaryError(msg)
        if not isinstance(self.traceguard_enabled, bool):
            msg = f"{self.name}: traceguard_enabled must be boolean"
            raise AtomicTaskBoundaryError(msg)
        if not isinstance(self.structured_claim_guard_enabled, bool):
            msg = f"{self.name}: structured_claim_guard_enabled must be boolean"
            raise AtomicTaskBoundaryError(msg)
        if self.structured_claim_guard_enabled and not self.traceguard_enabled:
            msg = f"{self.name}: structured claim guard requires TraceGuard"
            raise AtomicTaskBoundaryError(msg)
        if isinstance(self.max_depth, bool) or not isinstance(self.max_depth, int):
            msg = f"{self.name}: max_depth must be an integer"
            raise AtomicTaskBoundaryError(msg)
        if self.max_depth < 0:
            msg = f"{self.name}: max_depth must be non-negative"
            raise AtomicTaskBoundaryError(msg)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "condition_name": self.name,
            "traversal": self.traversal,
            "requires_atomic_leaf": self.requires_atomic_leaf,
            "traceguard_enabled": self.traceguard_enabled,
            "structured_claim_guard_enabled": self.structured_claim_guard_enabled,
            "max_depth": self.max_depth,
            "deterministic": True,
            "live_model_calls": 0,
        }


@dataclass(frozen=True, slots=True)
class AtomicTaskBoundaryResult:
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


def iter_atomic_task_boundary_conditions() -> tuple[AtomicTaskBoundaryCondition, ...]:
    """Return deterministic conditions in stable paper-table order."""
    return (
        AtomicTaskBoundaryCondition(
            name="broad-unchecked-parent",
            traversal="single-shot",
            requires_atomic_leaf=False,
            traceguard_enabled=False,
            structured_claim_guard_enabled=False,
            max_depth=0,
        ),
        AtomicTaskBoundaryCondition(
            name="one-level-atomic-traceguard",
            traversal="one-level",
            requires_atomic_leaf=True,
            traceguard_enabled=True,
            structured_claim_guard_enabled=False,
            max_depth=1,
        ),
        AtomicTaskBoundaryCondition(
            name="dfs-refinement-traceguard",
            traversal="dfs",
            requires_atomic_leaf=True,
            traceguard_enabled=True,
            structured_claim_guard_enabled=False,
            max_depth=2,
        ),
        AtomicTaskBoundaryCondition(
            name="dfs-traceguard-structured-claim-guard",
            traversal="dfs",
            requires_atomic_leaf=True,
            traceguard_enabled=True,
            structured_claim_guard_enabled=True,
            max_depth=2,
        ),
    )


def run_atomic_task_boundary_benchmark(
    *,
    output_dir: str | Path | None = None,
    conditions: tuple[AtomicTaskBoundaryCondition, ...] | None = None,
) -> AtomicTaskBoundaryResult:
    """Run deterministic atomic-boundary fixtures and persist JSON/CSV outputs."""
    if output_dir is None:
        msg = "output_dir must be provided for atomic task-boundary benchmark runs"
        raise AtomicTaskBoundaryError(msg)

    selected_conditions = conditions or iter_atomic_task_boundary_conditions()
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
    return AtomicTaskBoundaryResult(
        output_dir=resolved_output_dir,
        fixture_count=len(fixtures),
        condition_names=tuple(condition.name for condition in selected_conditions),
        rows=rows,
        summary_rows=summary_rows,
        output_paths=output_paths,
    )


def _validate_conditions(conditions: tuple[AtomicTaskBoundaryCondition, ...]) -> None:
    if not conditions:
        msg = "atomic task-boundary benchmark requires at least one condition"
        raise AtomicTaskBoundaryError(msg)
    if any(not isinstance(condition, AtomicTaskBoundaryCondition) for condition in conditions):
        msg = "conditions must be AtomicTaskBoundaryCondition objects"
        raise AtomicTaskBoundaryError(msg)
    names = [condition.name for condition in conditions]
    if len(set(names)) != len(names):
        msg = "atomic task-boundary benchmark rejects duplicate condition names"
        raise AtomicTaskBoundaryError(msg)


def _fixtures() -> tuple[dict[str, Any], ...]:
    domains = ("policy", "incident", "architecture", "experiment")
    return tuple(
        {
            "fixture_id": f"atomic-boundary-{domain}-{variant:02d}",
            "domain": domain,
            "variant": variant,
            "leaf_facts": tuple(
                {
                    "fact_id": f"AB-{domain[:3].upper()}-{variant:02d}-{index:03d}",
                    "chunk_id": (
                        f"atomic-boundary-{domain}-{variant:02d}.txt:{index}-{index}"
                    ),
                    "text": (
                        f"{domain} fixture {variant} atomic fact {index} is "
                        "supported by exactly one bounded input slice."
                    ),
                    "evidence_terms": {
                        "domain": domain,
                        "variant": f"{variant:02d}",
                        "atomic_fact": str(index),
                    },
                }
                for index in range(1, 4)
            ),
        }
        for domain in domains
        for variant in range(1, 4)
    )


def _run_condition_on_fixture(
    condition: AtomicTaskBoundaryCondition,
    fixture: dict[str, Any],
) -> dict[str, Any]:
    manifest = _manifest(fixture, condition)
    if condition.requires_atomic_leaf:
        leaf_count = len(manifest) * 2 if condition.traversal == "dfs" else len(manifest)
    else:
        leaf_count = 0
    parent = _parent_with_supported_facts(fixture, manifest)
    attempted_parent = _parent_with_unsupported_fact(parent, fixture)
    semantic_miss_parent = _parent_with_semantic_miss(parent, fixture, manifest)

    if condition.traceguard_enabled:
        validation = validate_parent_synthesis(
            evidence_manifest=manifest,
            parent_synthesis=attempted_parent,
        )
        unsupported_committed = validation.accepted
        final_validation = validate_parent_synthesis(
            evidence_manifest=manifest,
            parent_synthesis=parent,
        )
        semantic_traceguard_validation = validate_parent_synthesis(
            evidence_manifest=manifest,
            parent_synthesis=semantic_miss_parent,
        )
        structured_validation = _validate_structured_claim_terms(
            semantic_miss_parent,
            fixture,
        )
        semantic_miss_attempted = True
        traceguard_accepted_semantic_miss = semantic_traceguard_validation.accepted
        semantic_miss_rejected = (
            condition.structured_claim_guard_enabled
            and traceguard_accepted_semantic_miss
            and not structured_validation["accepted"]
        )
        semantic_miss_committed = traceguard_accepted_semantic_miss and not semantic_miss_rejected
        final_parent_committed = final_validation.accepted
        unsupported_rejected = not validation.accepted
    else:
        validation = None
        unsupported_committed = True
        final_parent_committed = True
        unsupported_rejected = False
        structured_validation = {"accepted": False, "missing_terms": {}, "claim_terms": {}}
        semantic_miss_attempted = False
        traceguard_accepted_semantic_miss = False
        semantic_miss_rejected = False
        semantic_miss_committed = False

    return {
        **condition.to_metadata(),
        "fixture_id": fixture["fixture_id"],
        "leaf_count": leaf_count,
        "leaf_atomic": condition.requires_atomic_leaf,
        "deterministic_acceptance_predicate": condition.requires_atomic_leaf
        and condition.traceguard_enabled,
        "unsupported_claim_attempted": True,
        "unsupported_rejected_before_commit": unsupported_rejected,
        "unsupported_committed": unsupported_committed,
        "semantic_miss_attempted": semantic_miss_attempted,
        "traceguard_accepted_semantic_miss": traceguard_accepted_semantic_miss,
        "semantic_miss_rejected_before_commit": semantic_miss_rejected,
        "semantic_miss_committed": semantic_miss_committed,
        "semantic_miss_claim_terms": structured_validation["claim_terms"],
        "structured_missing_terms": structured_validation["missing_terms"],
        "final_parent_committed": final_parent_committed,
        "initial_unsupported_claim_rate": (
            validation.unsupported_claim_rate if validation is not None else None
        ),
        "claim": (
            "stochastic child generation is wrapped by deterministic leaf acceptance"
            if condition.requires_atomic_leaf
            else "arbitrary broad parent task lacks a deterministic child boundary"
        ),
    }


def _manifest(
    fixture: dict[str, Any],
    condition: AtomicTaskBoundaryCondition,
) -> tuple[TraceGuardEvidence, ...]:
    if not condition.requires_atomic_leaf:
        return ()
    return tuple(
        TraceGuardEvidence(
            fact_id=fact["fact_id"],
            chunk_id=fact["chunk_id"],
            text=fact["text"],
            child_call_id=f"{fixture['fixture_id']}::dfs::{index:03d}"
            if condition.traversal == "dfs"
            else f"{fixture['fixture_id']}::child::{index:03d}",
        )
        for index, fact in enumerate(fixture["leaf_facts"], start=1)
    )


def _parent_with_supported_facts(
    fixture: dict[str, Any],
    manifest: tuple[TraceGuardEvidence, ...],
) -> dict[str, Any]:
    if not manifest:
        return {
            "mode": "broad_parent_synthesis",
            "result": {
                "summary": "A broad parent directly emits a final answer without child evidence."
            },
            "evidence_references": [],
        }
    return {
        "mode": "egri_parent_synthesis",
        "result": {
            "summary": (
                f"{fixture['fixture_id']} parent cites only accepted atomic child facts."
            ),
            "retained_facts": [
                {
                    "fact_id": item.fact_id,
                    "text": item.text,
                    "evidence_chunk_id": item.chunk_id,
                }
                for item in manifest
            ],
        },
        "evidence_references": [
            {
                "chunk_id": item.chunk_id,
                "supports_fact_ids": [item.fact_id],
                "quoted_evidence": item.text,
            }
            for item in manifest
        ],
        "residual_gaps": [],
    }


def _parent_with_unsupported_fact(
    parent: dict[str, Any], fixture: dict[str, Any]
) -> dict[str, Any]:
    copied = json.loads(json.dumps(parent))
    retained = copied.setdefault("result", {}).setdefault("retained_facts", [])
    retained.append(
        {
            "fact_id": f"UNSUPPORTED-{fixture['variant']:02d}",
            "text": "A tempting but unsupported parent claim was generated upstream.",
            "evidence_chunk_id": f"memory://{fixture['fixture_id']}/unsupported",
        }
    )
    return copied


def _parent_with_semantic_miss(
    parent: dict[str, Any],
    fixture: dict[str, Any],
    manifest: tuple[TraceGuardEvidence, ...],
) -> dict[str, Any]:
    copied = json.loads(json.dumps(parent))
    if not manifest:
        return copied
    evidence = manifest[0]
    retained = copied.setdefault("result", {}).setdefault("retained_facts", [])
    retained.append(
        {
            "fact_id": evidence.fact_id,
            "text": (
                "TraceGuard-admissible but semantically wrong structured claim: "
                "test_passed behavior=admin_delete_denied result=passed."
            ),
            "evidence_chunk_id": evidence.chunk_id,
            "claim_terms": {
                "behavior": "admin_delete_denied",
                "result": "passed",
            },
        }
    )
    return copied


def _validate_structured_claim_terms(
    parent_synthesis: dict[str, Any],
    fixture: dict[str, Any],
) -> dict[str, Any]:
    evidence_terms_by_fact = {
        fact["fact_id"]: fact.get("evidence_terms", {}) for fact in fixture["leaf_facts"]
    }
    claim_terms: dict[str, str] = {}
    missing_terms: dict[str, str] = {}
    result = parent_synthesis.get("result", {})
    retained = result.get("retained_facts", []) if isinstance(result, dict) else []
    if not isinstance(retained, list):
        return {"accepted": False, "claim_terms": {}, "missing_terms": {"retained_facts": "missing"}}
    for fact in retained:
        if not isinstance(fact, dict):
            continue
        terms = fact.get("claim_terms")
        if not isinstance(terms, dict) or not terms:
            continue
        fact_id = fact.get("fact_id")
        evidence_terms = evidence_terms_by_fact.get(fact_id, {}) if isinstance(fact_id, str) else {}
        normalized_evidence_terms = {
            str(key): str(value) for key, value in evidence_terms.items()
        }
        for key, value in terms.items():
            key_text = str(key)
            value_text = str(value)
            claim_terms[key_text] = value_text
            if normalized_evidence_terms.get(key_text) != value_text:
                missing_terms[key_text] = value_text
    return {
        "accepted": not missing_terms,
        "claim_terms": claim_terms,
        "missing_terms": missing_terms,
    }


def _summarize_rows(
    conditions: tuple[AtomicTaskBoundaryCondition, ...],
    rows: tuple[dict[str, Any], ...],
) -> tuple[dict[str, Any], ...]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition_name"]].append(row)
    return tuple(_summary_row(condition, by_condition[condition.name]) for condition in conditions)


def _summary_row(
    condition: AtomicTaskBoundaryCondition,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        **condition.to_metadata(),
        "fixture_count": len(rows),
        "leaf_atomic_rate": _mean_bool(row["leaf_atomic"] for row in rows),
        "deterministic_acceptance_predicate_rate": _mean_bool(
            row["deterministic_acceptance_predicate"] for row in rows
        ),
        "unsupported_rejection_rate": _mean_bool(
            row["unsupported_rejected_before_commit"] for row in rows
        ),
        "unsupported_commit_rate": _mean_bool(
            row["unsupported_committed"] for row in rows
        ),
        "semantic_miss_attempt_rate": _mean_bool(row["semantic_miss_attempted"] for row in rows),
        "traceguard_semantic_miss_accept_rate": _mean_bool(
            row["traceguard_accepted_semantic_miss"] for row in rows
        ),
        "semantic_miss_rejection_rate": _mean_bool(
            row["semantic_miss_rejected_before_commit"] for row in rows
        ),
        "semantic_miss_commit_rate": _mean_bool(row["semantic_miss_committed"] for row in rows),
        "final_parent_commit_rate": _mean_bool(row["final_parent_committed"] for row in rows),
        "mean_max_depth": float(condition.max_depth),
        "mean_leaf_count": _mean_number(row["leaf_count"] for row in rows),
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
    rows_path = output_dir / "atomic_task_boundary_rows.json"
    csv_path = output_dir / "atomic_task_boundary_summary.csv"
    metadata_path = output_dir / "atomic_task_boundary_metadata.json"

    rows_path.write_text(
        json.dumps(list(rows), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_summary_csv(csv_path, summary_rows)
    metadata_path.write_text(
        json.dumps(
            {
                "benchmark_name": "atomic-task-boundary",
                "fixture_count": fixture_count,
                "condition_names": list(condition_names),
                "deterministic": True,
                "live_model_calls": 0,
                "network_calls": 0,
                "claim": (
                    "Child model calls are not assumed deterministic; recursion refines "
                    "tasks until leaf outputs have deterministic acceptance predicates."
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
        "traversal",
        "requires_atomic_leaf",
        "traceguard_enabled",
        "structured_claim_guard_enabled",
        "max_depth",
        "deterministic",
        "live_model_calls",
        "fixture_count",
        "leaf_atomic_rate",
        "deterministic_acceptance_predicate_rate",
        "unsupported_rejection_rate",
        "unsupported_commit_rate",
        "semantic_miss_attempt_rate",
        "traceguard_semantic_miss_accept_rate",
        "semantic_miss_rejection_rate",
        "semantic_miss_commit_rate",
        "final_parent_commit_rate",
        "mean_max_depth",
        "mean_leaf_count",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)
