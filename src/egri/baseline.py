"""Self-contained deterministic baseline scorer used by the anonymous artifact.

The upstream workflow engine used during development exposed the same scorer
under ``ouroboros.rlm.baseline``. The review artifact vendors this small,
claim-aware subset so the deterministic scripts do not depend on an unreleased
or account-specific checkout.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

RLM_VANILLA_BASELINE_MODE = "vanilla_truncation_baseline"


@dataclass(frozen=True)
class TruncationBaselineScore:
    score: float
    retained_fact_citation_score: float
    truncation_boundary_score: float
    omitted_fact_safety_score: float
    claimed_omitted_fact_ids: list[str]
    missing_retained_fact_ids: list[str]
    cited_omitted_chunk_ids: list[str]
    reports_truncation_boundary: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "retained_fact_citation_score": self.retained_fact_citation_score,
            "truncation_boundary_score": self.truncation_boundary_score,
            "omitted_fact_safety_score": self.omitted_fact_safety_score,
            "claimed_omitted_fact_ids": self.claimed_omitted_fact_ids,
            "missing_retained_fact_ids": self.missing_retained_fact_ids,
            "cited_omitted_chunk_ids": self.cited_omitted_chunk_ids,
            "reports_truncation_boundary": self.reports_truncation_boundary,
        }


def _loads_completion(completion: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(completion, dict):
        return completion
    try:
        parsed = json.loads(completion)
    except json.JSONDecodeError:
        return {"result": {"summary": completion}, "evidence_references": []}
    return parsed if isinstance(parsed, dict) else {"result": {"summary": str(parsed)}}


def _fact_ids(items: list[dict[str, Any]]) -> list[str]:
    return [str(item.get("fact_id", "")) for item in items if item.get("fact_id")]


def _safe_text_for_omitted_scan(completion: dict[str, Any]) -> str:
    """Return text surfaces where omitted IDs are positive claims.

    Residual-gap and truncation-report surfaces are deliberately excluded: the
    scorer should not punish a completion for saying that omitted regions exist.
    """

    result = completion.get("result")
    if not isinstance(result, dict):
        return json.dumps(result, sort_keys=True)

    claim_surfaces: dict[str, Any] = {}
    for key in ("summary", "retained_facts", "observed_facts", "unsupported_claims", "evidence_references"):
        if key in result:
            claim_surfaces[key] = result[key]
    if "evidence_references" in completion:
        claim_surfaces["top_level_evidence_references"] = completion["evidence_references"]
    return json.dumps(claim_surfaces, sort_keys=True)


def _supported_fact_ids_from_completion(completion: dict[str, Any]) -> set[str]:
    result = completion.get("result", {})
    supported: set[str] = set()
    if isinstance(result, dict):
        for key in ("retained_facts", "observed_facts"):
            values = result.get(key, [])
            if isinstance(values, list):
                supported.update(_fact_ids([v for v in values if isinstance(v, dict)]))
    refs = completion.get("evidence_references", [])
    if isinstance(refs, list):
        for ref in refs:
            if isinstance(ref, dict):
                for fid in ref.get("supports_fact_ids", []) or []:
                    supported.add(str(fid))
    return supported


def score_vanilla_truncation_baseline_completion(
    fixture: dict[str, Any], completion: str | dict[str, Any]
) -> TruncationBaselineScore:
    """Score a deterministic truncation-baseline completion.

    This is a small provenance-oriented scorer, not a semantic evaluator. It
    checks whether retained fixture facts are cited, whether truncation metadata
    is reported, and whether omitted facts/chunks are positively claimed.
    """

    payload = _loads_completion(completion)
    retained = fixture.get("expected_retained_facts", [])
    omitted = fixture.get("expected_omitted_facts", [])
    retained_ids = _fact_ids([f for f in retained if isinstance(f, dict)])
    omitted_ids = _fact_ids([f for f in omitted if isinstance(f, dict)])
    omitted_chunks = {str(f.get("chunk_id")) for f in omitted if isinstance(f, dict) and f.get("chunk_id")}

    supported_ids = _supported_fact_ids_from_completion(payload)
    missing_retained = [fid for fid in retained_ids if fid not in supported_ids]
    retained_score = 1.0 if not retained_ids else (len(retained_ids) - len(missing_retained)) / len(retained_ids)

    result = payload.get("result", {})
    report = result.get("truncation_report") if isinstance(result, dict) else None
    boundary = fixture.get("truncation_config", {}).get("truncation_boundary", {})
    truncation_score = 0.0
    if isinstance(report, dict):
        required_line = boundary.get("last_retained_line")
        required_omitted = boundary.get("omitted_line_count")
        checks = [
            required_line is None or report.get("last_retained_line") == required_line,
            required_omitted is None or report.get("omitted_line_count") == required_omitted,
        ]
        truncation_score = 1.0 if all(checks) else 0.5

    claim_text = _safe_text_for_omitted_scan(payload)
    claimed_omitted = [fid for fid in omitted_ids if re.search(rf"\b{re.escape(fid)}\b", claim_text)]

    cited_omitted_chunks: list[str] = []
    refs = payload.get("evidence_references", [])
    if isinstance(refs, list):
        for ref in refs:
            if isinstance(ref, dict) and str(ref.get("chunk_id")) in omitted_chunks:
                cited_omitted_chunks.append(str(ref.get("chunk_id")))
                for fid in ref.get("supports_fact_ids", []) or []:
                    if fid in omitted_ids and fid not in claimed_omitted:
                        claimed_omitted.append(str(fid))

    omitted_safety = 1.0 if not claimed_omitted and not cited_omitted_chunks else 0.0
    score = (retained_score + truncation_score + omitted_safety) / 3.0

    return TruncationBaselineScore(
        score=score,
        retained_fact_citation_score=retained_score,
        truncation_boundary_score=truncation_score,
        omitted_fact_safety_score=omitted_safety,
        claimed_omitted_fact_ids=claimed_omitted,
        missing_retained_fact_ids=missing_retained,
        cited_omitted_chunk_ids=cited_omitted_chunks,
        reports_truncation_boundary=isinstance(report, dict),
    )
