from __future__ import annotations

from egri.traceguard import TraceGuardEvidence
from egri.traceguard import build_manifest_from_fixture
from egri.traceguard import normalize_allowed_evidence_manifest
from egri.traceguard import validate_parent_synthesis


def _fixture() -> dict[str, object]:
    return {
        "expected_retained_facts": [
            {
                "fact_id": "TG-001",
                "chunk_id": "traceguard.txt:1-2",
                "text": "FACT:TG-001 retained child evidence.",
            }
        ],
        "expected_omitted_facts": [
            {
                "fact_id": "TG-002",
                "chunk_id": "traceguard.txt:3-4",
                "text": "FACT:TG-002 omitted evidence.",
            }
        ],
    }


def test_traceguard_accepts_claims_backed_by_manifest() -> None:
    manifest = build_manifest_from_fixture(_fixture())
    parent = {
        "result": {
            "retained_facts": [
                {
                    "fact_id": "TG-001",
                    "text": "retained child evidence",
                    "evidence_chunk_id": "traceguard.txt:1-2",
                }
            ]
        },
        "evidence_references": [
            {
                "chunk_id": "traceguard.txt:1-2",
                "supports_fact_ids": ["TG-001"],
                "quoted_evidence": "FACT:TG-001 retained child evidence.",
            }
        ],
    }

    result = validate_parent_synthesis(
        evidence_manifest=manifest,
        parent_synthesis=parent,
    )

    assert result.accepted is True
    assert result.unsupported_claim_rate == 0.0
    assert len(result.accepted_claims) == 2
    assert result.rejected_claims == ()


def test_traceguard_rejects_omitted_fact_claims() -> None:
    manifest = build_manifest_from_fixture(_fixture())
    parent = {
        "result": {
            "observed_facts": [
                {
                    "fact_id": "TG-002",
                    "text": "omitted evidence",
                    "evidence_chunk_id": "traceguard.txt:3-4",
                }
            ]
        }
    }

    result = validate_parent_synthesis(
        evidence_manifest=manifest,
        parent_synthesis=parent,
    )

    assert result.accepted is False
    assert result.unsupported_claim_rate == 1.0
    assert result.rejected_claims[0].reason == "unsupported_fact_id"


def test_traceguard_rejects_chunk_handles_without_fact_evidence() -> None:
    manifest = build_manifest_from_fixture(_fixture())
    parent = {
        "result": {"summary": "chunk was read"},
        "evidence_references": [
            {"chunk_id": "traceguard.txt:1-2", "claim": "read traceguard.txt:1-2"}
        ],
    }

    result = validate_parent_synthesis(
        evidence_manifest=manifest,
        parent_synthesis=parent,
    )

    assert result.accepted is False
    assert result.rejected_claims[0].reason == "chunk_handle_without_fact"


def test_traceguard_rejects_parent_without_claim_bearing_surface() -> None:
    manifest = build_manifest_from_fixture(_fixture())
    parent = {
        "result": {"summary": "I read the evidence but emitted no structured facts."},
    }

    result = validate_parent_synthesis(
        evidence_manifest=manifest,
        parent_synthesis=parent,
    )

    assert result.accepted is False
    assert result.unsupported_claim_rate == 1.0
    assert result.accepted_claims == ()
    assert result.rejected_claims[0].reason == "no_claim_bearing_surface"


def test_traceguard_rejects_required_schema_wrapper_without_parent_claims() -> None:
    manifest = build_manifest_from_fixture(_fixture())
    parent = {
        "task": "synthesize_parent_answer_from_child_evidence",
        "required_schema": {
            "result": {
                "retained_facts": [
                    {
                        "fact_id": "TG-001",
                        "evidence_chunk_id": "traceguard.txt:1-2",
                    }
                ]
            }
        },
        "child_records": ["prompt-only wrapper; not actual parent output"],
    }

    result = validate_parent_synthesis(
        evidence_manifest=manifest,
        parent_synthesis=parent,
    )

    assert result.accepted is False
    assert result.accepted_claims == ()
    assert result.rejected_claims[0].reason == "no_claim_bearing_surface"


def test_allowed_evidence_manifest_normalization_is_canonical() -> None:
    manifest = (
        {
            "chunk_id": "traceguard.txt:9-10",
            "fact_id": "TG-003",
            "extra": "ignored",
        },
        TraceGuardEvidence(
            fact_id="TG-001",
            chunk_id="traceguard.txt:1-2",
            text="FACT:TG-001 retained child evidence.",
        ),
        {
            "fact_id": "TG-002",
            "chunk_id": "traceguard.txt:3-4",
            "text": "FACT:TG-002 retained child evidence.",
            "child_call_id": "child_0002",
        },
    )

    normalized = normalize_allowed_evidence_manifest(manifest)

    assert normalized == (
        {
            "fact_id": "TG-001",
            "chunk_id": "traceguard.txt:1-2",
            "text": "FACT:TG-001 retained child evidence.",
            "child_call_id": None,
        },
        {
            "fact_id": "TG-002",
            "chunk_id": "traceguard.txt:3-4",
            "text": "FACT:TG-002 retained child evidence.",
            "child_call_id": "child_0002",
        },
        {
            "fact_id": "TG-003",
            "chunk_id": "traceguard.txt:9-10",
            "text": "",
            "child_call_id": None,
        },
    )
    assert list(normalized[0]) == [
        "fact_id",
        "chunk_id",
        "text",
        "child_call_id",
    ]
