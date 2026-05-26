from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/run-live-atomic-task-boundary-pilot.py"
)


def _load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("live_atomic_task_boundary_pilot", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_live_atomic_boundary_pilot_writes_metadata_with_fake_live_calls(
    tmp_path, monkeypatch
) -> None:
    module = _load_script()

    fixture_ids = [fixture["fixture_id"] for fixture in module._fixtures()]
    assert len(fixture_ids) == 40
    assert len(set(fixture_ids)) == 40
    assert "adversarial-dfs-atomic" in module.CONDITIONS

    def fake_hermes_json(prompt: str, *, timeout: int = 180) -> tuple[dict[str, Any], dict[str, Any]]:
        assert "Return ONLY valid JSON" in prompt
        first_fact = re.search(
            r"\[(?P<chunk>[^\]]+)\] FACT (?P<fact>[^:]+): (?P<text>.+)", prompt
        )
        assert first_fact is not None
        retained = [
            {
                "fact_id": first_fact.group("fact"),
                "text": first_fact.group("text"),
                "evidence_chunk_id": first_fact.group("chunk"),
            }
        ]
        if (
            ("Task mode: broad-single-shot" in prompt or "Task mode: adversarial-dfs-atomic" in prompt)
            and "live-atomic-incident-01" in prompt
        ):
            retained.append(
                {
                    "fact_id": "NATURAL-UNSUPPORTED-FAKE",
                    "text": "A fake live parent naturally drifted to an unsupported memory claim.",
                    "evidence_chunk_id": "memory://fake-natural-drift",
                }
            )
        return {"summary": "fake live parent", "retained_facts": retained}, {
            "latency_ms": 1.0,
            "raw_output_chars": 10,
            "prompt_chars": len(prompt),
            "live_model_calls": 1,
        }

    monkeypatch.setattr(module, "_hermes_json", fake_hermes_json)

    result = module.run(tmp_path / "pilot")

    assert result["metadata"]["run_mode"] == "live-hermes-pilot"
    assert result["metadata"]["deterministic"] is False
    assert result["metadata"]["fixture_count"] == 40
    assert result["metadata"]["live_model_calls"] == 160
    rows = json.loads((tmp_path / "pilot/live_atomic_boundary_rows.json").read_text())
    summaries = json.loads((tmp_path / "pilot/live_atomic_boundary_summary.json").read_text())
    metadata = json.loads((tmp_path / "pilot/live_atomic_boundary_metadata.json").read_text())

    assert len(rows) == 160
    assert metadata == result["metadata"]
    assert {row["condition_name"] for row in summaries} == {
        "broad-single-shot",
        "one-level-atomic",
        "dfs-atomic",
        "adversarial-dfs-atomic",
    }
    assert all("natural_unsupported_claim_attempted" in row for row in rows)
    assert all("natural_unsupported_rejected_before_commit" in row for row in rows)
    assert all("natural_unsupported_commit_rate" in row for row in summaries)
    one_level = next(row for row in summaries if row["condition_name"] == "one-level-atomic")
    broad = next(row for row in summaries if row["condition_name"] == "broad-single-shot")
    adversarial_dfs = next(row for row in summaries if row["condition_name"] == "adversarial-dfs-atomic")
    assert broad["unsupported_commit_rate"] == 1.0
    assert broad["natural_unsupported_attempt_rate"] == 0.025
    assert broad["natural_unsupported_commit_rate"] == 0.025
    assert one_level["unsupported_rejection_rate"] == 1.0
    assert one_level["unsupported_commit_rate"] == 0.0
    assert one_level["natural_unsupported_commit_rate"] == 0.0
    assert adversarial_dfs["natural_unsupported_attempt_rate"] == 0.025
    assert adversarial_dfs["natural_unsupported_rejection_rate"] == 0.025
    assert adversarial_dfs["natural_unsupported_commit_rate"] == 0.0
    assert adversarial_dfs["unsupported_rejection_rate"] == 1.0
