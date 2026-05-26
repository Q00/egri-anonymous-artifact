from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from egri.benchmarks import generate_emnlp_additional_experiments
from egri.benchmarks.emnlp_additional import _posthoc_rows


_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_generate_emnlp_additional_experiments_runs_all_three_requested_studies(tmp_path) -> None:
    result = generate_emnlp_additional_experiments(
        repo_root=_REPO_ROOT,
        output_dir=tmp_path / "emnlp-extra",
        examples_per_dataset=20,
    )

    assert result.output_dir == (tmp_path / "emnlp-extra").resolve()
    assert result.metadata_path.is_file()
    assert result.scaled_examples_path.is_file()
    assert result.offline_run_dir.is_dir()
    assert result.posthoc_run_dir.is_dir()
    assert result.overhead_table_path.is_file()
    assert result.report_path.is_file()

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["run_mode"] == "offline-deterministic-emnlp-additional-experiments"
    assert metadata["deterministic"] is True
    assert metadata["live_model_calls"] == 0
    assert metadata["network_calls"] == 0
    assert metadata["requested_experiments"] == [
        "real_multidataset_scaleup",
        "posthoc_checker_baseline",
        "cost_token_latency_overhead",
    ]
    assert metadata["scaled_dataset"]["example_count"] == 80
    assert metadata["scaled_dataset"]["path"].endswith("real_multidataset_scaled_80.jsonl")
    assert not metadata["scaled_dataset"]["path"].startswith(("/home/", "<TMP>/"))
    assert metadata["offline_benchmark"]["dataset_path"].endswith("real_multidataset_scaled_80.jsonl")
    assert metadata["offline_benchmark"]["output_dir"].endswith("real_multidataset_scaled_offline")
    assert not metadata["offline_benchmark"]["dataset_path"].startswith(("/home/", "<TMP>/"))
    assert metadata["scaled_dataset"]["dataset_counts"] == {
        "2wikimultihopqa": 20,
        "asqa": 20,
        "hotpotqa": 20,
        "qasper": 20,
    }
    assert metadata["posthoc_baseline"]["posthoc_flag_rate"] == 1.0
    assert metadata["posthoc_baseline"]["precommit_block_rate"] == 0.0
    assert metadata["traceguard_precommit"]["posthoc_flag_rate"] == 1.0
    assert metadata["traceguard_precommit"]["precommit_block_rate"] == 1.0
    assert metadata["traceguard_precommit"]["unsupported_commit_rate"] == 0.0
    assert metadata["overhead"]["rows"] == 4

    assert len(result.scaled_examples_path.read_text(encoding="utf-8").splitlines()) == 80

    with result.overhead_table_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["condition"] for row in rows] == [
        "single-shot-context-window",
        "flat-topk-retrieval",
        "recursive-evidence-cache",
        "recursive-traceguard-dfs",
    ]
    ours = rows[-1]
    assert ours["model_calls"] == "8"
    assert float(ours["accepted_claim_recall"]) == 1.0
    assert float(ours["unsupported_commit_rate"]) == 0.0
    assert float(ours["latency_units"]) > float(rows[0]["latency_units"])

    report = result.report_path.read_text(encoding="utf-8")
    assert "80 deterministic examples" in report
    assert "post-hoc checker" in report
    assert "cost/token/latency" in report

    posthoc_rows = (result.posthoc_run_dir / "posthoc_rows.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    traceguard_row = next(
        json.loads(line)
        for line in posthoc_rows
        if '"condition": "recursive-traceguard-precommit"' in line
    )
    assert traceguard_row["traceguard_accepted"] is False
    assert traceguard_row["traceguard_rejected_reasons"] == ["unsupported_fact_id"]

    generated_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in result.output_dir.rglob("*")
        if path.is_file()
    )
    assert not any(
        local_path in generated_text
        for local_path in ("/home/", "<TMP>/")
    )


def test_generate_emnlp_additional_experiments_rejects_nonpositive_scale(tmp_path) -> None:
    with pytest.raises(ValueError, match="examples_per_dataset"):
        generate_emnlp_additional_experiments(
            repo_root=_REPO_ROOT,
            output_dir=tmp_path / "emnlp-extra",
            examples_per_dataset=0,
        )


def test_posthoc_rows_are_derived_from_citations_and_traceguard() -> None:
    example = {
        "example_id": "toy-001",
        "context_chunks": [
            {"chunk_id": "toy-001:gold", "metadata": {"role": "gold"}},
            {
                "chunk_id": "toy-001:bridge",
                "metadata": {"role": "unsupported_bridge"},
            },
        ],
        "gold_answers": ["toy answer"],
        "gold_evidence_chunk_ids": ["toy-001:gold"],
    }

    posthoc, traceguard = _posthoc_rows((example,))

    assert posthoc["posthoc_flagged"] is True
    assert posthoc["precommit_blocked"] is False
    assert posthoc["unsupported_committed"] is True
    assert traceguard["posthoc_flagged"] is True
    assert traceguard["precommit_blocked"] is True
    assert traceguard["unsupported_committed"] is False
    assert traceguard["traceguard_accepted"] is False
    assert traceguard["traceguard_rejected_reasons"] == ["unsupported_fact_id"]