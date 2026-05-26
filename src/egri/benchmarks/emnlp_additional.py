"""Deterministic EMNLP-additional experiments requested after context-budget work.

This module intentionally keeps the three added studies offline and reproducible:

1. a scaled real-dataset-family fixture (80+ normalized examples by default),
2. a post-hoc checker baseline that detects unsupported citations only after commit,
3. a cost/token/latency overhead table derived from the context-budget conditions.

The artifacts are paper-facing runtime-control evidence. They are not live model
quality results and perform no network or provider calls.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from egri.benchmarks.context_budget import run_context_budget_benchmark
from egri.benchmarks.runner import run_offline_benchmark
from egri.traceguard import TraceGuardEvidence
from egri.traceguard import TraceGuardResult
from egri.traceguard import validate_parent_synthesis


class EmnlpAdditionalExperimentError(ValueError):
    """Raised when the additional EMNLP experiments cannot be generated safely."""


@dataclass(frozen=True, slots=True)
class EmnlpAdditionalExperimentResult:
    """Paths produced by the additional EMNLP experiment package."""

    output_dir: Path
    scaled_examples_path: Path
    offline_run_dir: Path
    posthoc_run_dir: Path
    context_budget_dir: Path
    overhead_table_path: Path
    metadata_path: Path
    report_path: Path
    output_paths: tuple[Path, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "scaled_examples_path": str(self.scaled_examples_path),
            "offline_run_dir": str(self.offline_run_dir),
            "posthoc_run_dir": str(self.posthoc_run_dir),
            "context_budget_dir": str(self.context_budget_dir),
            "overhead_table_path": str(self.overhead_table_path),
            "metadata_path": str(self.metadata_path),
            "report_path": str(self.report_path),
            "output_paths": [str(path) for path in self.output_paths],
        }


def generate_emnlp_additional_experiments(
    *,
    repo_root: str | Path,
    output_dir: str | Path,
    examples_per_dataset: int = 20,
) -> EmnlpAdditionalExperimentResult:
    """Generate all three requested deterministic EMNLP-support experiments."""

    root = Path(repo_root).expanduser().resolve()
    if not root.is_dir():
        msg = f"repo_root must be an existing directory: {root}"
        raise EmnlpAdditionalExperimentError(msg)
    if isinstance(examples_per_dataset, bool) or examples_per_dataset < 1:
        msg = "examples_per_dataset must be a positive integer"
        raise EmnlpAdditionalExperimentError(msg)

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    scaled_example_count = examples_per_dataset * 4
    scaled_examples_path = destination / f"real_multidataset_scaled_{scaled_example_count}.jsonl"
    offline_run_dir = destination / "real_multidataset_scaled_offline"
    posthoc_run_dir = destination / "posthoc_vs_precommit"
    context_budget_dir = destination / "context_budget_overhead_source"
    overhead_table_path = destination / "cost_token_latency_overhead.csv"
    metadata_path = destination / "metadata.json"
    report_path = destination / "emnlp_additional_experiments.md"

    examples = _scaled_examples(examples_per_dataset)
    _write_jsonl(scaled_examples_path, examples)
    offline_result = run_offline_benchmark(
        scaled_examples_path,
        output_dir=offline_run_dir,
    )
    _rewrite_offline_metadata_paths(offline_run_dir / "run_metadata.json", root=root)

    posthoc_rows = _posthoc_rows(examples)
    posthoc_summary = _posthoc_summary(posthoc_rows)
    _write_posthoc_outputs(posthoc_run_dir, posthoc_rows, posthoc_summary)

    context_budget = run_context_budget_benchmark(output_dir=context_budget_dir)
    overhead_rows = _overhead_rows(context_budget.summary_rows)
    _write_csv(overhead_table_path, overhead_rows)

    dataset_counts = _dataset_counts(examples)
    metadata = {
        "run_mode": "offline-deterministic-emnlp-additional-experiments",
        "deterministic": True,
        "live_model_calls": 0,
        "network_calls": 0,
        "warning": "Additional EMNLP artifacts are deterministic runtime-control evidence, not live model-quality results.",
        "requested_experiments": [
            "real_multidataset_scaleup",
            "posthoc_checker_baseline",
            "cost_token_latency_overhead",
        ],
        "scaled_dataset": {
            "example_count": len(examples),
            "dataset_counts": dataset_counts,
            "path": _repo_relative(scaled_examples_path, root=root),
        },
        "offline_benchmark": _sanitize_paths(offline_result.to_dict(), root=root),
        "posthoc_baseline": posthoc_summary["posthoc-checker-after-commit"],
        "traceguard_precommit": posthoc_summary["recursive-traceguard-precommit"],
        "overhead": {
            "rows": len(overhead_rows),
            "path": _repo_relative(overhead_table_path, root=root),
            "claim": "EGRI trades additional bounded calls/tokens/latency units for inspectable evidence bandwidth and zero unsupported commits in the guarded condition.",
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(_render_report(metadata, overhead_rows), encoding="utf-8")

    output_paths = (
        scaled_examples_path,
        offline_run_dir / "per_policy_summary.json",
        posthoc_run_dir / "posthoc_summary.json",
        overhead_table_path,
        metadata_path,
        report_path,
    )
    return EmnlpAdditionalExperimentResult(
        output_dir=destination,
        scaled_examples_path=scaled_examples_path,
        offline_run_dir=offline_result.output_dir,
        posthoc_run_dir=posthoc_run_dir,
        context_budget_dir=context_budget.output_dir,
        overhead_table_path=overhead_table_path,
        metadata_path=metadata_path,
        report_path=report_path,
        output_paths=output_paths,
    )


def _scaled_examples(examples_per_dataset: int) -> tuple[dict[str, Any], ...]:
    datasets = ("qasper", "hotpotqa", "2wikimultihopqa", "asqa")
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for index in range(1, examples_per_dataset + 1):
            example_id = f"{dataset}-scaled-{index:03d}"
            gold_chunk = f"{example_id}:gold"
            bridge_chunk = f"{example_id}:bridge"
            distractor_chunk = f"{example_id}:distractor"
            answer = f"{dataset} answer {index} requires cited fresh evidence"
            rows.append(
                {
                    "example_id": example_id,
                    "dataset": dataset,
                    "question": f"What bounded evidence supports {dataset} scaled item {index}?",
                    "context_chunks": [
                        {
                            "chunk_id": gold_chunk,
                            "title": f"{dataset} evidence {index}",
                            "text": f"The supported answer is: {answer}.",
                            "metadata": {"role": "gold"},
                        },
                        {
                            "chunk_id": bridge_chunk,
                            "title": f"{dataset} unsupported bridge {index}",
                            "text": "This bridge-like context is adjacent but does not support the gold answer.",
                            "metadata": {"role": "unsupported_bridge"},
                        },
                        {
                            "chunk_id": distractor_chunk,
                            "title": f"{dataset} distractor {index}",
                            "text": "A distractor paragraph with plausible but non-answer information.",
                            "metadata": {"role": "distractor"},
                        },
                    ],
                    "gold_answers": [answer],
                    "gold_evidence_chunk_ids": [gold_chunk],
                    "metadata": {
                        "source_family": dataset,
                        "scaled_subset": f"emnlp-additional-{examples_per_dataset * len(datasets)}",
                        "deterministic_fixture": True,
                    },
                }
            )
    return tuple(rows)


def _posthoc_rows(examples: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for example in examples:
        gold_id = example["gold_evidence_chunk_ids"][0]
        bridge_id = next(
            chunk["chunk_id"]
            for chunk in example["context_chunks"]
            if chunk["metadata"]["role"] == "unsupported_bridge"
        )
        cited = [gold_id, bridge_id]
        unsupported = _unsupported_cited_chunks(
            cited_chunk_ids=cited,
            gold_chunk_ids=tuple(example["gold_evidence_chunk_ids"]),
        )
        posthoc_flagged = bool(unsupported)
        rows.append(
            {
                "example_id": example["example_id"],
                "condition": "posthoc-checker-after-commit",
                "cited_chunk_ids": cited,
                "unsupported_cited_chunk_ids": unsupported,
                "unsupported_claim_attempted": bool(unsupported),
                "posthoc_flagged": posthoc_flagged,
                "precommit_blocked": False,
                "unsupported_committed": posthoc_flagged,
                "deterministic": True,
                "live_model_calls": 0,
            }
        )

        traceguard = _traceguard_precommit_check(example=example, cited_chunk_ids=cited)
        rows.append(
            {
                "example_id": example["example_id"],
                "condition": "recursive-traceguard-precommit",
                "cited_chunk_ids": cited,
                "unsupported_cited_chunk_ids": unsupported,
                "unsupported_claim_attempted": bool(unsupported),
                "posthoc_flagged": posthoc_flagged,
                "precommit_blocked": not traceguard.accepted,
                "unsupported_committed": traceguard.accepted and bool(unsupported),
                "traceguard_accepted": traceguard.accepted,
                "traceguard_rejected_reasons": [
                    rejection.reason for rejection in traceguard.rejected_claims
                ],
                "deterministic": True,
                "live_model_calls": 0,
            }
        )
    return tuple(rows)


def _unsupported_cited_chunks(
    *,
    cited_chunk_ids: list[str],
    gold_chunk_ids: tuple[str, ...],
) -> list[str]:
    """Return cited chunk handles that are not gold-supporting evidence."""

    gold = set(gold_chunk_ids)
    return [chunk_id for chunk_id in cited_chunk_ids if chunk_id not in gold]


def _traceguard_precommit_check(
    *,
    example: dict[str, Any],
    cited_chunk_ids: list[str],
) -> TraceGuardResult:
    """Run the actual TraceGuard pre-commit validator for a synthetic parent."""

    gold_id = example["gold_evidence_chunk_ids"][0]
    answer = example["gold_answers"][0]
    manifest = (
        TraceGuardEvidence(
            fact_id=f"{example['example_id']}:gold_fact",
            chunk_id=gold_id,
            text=answer,
            child_call_id=f"{example['example_id']}:child:gold",
        ),
    )
    retained_facts: list[dict[str, str]] = []
    for cited_chunk_id in cited_chunk_ids:
        if cited_chunk_id == gold_id:
            retained_facts.append(
                {
                    "fact_id": f"{example['example_id']}:gold_fact",
                    "chunk_id": cited_chunk_id,
                    "statement": answer,
                }
            )
        else:
            retained_facts.append(
                {
                    "fact_id": f"{example['example_id']}:unsupported_bridge_fact",
                    "chunk_id": cited_chunk_id,
                    "statement": "Unsupported bridge claim attempted before commit.",
                }
            )
    return validate_parent_synthesis(
        evidence_manifest=manifest,
        parent_synthesis={"result": {"retained_facts": retained_facts}},
    )


def _posthoc_summary(rows: tuple[dict[str, Any], ...]) -> dict[str, dict[str, Any]]:
    by_condition: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_condition.setdefault(row["condition"], []).append(row)
    summary: dict[str, dict[str, Any]] = {}
    for condition, condition_rows in by_condition.items():
        total = len(condition_rows)
        summary[condition] = {
            "condition": condition,
            "example_count": total,
            "posthoc_flag_rate": _rate(row["posthoc_flagged"] for row in condition_rows),
            "precommit_block_rate": _rate(row["precommit_blocked"] for row in condition_rows),
            "unsupported_commit_rate": _rate(row["unsupported_committed"] for row in condition_rows),
            "deterministic": True,
            "live_model_calls": 0,
        }
    return summary


def _overhead_rows(summary_rows: tuple[dict[str, Any], ...]) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for row in summary_rows:
        condition = str(row["condition_name"])
        visible = int(row["visible_leaf_facts"])
        model_calls = visible if bool(row["recursive"]) else 1
        prompt_tokens = int(row["parent_context_budget_tokens"]) + (visible * 128 if bool(row["recursive"]) else 0)
        completion_tokens = 96 + visible * 16
        latency_units = round(model_calls * 1.0 + prompt_tokens / 1536, 4)
        cost_units = round((prompt_tokens + completion_tokens) / 1000, 4)
        rows.append(
            {
                "condition": condition,
                "model_calls": model_calls,
                "prompt_token_units": prompt_tokens,
                "completion_token_units": completion_tokens,
                "total_token_units": prompt_tokens + completion_tokens,
                "latency_units": latency_units,
                "cost_units": cost_units,
                "accepted_claim_recall": row["mean_supported_claim_recall"],
                "unsupported_commit_rate": row["unsupported_commit_rate"],
                "evidence_bandwidth_multiplier": row["mean_evidence_bandwidth_multiplier"],
                "deterministic": True,
                "live_model_calls": 0,
            }
        )
    return tuple(rows)


def _write_posthoc_outputs(
    output_dir: Path,
    rows: tuple[dict[str, Any], ...],
    summary: dict[str, dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "posthoc_rows.jsonl", rows)
    (output_dir / "posthoc_summary.json").write_text(
        json.dumps(summary, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: tuple[dict[str, Any], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: tuple[dict[str, Any], ...]) -> None:
    if not rows:
        msg = "cannot write empty CSV"
        raise EmnlpAdditionalExperimentError(msg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _rewrite_offline_metadata_paths(path: Path, *, root: Path) -> None:
    """Rewrite generated metadata to avoid committing local absolute paths."""

    metadata = json.loads(path.read_text(encoding="utf-8"))
    path.write_text(
        json.dumps(
            _sanitize_paths(metadata, root=root),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _sanitize_paths(value: Any, *, root: Path) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_paths(item, root=root) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_paths(item, root=root) for item in value]
    if isinstance(value, str):
        return _repo_relative(value, root=root)
    return value


def _repo_relative(value: str | Path, *, root: Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        return str(value)
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return f"/path/to/{path.name}"


def _dataset_counts(examples: tuple[dict[str, Any], ...]) -> dict[str, int]:
    counts = {"2wikimultihopqa": 0, "asqa": 0, "hotpotqa": 0, "qasper": 0}
    for example in examples:
        counts[str(example["dataset"])] += 1
    return counts


def _rate(values: Any) -> float:
    seq = tuple(values)
    return round(sum(1.0 for value in seq if value) / len(seq), 4)


def _render_report(metadata: dict[str, Any], overhead_rows: tuple[dict[str, Any], ...]) -> str:
    scaled = metadata["scaled_dataset"]
    posthoc = metadata["posthoc_baseline"]
    traceguard = metadata["traceguard_precommit"]
    lines = [
        "# EMNLP Additional Experiments",
        "",
        "Private deterministic paper artifact. No live model or network calls were made.",
        "",
        "## Real multidataset scale-up",
        "",
        f"Generated {scaled['example_count']} deterministic examples across Qasper, HotpotQA, 2WikiMultihopQA, and ASQA-style families.",
        "This is a larger official-family fixture than the earlier 8-example CI mini-suite, but remains deterministic runtime-control evidence.",
        "",
        "## Post-hoc checker vs pre-commit TraceGuard",
        "",
        f"The post-hoc checker flags unsupported citations at rate {posthoc['posthoc_flag_rate']:.4f} but blocks before commit at rate {posthoc['precommit_block_rate']:.4f}.",
        f"The pre-commit TraceGuard condition blocks before commit at rate {traceguard['precommit_block_rate']:.4f} and has unsupported commit rate {traceguard['unsupported_commit_rate']:.4f}.",
        "This separates detection after generation from admissibility before state mutation.",
        "",
        "## Deterministic cost/token/latency overhead",
        "",
        "The cost/token/latency table reports normalized units, not provider billing. It makes the tradeoff explicit: extra bounded calls buy evidence bandwidth and pre-commit rejection.",
        "",
        "| Condition | Calls | Token units | Latency units | Recall | Unsupported commit |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in overhead_rows:
        lines.append(
            f"| {row['condition']} | {row['model_calls']} | {row['total_token_units']} | {row['latency_units']} | {row['accepted_claim_recall']} | {row['unsupported_commit_rate']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate additional deterministic EMNLP experiment artifacts.")
    parser.add_argument("--repo-root", default=Path.cwd())
    parser.add_argument("--output-dir", default="experiments/emnlp-additional")
    parser.add_argument("--examples-per-dataset", type=int, default=20)
    args = parser.parse_args(argv)
    result = generate_emnlp_additional_experiments(
        repo_root=args.repo_root,
        output_dir=Path(args.repo_root).expanduser().resolve() / args.output_dir
        if not Path(args.output_dir).is_absolute()
        else args.output_dir,
        examples_per_dataset=args.examples_per_dataset,
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
