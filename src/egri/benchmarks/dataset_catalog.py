"""Dataset catalog and manifest loading for scaled benchmark subsets."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from egri.benchmarks.jsonl_loader import load_jsonl_examples
from egri.benchmarks.schema import BenchmarkExample


class BenchmarkDatasetCatalogError(ValueError):
    """Raised when benchmark dataset catalog or manifest validation fails."""


@dataclass(frozen=True, slots=True)
class BenchmarkDatasetSpec:
    """Metadata for one supported external benchmark family."""

    name: str
    display_name: str
    task_type: str
    citation: str
    license_name: str
    homepage: str
    requires_multihop_reasoning: bool = False
    requires_abstractive_synthesis: bool = False

    def __post_init__(self) -> None:
        for field_name in (
            "name",
            "display_name",
            "task_type",
            "citation",
            "license_name",
            "homepage",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                msg = f"dataset spec {field_name} must be a non-empty string"
                raise BenchmarkDatasetCatalogError(msg)
        if not isinstance(self.requires_multihop_reasoning, bool):
            msg = "requires_multihop_reasoning must be boolean"
            raise BenchmarkDatasetCatalogError(msg)
        if not isinstance(self.requires_abstractive_synthesis, bool):
            msg = "requires_abstractive_synthesis must be boolean"
            raise BenchmarkDatasetCatalogError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable catalog metadata."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "task_type": self.task_type,
            "citation": self.citation,
            "license_name": self.license_name,
            "homepage": self.homepage,
            "requires_multihop_reasoning": self.requires_multihop_reasoning,
            "requires_abstractive_synthesis": self.requires_abstractive_synthesis,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkManifestSource:
    """One normalized JSONL source in a benchmark suite manifest."""

    dataset: str
    split: str
    path: Path
    example_count: int
    license_name: str

    def __post_init__(self) -> None:
        if self.dataset not in _DATASET_SPECS_BY_NAME:
            msg = f"unknown dataset in manifest source: {self.dataset}"
            raise BenchmarkDatasetCatalogError(msg)
        if not isinstance(self.split, str) or not self.split:
            msg = "manifest source split must be a non-empty string"
            raise BenchmarkDatasetCatalogError(msg)
        if not isinstance(self.path, Path):
            msg = "manifest source path must be a Path"
            raise BenchmarkDatasetCatalogError(msg)
        if self.path.suffix != ".jsonl":
            msg = f"manifest source path must be .jsonl: {self.path}"
            raise BenchmarkDatasetCatalogError(msg)
        if (
            isinstance(self.example_count, bool)
            or not isinstance(self.example_count, int)
            or self.example_count <= 0
        ):
            msg = "manifest source example_count must be a positive integer"
            raise BenchmarkDatasetCatalogError(msg)
        if not isinstance(self.license_name, str) or not self.license_name:
            msg = "manifest source license_name must be a non-empty string"
            raise BenchmarkDatasetCatalogError(msg)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable manifest-source metadata."""
        return {
            "dataset": self.dataset,
            "split": self.split,
            "path": str(self.path),
            "example_count": self.example_count,
            "license_name": self.license_name,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkSuiteManifest:
    """A reproducible offline benchmark suite assembled from real dataset families."""

    name: str
    version: str
    description: str
    sources: tuple[BenchmarkManifestSource, ...]
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            msg = "manifest name must be a non-empty string"
            raise BenchmarkDatasetCatalogError(msg)
        if not isinstance(self.version, str) or not self.version:
            msg = "manifest version must be a non-empty string"
            raise BenchmarkDatasetCatalogError(msg)
        if not isinstance(self.description, str) or not self.description:
            msg = "manifest description must be a non-empty string"
            raise BenchmarkDatasetCatalogError(msg)
        if not isinstance(self.sources, tuple) or not self.sources:
            msg = "manifest sources must be a non-empty tuple"
            raise BenchmarkDatasetCatalogError(msg)
        if any(
            not isinstance(source, BenchmarkManifestSource) for source in self.sources
        ):
            msg = "manifest sources entries must be BenchmarkManifestSource objects"
            raise BenchmarkDatasetCatalogError(msg)
        if self.metadata is not None and not isinstance(self.metadata, Mapping):
            msg = "manifest metadata must be a mapping when present"
            raise BenchmarkDatasetCatalogError(msg)

    @property
    def example_count(self) -> int:
        """Return total declared example count."""
        return sum(source.example_count for source in self.sources)

    @property
    def dataset_counts(self) -> dict[str, int]:
        """Return declared example counts by dataset name."""
        counts: Counter[str] = Counter()
        for source in self.sources:
            counts[source.dataset] += source.example_count
        return dict(counts)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable manifest metadata."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "example_count": self.example_count,
            "dataset_counts": self.dataset_counts,
            "sources": [source.to_dict() for source in self.sources],
            "metadata": dict(self.metadata or {}),
        }


@dataclass(frozen=True, slots=True)
class MaterializedBenchmarkSubset:
    """Paths and summary for one materialized normalized benchmark subset."""

    suite_name: str
    examples_path: Path
    metadata_path: Path
    example_count: int
    dataset_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable materialization metadata."""
        return {
            "suite_name": self.suite_name,
            "examples_path": str(self.examples_path),
            "metadata_path": str(self.metadata_path),
            "example_count": self.example_count,
            "dataset_counts": dict(self.dataset_counts),
        }


_DATASET_SPECS: tuple[BenchmarkDatasetSpec, ...] = (
    BenchmarkDatasetSpec(
        name="qasper",
        display_name="Qasper",
        task_type="long-document evidence QA",
        citation="Dasigi et al., 2021",
        license_name="CC BY 4.0",
        homepage="allenai-qasper",
    ),
    BenchmarkDatasetSpec(
        name="hotpotqa",
        display_name="HotpotQA",
        task_type="multi-hop evidence QA",
        citation="Yang et al., 2018",
        license_name="CC BY-SA 4.0",
        homepage="hotpotqa-official-site",
        requires_multihop_reasoning=True,
    ),
    BenchmarkDatasetSpec(
        name="2wikimultihopqa",
        display_name="2WikiMultihopQA",
        task_type="multi-hop evidence QA",
        citation="Ho et al., 2020",
        license_name="Apache-2.0",
        homepage="alab-nii-2wikimultihop-repository",
        requires_multihop_reasoning=True,
    ),
    BenchmarkDatasetSpec(
        name="asqa",
        display_name="ASQA",
        task_type="ambiguous abstractive QA with citations",
        citation="Stelmakh et al., 2022",
        license_name="MIT",
        homepage="google-research-datasets-asqa-repository",
        requires_abstractive_synthesis=True,
    ),
)
_DATASET_SPECS_BY_NAME = {spec.name: spec for spec in _DATASET_SPECS}


def iter_dataset_specs() -> tuple[BenchmarkDatasetSpec, ...]:
    """Return supported real benchmark families in stable paper-table order."""
    return _DATASET_SPECS


def get_dataset_spec(name: str) -> BenchmarkDatasetSpec:
    """Return one dataset spec by canonical name."""
    try:
        return _DATASET_SPECS_BY_NAME[name]
    except KeyError as exc:
        msg = f"unknown dataset: {name}"
        raise BenchmarkDatasetCatalogError(msg) from exc


def load_benchmark_suite_manifest(
    path: str | Path,
    *,
    base_dir: str | Path | None = None,
) -> BenchmarkSuiteManifest:
    """Load and validate a benchmark suite manifest."""
    manifest_path = _resolve_manifest_path(path, base_dir=base_dir)
    if not manifest_path.exists():
        msg = f"benchmark suite manifest does not exist: {manifest_path}"
        raise FileNotFoundError(msg)
    if manifest_path.suffix != ".json":
        msg = f"benchmark suite manifest must use .json suffix: {manifest_path}"
        raise BenchmarkDatasetCatalogError(msg)
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        msg = f"{manifest_path}: invalid JSON: {exc.msg}"
        raise BenchmarkDatasetCatalogError(msg) from exc
    if not isinstance(raw, Mapping):
        msg = f"{manifest_path}: manifest must be a JSON object"
        raise BenchmarkDatasetCatalogError(msg)
    manifest_base_dir = Path(base_dir).resolve() if base_dir else manifest_path.parent
    return _manifest_from_mapping(raw, manifest_base_dir=manifest_base_dir)


def load_manifest_examples(manifest: BenchmarkSuiteManifest) -> list[BenchmarkExample]:
    """Load all normalized examples referenced by a manifest."""
    examples: list[BenchmarkExample] = []
    for source in manifest.sources:
        source_examples = load_jsonl_examples(source.path)
        if len(source_examples) != source.example_count:
            msg = (
                f"{source.path}: example_count declared {source.example_count} "
                f"but loaded {len(source_examples)}"
            )
            raise BenchmarkDatasetCatalogError(msg)
        for example in source_examples:
            if example.dataset != source.dataset:
                msg = (
                    f"{source.path}: example {example.example_id} dataset "
                    f"{example.dataset!r} does not match manifest dataset {source.dataset!r}"
                )
                raise BenchmarkDatasetCatalogError(msg)
        examples.extend(source_examples)
    return examples


def materialize_manifest_subset(
    manifest: BenchmarkSuiteManifest,
    *,
    output_dir: str | Path,
) -> MaterializedBenchmarkSubset:
    """Write a merged normalized JSONL subset and companion metadata file."""
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    examples = load_manifest_examples(manifest)
    examples_path = destination / f"{manifest.name}.jsonl"
    metadata_path = destination / f"{manifest.name}.metadata.json"
    with examples_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(
                json.dumps(example.to_dict(), allow_nan=False, sort_keys=True) + "\n"
            )
    metadata = {
        "suite_name": manifest.name,
        "version": manifest.version,
        "description": manifest.description,
        "example_count": len(examples),
        "dataset_counts": manifest.dataset_counts,
        "normalized_schema": "egri.benchmarks.BenchmarkExample",
        "sources": [source.to_dict() for source in manifest.sources],
        "catalog": [spec.to_dict() for spec in iter_dataset_specs()],
        "metadata": dict(manifest.metadata or {}),
    }
    metadata_path.write_text(
        json.dumps(metadata, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return MaterializedBenchmarkSubset(
        suite_name=manifest.name,
        examples_path=examples_path,
        metadata_path=metadata_path,
        example_count=len(examples),
        dataset_counts=manifest.dataset_counts,
    )


def _resolve_manifest_path(path: str | Path, *, base_dir: str | Path | None) -> Path:
    manifest_path = Path(path).expanduser().resolve()
    if base_dir is None:
        return manifest_path
    trusted_root = Path(base_dir).expanduser().resolve()
    try:
        manifest_path.relative_to(trusted_root)
    except ValueError as exc:
        msg = f"benchmark suite manifest is outside trusted base_dir: {manifest_path}"
        raise BenchmarkDatasetCatalogError(msg) from exc
    return manifest_path


def _manifest_from_mapping(
    raw: Mapping[str, Any], *, manifest_base_dir: Path
) -> BenchmarkSuiteManifest:
    name = _required_str(raw, "name", manifest_id="<unknown>")
    sources = tuple(
        _source_from_mapping(
            source, manifest_base_dir=manifest_base_dir, manifest_id=name
        )
        for source in _required_list(raw, "sources", manifest_id=name)
    )
    return BenchmarkSuiteManifest(
        name=name,
        version=_required_str(raw, "version", manifest_id=name),
        description=_required_str(raw, "description", manifest_id=name),
        sources=sources,
        metadata=_optional_mapping(raw, "metadata", manifest_id=name),
    )


def _source_from_mapping(
    raw: Any, *, manifest_base_dir: Path, manifest_id: str
) -> BenchmarkManifestSource:
    if not isinstance(raw, Mapping):
        msg = f"{manifest_id}: sources entries must be objects"
        raise BenchmarkDatasetCatalogError(msg)
    dataset = _required_str(raw, "dataset", manifest_id=manifest_id)
    spec = get_dataset_spec(dataset)
    source_path = _resolve_source_path(
        _required_str(raw, "path", manifest_id=manifest_id),
        base_dir=manifest_base_dir,
    )
    return BenchmarkManifestSource(
        dataset=dataset,
        split=_required_str(raw, "split", manifest_id=manifest_id),
        path=source_path,
        example_count=_required_positive_int(
            raw, "example_count", manifest_id=manifest_id
        ),
        license_name=str(raw.get("license_name") or spec.license_name),
    )


def _resolve_source_path(path: str, *, base_dir: Path) -> Path:
    source_path = Path(path)
    if not source_path.is_absolute():
        source_path = base_dir / source_path
    source_path = source_path.expanduser().resolve()
    try:
        source_path.relative_to(base_dir)
    except ValueError as exc:
        msg = f"benchmark manifest source is outside trusted base_dir: {source_path}"
        raise BenchmarkDatasetCatalogError(msg) from exc
    return source_path


def _required_str(raw: Mapping[str, Any], field: str, *, manifest_id: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value:
        msg = f"{manifest_id}: {field} must be a non-empty string"
        raise BenchmarkDatasetCatalogError(msg)
    return value


def _required_positive_int(
    raw: Mapping[str, Any], field: str, *, manifest_id: str
) -> int:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        msg = f"{manifest_id}: {field} must be a positive integer"
        raise BenchmarkDatasetCatalogError(msg)
    return value


def _required_list(
    raw: Mapping[str, Any], field: str, *, manifest_id: str
) -> list[Any]:
    value = raw.get(field)
    if not isinstance(value, list) or not value:
        msg = f"{manifest_id}: {field} must be a non-empty list"
        raise BenchmarkDatasetCatalogError(msg)
    return value


def _optional_mapping(
    raw: Mapping[str, Any], field: str, *, manifest_id: str
) -> Mapping[str, Any] | None:
    value = raw.get(field)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        msg = f"{manifest_id}: {field} must be an object when present"
        raise BenchmarkDatasetCatalogError(msg)
    return dict(value)
