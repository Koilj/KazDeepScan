"""Reproducible personal-research intake for the full pinned RuASD release.

RuASD does not provide verified speaker or spoof-voice groups.  This module therefore
protects only source-record and transcript leakage; it must never be described as a
speaker-disjoint benchmark.
"""

from __future__ import annotations

import hashlib
import heapq
import shutil
import tarfile
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import soundfile as sf  # type: ignore[import-untyped]

from kds.data.manifest import ManifestRow
from kds.data.ruasd_catalog import (
    RuAsdArchiveSpec,
    RuAsdCatalogError,
    RuAsdRecordMetadata,
    _safe_member_name,
    _validate_archive_set,
    _validate_archive_size,
    load_ruasd_archive_records,
    sha256_file,
)

RUASD_RESEARCH_SOURCE_ID = "ruasd_ru_v1_full"
RUASD_RESEARCH_SOURCE_LICENSE = "CC-BY-NC-SA-4.0"
RUASD_RESEARCH_RIGHTS_BASIS = (
    "RuASD full pinned release; CC-BY-NC-SA-4.0; personal research; "
    "raw binary subset; no verified speaker or spoof-voice groups"
)
RUASD_RESEARCH_CAPTURE_ROUTE = "ruasd_source_audio"
RUASD_RESEARCH_UNKNOWN = "unknown"
RuAsdResearchProgressCallback = Callable[[int, int, str], None]


class RuAsdResearchError(ValueError):
    """Raised when a full RuASD research slice cannot be safely created."""


@dataclass(frozen=True, slots=True)
class RuAsdResearchRecord:
    """One eligible raw binary RuASD record, without retaining transcript text."""

    archive_name: str
    sample_id: str
    label: str
    subset: str
    model: str
    text_hash: str
    has_source_text: bool

    @property
    def record_key(self) -> str:
        return f"{self.archive_name.removesuffix('.tar')}:{self.sample_id}"

    @property
    def stratum(self) -> str:
        if self.label == "bonafide":
            return f"bonafide/{self.subset}"
        return f"spoof/{self.subset}/{self.model or 'unspecified_by_source'}"


@dataclass(frozen=True, slots=True)
class RuAsdResearchSelection:
    """Selected records and transparent class/stratum accounting."""

    records: tuple[RuAsdResearchRecord, ...]
    available_stratum_counts: dict[str, int]
    selected_stratum_counts: dict[str, int]
    sha256_verified_archives: int


@dataclass(frozen=True, slots=True)
class ExtractedRuAsdResearchAsset:
    record_key: str
    relative_path: str
    sha256: str
    duration_s: float
    original_sr: int


def select_ruasd_research_records(
    archive_dir: Path,
    catalog: Mapping[str, RuAsdArchiveSpec],
    *,
    limit_per_label: int,
    min_per_stratum: int,
    seed: str,
    verify_sha256: bool = True,
    progress_callback: RuAsdResearchProgressCallback | None = None,
) -> RuAsdResearchSelection:
    """Choose a balanced, deterministic raw binary slice from every pinned artifact.

    A first TAR walk verifies the release and calculates available strata.  A second walk
    retains only the deterministic lowest-hash candidates required by the quota, avoiding a
    large in-memory index of the complete release.  SHA-256 is verified in the first walk;
    extraction in the same process later repeats TAR safety and size checks.
    """

    if limit_per_label <= 0 or min_per_stratum <= 0 or not seed:
        raise ValueError("limit_per_label, min_per_stratum, and seed must be positive.")
    archive_paths = _validate_archive_set(archive_dir, catalog)
    available_counts: Counter[str] = Counter()
    verified_archives = 0
    for completed, archive_name in enumerate(sorted(catalog), start=1):
        spec = catalog[archive_name]
        archive_path = archive_paths[archive_name]
        _validate_archive_size(archive_path, spec)
        if verify_sha256:
            if sha256_file(archive_path) != spec.sha256:
                raise RuAsdResearchError(
                    f"RuASD archive SHA-256 does not match catalog: {archive_name!r}."
                )
            verified_archives += 1
        for record in _eligible_records(archive_name, load_ruasd_archive_records(archive_path)):
            available_counts[record.stratum] += 1
        if progress_callback is not None:
            progress_callback(completed, len(catalog), archive_name)
    quotas = _allocate_stratum_quotas(available_counts, limit_per_label, min_per_stratum)
    selected = _deterministic_candidates(archive_paths, quotas, seed)
    selected_counts = Counter(record.stratum for record in selected)
    if dict(sorted(selected_counts.items())) != dict(sorted(quotas.items())):
        raise RuAsdResearchError("RuASD selection did not meet its deterministic stratum quotas.")
    return RuAsdResearchSelection(
        records=tuple(sorted(selected, key=lambda record: record.record_key)),
        available_stratum_counts=dict(sorted(available_counts.items())),
        selected_stratum_counts=dict(sorted(selected_counts.items())),
        sha256_verified_archives=verified_archives,
    )


def extract_ruasd_research_slice(
    archive_dir: Path,
    catalog: Mapping[str, RuAsdArchiveSpec],
    records: tuple[RuAsdResearchRecord, ...],
    destination: Path,
) -> dict[str, Path]:
    """Atomically extract only selected direct-file WAVs from the validated release."""

    if not records or destination.exists() or not destination.parent.is_dir():
        raise RuAsdResearchError(f"Unsafe RuASD research destination: {destination}")
    archive_paths = _validate_archive_set(archive_dir, catalog)
    requested: dict[str, set[str]] = defaultdict(set)
    for record in records:
        if record.archive_name not in catalog:
            raise RuAsdResearchError(
                f"Selected record has unknown archive: {record.archive_name!r}."
            )
        requested[record.archive_name].add(record.sample_id)
    try:
        with tempfile.TemporaryDirectory(
            prefix="kds-ruasd-research-", dir=destination.parent
        ) as stage_dir:
            stage = Path(stage_dir)
            extracted: set[str] = set()
            for archive_name in sorted(requested):
                archive_path = archive_paths[archive_name]
                _validate_archive_size(archive_path, catalog[archive_name])
                with tarfile.open(archive_path, mode="r:") as archive:
                    for member in archive:
                        sample_id, suffix = _safe_member_name(member)
                        if suffix != ".wav" or sample_id not in requested[archive_name]:
                            continue
                        source = archive.extractfile(member)
                        if source is None:
                            raise RuAsdResearchError(
                                f"Cannot read RuASD audio: {member.name!r}."
                            )
                        relative_path = Path(Path(archive_name).stem) / f"{sample_id}.wav"
                        output_path = stage / relative_path
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        with source, output_path.open("wb") as output:
                            shutil.copyfileobj(source, output, length=1024 * 1024)
                        extracted.add(f"{Path(archive_name).stem}:{sample_id}")
            expected = {record.record_key for record in records}
            if extracted != expected:
                raise RuAsdResearchError("RuASD extraction did not produce every selected WAV.")
            stage.replace(destination)
    except (OSError, tarfile.TarError, RuAsdCatalogError) as error:
        raise RuAsdResearchError(f"RuASD research extraction failed safely: {error}") from error
    return {
        record.record_key: destination / Path(record.archive_name).stem / f"{record.sample_id}.wav"
        for record in records
    }


def inspect_extracted_ruasd_research_audio(path: Path) -> tuple[float, int]:
    """Return basic facts required for a manifest after successful extraction."""

    try:
        info = sf.info(str(path))
    except RuntimeError as error:
        raise RuAsdResearchError(f"Cannot inspect RuASD audio: {path}") from error
    if info.duration <= 0 or info.samplerate <= 0:
        raise RuAsdResearchError(f"Invalid RuASD audio properties: {path}")
    return float(info.duration), int(info.samplerate)


def ruasd_research_manifest_rows(
    records: tuple[RuAsdResearchRecord, ...],
    assets: Mapping[str, ExtractedRuAsdResearchAsset],
    *,
    created_at: str | None = None,
) -> list[ManifestRow]:
    """Build binary RuASD research rows with text-safe but not speaker-safe provenance."""

    timestamp = created_at or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    rows: list[ManifestRow] = []
    for record in records:
        asset = assets.get(record.record_key)
        if asset is None:
            raise RuAsdResearchError(f"Missing RuASD asset: {record.record_key!r}.")
        record_id = f"{RUASD_RESEARCH_SOURCE_ID}:source-record:{record.record_key}"
        rows.append(
            ManifestRow(
                sample_id=f"{RUASD_RESEARCH_SOURCE_ID}:{record.record_key}",
                relative_path=asset.relative_path,
                sha256=asset.sha256,
                split="train",
                label=record.label,
                language="ru",
                code_switch="unknown",
                parent_group_id=record_id,
                source_name=RUASD_RESEARCH_SOURCE_ID,
                source_license=RUASD_RESEARCH_SOURCE_LICENSE,
                rights_basis=RUASD_RESEARCH_RIGHTS_BASIS,
                speaker_pseudo_id=record_id,
                text_id=(
                    f"{RUASD_RESEARCH_SOURCE_ID}:text:{record.text_hash}"
                    if record.has_source_text
                    else f"{RUASD_RESEARCH_SOURCE_ID}:content:{record.record_key}"
                ),
                text_hash=record.text_hash,
                duration_s=asset.duration_s,
                generator_family="tts" if record.label == "spoof" else "",
                generator_name=record.subset if record.label == "spoof" else "",
                generator_version=(
                    record.model or "unspecified_by_source" if record.label == "spoof" else ""
                ),
                voice_id=(
                    f"{RUASD_RESEARCH_SOURCE_ID}:voice:{RUASD_RESEARCH_UNKNOWN}"
                    if record.label == "spoof"
                    else ""
                ),
                clone_consent_id="",
                device=RUASD_RESEARCH_UNKNOWN,
                capture_route=RUASD_RESEARCH_CAPTURE_ROUTE,
                original_sr=asset.original_sr,
                codec="wav",
                augmentation_chain="",
                augmentation_seed="",
                created_at=timestamp,
            )
        )
    return rows


def _eligible_records(
    archive_name: str, metadata: list[RuAsdRecordMetadata]
) -> list[RuAsdResearchRecord]:
    records: list[RuAsdResearchRecord] = []
    for item in metadata:
        # ``load_ruasd_archive_records`` has already schema-validated every attribute.  The
        # explicit allow-list below prevents augmented or differently typed future records
        # from silently entering a binary training slice.
        label = item.label
        group = item.group
        source_type = item.source_type
        if group != "raw" or (label, source_type) not in {
            ("real", "real_speech"),
            ("fake", "tts"),
        }:
            continue
        sample_id = item.sample_id
        source_text_hash = item.source_text_hash
        record_key = f"{archive_name.removesuffix('.tar')}:{sample_id}"
        records.append(
            RuAsdResearchRecord(
                archive_name=archive_name,
                sample_id=sample_id,
                label="bonafide" if label == "real" else "spoof",
                subset=item.subset,
                model=item.model,
                text_hash=source_text_hash
                or hashlib.sha256(
                    f"{RUASD_RESEARCH_SOURCE_ID}:content:{record_key}".encode()
                ).hexdigest(),
                has_source_text=item.has_source_text,
            )
        )
    return records
def _allocate_stratum_quotas(
    counts: Mapping[str, int], limit_per_label: int, min_per_stratum: int
) -> dict[str, int]:
    by_label: dict[str, dict[str, int]] = {"bonafide": {}, "spoof": {}}
    for stratum, count in counts.items():
        label = stratum.split("/", 1)[0]
        if label not in by_label or count <= 0:
            raise RuAsdResearchError(f"Invalid RuASD research stratum: {stratum!r}.")
        by_label[label][stratum] = count
    quotas: dict[str, int] = {}
    for label, label_counts in by_label.items():
        if not label_counts:
            raise RuAsdResearchError(f"RuASD has no eligible {label} records.")
        total = sum(label_counts.values())
        base = min_per_stratum * len(label_counts)
        if limit_per_label > total or base > limit_per_label:
            raise RuAsdResearchError(
                f"Cannot allocate {limit_per_label} {label} rows across "
                f"{len(label_counts)} strata with min_per_stratum={min_per_stratum}."
            )
        label_quotas = {stratum: min_per_stratum for stratum in label_counts}
        remaining = limit_per_label - base
        capacities = {
            stratum: count - min_per_stratum for stratum, count in label_counts.items()
        }
        if any(capacity < 0 for capacity in capacities.values()):
            raise RuAsdResearchError(
                f"A RuASD {label} stratum contains fewer than {min_per_stratum} rows."
            )
        capacity_total = sum(capacities.values())
        if remaining > capacity_total:
            raise RuAsdResearchError(f"RuASD has too few eligible {label} records.")
        exact = {
            stratum: remaining * capacity / capacity_total if capacity_total else 0.0
            for stratum, capacity in capacities.items()
        }
        for stratum, value in exact.items():
            addition = min(capacities[stratum], int(value))
            label_quotas[stratum] += addition
        unassigned = limit_per_label - sum(label_quotas.values())
        for stratum in sorted(
            label_counts,
            key=lambda name: (-(exact[name] - int(exact[name])), name),
        ):
            if not unassigned:
                break
            if label_quotas[stratum] < label_counts[stratum]:
                label_quotas[stratum] += 1
                unassigned -= 1
        if unassigned:
            raise RuAsdResearchError(f"Cannot finish RuASD {label} stratum allocation.")
        quotas.update(label_quotas)
    return quotas


def _deterministic_candidates(
    archive_paths: Mapping[str, Path], quotas: Mapping[str, int], seed: str
) -> list[RuAsdResearchRecord]:
    heaps: dict[str, list[tuple[int, str, RuAsdResearchRecord]]] = {
        stratum: [] for stratum in quotas
    }
    for archive_name in sorted(archive_paths):
        for record in _eligible_records(
            archive_name, load_ruasd_archive_records(archive_paths[archive_name])
        ):
            quota = quotas.get(record.stratum)
            if quota is None:
                continue
            key = int.from_bytes(
                hashlib.sha256(f"{seed}:{record.record_key}".encode()).digest(), "big"
            )
            entry = (-key, record.record_key, record)
            heap = heaps[record.stratum]
            if len(heap) < quota:
                heapq.heappush(heap, entry)
            elif entry[0] > heap[0][0]:
                heapq.heapreplace(heap, entry)
    return [record for heap in heaps.values() for _, _, record in heap]
