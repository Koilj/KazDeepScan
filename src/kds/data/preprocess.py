from __future__ import annotations

import tempfile
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from kds.audio.contracts import PreparationStatus
from kds.audio.pipeline import PreparedAudio
from kds.data.assets import resolve_asset_path, sha256_file
from kds.data.manifest import ManifestRow


class AudioPreparer(Protocol):
    def prepare_to_wav(self, source: Path, destination: Path) -> PreparedAudio:
        """Write a new normalized WAV and return its readiness result."""


@dataclass(frozen=True, slots=True)
class PreprocessIssue:
    sample_id: str
    relative_path: str
    detail: str


@dataclass(frozen=True, slots=True)
class PreprocessReport:
    processed_rows: tuple[ManifestRow, ...]
    issues: tuple[PreprocessIssue, ...]

    @property
    def is_successful(self) -> bool:
        return not self.issues


@dataclass(frozen=True, slots=True)
class PreprocessReuse:
    reused_rows: tuple[ManifestRow, ...]
    remaining_rows: tuple[ManifestRow, ...]


def processed_relative_path(row: ManifestRow) -> str:
    return f"processed/{row.sha256[:2]}/{row.sha256}.wav"


def reuse_preprocessed_rows(
    rows: Iterable[ManifestRow],
    prior_raw_rows: Iterable[ManifestRow],
    prior_ready_rows: Iterable[ManifestRow],
) -> PreprocessReuse:
    """Reuse an exact prior normalized asset while retaining new manifest provenance.

    Reuse is permitted only for the same sample ID and identical raw bytes.  This prevents an
    existing content-addressed destination from silently turning a different source record into
    a duplicate sample.
    """

    prior_raw = {row.sample_id: row for row in prior_raw_rows}
    prior_ready = {row.sample_id: row for row in prior_ready_rows}
    reused: list[ManifestRow] = []
    remaining: list[ManifestRow] = []
    for row in rows:
        old_raw = prior_raw.get(row.sample_id)
        old_ready = prior_ready.get(row.sample_id)
        if old_raw is None or old_ready is None or old_raw.sha256 != row.sha256:
            remaining.append(row)
            continue
        if old_ready.relative_path != processed_relative_path(old_raw):
            raise ValueError(
                f"Prior ready asset path is not derived from raw bytes: {row.sample_id!r}."
            )
        reused.append(
            replace(
                row,
                relative_path=old_ready.relative_path,
                sha256=old_ready.sha256,
                duration_s=old_ready.duration_s,
                codec="wav",
            )
        )
    return PreprocessReuse(tuple(reused), tuple(remaining))


def preprocess_rows(
    rows: Iterable[ManifestRow],
    data_root: Path,
    preparer: AudioPreparer,
    allow_rejections: bool = False,
) -> PreprocessReport:
    """Stage normalized assets before publishing them.

    By default a rejected asset prevents publication of every staged WAV. Callers that choose to
    keep ready assets despite rejections must record every issue outside this function.
    """

    processed_rows: list[ManifestRow] = []
    issues: list[PreprocessIssue] = []
    resolved_root = data_root.resolve(strict=True)

    with tempfile.TemporaryDirectory(prefix="kds-preprocess-batch-", dir=resolved_root) as stage:
        stage_root = Path(stage)
        staged_assets: list[tuple[Path, Path]] = []
        for row in rows:
            try:
                source = resolve_asset_path(resolved_root, row.relative_path)
                if not source.is_file():
                    raise ValueError("Raw audio asset does not exist.")
                relative_destination = processed_relative_path(row)
                destination = resolve_asset_path(resolved_root, relative_destination)
                if destination.exists():
                    raise ValueError(
                        f"Refusing to overwrite processed asset: {relative_destination}"
                    )
                staged_path = stage_root / relative_destination
                staged_path.parent.mkdir(parents=True, exist_ok=True)
                prepared = preparer.prepare_to_wav(source, staged_path)
                if prepared.status is not PreparationStatus.READY:
                    raise ValueError(
                        f"Audio is not trainable: {prepared.status.value} "
                        f"({', '.join(prepared.quality_flags) or 'no quality flags'})."
                    )
                processed_rows.append(
                    replace(
                        row,
                        relative_path=relative_destination,
                        sha256=sha256_file(staged_path),
                        duration_s=prepared.waveform.duration_seconds,
                        codec="wav",
                    )
                )
                staged_assets.append((staged_path, destination))
            except (OSError, ValueError, RuntimeError) as error:
                issues.append(PreprocessIssue(row.sample_id, row.relative_path, str(error)))

        if issues and not allow_rejections:
            return PreprocessReport(processed_rows=(), issues=tuple(issues))

        for _staged_path, destination in staged_assets:
            if destination.exists():
                return PreprocessReport(
                    processed_rows=(),
                    issues=(
                        PreprocessIssue(
                            sample_id="batch",
                            relative_path=destination.relative_to(resolved_root).as_posix(),
                            detail="Processed destination appeared while batch was staging.",
                        ),
                    ),
                )
        for staged_path, destination in staged_assets:
            destination.parent.mkdir(parents=True, exist_ok=True)
            staged_path.replace(destination)

    return PreprocessReport(processed_rows=tuple(processed_rows), issues=tuple(issues))
