"""Integrity intake for the pinned Dialogs Russian conversation release.

The audit verifies a locally downloaded Hugging Face snapshot against its exact
snapshot-tree hash, then verifies the relationship between the published CSVs and
the WAV tree.  It deliberately does not create a manifest: the pinned release is
incomplete relative to its own metadata and is therefore ineligible for a
bona-fide final layer.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from kds.data.assets import sha256_file

DIALOGS_SOURCE_ID = "dialogs_ru_v1"
DIALOGS_DATASET = "langswap/dialogs-ru-emotional-conversations"
DIALOGS_REVISION = "e25ba617b2b56bd1dbf255d3905c51bd8da3d31f"
DIALOGS_SNAPSHOT_TREE_SHA256 = "0e44fb25264c8b4486d968923cf7fcd9bf08c35d5d354a068a723aa02003eff0"
DIALOGS_EXPECTED_SOURCE_FILES = 10_004
DIALOGS_EXPECTED_LFS_FILES = 9_997
DIALOGS_EXPECTED_WAV_FILES = 9_996
DIALOGS_EXPECTED_SOURCE_BYTES = 5_571_006_447
DIALOGS_EXPECTED_CSV_COLUMNS = (
    "audio_path",
    "speaker_id",
    "text",
    "emotion",
    "emotion_on_russian",
    "accent_text",
    "duration",
)
DIALOGS_EXPECTED_ROWS_BY_CSV = {
    "metadata.csv": 11_796,
    "train.csv": 11_428,
    "val.csv": 180,
    "test.csv": 188,
}
DIALOGS_EXPECTED_ROWS_BY_SPEAKER = {"D": 2_245, "M": 5_935, "S": 3_616}


class DialogsAuditError(ValueError):
    """Raised when a Dialogs artifact differs from the pinned release."""


@dataclass(frozen=True, slots=True)
class DialogsAudit:
    """Hash and membership result for one local Dialogs snapshot."""

    source_id: str
    dataset: str
    revision: str
    snapshot_tree_sha256: str
    source_artifact_files: int
    source_artifact_bytes: int
    lfs_files_verified: int
    git_blob_files_verified: int
    wav_files: int
    rows_by_csv: dict[str, int]
    rows_by_speaker: dict[str, int]
    split_overlap_rows: dict[str, int]
    metadata_unique_audio_paths: int
    split_union_audio_paths: int
    metadata_missing_wavs: int
    metadata_missing_duration_s: float
    metadata_missing_by_speaker: dict[str, int]
    metadata_missing_path_samples: tuple[str, ...]
    available_wav_duration_s: float
    available_wav_by_speaker: dict[str, int]
    intake_status: str
    eligible_for_bonafide_final: bool


def _git_blob_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    digest.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or value != path.as_posix()
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
    ):
        raise DialogsAuditError(f"Unsafe path in Dialogs snapshot tree: {value!r}.")
    return path


def _load_snapshot_tree(root: Path) -> dict[str, dict[str, object]]:
    tree_path = root / ".cache" / "huggingface" / "trees" / f"{DIALOGS_REVISION}.json"
    if not tree_path.is_file():
        raise DialogsAuditError(
            "Dialogs pinned snapshot tree is absent. Download with "
            "huggingface_hub.snapshot_download at the required revision."
        )
    actual_tree_sha256 = sha256_file(tree_path)
    if actual_tree_sha256 != DIALOGS_SNAPSHOT_TREE_SHA256:
        raise DialogsAuditError(
            "Dialogs snapshot-tree SHA-256 differs from the pinned release: "
            f"expected {DIALOGS_SNAPSHOT_TREE_SHA256}, got {actual_tree_sha256}."
        )
    try:
        payload: Any = json.loads(tree_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DialogsAuditError("Dialogs snapshot tree cannot be parsed as JSON.") from error
    if (
        not isinstance(payload, dict)
        or payload.get("format_version") != 1
        or not isinstance(payload.get("files"), dict)
    ):
        raise DialogsAuditError("Dialogs snapshot tree has an unexpected schema.")
    files = payload["files"]
    typed_files: dict[str, dict[str, object]] = {}
    for relative_path, entry in files.items():
        if not isinstance(relative_path, str) or not isinstance(entry, dict):
            raise DialogsAuditError("Dialogs snapshot tree has an invalid file entry.")
        _safe_relative_path(relative_path)
        typed_files[relative_path] = entry
    return typed_files


def _actual_source_paths(root: Path) -> set[str]:
    if not root.is_dir():
        raise DialogsAuditError(f"Dialogs artifact root does not exist: {root}")
    actual_paths: set[str] = set()
    for path in root.rglob("*"):
        relative_path = path.relative_to(root).as_posix()
        if relative_path == ".cache" or relative_path.startswith(".cache/"):
            continue
        if path.is_symlink():
            raise DialogsAuditError(f"Dialogs artifact tree has a symlink: {relative_path!r}.")
        if path.is_file():
            actual_paths.add(relative_path)
        elif not path.is_dir():
            raise DialogsAuditError(
                f"Dialogs artifact tree has an unexpected entry: {relative_path!r}."
            )
    return actual_paths


def _verify_source_artifacts(root: Path, tree: dict[str, dict[str, object]]) -> tuple[int, int]:
    actual_paths = _actual_source_paths(root)
    tree_paths = set(tree)
    if actual_paths != tree_paths:
        missing = sorted(tree_paths.difference(actual_paths))
        unexpected = sorted(actual_paths.difference(tree_paths))
        raise DialogsAuditError(
            "Dialogs artifact tree differs from the pinned snapshot: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}."
        )
    if len(tree) != DIALOGS_EXPECTED_SOURCE_FILES:
        raise DialogsAuditError(
            "Dialogs snapshot file count differs from the pinned release: "
            f"expected {DIALOGS_EXPECTED_SOURCE_FILES}, got {len(tree)}."
        )

    source_bytes = 0
    lfs_files_verified = 0
    git_blob_files_verified = 0
    for relative_path, entry in sorted(tree.items()):
        path = root / _safe_relative_path(relative_path)
        expected_size = entry.get("size")
        if not isinstance(expected_size, int) or expected_size < 0:
            raise DialogsAuditError(
                f"Dialogs snapshot entry has an invalid size: {relative_path!r}."
            )
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise DialogsAuditError(
                f"Dialogs size mismatch for {relative_path}: expected {expected_size}, "
                f"got {actual_size}."
            )
        source_bytes += actual_size
        expected_lfs_sha256 = entry.get("lfs_sha256")
        if isinstance(expected_lfs_sha256, str):
            expected_lfs_size = entry.get("lfs_size")
            if expected_lfs_size != expected_size:
                raise DialogsAuditError(
                    f"Dialogs LFS size metadata differs for {relative_path!r}."
                )
            actual_sha256 = sha256_file(path)
            if actual_sha256 != expected_lfs_sha256:
                raise DialogsAuditError(
                    f"Dialogs LFS SHA-256 mismatch for {relative_path}: "
                    f"expected {expected_lfs_sha256}, got {actual_sha256}."
                )
            lfs_files_verified += 1
            continue
        expected_blob_id = entry.get("blob_id")
        if not isinstance(expected_blob_id, str):
            raise DialogsAuditError(f"Dialogs entry lacks a usable hash: {relative_path!r}.")
        actual_blob_id = _git_blob_sha1(path)
        if actual_blob_id != expected_blob_id:
            raise DialogsAuditError(
                f"Dialogs Git blob SHA-1 mismatch for {relative_path}: "
                f"expected {expected_blob_id}, got {actual_blob_id}."
            )
        git_blob_files_verified += 1
    if source_bytes != DIALOGS_EXPECTED_SOURCE_BYTES:
        raise DialogsAuditError(
            "Dialogs source-byte total differs from the pinned release: "
            f"expected {DIALOGS_EXPECTED_SOURCE_BYTES}, got {source_bytes}."
        )
    if lfs_files_verified != DIALOGS_EXPECTED_LFS_FILES:
        raise DialogsAuditError(
            "Dialogs LFS file count differs from the pinned release: "
            f"expected {DIALOGS_EXPECTED_LFS_FILES}, got {lfs_files_verified}."
        )
    return lfs_files_verified, git_blob_files_verified


def _read_csv(root: Path, name: str) -> list[dict[str, str]]:
    path = root / name
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="|")
            if tuple(reader.fieldnames or ()) != DIALOGS_EXPECTED_CSV_COLUMNS:
                raise DialogsAuditError(
                    f"Dialogs CSV columns differ in {name}: {reader.fieldnames!r}."
                )
            rows = list(reader)
    except OSError as error:
        raise DialogsAuditError(f"Dialogs CSV cannot be read: {name!r}.") from error
    if len(rows) != DIALOGS_EXPECTED_ROWS_BY_CSV[name]:
        raise DialogsAuditError(
            f"Dialogs row count differs in {name}: expected {DIALOGS_EXPECTED_ROWS_BY_CSV[name]}, "
            f"got {len(rows)}."
        )
    for row_number, row in enumerate(rows, start=2):
        if None in row or not row["audio_path"] or not row["speaker_id"]:
            raise DialogsAuditError(f"Dialogs {name} has an incomplete row {row_number}.")
        try:
            duration_s = float(row["duration"])
        except ValueError as error:
            raise DialogsAuditError(
                f"Dialogs {name} has an invalid duration at row {row_number}."
            ) from error
        if not math.isfinite(duration_s) or duration_s <= 0:
            raise DialogsAuditError(
                f"Dialogs {name} has a non-positive duration at row {row_number}."
            )
        audio_path = _safe_relative_path(row["audio_path"])
        if (
            len(audio_path.parts) != 2
            or audio_path.parts[0] != "wavs"
            or audio_path.suffix != ".wav"
        ):
            raise DialogsAuditError(
                f"Dialogs {name} has an unexpected audio path at row {row_number}."
            )
    return rows


def audit_dialogs_release(root: Path) -> DialogsAudit:
    """Audit the pinned Dialogs release and record its publication defect.

    A structurally valid local snapshot can still be ineligible: every published
    metadata row must resolve to a supplied WAV before it may enter a protocol.
    """

    tree = _load_snapshot_tree(root)
    lfs_files_verified, git_blob_files_verified = _verify_source_artifacts(root, tree)
    tree_wavs = {path for path in tree if path.startswith("wavs/")}
    if len(tree_wavs) != DIALOGS_EXPECTED_WAV_FILES:
        raise DialogsAuditError(
            "Dialogs WAV count differs from the pinned release: "
            f"expected {DIALOGS_EXPECTED_WAV_FILES}, got {len(tree_wavs)}."
        )

    csv_rows = {name: _read_csv(root, name) for name in DIALOGS_EXPECTED_ROWS_BY_CSV}
    metadata_rows = csv_rows["metadata.csv"]
    metadata_paths = [row["audio_path"] for row in metadata_rows]
    if len(metadata_paths) != len(set(metadata_paths)):
        raise DialogsAuditError("Dialogs metadata repeats an audio_path.")
    rows_by_speaker = dict(sorted(Counter(row["speaker_id"] for row in metadata_rows).items()))
    if rows_by_speaker != DIALOGS_EXPECTED_ROWS_BY_SPEAKER:
        raise DialogsAuditError(
            "Dialogs speaker-row counts differ from the pinned release: "
            f"expected {DIALOGS_EXPECTED_ROWS_BY_SPEAKER}, got {rows_by_speaker}."
        )

    split_paths = {
        name: {row["audio_path"] for row in rows}
        for name, rows in csv_rows.items()
        if name != "metadata.csv"
    }
    overlaps = {
        "train_val": len(split_paths["train.csv"] & split_paths["val.csv"]),
        "train_test": len(split_paths["train.csv"] & split_paths["test.csv"]),
        "val_test": len(split_paths["val.csv"] & split_paths["test.csv"]),
    }
    if any(overlaps.values()):
        raise DialogsAuditError(f"Dialogs split membership overlaps: {overlaps}.")
    split_union = set().union(*split_paths.values())
    if set(metadata_paths) != split_union:
        raise DialogsAuditError("Dialogs metadata paths differ from the split union.")

    missing_rows = [row for row in metadata_rows if row["audio_path"] not in tree_wavs]
    available_rows = [row for row in metadata_rows if row["audio_path"] in tree_wavs]
    missing_by_speaker = dict(sorted(Counter(row["speaker_id"] for row in missing_rows).items()))
    available_by_speaker = dict(
        sorted(Counter(row["speaker_id"] for row in available_rows).items())
    )
    missing_duration_s = sum(float(row["duration"]) for row in missing_rows)
    available_duration_s = sum(float(row["duration"]) for row in available_rows)
    eligible_for_bonafide_final = not missing_rows
    return DialogsAudit(
        source_id=DIALOGS_SOURCE_ID,
        dataset=DIALOGS_DATASET,
        revision=DIALOGS_REVISION,
        snapshot_tree_sha256=DIALOGS_SNAPSHOT_TREE_SHA256,
        source_artifact_files=len(tree),
        source_artifact_bytes=DIALOGS_EXPECTED_SOURCE_BYTES,
        lfs_files_verified=lfs_files_verified,
        git_blob_files_verified=git_blob_files_verified,
        wav_files=len(tree_wavs),
        rows_by_csv={name: len(rows) for name, rows in sorted(csv_rows.items())},
        rows_by_speaker=rows_by_speaker,
        split_overlap_rows=overlaps,
        metadata_unique_audio_paths=len(set(metadata_paths)),
        split_union_audio_paths=len(split_union),
        metadata_missing_wavs=len(missing_rows),
        metadata_missing_duration_s=missing_duration_s,
        metadata_missing_by_speaker=missing_by_speaker,
        metadata_missing_path_samples=tuple(
            sorted(row["audio_path"] for row in missing_rows)[:12]
        ),
        available_wav_duration_s=available_duration_s,
        available_wav_by_speaker=available_by_speaker,
        intake_status=(
            "accepted" if eligible_for_bonafide_final else "rejected_incomplete_published_release"
        ),
        eligible_for_bonafide_final=eligible_for_bonafide_final,
    )


def require_dialogs_bonafide_final(audit: DialogsAudit) -> None:
    """Refuse use of a Dialogs audit result that lacks every published WAV."""

    if audit.eligible_for_bonafide_final:
        return
    raise DialogsAuditError(
        "Dialogs is ineligible for a bona-fide final layer: "
        f"{audit.metadata_missing_wavs} published metadata paths have no WAV."
    )
