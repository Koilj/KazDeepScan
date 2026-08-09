from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from kds.data.manifest import ManifestRow

HASH_CHUNK_BYTES = 1024 * 1024


class AssetValidationError(ValueError):
    """Raised when manifest entries do not resolve to trustworthy local assets."""


@dataclass(frozen=True, slots=True)
class AssetIssue:
    sample_id: str
    relative_path: str
    detail: str


@dataclass(frozen=True, slots=True)
class AssetValidationReport:
    checked: int
    verified: int
    issues: tuple[AssetIssue, ...]

    @property
    def is_valid(self) -> bool:
        return not self.issues


def resolve_asset_path(audio_root: Path, relative_path: str) -> Path:
    """Resolve a manifest path while preventing a symlink escape from ``audio_root``."""

    resolved_root = audio_root.resolve(strict=True)
    candidate = (resolved_root / relative_path).resolve(strict=False)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise AssetValidationError(
            f"Manifest path escapes audio root: {relative_path!r}."
        ) from error
    return candidate


def sha256_file(path: Path, chunk_bytes: int = HASH_CHUNK_BYTES) -> str:
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive.")
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        while chunk := file_handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def validate_assets(
    rows: Iterable[ManifestRow], audio_root: Path, verify_sha256: bool = True
) -> AssetValidationReport:
    """Check that every manifest asset exists and, by default, matches its recorded digest."""

    issues: list[AssetIssue] = []
    checked = 0
    verified = 0
    for row in rows:
        checked += 1
        try:
            path = resolve_asset_path(audio_root, row.relative_path)
        except (AssetValidationError, FileNotFoundError) as error:
            issues.append(AssetIssue(row.sample_id, row.relative_path, str(error)))
            continue
        if not path.is_file():
            issues.append(
                AssetIssue(row.sample_id, row.relative_path, "Audio asset does not exist.")
            )
            continue
        if verify_sha256:
            actual_hash = sha256_file(path)
            if actual_hash != row.sha256:
                issues.append(
                    AssetIssue(
                        row.sample_id,
                        row.relative_path,
                        f"SHA-256 mismatch: expected {row.sha256}, got {actual_hash}.",
                    )
                )
                continue
        verified += 1
    return AssetValidationReport(checked=checked, verified=verified, issues=tuple(issues))


def require_valid_assets(
    rows: Iterable[ManifestRow], audio_root: Path, verify_sha256: bool = True
) -> None:
    report = validate_assets(rows, audio_root, verify_sha256=verify_sha256)
    if report.is_valid:
        return
    details = "\n".join(
        f"{issue.sample_id} ({issue.relative_path}): {issue.detail}" for issue in report.issues
    )
    raise AssetValidationError(details)
