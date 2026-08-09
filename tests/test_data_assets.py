from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from kds.data.assets import AssetValidationError, resolve_asset_path, validate_assets
from kds.data.manifest import ManifestRow
from tests.factories import manifest_mapping


def test_asset_validator_checks_file_and_sha256(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    asset_path = audio_root / "processed" / "ru" / "sample.wav"
    asset_path.parent.mkdir(parents=True)
    asset_path.write_bytes(b"audio-content")
    digest = hashlib.sha256(b"audio-content").hexdigest()
    row = ManifestRow.from_mapping(
        manifest_mapping(relative_path="processed/ru/sample.wav", sha256=digest), row_number=2
    )

    report = validate_assets([row], audio_root)

    assert report.is_valid
    assert report.checked == 1
    assert report.verified == 1


def test_asset_validator_reports_hash_mismatch(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    asset_path = audio_root / "processed" / "ru" / "sample.wav"
    asset_path.parent.mkdir(parents=True)
    asset_path.write_bytes(b"actual")
    row = ManifestRow.from_mapping(
        manifest_mapping(relative_path="processed/ru/sample.wav", sha256="0" * 64), row_number=2
    )

    report = validate_assets([row], audio_root)

    assert not report.is_valid
    assert "SHA-256 mismatch" in report.issues[0].detail


def test_symlink_asset_cannot_escape_audio_root(tmp_path: Path) -> None:
    audio_root = tmp_path / "audio"
    audio_root.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"outside")
    (audio_root / "processed").mkdir()
    (audio_root / "processed" / "link.wav").symlink_to(outside)

    with pytest.raises(AssetValidationError, match="escapes audio root"):
        resolve_asset_path(audio_root, "processed/link.wav")
