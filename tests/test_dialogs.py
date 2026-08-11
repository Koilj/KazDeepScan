from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import pytest

import kds.data.dialogs as dialogs


def _git_blob_id(path: Path) -> str:
    payload = path.read_bytes()
    return hashlib.sha1(f"blob {len(payload)}\0".encode("ascii") + payload).hexdigest()


def _wav(path: Path) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(b"\0\0" * 160)


def _csv_content(paths: list[str]) -> str:
    rows = ["|".join(dialogs.DIALOGS_EXPECTED_CSV_COLUMNS)]
    for path in paths:
        rows.append(f"{path}|M|Текст|neutral|нейтральная|Текст|0.01")
    return "\n".join(rows) + "\n"


def _fixture_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, include_missing_metadata: bool = False
) -> Path:
    root = tmp_path / "release"
    wav_root = root / "wavs"
    wav_root.mkdir(parents=True)
    wav_path = wav_root / "present.wav"
    _wav(wav_path)
    paths = ["wavs/present.wav"]
    if include_missing_metadata:
        paths.append("wavs/missing.wav")
    files = {
        ".gitattributes": "*.wav filter=lfs\n",
        "LICENSE.md": "license\n",
        "README.md": "# Dialogs\n",
        "preview.parquet": "preview\n",
        "metadata.csv": _csv_content(paths),
        "train.csv": _csv_content(paths),
        "val.csv": _csv_content([]),
        "test.csv": _csv_content([]),
    }
    for relative_path, content in files.items():
        (root / relative_path).write_text(content, encoding="utf-8")

    tree_entries: dict[str, dict[str, object]] = {}
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        relative_path = path.relative_to(root).as_posix()
        if relative_path.endswith(".wav"):
            tree_entries[relative_path] = {
                "size": path.stat().st_size,
                "lfs_size": path.stat().st_size,
                "lfs_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        else:
            tree_entries[relative_path] = {
                "size": path.stat().st_size,
                "blob_id": _git_blob_id(path),
            }
    revision = "test-revision"
    tree_path = root / ".cache" / "huggingface" / "trees" / f"{revision}.json"
    tree_path.parent.mkdir(parents=True)
    tree_path.write_text(
        json.dumps({"format_version": 1, "files": tree_entries}, sort_keys=True), encoding="utf-8"
    )
    monkeypatch.setattr(dialogs, "DIALOGS_REVISION", revision)
    monkeypatch.setattr(
        dialogs, "DIALOGS_SNAPSHOT_TREE_SHA256", hashlib.sha256(tree_path.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(dialogs, "DIALOGS_EXPECTED_SOURCE_FILES", len(tree_entries))
    monkeypatch.setattr(dialogs, "DIALOGS_EXPECTED_LFS_FILES", 1)
    monkeypatch.setattr(dialogs, "DIALOGS_EXPECTED_WAV_FILES", 1)
    source_bytes = sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and ".cache" not in path.parts
    )
    monkeypatch.setattr(dialogs, "DIALOGS_EXPECTED_SOURCE_BYTES", source_bytes)
    monkeypatch.setattr(
        dialogs,
        "DIALOGS_EXPECTED_ROWS_BY_CSV",
        {"metadata.csv": len(paths), "train.csv": len(paths), "val.csv": 0, "test.csv": 0},
    )
    monkeypatch.setattr(dialogs, "DIALOGS_EXPECTED_ROWS_BY_SPEAKER", {"M": len(paths)})
    return root


def test_audit_dialogs_release_accepts_complete_hash_pinned_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture_release(tmp_path, monkeypatch)

    audit = dialogs.audit_dialogs_release(root)

    assert audit.eligible_for_bonafide_final
    assert audit.intake_status == "accepted"
    assert audit.lfs_files_verified == 1
    assert audit.git_blob_files_verified == 8
    assert audit.metadata_missing_wavs == 0


def test_audit_dialogs_release_records_incomplete_published_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture_release(tmp_path, monkeypatch, include_missing_metadata=True)

    audit = dialogs.audit_dialogs_release(root)

    assert not audit.eligible_for_bonafide_final
    assert audit.intake_status == "rejected_incomplete_published_release"
    assert audit.metadata_missing_wavs == 1
    assert audit.metadata_missing_by_speaker == {"M": 1}
    assert audit.metadata_missing_path_samples == ("wavs/missing.wav",)
    with pytest.raises(dialogs.DialogsAuditError, match="ineligible"):
        dialogs.require_dialogs_bonafide_final(audit)


def test_audit_dialogs_release_rejects_changed_lfs_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _fixture_release(tmp_path, monkeypatch)
    original_size = (root / "wavs/present.wav").stat().st_size
    (root / "wavs/present.wav").write_bytes(b"x" * original_size)

    with pytest.raises(dialogs.DialogsAuditError, match="LFS SHA-256 mismatch"):
        dialogs.audit_dialogs_release(root)
