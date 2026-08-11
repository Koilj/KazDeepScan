from __future__ import annotations

import hashlib
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

from kds.data.kazemotts import (
    _SOURCE_FILES,
    extract_verified_kazemotts_source,
    extract_verified_zip_member,
    load_kazemotts_runtime,
)
from kds.data.research_tts import ResearchTtsError, load_research_tts_model_lock


def _runtime():
    lock = load_research_tts_model_lock(Path("configs/research/kazemotts_kk_v1_models.json"))
    assert len(lock.models) == 1
    return load_kazemotts_runtime(lock.models[0])


def test_kazemotts_lock_declares_one_independent_family_and_balanced_profiles() -> None:
    runtime = _runtime()

    assert runtime.sample_rate == 22050
    assert runtime.n_timesteps == 10
    assert len(runtime.profiles) == 18
    assert {profile.speaker_id for profile in runtime.profiles} == {0, 1, 2}
    assert {profile.emotion_id for profile in runtime.profiles} == {0, 1, 2, 3, 4, 5}


def test_kazemotts_zip_member_requires_crc_size_and_sha256(tmp_path: Path) -> None:
    payload = b"pinned-checkpoint"
    archive = tmp_path / "model.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("bundle/checkpoint.pt", payload)
    destination = tmp_path / "checkpoint.pt"

    extract_verified_zip_member(
        archive,
        "bundle/checkpoint.pt",
        len(payload),
        hashlib.sha256(payload).hexdigest(),
        destination,
    )

    assert destination.read_bytes() == payload
    with pytest.raises(ResearchTtsError, match="overwrite"):
        extract_verified_zip_member(
            archive,
            "bundle/checkpoint.pt",
            len(payload),
            hashlib.sha256(payload).hexdigest(),
            destination,
        )


def test_kazemotts_source_extraction_allowlists_files_and_adds_inference_shim(
    tmp_path: Path,
) -> None:
    runtime = _runtime()
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for relative_path in sorted(_SOURCE_FILES):
            content = f"source:{relative_path}".encode()
            member = tarfile.TarInfo(f"{runtime.source_archive_root}/{relative_path}")
            member.size = len(content)
            handle.addfile(member, io.BytesIO(content))
        ignored = tarfile.TarInfo(f"{runtime.source_archive_root}/untrusted.so")
        ignored.size = len(b"not extracted")
        handle.addfile(ignored, io.BytesIO(b"not extracted"))

    source_root = extract_verified_kazemotts_source(archive, runtime, tmp_path / "source")

    assert (source_root / "model" / "tts.py").read_text(encoding="utf-8") == "source:model/tts.py"
    assert not (source_root / "untrusted.so").exists()
    shim = source_root / "model" / "monotonic_align" / "model" / "monotonic_align" / "core.py"
    assert "training-only" in shim.read_text(encoding="utf-8")
