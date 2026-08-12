from __future__ import annotations

import hashlib
import zipfile
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from kds.data.kazakhtts import (
    KAZAKHTTS_GENERATOR_FAMILY,
    VerifiedZipMember,
    extract_verified_kazakhtts_runtime,
    load_kazakhtts_runtime,
    validate_kazakhtts_text,
)
from kds.data.research_tts import ResearchTtsError, load_research_tts_model_lock


def _model_and_runtime():
    lock = load_research_tts_model_lock(
        Path("configs/research/kazakhtts_tacotron2_pwg_v1_models.json")
    )
    assert len(lock.models) == 1
    return lock.models[0], load_kazakhtts_runtime(lock.models[0])


def test_kazakhtts_lock_is_fixed_voice_no_reference_and_language_honest() -> None:
    model, runtime = _model_and_runtime()

    assert model.generator_family == KAZAKHTTS_GENERATOR_FAMILY
    assert runtime.fixed_voice_id == "ISSAI_KazakhTTS2_M2"
    assert runtime.supported_languages == ("kk",)
    assert runtime.conditional_smoke_languages == ("mixed", "ru")
    assert runtime.sample_rate == 22050
    assert runtime.espnet_version == "0.10.6"
    assert runtime.parallel_wavegan_version == "0.6.1"


def test_kazakhtts_extracts_only_verified_members_and_validates_configs(tmp_path: Path) -> None:
    _model, runtime = _model_and_runtime()
    acoustic_archive = tmp_path / "acoustic.zip"
    vocoder_archive = tmp_path / "vocoder.zip"
    acoustic_payloads = {
        "meta.yaml": yaml.safe_dump({"espnet": "0.10.3a4"}).encode(),
        "config.yaml": yaml.safe_dump(
            {
                "tts": "tacotron2",
                "token_type": "char",
                "g2p": None,
                "token_list": ["<blank>", "<unk>", "<space>", "а", "ә", "<sos/eos>"],
                "feats_extract_conf": {"fs": 22050},
            },
            allow_unicode=True,
        ).encode(),
        "checkpoint.pth": b"acoustic-checkpoint",
        "stats.npz": b"stats",
    }
    vocoder_payloads = {
        "config.yml": yaml.safe_dump(
            {
                "version": "0.4.8",
                "sampling_rate": 22050,
                "num_mels": 80,
                "generator_params": {"aux_channels": 80, "out_channels": 1},
            }
        ).encode(),
        "checkpoint.pkl": b"vocoder-checkpoint",
    }
    with zipfile.ZipFile(acoustic_archive, "w") as archive:
        for name, payload in acoustic_payloads.items():
            archive.writestr(name, payload)
        archive.writestr("ignored.py", b"must not be extracted")
    with zipfile.ZipFile(vocoder_archive, "w") as archive:
        for name, payload in vocoder_payloads.items():
            archive.writestr(name, payload)

    def member(name: str, payload: bytes) -> VerifiedZipMember:
        return VerifiedZipMember(name, len(payload), hashlib.sha256(payload).hexdigest())

    test_runtime = replace(
        runtime,
        acoustic_archive_path="acoustic.zip",
        acoustic_meta=member("meta.yaml", acoustic_payloads["meta.yaml"]),
        acoustic_config=member("config.yaml", acoustic_payloads["config.yaml"]),
        acoustic_checkpoint=member("checkpoint.pth", acoustic_payloads["checkpoint.pth"]),
        acoustic_stats=member("stats.npz", acoustic_payloads["stats.npz"]),
        vocoder_archive_path="vocoder.zip",
        vocoder_config=member("config.yml", vocoder_payloads["config.yml"]),
        vocoder_checkpoint=member("checkpoint.pkl", vocoder_payloads["checkpoint.pkl"]),
    )

    extracted = extract_verified_kazakhtts_runtime(
        verified_paths={"acoustic.zip": acoustic_archive, "vocoder.zip": vocoder_archive},
        runtime=test_runtime,
        destination=tmp_path / "runtime",
    )

    assert extracted.acoustic_checkpoint.read_bytes() == b"acoustic-checkpoint"
    assert not (tmp_path / "runtime" / "ignored.py").exists()
    assert validate_kazakhtts_text("  А Ә  ", extracted) == "а ә"
    with pytest.raises(ResearchTtsError, match="unsupported characters"):
        validate_kazakhtts_text("a", extracted)
    with pytest.raises(ResearchTtsError, match="overwrite"):
        extract_verified_kazakhtts_runtime(
            verified_paths={"acoustic.zip": acoustic_archive, "vocoder.zip": vocoder_archive},
            runtime=test_runtime,
            destination=tmp_path / "runtime",
        )
