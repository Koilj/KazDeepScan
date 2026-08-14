from __future__ import annotations

import json
from array import array
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from kds.audio.contracts import (
    AudioQuality,
    MediaInfo,
    PreparationStatus,
    SpeechSegment,
    WindowDescriptor,
)
from kds.audio.pipeline import PreparedAudio
from kds.audio.waveform import Waveform
from kds.cli import main
from kds.inference import (
    RESEARCH_ONLY_WARNING,
    ResearchInferenceContractError,
    assert_user_audio_path_allowed,
    file_sha256,
    load_research_inference_contract,
    load_research_inference_engine,
)
from kds.models import B0Config, B0LogMelCnn
from kds.training.frozen_b0 import state_dict_sha256


def _write_contract(root: Path, *, warning: str = RESEARCH_ONLY_WARNING) -> Path:
    model_dir = root / "models"
    contract_dir = root / "configs" / "inference"
    model_dir.mkdir(parents=True)
    contract_dir.mkdir(parents=True)
    for directory in (root / "data", root / "artifacts", root / "checkpoints"):
        directory.mkdir()

    config = B0Config()
    model = B0LogMelCnn(config)
    state_dict = model.state_dict()
    for tensor in state_dict.values():
        tensor.zero_()
    state_dict["classifier.2.bias"].fill_(2.0)
    checkpoint = model_dir / "checkpoint.pt"
    torch.save(
        {
            "model_name": "b0_logmel_cnn",
            "model_config": asdict(config),
            "training_seed": "test",
            "training_purpose": "research",
            "source_mixed_research_matrix": {},
            "best_dev_loss": 0.5,
            "final_test_metrics": {},
            "state_dict": state_dict,
        },
        checkpoint,
    )
    payload = {
        "schema_version": 1,
        "contract_id": "test-user-audio-research-v1",
        "purpose": "research_user_audio_only",
        "input_scope": {
            "allowed": "user_supplied_external_audio_only",
            "prohibited_project_roots": [
                "../../data",
                "../../models",
                "../../artifacts",
                "../../checkpoints",
            ],
            "training_data_overlap": "unverified",
        },
        "checkpoint": {
            "path": "../../models/checkpoint.pt",
            "sha256": file_sha256(checkpoint),
            "state_dict_sha256": state_dict_sha256(state_dict),
            "model_id": "test-b0",
            "architecture": "b0_logmel_cnn",
            "training_purpose": "research",
            "model_config": asdict(config),
        },
        "preprocessing": {
            "target_sample_rate": 16_000,
            "minimum_speech_seconds": 2.5,
            "window_samples": 64_600,
            "hop_samples": 32_000,
            "short_window_policy": "repeat_to_window",
            "vad_scope": "speech_segments_only",
        },
        "inference": {
            "device": "cpu",
            "batch_size": 16,
            "aggregation": "duration_weighted_mean_raw_logit",
            "score_transform": "sigmoid_uncalibrated",
            "raw_logit_boundary": 0.0,
            "repeat_completed_evaluation_prohibited": True,
        },
        "output_semantics": {
            "score_name": "uncalibrated_spoof_score",
            "calibrated": False,
            "probability_claim": False,
            "fraud_claim": False,
            "product_grade": False,
            "warning": warning,
        },
        "limitations": [
            "uncalibrated_score_not_probability",
            "training_data_overlap_unverified",
            "not_speaker_independent",
            "not_fraud_determination",
            "not_product_grade",
        ],
    }
    contract = contract_dir / "contract.json"
    contract.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return contract


def _prepared_audio() -> PreparedAudio:
    sample_count = 48_000
    segment = SpeechSegment(0, sample_count, 16_000)
    window = WindowDescriptor(0, sample_count, 64_600, 16_000)
    return PreparedAudio(
        media=MediaInfo(Path("user.wav"), ("wav",), 3.0, 1),
        waveform=Waveform(array("h", [1_000] * sample_count), 16_000),
        quality=AudioQuality(peak=0.1, rms_dbfs=-20.0, clipped_fraction=0.0, dc_offset=0.0),
        speech_segments=(segment,),
        speech_seconds=3.0,
        windows=(window,),
        status=PreparationStatus.READY,
        quality_flags=(),
    )


def test_engine_loads_pinned_checkpoint_and_scores_without_calibration(tmp_path: Path) -> None:
    engine = load_research_inference_engine(_write_contract(tmp_path))

    result = engine.score(_prepared_audio())

    assert result.raw_spoof_logit == pytest.approx(2.0)
    assert result.uncalibrated_spoof_score == pytest.approx(torch.sigmoid(torch.tensor(2.0)).item())
    assert result.interpretation == "spoof_like"
    assert len(result.windows) == 1
    assert not engine.contract.calibrated
    assert not engine.contract.fraud_claim
    assert not engine.contract.product_grade


def test_engine_rejects_checkpoint_hash_mismatch(tmp_path: Path) -> None:
    contract_path = _write_contract(tmp_path)
    contract = load_research_inference_contract(contract_path)
    contract.checkpoint.path.write_bytes(b"changed")

    with pytest.raises(ResearchInferenceContractError, match="checkpoint SHA-256 mismatch"):
        load_research_inference_engine(contract)


def test_contract_rejects_weakened_research_warning(tmp_path: Path) -> None:
    contract_path = _write_contract(tmp_path, warning="safe")

    with pytest.raises(ResearchInferenceContractError, match="warning was changed"):
        load_research_inference_contract(contract_path)


def test_user_audio_guard_rejects_project_data_and_allows_external_file(tmp_path: Path) -> None:
    contract = load_research_inference_contract(_write_contract(tmp_path))
    project_audio = tmp_path / "data" / "frozen.wav"
    project_audio.write_bytes(b"audio")
    external_audio = tmp_path / "external-user.wav"
    external_audio.write_bytes(b"audio")

    with pytest.raises(ResearchInferenceContractError, match="refuses project data/model roots"):
        assert_user_audio_path_allowed(contract, project_audio)

    assert assert_user_audio_path_allowed(contract, external_audio) == external_audio.resolve()


def test_cli_requires_explicit_research_acknowledgement(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["research-infer", "missing.wav"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert payload["code"] == "research_acknowledgement_required"
    assert payload["warning"] == RESEARCH_ONLY_WARNING


def test_cli_validates_separate_contract_and_checkpoint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    contract_path = _write_contract(tmp_path)

    exit_code = main(["validate-research-inference", "--contract", str(contract_path)])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["status"] == "ready"
    assert payload["research_only"] is True
    assert payload["calibrated"] is False
    assert payload["fraud_claim"] is False
    assert payload["product_grade"] is False
