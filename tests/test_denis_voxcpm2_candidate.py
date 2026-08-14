from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from kds.data.assets import sha256_file
from kds.data.denis_voxcpm2_candidate import (
    DENIS_VOXCPM2_SOURCE_ID,
    DENIS_VOXCPM2_VOICE_ID,
    DenisVoxCPM2CandidateError,
    denis_voxcpm2_spoof_row,
)
from kds.data.manifest import ManifestRow, load_manifest
from kds.data.research_tts import ResearchTtsModel
from kds.data.voxcpm2_text_only import bind_text, generation_kwargs
from scripts.publish_denis_mdc_voxcpm2_pre_qa_spoof_ready import rejection_accounting
from scripts.synthesize_denis_mdc_voxcpm2_pre_qa import (
    CandidateCallAudit,
    DenisVoxCPM2SynthesisError,
)


def _base_row() -> ManifestRow:
    text_hash = bind_text("  Точный\u00a0 текст  ").collapse_whitespace_sha256
    return ManifestRow(
        sample_id="denis_1_0_mdc:category/0001",
        relative_path="processed/aa/audio.wav",
        sha256="a" * 64,
        split="ood",
        label="bonafide",
        language="ru",
        code_switch="false",
        parent_group_id="denis_1_0_mdc:speaker:single",
        source_name="denis_1_0_mdc",
        source_license="CC0-1.0",
        rights_basis="fixture",
        speaker_pseudo_id="denis_1_0_mdc:speaker:single",
        text_id=f"denis_1_0_mdc:text:{text_hash}",
        text_hash=text_hash,
        duration_s=3.0,
        generator_family="",
        generator_name="",
        generator_version="",
        voice_id="",
        clone_consent_id="",
        device="unknown",
        capture_route="fixture",
        original_sr=16_000,
        codec="wav",
        augmentation_chain="",
        augmentation_seed="",
        created_at="2026-08-14T08:00:00+05:00",
    )


def _model() -> ResearchTtsModel:
    return ResearchTtsModel(
        model_id="openbmb_voxcpm2_official_text_only",
        destination="voxcpm2_official_v1",
        generator_family="openbmb_voxcpm2_official_text_only",
        generator_name="Official OpenBMB VoxCPM2 text-only default voice",
        generator_version="pinned-test-version",
        license="Apache-2.0; training-source overlap unverified",
        source_url="https://example.test/model",
        runtime={},
        artifacts=(),
    )


def test_spoof_row_preserves_pair_text_and_marks_unknown_fixed_profile() -> None:
    base = _base_row()
    binding = bind_text("  Точный\u00a0 текст  ")

    row = denis_voxcpm2_spoof_row(
        base_row=base,
        model=_model(),
        binding=binding,
        relative_path="raw/voxcpm2/candidate.wav",
        sha256="b" * 64,
        duration_s=2.5,
        created_at="2026-08-14T08:10:00+05:00",
    )

    assert row.source_name == DENIS_VOXCPM2_SOURCE_ID
    assert row.voice_id == DENIS_VOXCPM2_VOICE_ID
    assert row.text_id == base.text_id
    assert row.text_hash == base.text_hash
    assert row.split == "ood"
    assert row.label == "spoof"
    assert row.augmentation_seed == "20260814"
    assert "reference_audio=forbidden" in row.augmentation_chain
    assert "identity_unknown" in row.voice_id


def test_spoof_row_rejects_text_or_source_drift() -> None:
    base = _base_row()
    with pytest.raises(DenisVoxCPM2CandidateError, match="outside the frozen"):
        denis_voxcpm2_spoof_row(
            base_row=replace(base, source_name="different"),
            model=_model(),
            binding=bind_text("  Точный\u00a0 текст  "),
            relative_path="raw/voxcpm2/candidate.wav",
            sha256="b" * 64,
            duration_s=2.5,
            created_at="2026-08-14T08:10:00+05:00",
        )


class _FakeModel:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, **kwargs: object) -> Any:
        self.calls += 1
        return [float(len(kwargs))]


def test_candidate_call_audit_binds_one_text_and_never_persists_plaintext() -> None:
    text = "  Точный\u00a0 текст  "
    binding = bind_text(text)
    model = _FakeModel()
    audit = CandidateCallAudit(model, max_calls=1)
    audit.expect(binding.collapse_whitespace_sha256)

    assert audit.generate(**generation_kwargs(text, binding))
    assert audit.calls == 1
    assert model.calls == 1
    assert audit.sanitized_kwargs is not None
    assert audit.sanitized_kwargs["text"] == {
        "sha256": binding.collapse_whitespace_sha256,
        "utf8_bytes": len("Точный текст".encode()),
        "plaintext_persisted": False,
    }
    assert text not in str(audit.sanitized_kwargs)
    with pytest.raises(DenisVoxCPM2SynthesisError, match="Unexpected or excess"):
        audit.generate(**generation_kwargs(text, binding))


def test_rejection_accounting_forbids_reuse_or_unaccounted_raw_row(tmp_path: Path) -> None:
    raw = denis_voxcpm2_spoof_row(
        base_row=_base_row(),
        model=_model(),
        binding=bind_text("  Точный\u00a0 текст  "),
        relative_path="raw/voxcpm2/candidate.wav",
        sha256="b" * 64,
        duration_s=2.5,
        created_at="2026-08-14T08:10:00+05:00",
    )
    raw_manifest = tmp_path / "raw.csv"
    rejected: list[dict[str, object]] = [
        {
            "sample_id": raw.sample_id,
            "relative_path": raw.relative_path,
            "detail": "Audio is not trainable: insufficient_speech.",
        }
    ]
    report: dict[str, object] = {
        "input_manifest": str(raw_manifest),
        "reused_rows": 0,
        "published_rows": 0,
        "rejected_rows": rejected,
    }

    assert rejection_accounting(
        raw_rows=(raw,),
        ready_rows=(),
        report=report,
        raw_manifest=raw_manifest,
    ) == tuple(rejected)
    report["reused_rows"] = 1
    with pytest.raises(ValueError, match="count accounting"):
        rejection_accounting(
            raw_rows=(raw,),
            ready_rows=(),
            report=report,
            raw_manifest=raw_manifest,
        )


def test_current_64_row_binding_pins_every_future_program_before_synthesis() -> None:
    path = Path(
        "data/manifests/denis_1_0_mdc_voxcpm2_official_pre_qa_text_binding_v1.json"
    )
    receipt = json.loads(path.read_text(encoding="utf-8"))

    assert sha256_file(path) == (
        "943a9595968996f29da1a13f213e28419fc2c7b5215df790e4d4c440528f2b7b"
    )
    assert len(receipt["rows"]) == 64
    assert receipt["text_binding_sha256"] == (
        "b28d1ff99bc50b5dc6879b75a7dee018cef3a0767508cfde2fc660f9156204c0"
    )
    assert receipt["claims"]["synthetic_audio_generated"] is False
    assert receipt["claims"]["detector_inference_authorized"] is False
    assert all(
        not {"text", "transcript", "prompt_text"}.intersection(row)
        for row in receipt["rows"]
    )
    for program in receipt["frozen_programs"].values():
        assert sha256_file(Path(program["path"])) == program["sha256"]


def test_current_one_shot_synthesis_and_qa_stop_below_frozen_minimum() -> None:
    manifest_directory = Path("data/manifests")
    raw_manifest = manifest_directory / (
        "denis_1_0_mdc_voxcpm2_official_pre_qa_raw_v1.csv"
    )
    synthesis_receipt = manifest_directory / (
        "denis_1_0_mdc_voxcpm2_official_pre_qa_synthesis_v1.json"
    )
    ready_manifest = manifest_directory / (
        "denis_1_0_mdc_voxcpm2_official_pre_qa_ready_v1.csv"
    )
    rejection_report = manifest_directory / (
        "denis_1_0_mdc_voxcpm2_official_pre_qa_technical_qa_rejections_v1.json"
    )
    qa_receipt = manifest_directory / (
        "denis_1_0_mdc_voxcpm2_official_pre_qa_technical_qa_v1.json"
    )

    assert sha256_file(raw_manifest) == (
        "45c8d5c9fb4d9f9bd9b5745add9b6e738111928b2b5c42a8779e030377195362"
    )
    assert sha256_file(synthesis_receipt) == (
        "b827ba8208d4d44fdaeefaabeaa841355ed580aa253b261dead766a3a16ee83b"
    )
    assert sha256_file(ready_manifest) == (
        "f90a634b80364a3a70046cf66354dbc7c11459f15a375b1e3a61c1f440e3028a"
    )
    assert sha256_file(rejection_report) == (
        "38c4da79e2bd0a50168fabb1817f866c6dacbbcd657c8ee18e6846a45e058ecb"
    )
    assert sha256_file(qa_receipt) == (
        "ca46362313f50f79043dd559f8d739185b51d8cb0dc9dcc0f5dc659e5b02951c"
    )

    synthesis = json.loads(synthesis_receipt.read_text(encoding="utf-8"))
    qa = json.loads(qa_receipt.read_text(encoding="utf-8"))
    assert len(load_manifest(raw_manifest)) == 64
    assert len(load_manifest(ready_manifest)) == 53
    policy = synthesis["generation_policy"]
    assert policy["bound_rows"] == 64
    assert policy["attempted_rows"] == 64
    assert policy["successful_rows"] == 64
    assert policy["failed_attempt_rows"] == 0
    assert policy["model_loads"] == 1
    assert policy["retry_or_resynthesis_used"] is False
    assert policy["post_selection_replacement_or_backfill"] is False
    assert synthesis["network_policy"]["observed_upstream_network_attempts"] == 0
    assert qa["technical_qa"]["ready_rows"] == 53
    assert qa["technical_qa"]["rejected_rows"] == 11
    assert qa["technical_qa"]["rejection_reason_counts"] == {
        "insufficient_speech": 11
    }
    assert qa["technical_qa"]["reused_rows"] == 0
    assert qa["technical_qa"]["resynthesis_replacement_or_backfill"] is False
    assert qa["target_outcome"] == {
        "minimum_ready_pairs": 60,
        "target_ready_pairs": 64,
        "actual_ready_pairs": 53,
        "status": "stop_below_minimum_60",
    }
    assert qa["claims"]["binary_pairing_performed"] is False
    assert qa["claims"]["detector_inference_authorized"] is False
    assert qa["claims"]["speaker_independent"] is False
