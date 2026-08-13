"""Manifest provenance for the fixed Qwen3-TTS CustomVoice / Aiden derivative."""

from __future__ import annotations

import hashlib

from kds.data.manifest import ManifestRow
from kds.data.qwen3_tts_customvoice import Qwen3TtsCustomVoice, Qwen3TtsCustomVoiceText
from kds.data.research_tts import ResearchTtsModel

QWEN3_TTS_CUSTOMVOICE_AIDEN_SOURCE_ID = "voxforge_ru_mdc_qwen3_tts_customvoice_aiden_v1"
QWEN3_TTS_CUSTOMVOICE_AIDEN_SOURCE_LICENSE = (
    "VoxForge Russian GPL-3.0-or-later; Qwen CustomVoice/tokenizer Apache-2.0; "
    "cstr GGUF conversion Apache-2.0; CrispASR MIT"
)


class Qwen3TtsCustomVoiceCandidateError(ValueError):
    """Raised when a generated row breaks the frozen literal-text contract."""


def qwen3_tts_customvoice_spoof_row(
    *,
    base_row: ManifestRow,
    model: ResearchTtsModel,
    runtime: Qwen3TtsCustomVoice,
    prepared: Qwen3TtsCustomVoiceText,
    relative_path: str,
    sha256: str,
    duration_s: float,
    created_at: str,
) -> ManifestRow:
    """Return one fixed-Aiden text-only spoof row paired with a frozen base text."""

    if (
        base_row.split != "test"
        or base_row.label != "bonafide"
        or base_row.language != "ru"
        or base_row.source_name != "voxforge_ru_mdc_2026_05"
    ):
        raise Qwen3TtsCustomVoiceCandidateError("Qwen3-TTS base row is outside frozen VoxForge RU.")
    source_text_sha256 = hashlib.sha256(prepared.source_text.encode("utf-8")).hexdigest()
    expected_seed = int.from_bytes(
        hashlib.sha256(prepared.source_text.encode("utf-8")).digest()[:4], "big"
    )
    if source_text_sha256 != base_row.text_hash or prepared.seed != expected_seed:
        raise Qwen3TtsCustomVoiceCandidateError(
            "Qwen3-TTS synthesis input changes the frozen literal text or seed."
        )
    profile_id = "qwen3_tts_customvoice:aiden"
    identity = f"{base_row.sample_id}:{model.generator_version}:{profile_id}:{source_text_sha256}"
    sample_key = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return ManifestRow(
        sample_id=f"{QWEN3_TTS_CUSTOMVOICE_AIDEN_SOURCE_ID}:{sample_key}",
        relative_path=relative_path,
        sha256=sha256,
        split="test",
        label="spoof",
        language="ru",
        code_switch=base_row.code_switch,
        parent_group_id=f"{QWEN3_TTS_CUSTOMVOICE_AIDEN_SOURCE_ID}:fixed-profile:{profile_id}",
        source_name=QWEN3_TTS_CUSTOMVOICE_AIDEN_SOURCE_ID,
        source_license=QWEN3_TTS_CUSTOMVOICE_AIDEN_SOURCE_LICENSE,
        rights_basis=(
            f"Offline literal-text-only derivative of {base_row.source_name} text "
            f"{base_row.text_id}; {model.license}; fixed documented English Aiden token; "
            "no reference audio, voice cloning, VoiceDesign or auto-download; language "
            "provenance is intended input text until acoustic/language review"
        ),
        speaker_pseudo_id=(
            f"{QWEN3_TTS_CUSTOMVOICE_AIDEN_SOURCE_ID}:synthetic-profile:{profile_id}"
        ),
        text_id=base_row.text_id,
        text_hash=base_row.text_hash,
        duration_s=duration_s,
        generator_family=model.generator_family,
        generator_name=model.generator_name,
        generator_version=model.generator_version,
        voice_id=profile_id,
        clone_consent_id="not_applicable:fixed_pretrained_tts_no_reference_audio",
        device="cuda:0",
        capture_route="offline_local_gguf_qwen3_tts_customvoice_fixed_aiden",
        original_sr=runtime.sample_rate,
        codec="wav",
        augmentation_chain=(
            "language_provenance=intended_input_text_only;"
            f"literal_source_text_sha256={source_text_sha256};"
            f"rng_seed={prepared.seed};"
            "reference_audio=forbidden;voice_design=forbidden"
        ),
        augmentation_seed=str(prepared.seed),
        created_at=created_at,
    )
