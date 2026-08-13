"""Manifest rows for the fixed-profile Stage-D Dialogs-RU derivative."""

from __future__ import annotations

import hashlib

from kds.data.dialogs_ru_vits2 import DialogsRuVits2, DialogsRuVits2Text
from kds.data.manifest import ManifestRow
from kds.data.research_tts import ResearchTtsModel

DIALOGS_RU_VITS2_STAGE_D_SOURCE_ID = "stage_d_ru_dialogs_vits2_masha_neutral_v1"
DIALOGS_RU_VITS2_STAGE_D_SOURCE_LICENSE = (
    "Common Voice RU CC0-1.0; Dialogs-RU model card OpenRAIL declaration; "
    "Dialogs dataset OpenRAIL license"
)


class DialogsRuVits2CandidateError(ValueError):
    """Raised when a generated Dialogs-RU row would violate the frozen pair contract."""


def dialogs_ru_vits2_spoof_row(
    *,
    base_row: ManifestRow,
    model: ResearchTtsModel,
    runtime: DialogsRuVits2,
    prepared: DialogsRuVits2Text,
    relative_path: str,
    sha256: str,
    duration_s: float,
    created_at: str,
) -> ManifestRow:
    """Create one text-only spoof row paired to exactly one frozen Common Voice row."""

    if (
        base_row.split != "test"
        or base_row.label != "bonafide"
        or base_row.language != "ru"
        or base_row.source_name != "common_voice_ru_v24"
    ):
        raise DialogsRuVits2CandidateError("Dialogs-RU base row is outside frozen Stage-D roles.")
    source_text_sha256 = hashlib.sha256(prepared.source_text.encode("utf-8")).hexdigest()
    if source_text_sha256 != base_row.text_hash:
        raise DialogsRuVits2CandidateError(
            "Dialogs-RU synthesis input changes the frozen text hash."
        )
    profile_id = "dialogs_ru_vits2:Masha:neutral"
    identity = f"{base_row.sample_id}:{model.generator_version}:{profile_id}"
    sample_key = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    dropped = ",".join(prepared.dropped_characters) if prepared.dropped_characters else "none"
    return ManifestRow(
        sample_id=f"{DIALOGS_RU_VITS2_STAGE_D_SOURCE_ID}:{sample_key}",
        relative_path=relative_path,
        sha256=sha256,
        split="test",
        label="spoof",
        language="ru",
        code_switch="false",
        parent_group_id=f"{DIALOGS_RU_VITS2_STAGE_D_SOURCE_ID}:fixed-profile:{profile_id}",
        source_name=DIALOGS_RU_VITS2_STAGE_D_SOURCE_ID,
        source_license=DIALOGS_RU_VITS2_STAGE_D_SOURCE_LICENSE,
        rights_basis=(
            f"Offline literal-text-only derivative of {base_row.source_name} text "
            f"{base_row.text_id}; {model.license}; fixed Masha/neutral profile; "
            "no reference audio or voice cloning; upstream basic cleaner only; language "
            "provenance is intended input text until acoustic "
            f"review; tokenizer_dropped_characters={dropped}"
        ),
        speaker_pseudo_id=(
            f"{DIALOGS_RU_VITS2_STAGE_D_SOURCE_ID}:synthetic-profile:{profile_id}"
        ),
        text_id=base_row.text_id,
        text_hash=base_row.text_hash,
        duration_s=duration_s,
        generator_family=model.generator_family,
        generator_name=model.generator_name,
        generator_version=model.generator_version,
        voice_id=f"{model.model_id}:{profile_id}",
        clone_consent_id="not_applicable:fixed_pretrained_tts_no_reference_audio",
        device="cpu",
        capture_route="offline_text_only_vits2_weights_only_fixed_profile",
        original_sr=runtime.sample_rate,
        codec="wav",
        augmentation_chain=(
            "language_provenance=intended_input_text_only;"
            f"literal_source_text_sha256={source_text_sha256};"
            f"upstream_tokenizer_dropped_characters={dropped};"
            "rng_seed=sha256_literal_source_text"
        ),
        augmentation_seed=source_text_sha256[:16],
        created_at=created_at,
    )
