"""Manifest provenance for the frozen Denis / official VoxCPM2 candidate route."""

from __future__ import annotations

import hashlib

from kds.data.manifest import ManifestRow
from kds.data.research_tts import ResearchTtsModel
from kds.data.voxcpm2_text_only import VOXCPM2_FIXED_SEED, BoundText

DENIS_VOXCPM2_SOURCE_ID = "voxcpm2_official_text_only_v1"
DENIS_VOXCPM2_VOICE_ID = "voxcpm2:default_voice_identity_unknown"
DENIS_VOXCPM2_TEXT_BINDING_PROTOCOL_ID = "denis-1-0-mdc-voxcpm2-official-pre-qa-text-binding-v1"
DENIS_VOXCPM2_SYNTHESIS_PROTOCOL_ID = "denis-1-0-mdc-voxcpm2-official-pre-qa-synthesis-v1"
DENIS_VOXCPM2_TECHNICAL_QA_PROTOCOL_ID = "denis-1-0-mdc-voxcpm2-official-pre-qa-technical-qa-v1"


class DenisVoxCPM2CandidateError(ValueError):
    """Raised when a candidate row leaves the frozen text-only route."""


def denis_voxcpm2_spoof_row(
    *,
    base_row: ManifestRow,
    model: ResearchTtsModel,
    binding: BoundText,
    relative_path: str,
    sha256: str,
    duration_s: float,
    created_at: str,
) -> ManifestRow:
    """Return one default-voice text-only spoof row paired to one Denis ready text."""

    if (
        base_row.split != "ood"
        or base_row.label != "bonafide"
        or base_row.language != "ru"
        or base_row.source_name != "denis_1_0_mdc"
        or base_row.text_hash != binding.collapse_whitespace_sha256
    ):
        raise DenisVoxCPM2CandidateError(
            "VoxCPM2 base row is outside the frozen Denis ready-text contract."
        )
    profile_id = DENIS_VOXCPM2_VOICE_ID
    identity = (
        f"{base_row.sample_id}:{model.generator_version}:{profile_id}:"
        f"{binding.literal_sha256}:{binding.collapse_whitespace_sha256}"
    )
    sample_key = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return ManifestRow(
        sample_id=f"{DENIS_VOXCPM2_SOURCE_ID}:{sample_key}",
        relative_path=relative_path,
        sha256=sha256,
        split="ood",
        label="spoof",
        language="ru",
        code_switch=base_row.code_switch,
        parent_group_id=(f"{DENIS_VOXCPM2_SOURCE_ID}:fixed-profile:{DENIS_VOXCPM2_VOICE_ID}"),
        source_name=DENIS_VOXCPM2_SOURCE_ID,
        source_license=model.license,
        rights_basis=(
            f"Offline collapse-whitespace text-only derivative of CC0 Denis text "
            f"{base_row.text_id}; {model.license}; personal-research external "
            "generator-family holdout only; no reference/prompt audio, voice cloning, "
            "LoRA, denoiser, semantic normalizer, retry, replacement or network access; "
            "language provenance is intended input until independent acoustic review"
        ),
        speaker_pseudo_id=(f"{DENIS_VOXCPM2_SOURCE_ID}:synthetic-profile:{DENIS_VOXCPM2_VOICE_ID}"),
        text_id=base_row.text_id,
        text_hash=base_row.text_hash,
        duration_s=duration_s,
        generator_family=model.generator_family,
        generator_name=model.generator_name,
        generator_version=model.generator_version,
        voice_id=profile_id,
        clone_consent_id="not_applicable:fixed_pretrained_tts_no_reference_audio",
        device="cuda:0",
        capture_route="offline_local_voxcpm2_official_text_only_default_voice",
        original_sr=48_000,
        codec="wav",
        augmentation_chain=(
            "language_provenance=intended_input_text_only;"
            f"literal_source_text_sha256={binding.literal_sha256};"
            f"collapse_whitespace_text_sha256={binding.collapse_whitespace_sha256};"
            f"rng_seed={VOXCPM2_FIXED_SEED};"
            "reference_audio=forbidden;prompt_audio=forbidden;prompt_text=forbidden;"
            "lora=forbidden;normalize=false;denoise=false;retry=false"
        ),
        augmentation_seed=str(VOXCPM2_FIXED_SEED),
        created_at=created_at,
    )
