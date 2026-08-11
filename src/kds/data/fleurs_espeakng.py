"""Strict FLEURS RU selection and provenance for a text-only eSpeak NG stress source."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from kds.data.espeakng import EspeakNgProfile
from kds.data.manifest import ManifestRow
from kds.data.research_tts import ResearchTtsModel

FLEURS_RU_ESPEAKNG_SOURCE_ID = "fleurs_ru_v1_espeakng"
FLEURS_RU_ESPEAKNG_SOURCE_LICENSE = "FLEURS CC-BY-4.0; eSpeak NG GPL-3.0-or-later"
FLEURS_RU_SOURCE_ID = "google_fleurs_ru_v1"
FLEURS_RU_ESPEAKNG_HOLDOUT_REASON = "reserved_by_fleurs_ru_v1_silero_v4_candidate"


class FleursEspeakNgError(ValueError):
    """Raised when the Russian formant candidate would reuse or misrepresent FLEURS data."""


def select_fleurs_ru_espeakng_base(
    full_base_rows: Iterable[ManifestRow], existing_candidate_rows: Iterable[ManifestRow]
) -> tuple[list[ManifestRow], list[ManifestRow]]:
    """Reserve every FLEURS row already exposed by the Silero candidate and return the remainder."""

    full = list(full_base_rows)
    existing = list(existing_candidate_rows)
    if not full or any(
        row.source_name != FLEURS_RU_SOURCE_ID
        or row.language != "ru"
        or row.split != "test"
        or row.label != "bonafide"
        or row.code_switch != "false"
        or row.codec != "wav"
        or not row.relative_path.startswith("processed/")
        for row in full
    ):
        raise FleursEspeakNgError(
            "Full base must contain only ready non-code-switched FLEURS RU test bona-fide WAVs."
        )
    full_by_text = {row.text_hash: row for row in full}
    if len(full_by_text) != len(full):
        raise FleursEspeakNgError("Full FLEURS RU base has duplicate text hashes.")
    existing_by_text: dict[str, dict[str, ManifestRow]] = {}
    for row in existing:
        if row.language != "ru" or row.split != "test" or row.code_switch != "false":
            raise FleursEspeakNgError("Existing FLEURS candidate has an invalid RU test row.")
        existing_by_text.setdefault(row.text_hash, {})[row.label] = row
    held_texts: set[str] = set()
    for text_hash, pair in existing_by_text.items():
        bonafide = pair.get("bonafide")
        spoof = pair.get("spoof")
        if (
            bonafide is None
            or spoof is None
            or bonafide.source_name != FLEURS_RU_SOURCE_ID
            or bonafide.text_id != spoof.text_id
            or text_hash not in full_by_text
            or full_by_text[text_hash].sample_id != bonafide.sample_id
        ):
            raise FleursEspeakNgError(
                "Existing FLEURS candidate must be exact paired FLEURS RU bona-fide/spoof data."
            )
        held_texts.add(text_hash)
    if not held_texts:
        raise FleursEspeakNgError("Existing candidate has no FLEURS RU pairs to hold out.")
    selected = [row for row in full if row.text_hash not in held_texts]
    held = [full_by_text[text_hash] for text_hash in sorted(held_texts)]
    if not selected:
        raise FleursEspeakNgError("No FLEURS RU rows remain after the existing candidate holdout.")
    if {row.text_hash for row in selected}.intersection(held_texts):
        raise FleursEspeakNgError("Selected FLEURS RU base overlaps the existing candidate.")
    return sorted(selected, key=lambda row: row.sample_id), held


def fleurs_ru_espeakng_spoof_row(
    *,
    base_row: ManifestRow,
    model: ResearchTtsModel,
    profile: EspeakNgProfile,
    relative_path: str,
    sha256: str,
    duration_s: float,
    original_sr: int,
    created_at: str,
) -> ManifestRow:
    """Make one deterministic FLEURS RU formant-TTS row without text modification or cloning."""

    if (
        base_row.source_name != FLEURS_RU_SOURCE_ID
        or base_row.language != "ru"
        or base_row.code_switch != "false"
        or not profile.voice_id.startswith("ru:")
    ):
        raise FleursEspeakNgError("Russian eSpeak profile does not match its FLEURS RU base row.")
    profile_id = f"{model.model_id}:{profile.voice_id}"
    sample_key = hashlib.sha256(
        f"{base_row.sample_id}:{model.model_id}:{profile.voice_id}".encode()
    ).hexdigest()[:16]
    return ManifestRow(
        sample_id=f"{FLEURS_RU_ESPEAKNG_SOURCE_ID}:{sample_key}",
        relative_path=relative_path,
        sha256=sha256,
        split="test",
        label="spoof",
        language="ru",
        code_switch="false",
        parent_group_id=f"{FLEURS_RU_ESPEAKNG_SOURCE_ID}:control:{profile_id}",
        source_name=FLEURS_RU_ESPEAKNG_SOURCE_ID,
        source_license=FLEURS_RU_ESPEAKNG_SOURCE_LICENSE,
        rights_basis=(
            "Offline text-only personal-research derivative from Google FLEURS transcript "
            f"{base_row.text_id}; {model.license}; deterministic formant controls; "
            "no reference audio or voice cloning"
        ),
        speaker_pseudo_id=f"{FLEURS_RU_ESPEAKNG_SOURCE_ID}:synthetic-control:{profile_id}",
        text_id=base_row.text_id,
        text_hash=base_row.text_hash,
        duration_s=duration_s,
        generator_family=model.generator_family,
        generator_name=model.generator_name,
        generator_version=model.generator_version,
        voice_id=profile_id,
        clone_consent_id="not_applicable:text_only_formant_tts_no_reference_audio",
        device="local_cpu_espeakng_formant",
        capture_route="offline_text_only_formant_tts",
        original_sr=original_sr,
        codec="wav",
        augmentation_chain="none",
        augmentation_seed="",
        created_at=created_at,
    )
