"""Provenance helpers for a narrow KSC2 mixed-text Silero V4 research candidate."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from kds.data.ksc2_mixed_candidate import Ksc2MixedCandidateError
from kds.data.manifest import ManifestRow
from kds.data.research_tts import ResearchTtsModel
from kds.data.silero_v4 import SILERO_V4_TEXT_NORMALIZER_ID, SileroV4Profile

KSC2_MIXED_SILERO_V4_SOURCE_ID = "ksc2_mixed_v1_silero_v4"
KSC2_MIXED_SILERO_V4_SOURCE_LICENSE = "KSC2 CC-BY-4.0; Silero V4 model CC-BY-NC-SA-4.0"


def mixed_silero_v4_spoof_row(
    *,
    base_row: ManifestRow,
    model: ResearchTtsModel,
    profile: SileroV4Profile,
    relative_path: str,
    sha256: str,
    duration_s: float,
    original_sr: int,
    created_at: str,
    device: str,
) -> ManifestRow:
    """Create a research-only spoof row from a known mixed *input* transcript.

    ``language=mixed`` records the provenance of the supplied transcript. It is not an acoustic
    language-preservation certification for this generated waveform.
    """

    if (
        base_row.source_name != "ksc2_v1"
        or base_row.split != "test"
        or base_row.label != "bonafide"
        or base_row.language != "mixed"
        or base_row.code_switch != "true"
        or profile.language != "kk"
    ):
        raise Ksc2MixedCandidateError(
            ["KSC2 mixed Silero spoof needs a mixed KSC2 bona-fide row and fixed KK profile."]
        )
    profile_id = f"{model.model_id}:{profile.voice_id}"
    sample_key = hashlib.sha256(f"{base_row.sample_id}:{profile_id}".encode()).hexdigest()[:16]
    return ManifestRow(
        sample_id=f"{KSC2_MIXED_SILERO_V4_SOURCE_ID}:{sample_key}",
        relative_path=relative_path,
        sha256=sha256,
        split="test",
        label="spoof",
        language="mixed",
        code_switch="true",
        parent_group_id=f"{KSC2_MIXED_SILERO_V4_SOURCE_ID}:profile:{profile_id}",
        source_name=KSC2_MIXED_SILERO_V4_SOURCE_ID,
        source_license=KSC2_MIXED_SILERO_V4_SOURCE_LICENSE,
        rights_basis=(
            "Offline text-only personal-research derivative from explicit KSC2 mixed transcript "
            f"{base_row.text_id}; {model.license}; fixed pretrained profile; no reference audio "
            "or voice cloning; language provenance is intended input text only"
        ),
        speaker_pseudo_id=f"{KSC2_MIXED_SILERO_V4_SOURCE_ID}:synthetic-profile:{profile_id}",
        text_id=base_row.text_id,
        text_hash=base_row.text_hash,
        duration_s=duration_s,
        generator_family=model.generator_family,
        generator_name=model.generator_name,
        generator_version=model.generator_version,
        voice_id=profile_id,
        clone_consent_id="not_applicable:fixed-pretrained-tts-no-reference-audio",
        device=device,
        capture_route="offline_text_only_fastpitch_hifigan_tts",
        original_sr=original_sr,
        codec="wav",
        augmentation_chain=(
            f"text_normalization={SILERO_V4_TEXT_NORMALIZER_ID};"
            "language_provenance=intended_input_text_only"
        ),
        augmentation_seed="",
        created_at=created_at,
    )


def build_paired_mixed_candidate_rows(
    *,
    base_rows: Iterable[ManifestRow],
    raw_spoof_rows: Iterable[ManifestRow],
    ready_spoof_rows: Iterable[ManifestRow],
    text_rejected_base_ids: set[str],
    audio_rejected_spoof_ids: set[str],
) -> list[ManifestRow]:
    """Return only exactly paired ready rows with all omissions accounted for."""

    base = list(base_rows)
    raw = list(raw_spoof_rows)
    ready = list(ready_spoof_rows)
    if not base or any(
        row.source_name != "ksc2_v1"
        or row.split != "test"
        or row.label != "bonafide"
        or row.language != "mixed"
        or row.code_switch != "true"
        for row in base
    ):
        raise Ksc2MixedCandidateError(["Base rows must be ready mixed KSC2 bona-fide test rows."])
    base_by_text = {row.text_id: row for row in base}
    if len(base_by_text) != len(base):
        raise Ksc2MixedCandidateError(["KSC2 mixed base has duplicate text IDs."])
    base_ids = {row.sample_id for row in base}
    if not text_rejected_base_ids.issubset(base_ids):
        raise Ksc2MixedCandidateError(["Text-rejection report names an unknown KSC2 base row."])
    if any(
        row.source_name != KSC2_MIXED_SILERO_V4_SOURCE_ID
        or row.split != "test"
        or row.label != "spoof"
        or row.language != "mixed"
        or row.code_switch != "true"
        for row in [*raw, *ready]
    ):
        raise Ksc2MixedCandidateError(["Silero rows have an invalid mixed spoof contract."])
    raw_by_text = {row.text_id: row for row in raw}
    ready_by_text = {row.text_id: row for row in ready}
    if len(raw_by_text) != len(raw) or len(ready_by_text) != len(ready):
        raise Ksc2MixedCandidateError(["Silero raw/ready rows have duplicate text IDs."])
    expected_raw_texts = {
        row.text_id for row in base if row.sample_id not in text_rejected_base_ids
    }
    if set(raw_by_text) != expected_raw_texts:
        raise Ksc2MixedCandidateError(
            ["Raw Silero rows do not match base rows minus text rejections."]
        )
    raw_ids = {row.sample_id for row in raw}
    if not audio_rejected_spoof_ids.issubset(raw_ids):
        raise Ksc2MixedCandidateError(["Audio-rejection report names an unknown raw spoof row."])
    expected_ready_texts = {
        row.text_id for row in raw if row.sample_id not in audio_rejected_spoof_ids
    }
    if set(ready_by_text) != expected_ready_texts:
        raise Ksc2MixedCandidateError(
            ["Ready Silero rows do not match raw rows minus audio rejections."]
        )
    paired_base = [base_by_text[text_id] for text_id in sorted(ready_by_text)]
    if any(base.text_hash != ready_by_text[base.text_id].text_hash for base in paired_base):
        raise Ksc2MixedCandidateError(["KSC2/Silero pair has a text-hash mismatch."])
    return paired_base + [ready_by_text[row.text_id] for row in paired_base]
