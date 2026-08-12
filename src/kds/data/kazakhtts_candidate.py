"""Manifest provenance and exact pairing for the Stage-C KazakhTTS candidate."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

from kds.data.kazakhtts import KAZAKHTTS_SOURCE_ID, KazakhTtsRuntime
from kds.data.manifest import ManifestRow
from kds.data.research_tts import ResearchTtsModel

KAZAKHTTS_SOURCE_LICENSE = (
    "FLEURS/KSC2 CC-BY-4.0; ISSAI Kazakh_TTS model CC-BY-4.0; "
    "ESPnet Apache-2.0; ParallelWaveGAN MIT"
)


class KazakhTtsCandidateError(ValueError):
    """Raised when generated assets cannot form exact fresh Stage-C pairs."""


def kazakhtts_spoof_row(
    *,
    base_row: ManifestRow,
    model: ResearchTtsModel,
    runtime: KazakhTtsRuntime,
    relative_path: str,
    sha256: str,
    duration_s: float,
    created_at: str,
    device: str,
    normalizer_id: str = "",
    synthesis_text_sha256: str = "",
) -> ManifestRow:
    """Create one fixed-voice text-only spoof row paired to a frozen ready base row."""

    if (
        base_row.split != "test"
        or base_row.label != "bonafide"
        or base_row.language not in {"ru", "kk", "mixed"}
        or base_row.code_switch != ("true" if base_row.language == "mixed" else "false")
        or base_row.source_name
        not in {"google_fleurs_ru_v1", "google_fleurs_kk_v1", "ksc2_v1"}
    ):
        raise KazakhTtsCandidateError("KazakhTTS base row is outside the frozen Stage-C roles.")
    if bool(normalizer_id) != bool(synthesis_text_sha256) or (
        synthesis_text_sha256
        and (
            len(synthesis_text_sha256) != 64
            or any(character not in "0123456789abcdef" for character in synthesis_text_sha256)
        )
    ):
        raise KazakhTtsCandidateError("KazakhTTS normalizer binding is incomplete or invalid.")
    profile_id = f"{model.model_id}:{runtime.fixed_voice_id}"
    identity = f"{base_row.sample_id}:{model.generator_version}:{profile_id}"
    if normalizer_id:
        identity += f":{normalizer_id}:{synthesis_text_sha256}"
    sample_key = hashlib.sha256(identity.encode()).hexdigest()[:20]
    return ManifestRow(
        sample_id=f"{KAZAKHTTS_SOURCE_ID}:{sample_key}",
        relative_path=relative_path,
        sha256=sha256,
        split="test",
        label="spoof",
        language=base_row.language,
        code_switch=base_row.code_switch,
        parent_group_id=f"{KAZAKHTTS_SOURCE_ID}:fixed-profile:{profile_id}",
        source_name=KAZAKHTTS_SOURCE_ID,
        source_license=KAZAKHTTS_SOURCE_LICENSE,
        rights_basis=(
            f"Offline text-only derivative of {base_row.source_name} text {base_row.text_id}; "
            f"{model.license}; fixed published voice; no reference audio or voice cloning; "
            "language provenance is intended input text until full acoustic review; "
            f"normalizer={normalizer_id or 'none'}"
        ),
        speaker_pseudo_id=f"{KAZAKHTTS_SOURCE_ID}:synthetic-profile:{profile_id}",
        text_id=base_row.text_id,
        text_hash=base_row.text_hash,
        duration_s=duration_s,
        generator_family=model.generator_family,
        generator_name=model.generator_name,
        generator_version=model.generator_version,
        voice_id=profile_id,
        clone_consent_id="not_applicable:fixed_pretrained_tts_no_reference_audio",
        device=device,
        capture_route="offline_text_only_tacotron2_parallelwavegan_tts",
        original_sr=runtime.sample_rate,
        codec="wav",
        augmentation_chain=(
            "language_provenance=intended_input_text_only"
            + (
                f";text_normalizer={normalizer_id};synthesis_text_sha256={synthesis_text_sha256}"
                if normalizer_id
                else ""
            )
        ),
        augmentation_seed="",
        created_at=created_at,
    )


def build_kazakhtts_pairs(
    *,
    base_rows: Iterable[ManifestRow],
    raw_spoof_rows: Iterable[ManifestRow],
    ready_spoof_rows: Iterable[ManifestRow],
    text_rejected_base_ids: set[str],
    rejected_spoof_ids: set[str],
) -> list[ManifestRow]:
    """Return balanced exact pairs after accounting every generated QA rejection."""

    base = list(base_rows)
    raw = list(raw_spoof_rows)
    ready = list(ready_spoof_rows)
    base_by_text = {row.text_id: row for row in base}
    raw_by_text = {row.text_id: row for row in raw}
    ready_by_text = {row.text_id: row for row in ready}
    if (
        not base
        or len(base_by_text) != len(base)
        or len(raw_by_text) != len(raw)
        or len(ready_by_text) != len(ready)
        or not text_rejected_base_ids.issubset({row.sample_id for row in base})
        or not rejected_spoof_ids.issubset({row.sample_id for row in raw})
    ):
        raise KazakhTtsCandidateError("KazakhTTS base/raw/ready pairing contract is invalid.")
    expected_raw = {
        row.text_id for row in base if row.sample_id not in text_rejected_base_ids
    }
    if set(raw_by_text) != expected_raw:
        raise KazakhTtsCandidateError(
            "KazakhTTS raw rows differ from base rows minus accounted text rejections."
        )
    expected_ready = {
        row.text_id for row in raw if row.sample_id not in rejected_spoof_ids
    }
    if set(ready_by_text) != expected_ready:
        raise KazakhTtsCandidateError(
            "KazakhTTS ready rows differ from raw rows minus accounted rejections."
        )
    paired_base = [base_by_text[text_id] for text_id in sorted(expected_ready)]
    if any(
        base_row.text_hash != ready_by_text[base_row.text_id].text_hash
        for base_row in paired_base
    ):
        raise KazakhTtsCandidateError("KazakhTTS pair changes its frozen text hash.")
    return paired_base + [ready_by_text[row.text_id] for row in paired_base]
