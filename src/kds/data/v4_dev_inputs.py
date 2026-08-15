"""Invariants for the isolated XLS-R+SLS model-v4 bilingual development inputs."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import replace

from kds.data.manifest import ManifestError, ManifestRow, validate_manifest
from kds.data.research_tts import ResearchTtsModel
from kds.data.silero_v4 import SILERO_V4_TEXT_NORMALIZER_ID, SileroV4Profile

V4_KK_DEV_SILERO_SOURCE_ID = "ksc_slr102_silero_v4"
V4_KK_DEV_SILERO_SOURCE_LICENSE = "KSC CC-BY-4.0; Silero V4 model CC-BY-NC-SA-4.0"


class V4DevInputsError(ValueError):
    """Raised when the v4 dev inputs lose their role or pair isolation."""


def v4_kk_dev_silero_spoof_row(
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
    """Make a KSC-SLR102 text-only Silero row for the v4 ``dev`` role only."""

    if (
        base_row.source_name != "ksc_slr102"
        or base_row.split != "dev"
        or base_row.label != "bonafide"
        or base_row.language != "kk"
        or profile.language != "kk"
    ):
        raise V4DevInputsError(
            "v4 KK Silero dev synthesis requires a KSC-SLR102 KK bona-fide dev row."
        )
    profile_id = f"{model.model_id}:{profile.voice_id}"
    sample_key = hashlib.sha256(f"{base_row.sample_id}:{profile_id}:v4-dev".encode()).hexdigest()[
        :16
    ]
    return ManifestRow(
        sample_id=f"{V4_KK_DEV_SILERO_SOURCE_ID}:{sample_key}",
        relative_path=relative_path,
        sha256=sha256,
        split="dev",
        label="spoof",
        language="kk",
        code_switch=base_row.code_switch,
        parent_group_id=f"{V4_KK_DEV_SILERO_SOURCE_ID}:profile:{profile_id}",
        source_name=V4_KK_DEV_SILERO_SOURCE_ID,
        source_license=V4_KK_DEV_SILERO_SOURCE_LICENSE,
        rights_basis=(
            "Offline text-only personal-research derivative from KSC SLR102 transcript "
            f"{base_row.text_id}; {model.license}; fixed pretrained profile; no reference "
            "audio or voice cloning; source code-switch annotation remains unchanged"
        ),
        speaker_pseudo_id=(f"{V4_KK_DEV_SILERO_SOURCE_ID}:synthetic-profile:{profile_id}"),
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
            "language_provenance=intended_kk_input_text"
        ),
        augmentation_seed="",
        created_at=created_at,
    )


def replace_with_decoded_v4_dev_row(
    row: ManifestRow,
    *,
    relative_path: str,
    sha256: str,
    duration_s: float,
    created_at: str,
) -> ManifestRow:
    """Retain row provenance while binding it to its canonical 16-kHz decoded asset."""

    return replace(
        row,
        relative_path=relative_path,
        sha256=sha256,
        duration_s=duration_s,
        original_sr=16_000,
        codec="wav",
        created_at=created_at,
    )


def freeze_v4_kk_dev_pairs(
    source_ready_rows: Sequence[ManifestRow],
    spoof_ready_rows: Sequence[ManifestRow],
    *,
    source_ranks: Mapping[str, int],
    target_pairs: int,
) -> tuple[ManifestRow, ...]:
    """Freeze the first fully eligible KSC/Silero pairs in the predeclared rank order."""

    if target_pairs <= 0:
        raise V4DevInputsError("v4 KK dev target_pairs must be positive.")
    source = tuple(source_ready_rows)
    spoof = tuple(spoof_ready_rows)
    if not source or not spoof:
        raise V4DevInputsError("v4 KK dev source and Silero inputs must both be non-empty.")
    try:
        validate_manifest(source)
        validate_manifest(spoof)
    except ManifestError as error:
        raise V4DevInputsError(error.issues) from error
    if any(
        row.source_name != "ksc_slr102"
        or row.split != "dev"
        or row.label != "bonafide"
        or row.language != "kk"
        for row in source
    ):
        raise V4DevInputsError("v4 KK dev source manifest has an invalid role or source.")
    if any(
        row.source_name != V4_KK_DEV_SILERO_SOURCE_ID
        or row.split != "dev"
        or row.label != "spoof"
        or row.language != "kk"
        for row in spoof
    ):
        raise V4DevInputsError("v4 KK dev Silero manifest has an invalid role or source.")
    source_by_text = {row.text_id: row for row in source}
    spoof_by_text = {row.text_id: row for row in spoof}
    if len(source_by_text) != len(source) or len(spoof_by_text) != len(spoof):
        raise V4DevInputsError("v4 KK dev inputs contain duplicate text IDs.")
    if not set(source_by_text).issubset(source_ranks):
        raise V4DevInputsError("v4 KK dev source ranks do not cover every ready source row.")
    pairs: list[tuple[int, str, ManifestRow, ManifestRow]] = []
    for text_id, source_row in source_by_text.items():
        spoof_row = spoof_by_text.get(text_id)
        if spoof_row is None:
            continue
        if source_row.text_hash != spoof_row.text_hash:
            raise V4DevInputsError("v4 KK dev pair has a mismatched transcript hash.")
        pairs.append((source_ranks[source_row.text_id], text_id, source_row, spoof_row))
    pairs.sort(key=lambda item: (item[0], item[1]))
    if len(pairs) < target_pairs:
        raise V4DevInputsError(
            f"v4 KK dev has {len(pairs)} QA-eligible pairs; needs {target_pairs}."
        )
    frozen = tuple(
        row
        for _rank, _text, source_row, spoof_row in pairs[:target_pairs]
        for row in (source_row, spoof_row)
    )
    try:
        validate_manifest(frozen)
    except ManifestError as error:
        raise V4DevInputsError(error.issues) from error
    return frozen


def build_v4_combined_dev_manifest(
    pyara_rows: Sequence[ManifestRow], kk_pairs: Sequence[ManifestRow]
) -> tuple[ManifestRow, ...]:
    """Combine the explicitly reused RU dev role with newly frozen KK dev pairs."""

    ru = tuple(pyara_rows)
    kk = tuple(kk_pairs)
    try:
        validate_manifest(ru)
        validate_manifest(kk)
    except ManifestError as error:
        raise V4DevInputsError(error.issues) from error
    if not ru or any(
        row.source_name != "pyara_ru_v7"
        or row.split != "dev"
        or row.language != "ru"
        or row.label not in {"bonafide", "spoof"}
        for row in ru
    ):
        raise V4DevInputsError("The reused PyAra input is not the reserved RU dev role.")
    if len(kk) % 2 or any(row.split != "dev" or row.language != "kk" for row in kk):
        raise V4DevInputsError("The KK input is not a complete v4 dev pair set.")
    combined = tuple(sorted((*ru, *kk), key=lambda row: (row.language, row.label, row.sample_id)))
    try:
        validate_manifest(combined)
    except ManifestError as error:
        raise V4DevInputsError(error.issues) from error
    return combined
