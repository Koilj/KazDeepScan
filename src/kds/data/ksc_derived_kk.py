"""Provenance-safe helpers for the KSC-derived Kazakh TTS stress source."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from kds.data.manifest import ManifestRow
from kds.data.research_tts import ResearchTtsError, ResearchTtsModel, ResearchTtsModelLock

KSC_DERIVED_KK_SOURCE_ID = "ksc_derived_kk_v1"
KSC_DERIVED_KK_SOURCE_LICENSE = (
    "KSC text CC-BY-4.0; Piper voice CC-BY-4.0; MMS-TTS model CC-BY-NC-4.0"
)


@dataclass(frozen=True, slots=True)
class SynthesisProfile:
    model: ResearchTtsModel
    voice_id: str
    speaker_id: int | None


def synthesis_profiles(lock: ResearchTtsModelLock) -> tuple[SynthesisProfile, ...]:
    """Derive fixed Piper multi-voice and MMS/VITS profiles from the pinned model lock."""

    profiles: list[SynthesisProfile] = []
    for model in lock.models:
        runtime_kind = _runtime_string(model.runtime, "kind", model.model_id)
        if runtime_kind == "piper_cli":
            speaker_ids = model.runtime.get("speaker_ids")
            if not isinstance(speaker_ids, dict) or not speaker_ids:
                raise ResearchTtsError(
                    f"Piper model {model.model_id!r} needs a non-empty runtime speaker_ids map."
                )
            parsed_speakers: list[tuple[str, int]] = []
            for voice_id, speaker_id in speaker_ids.items():
                if not isinstance(voice_id, str) or not voice_id.strip():
                    raise ResearchTtsError(
                        f"Piper model {model.model_id!r} has an invalid speaker voice id."
                    )
                if (
                    not isinstance(speaker_id, int)
                    or isinstance(speaker_id, bool)
                    or speaker_id < 0
                ):
                    raise ResearchTtsError(
                        f"Piper model {model.model_id!r} has an invalid speaker id."
                    )
                parsed_speakers.append((voice_id.strip(), speaker_id))
            profiles.extend(
                SynthesisProfile(model=model, voice_id=voice_id, speaker_id=speaker_id)
                for voice_id, speaker_id in sorted(parsed_speakers, key=lambda item: item[1])
            )
        elif runtime_kind == "transformers_vits":
            voices = model.runtime.get("voices")
            if not isinstance(voices, list) or not voices:
                raise ResearchTtsError(
                    f"MMS model {model.model_id!r} needs a non-empty runtime voices list."
                )
            for voice_id in voices:
                if not isinstance(voice_id, str) or not voice_id.strip():
                    raise ResearchTtsError(f"MMS model {model.model_id!r} has an invalid voice id.")
                profiles.append(
                    SynthesisProfile(model=model, voice_id=voice_id.strip(), speaker_id=None)
                )
        else:
            raise ResearchTtsError(
                f"Model {model.model_id!r} has unsupported runtime kind {runtime_kind!r}."
            )
    families = {profile.model.generator_family for profile in profiles}
    if len(families) < 2:
        raise ResearchTtsError("Derived KSC research source requires at least two TTS families.")
    return tuple(profiles)


def select_ksc_bonafide_rows(
    rows: Iterable[ManifestRow], *, limit: int, seed: str
) -> list[ManifestRow]:
    """Select a deterministic KSC test subset without relabelling or changing source provenance."""

    if limit <= 0:
        raise ValueError("limit must be positive.")
    if not seed:
        raise ValueError("seed must not be empty.")
    candidates = [
        row
        for row in rows
        if row.source_name == "ksc_slr102"
        and row.split == "test"
        and row.label == "bonafide"
        and row.language == "kk"
    ]
    if len(candidates) < limit:
        raise ValueError(f"Need {limit} KSC kk test bona-fide rows, found only {len(candidates)}.")
    return sorted(
        candidates,
        key=lambda row: hashlib.sha256(f"{seed}:{row.sample_id}".encode()).digest(),
    )[:limit]


def assign_synthesis_profiles(
    rows: Iterable[ManifestRow], profiles: Iterable[SynthesisProfile]
) -> list[tuple[ManifestRow, SynthesisProfile]]:
    """Balance rows first across TTS families, then across voices within each family."""

    rows = list(rows)
    by_family: dict[str, list[SynthesisProfile]] = {}
    for profile in profiles:
        by_family.setdefault(profile.model.generator_family, []).append(profile)
    if len(by_family) < 2:
        raise ResearchTtsError("Derived KSC research source requires at least two TTS families.")
    assignments: list[tuple[ManifestRow, SynthesisProfile]] = []
    family_positions = {family: 0 for family in by_family}
    families = sorted(by_family)
    for index, row in enumerate(rows):
        family = families[index % len(families)]
        options = by_family[family]
        position = family_positions[family]
        assignments.append((row, options[position % len(options)]))
        family_positions[family] = position + 1
    return assignments


def merge_prepared_ksc_rows(
    raw_rows: Iterable[ManifestRow],
    newly_prepared_rows: Iterable[ManifestRow],
    reusable_prepared_rows: Iterable[ManifestRow],
) -> tuple[list[ManifestRow], set[str]]:
    """Reuse an identical prior KSC WAV only when it matches a selected raw row's provenance.

    This repairs a deliberate non-overwrite collision from batch preprocessing without turning a
    path collision into permission to mix arbitrary recordings into a new test slice.
    """

    return merge_prepared_rows(
        raw_rows,
        newly_prepared_rows,
        reusable_prepared_rows,
        source_name="ksc_slr102",
        label="bonafide",
        language="kk",
        source_description="Raw KSC base",
    )


def merge_prepared_rows(
    raw_rows: Iterable[ManifestRow],
    newly_prepared_rows: Iterable[ManifestRow],
    reusable_prepared_rows: Iterable[ManifestRow],
    *,
    source_name: str,
    label: str,
    language: str,
    source_description: str,
) -> tuple[list[ManifestRow], set[str]]:
    """Merge new and reusable ready audio only for one explicit source/label/split contract."""

    raw_rows = list(raw_rows)
    expected = [
        row
        for row in raw_rows
        if row.source_name == source_name
        and row.split == "test"
        and row.label == label
        and row.language == language
    ]
    if len(expected) != len(raw_rows):
        raise ValueError(
            f"{source_description} manifest must contain only {language} test {label} rows."
        )
    new_by_id = _unique_rows_by_sample_id(newly_prepared_rows, "newly prepared")
    reusable_by_id = _unique_rows_by_sample_id(reusable_prepared_rows, "reusable")
    merged: list[ManifestRow] = []
    reused_ids: set[str] = set()
    for raw_row in expected:
        prepared = new_by_id.get(raw_row.sample_id)
        if prepared is None:
            prepared = reusable_by_id.get(raw_row.sample_id)
            if prepared is not None:
                reused_ids.add(raw_row.sample_id)
        if prepared is None:
            continue
        _require_matching_provenance(raw_row, prepared)
        merged.append(prepared)
    return merged, reused_ids


def load_verified_ksc_transcript(row: ManifestRow, transcript_root: Path) -> str:
    """Load a paired KSC transcript only after proving it matches the manifest text hash."""

    expected_prefix = "ksc_slr102:"
    if not row.text_id.startswith(expected_prefix):
        raise ValueError(f"Row {row.sample_id!r} has no KSC text_id.")
    utterance_id = row.text_id.removeprefix(expected_prefix)
    path = PurePosixPath(utterance_id)
    if (
        not utterance_id
        or path.is_absolute()
        or len(path.parts) != 1
        or path.name != utterance_id
        or "\\" in utterance_id
        or utterance_id in {".", ".."}
    ):
        raise ValueError(f"Row {row.sample_id!r} has unsafe KSC utterance id.")
    root = transcript_root.resolve(strict=True)
    transcript_path = (root / "Transcriptions" / f"{utterance_id}.txt").resolve(strict=False)
    try:
        transcript_path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Transcript path escapes declared root for {row.sample_id!r}.") from error
    if not transcript_path.is_file():
        raise ValueError(f"Missing transcript for {row.sample_id!r}: {transcript_path}")
    try:
        transcript = " ".join(transcript_path.read_text(encoding="utf-8").split())
    except UnicodeDecodeError as error:
        raise ValueError(
            f"Transcript is not UTF-8 for {row.sample_id!r}: {transcript_path}"
        ) from error
    if not transcript:
        raise ValueError(f"Transcript is empty for {row.sample_id!r}: {transcript_path}")
    actual_hash = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
    if actual_hash != row.text_hash:
        raise ValueError(
            f"Transcript SHA-256 mismatch for {row.sample_id!r}: "
            f"expected {row.text_hash}, got {actual_hash}."
        )
    return transcript


def synthesis_seed(base_seed: str, row: ManifestRow, profile: SynthesisProfile) -> int:
    """Return a stable non-negative seed for stochastic TTS runtimes such as MMS/VITS."""

    digest = hashlib.sha256(
        f"{base_seed}:{row.sample_id}:{profile.model.model_id}:{profile.voice_id}".encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def derived_spoof_row(
    *,
    base_row: ManifestRow,
    profile: SynthesisProfile,
    relative_path: str,
    sha256: str,
    duration_s: float,
    original_sr: int,
    created_at: str,
    device: str,
    seed: int,
    source_name: str = KSC_DERIVED_KK_SOURCE_ID,
    source_license: str = KSC_DERIVED_KK_SOURCE_LICENSE,
    include_tts_seed: bool | None = None,
    capture_route: str = "offline_neural_tts",
) -> ManifestRow:
    """Make one spoof manifest row without hiding the KSC text or TTS provenance."""

    sample_key = hashlib.sha256(
        f"{base_row.sample_id}:{profile.model.model_id}:{profile.voice_id}".encode()
    ).hexdigest()[:16]
    profile_id = f"{profile.model.model_id}:{profile.voice_id}"
    return ManifestRow(
        sample_id=f"{source_name}:{sample_key}",
        relative_path=relative_path,
        sha256=sha256,
        split="test",
        label="spoof",
        language="kk",
        code_switch=base_row.code_switch,
        parent_group_id=f"{source_name}:profile:{profile_id}",
        source_name=source_name,
        source_license=source_license,
        rights_basis=(
            "Offline derivative for personal research from KSC transcript "
            f"{base_row.text_id}; {profile.model.license}; no voice cloning"
        ),
        speaker_pseudo_id=f"{source_name}:synthetic-voice:{profile_id}",
        text_id=base_row.text_id,
        text_hash=base_row.text_hash,
        duration_s=duration_s,
        generator_family=profile.model.generator_family,
        generator_name=profile.model.generator_name,
        generator_version=profile.model.generator_version,
        voice_id=profile_id,
        clone_consent_id="not_applicable:pretrained-tts-no-local-voice-cloning",
        device=device,
        capture_route=capture_route,
        original_sr=original_sr,
        codec="wav",
        augmentation_chain="none",
        augmentation_seed=(
            f"tts_seed={seed}"
            if (profile.speaker_id is None if include_tts_seed is None else include_tts_seed)
            else ""
        ),
        created_at=created_at,
    )


def _runtime_string(runtime: Mapping[str, object], key: str, model_id: str) -> str:
    value = runtime.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ResearchTtsError(f"Model {model_id!r} needs a non-empty runtime {key!r}.")
    return value.strip()


def _unique_rows_by_sample_id(rows: Iterable[ManifestRow], label: str) -> dict[str, ManifestRow]:
    indexed: dict[str, ManifestRow] = {}
    for row in rows:
        if row.sample_id in indexed:
            raise ValueError(f"{label} manifest contains duplicate sample_id={row.sample_id!r}.")
        indexed[row.sample_id] = row
    return indexed


def _require_matching_provenance(raw_row: ManifestRow, ready_row: ManifestRow) -> None:
    fields = (
        "sample_id",
        "split",
        "label",
        "language",
        "code_switch",
        "parent_group_id",
        "source_name",
        "source_license",
        "rights_basis",
        "speaker_pseudo_id",
        "text_id",
        "text_hash",
        "original_sr",
        "device",
        "capture_route",
    )
    mismatched = [field for field in fields if getattr(raw_row, field) != getattr(ready_row, field)]
    if mismatched:
        raise ValueError(
            f"Prepared row {ready_row.sample_id!r} does not match raw provenance fields: "
            + ", ".join(mismatched)
            + "."
        )
    if ready_row.codec != "wav":
        raise ValueError(f"Prepared row {ready_row.sample_id!r} must use codec='wav'.")
