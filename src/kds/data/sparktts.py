"""Pinned, non-cloning Spark-TTS controls for personal-research generation.

The upstream convenience class always initializes wav2vec2, including for controlled generation.
That model is only needed to encode reference audio for cloning.  The controlled path generates
both token streams from text and control labels, then only needs BiCodec to detokenize them.
This module therefore extracts the minimal audited inference closure and never exposes a path
that accepts, reads, or encodes reference audio.
"""

from __future__ import annotations

import hashlib
import tarfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol, cast

from kds.data.manifest import ManifestRow
from kds.data.research_tts import ResearchTtsError, ResearchTtsModel

SPARKTTS_SOURCE_ID = "ksc_derived_kk_v3_sparktts"
SPARKTTS_SOURCE_LICENSE = (
    "KSC text CC-BY-4.0; Spark-TTS-Kazakh model CC-BY-NC-SA-4.0; "
    "Spark-TTS inference source Apache-2.0"
)
SPARKTTS_RUNTIME_KIND = "sparktts_controlled_bicodec"
MAX_SOURCE_UNPACKED_BYTES = 5 * 1024 * 1024

_SOURCE_FILES = frozenset(
    {
        "models/bicodec.py",
        "modules/blocks/layers.py",
        "modules/blocks/samper.py",
        "modules/blocks/vocos.py",
        "modules/encoder_decoder/feat_decoder.py",
        "modules/encoder_decoder/feat_encoder.py",
        "modules/encoder_decoder/wave_generator.py",
        "modules/fsq/finite_scalar_quantization.py",
        "modules/fsq/residual_fsq.py",
        "modules/speaker/ecapa_tdnn.py",
        "modules/speaker/perceiver_encoder.py",
        "modules/speaker/pooling_layers.py",
        "modules/speaker/speaker_encoder.py",
        "modules/vq/factorized_vector_quantize.py",
        "utils/__init__.py",
        "utils/file.py",
    }
)
_GENDERS = frozenset({"female", "male"})
_LEVELS = frozenset({"very_low", "low", "moderate", "high", "very_high"})


@dataclass(frozen=True, slots=True)
class SparkTtsProfile:
    """One virtual voice control; it is not a human identity or source voice."""

    voice_id: str
    gender: str
    pitch: str
    speed: str


@dataclass(frozen=True, slots=True)
class SparkTtsRuntime:
    """Exact no-reference runtime contract for the Spark-TTS controlled branch."""

    source_archive_path: str
    source_archive_root: str
    root_config_path: str
    bicodec_config_path: str
    bicodec_checkpoint_path: str
    llm_config_path: str
    llm_merges_path: str
    llm_tokenizer_config_path: str
    llm_vocab_path: str
    llm_checkpoint_path: str
    sample_rate: int
    max_new_tokens: int
    generation_attempts: int
    profile_attempts: int
    temperature: float
    top_k: int
    top_p: float
    profiles: tuple[SparkTtsProfile, ...]


class _ByteReader(Protocol):
    def read(self, size: int = -1) -> bytes: ...


class _ByteWriter(Protocol):
    def write(self, data: bytes) -> int: ...


def load_sparktts_runtime(model: ResearchTtsModel) -> SparkTtsRuntime:
    """Parse the complete fixed runtime and reject cloning-related configuration fields."""

    runtime = model.runtime
    expected = {
        "kind",
        "source_archive_path",
        "source_archive_root",
        "root_config_path",
        "bicodec_config_path",
        "bicodec_checkpoint_path",
        "llm_config_path",
        "llm_merges_path",
        "llm_tokenizer_config_path",
        "llm_vocab_path",
        "llm_checkpoint_path",
        "sample_rate",
        "max_new_tokens",
        "generation_attempts",
        "profile_attempts",
        "temperature",
        "top_k",
        "top_p",
        "profiles",
    }
    _expect_exact_keys(runtime, expected, model.model_id, "runtime")
    if _runtime_string(runtime, "kind", model.model_id) != SPARKTTS_RUNTIME_KIND:
        raise ResearchTtsError(
            f"Spark-TTS model {model.model_id!r} must use {SPARKTTS_RUNTIME_KIND!r}."
        )
    artifact_paths = {artifact.relative_path for artifact in model.artifacts}
    profiles_value = runtime["profiles"]
    if not isinstance(profiles_value, list) or not profiles_value:
        raise ResearchTtsError(f"Spark-TTS model {model.model_id!r} needs non-empty profiles.")
    profiles = tuple(_parse_profile(value, model.model_id) for value in profiles_value)
    voice_ids = [profile.voice_id for profile in profiles]
    if len(voice_ids) != len(set(voice_ids)):
        raise ResearchTtsError(
            f"Spark-TTS model {model.model_id!r} has duplicate profile voice_id."
        )
    profile_attempts = _positive_int(runtime, "profile_attempts", model.model_id)
    if profile_attempts > len(profiles):
        raise ResearchTtsError(
            f"Spark-TTS model {model.model_id!r} profile_attempts exceeds declared profiles."
        )
    return SparkTtsRuntime(
        source_archive_path=_artifact_path(runtime, "source_archive_path", model, artifact_paths),
        source_archive_root=_single_directory(
            _runtime_string(runtime, "source_archive_root", model.model_id), model.model_id
        ),
        root_config_path=_artifact_path(runtime, "root_config_path", model, artifact_paths),
        bicodec_config_path=_artifact_path(runtime, "bicodec_config_path", model, artifact_paths),
        bicodec_checkpoint_path=_artifact_path(
            runtime, "bicodec_checkpoint_path", model, artifact_paths
        ),
        llm_config_path=_artifact_path(runtime, "llm_config_path", model, artifact_paths),
        llm_merges_path=_artifact_path(runtime, "llm_merges_path", model, artifact_paths),
        llm_tokenizer_config_path=_artifact_path(
            runtime, "llm_tokenizer_config_path", model, artifact_paths
        ),
        llm_vocab_path=_artifact_path(runtime, "llm_vocab_path", model, artifact_paths),
        llm_checkpoint_path=_artifact_path(runtime, "llm_checkpoint_path", model, artifact_paths),
        sample_rate=_positive_int(runtime, "sample_rate", model.model_id),
        max_new_tokens=_positive_int(runtime, "max_new_tokens", model.model_id),
        generation_attempts=_positive_int(runtime, "generation_attempts", model.model_id),
        profile_attempts=profile_attempts,
        temperature=_positive_float(runtime, "temperature", model.model_id),
        top_k=_positive_int(runtime, "top_k", model.model_id),
        top_p=_unit_interval(runtime, "top_p", model.model_id),
        profiles=profiles,
    )


def assign_sparktts_profiles(
    rows: Iterable[ManifestRow], runtime: SparkTtsRuntime
) -> list[tuple[ManifestRow, SparkTtsProfile]]:
    """Assign fixed virtual controls round-robin without treating them as people."""

    return [
        (row, runtime.profiles[index % len(runtime.profiles)]) for index, row in enumerate(rows)
    ]


def extract_verified_sparktts_source(
    archive_path: Path, runtime: SparkTtsRuntime, destination_root: Path
) -> Path:
    """Extract only the source closure needed for no-reference BiCodec detokenization."""

    if destination_root.exists():
        raise ResearchTtsError(
            f"Refusing to overwrite Spark-TTS source extraction: {destination_root}"
        )
    destination_root.mkdir(parents=True, exist_ok=False)
    resolved_root = destination_root.resolve(strict=True)
    prefix = f"{runtime.source_archive_root}/sparktts/"
    found: set[str] = set()
    total_size = 0
    try:
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive.getmembers():
                if not member.name.startswith(prefix):
                    continue
                relative = member.name.removeprefix(prefix)
                if relative not in _SOURCE_FILES:
                    continue
                if not member.isfile() or member.issym() or member.islnk():
                    raise ResearchTtsError(
                        f"Spark-TTS source member is not a regular file: {member.name}"
                    )
                if relative in found:
                    raise ResearchTtsError(f"Spark-TTS source has duplicate member: {relative}")
                total_size += member.size
                if total_size > MAX_SOURCE_UNPACKED_BYTES:
                    raise ResearchTtsError("Spark-TTS source exceeds safe unpacked-size limit.")
                output_path = _resolve_below(resolved_root, f"sparktts/{relative}")
                output_path.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ResearchTtsError(f"Cannot read Spark-TTS source member: {member.name}")
                with source, output_path.open("xb") as output:
                    _copy_and_verify(source, output, member.size, member.name)
                found.add(relative)
    except (OSError, tarfile.TarError) as error:
        raise ResearchTtsError(
            f"Cannot safely read Spark-TTS source archive {archive_path}: {error}"
        ) from error
    missing = sorted(_SOURCE_FILES.difference(found))
    if missing:
        raise ResearchTtsError("Spark-TTS source is missing required files: " + ", ".join(missing))
    return resolved_root


def _copy_and_verify(
    source: _ByteReader, output: _ByteWriter, expected_size_bytes: int, label: str
) -> None:
    digest = hashlib.sha256()
    received = 0
    while chunk := source.read(1024 * 1024):
        received += len(chunk)
        if received > expected_size_bytes:
            raise ResearchTtsError(f"Spark-TTS extracted member exceeds expected size: {label}")
        digest.update(chunk)
        output.write(chunk)
    if received != expected_size_bytes:
        raise ResearchTtsError(
            f"Spark-TTS extracted member size mismatch for {label}: "
            f"expected {expected_size_bytes}, got {received}."
        )


def _parse_profile(value: object, model_id: str) -> SparkTtsProfile:
    if not isinstance(value, dict):
        raise ResearchTtsError(f"Spark-TTS model {model_id!r} profile must be an object.")
    raw = cast(dict[str, object], value)
    _expect_exact_keys(raw, {"voice_id", "gender", "pitch", "speed"}, model_id, "profile")
    gender = _runtime_string(raw, "gender", model_id)
    pitch = _runtime_string(raw, "pitch", model_id)
    speed = _runtime_string(raw, "speed", model_id)
    if gender not in _GENDERS or pitch not in _LEVELS or speed not in _LEVELS:
        raise ResearchTtsError(
            f"Spark-TTS model {model_id!r} profile has unsupported control value."
        )
    return SparkTtsProfile(
        voice_id=_runtime_string(raw, "voice_id", model_id),
        gender=gender,
        pitch=pitch,
        speed=speed,
    )


def _expect_exact_keys(
    raw: Mapping[str, object], expected: set[str], model_id: str, label: str
) -> None:
    unknown = sorted(set(raw).difference(expected))
    missing = sorted(expected.difference(raw))
    if unknown or missing:
        details: list[str] = []
        if missing:
            details.append("missing fields: " + ", ".join(missing))
        if unknown:
            details.append("unknown fields: " + ", ".join(unknown))
        raise ResearchTtsError(
            f"Spark-TTS model {model_id!r} {label} has " + "; ".join(details) + "."
        )


def _artifact_path(
    runtime: Mapping[str, object], name: str, model: ResearchTtsModel, artifact_paths: set[str]
) -> str:
    value = _safe_member_path(_runtime_string(runtime, name, model.model_id), model.model_id)
    if value not in artifact_paths:
        raise ResearchTtsError(
            f"Spark-TTS model {model.model_id!r} runtime {name!r} is not a locked artifact."
        )
    return value


def _runtime_string(runtime: Mapping[str, object], name: str, model_id: str) -> str:
    value = runtime.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ResearchTtsError(f"Spark-TTS model {model_id!r} needs non-empty runtime {name!r}.")
    return value.strip()


def _positive_int(runtime: Mapping[str, object], name: str, model_id: str) -> int:
    value = runtime.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ResearchTtsError(
            f"Spark-TTS model {model_id!r} runtime {name!r} must be positive int."
        )
    return value


def _positive_float(runtime: Mapping[str, object], name: str, model_id: str) -> float:
    value = runtime.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ResearchTtsError(
            f"Spark-TTS model {model_id!r} runtime {name!r} must be positive number."
        )
    return float(value)


def _unit_interval(runtime: Mapping[str, object], name: str, model_id: str) -> float:
    value = _positive_float(runtime, name, model_id)
    if value > 1:
        raise ResearchTtsError(f"Spark-TTS model {model_id!r} runtime {name!r} must be at most 1.")
    return value


def _single_directory(value: str, model_id: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 1 or path.name != value or "\\" in value:
        raise ResearchTtsError(
            f"Spark-TTS model {model_id!r} source_archive_root must name one directory."
        )
    return value


def _safe_member_path(value: str, model_id: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or value in {"", "."}:
        raise ResearchTtsError(f"Spark-TTS model {model_id!r} has unsafe artifact path.")
    return path.as_posix()


def _resolve_below(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ResearchTtsError(
            f"Spark-TTS source path escapes extraction root: {relative_path}"
        ) from error
    return candidate
