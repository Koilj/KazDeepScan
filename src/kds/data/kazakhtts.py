"""Pinned, fixed-voice ISSAI KazakhTTS inference controls.

The upstream checkpoints are ZIP archives produced by ESPnet 0.10.x and
ParallelWaveGAN 0.4.x.  This module extracts only the six files required for
inference, verifies their inner hashes and rejects every route that could accept
reference audio or select an undeclared speaker.
"""

from __future__ import annotations

import hashlib
import shutil
import stat
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol, cast

import yaml  # type: ignore[import-untyped]

from kds.data.research_tts import ResearchTtsError, ResearchTtsModel

KAZAKHTTS_RUNTIME_KIND = "kazakhtts_tacotron2_parallelwavegan"
KAZAKHTTS_GENERATOR_FAMILY = "tacotron2_parallelwavegan_fixed_voice_tts"
KAZAKHTTS_SOURCE_ID = "fresh_suite_v1_kazakhtts_tacotron2_pwg"
KAZAKHTTS_SOURCE_LICENSE = (
    "source text license; ISSAI Kazakh_TTS model CC-BY-4.0; "
    "ESPnet Apache-2.0; ParallelWaveGAN MIT"
)


@dataclass(frozen=True, slots=True)
class VerifiedZipMember:
    """One regular ZIP member bound to an exact size and digest."""

    member_name: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class KazakhTtsRuntime:
    """Complete fixed inference route for the accepted Stage-C candidate."""

    source_archive_path: str
    source_archive_root: str
    acoustic_archive_path: str
    acoustic_meta: VerifiedZipMember
    acoustic_config: VerifiedZipMember
    acoustic_checkpoint: VerifiedZipMember
    acoustic_stats: VerifiedZipMember
    vocoder_archive_path: str
    vocoder_config: VerifiedZipMember
    vocoder_checkpoint: VerifiedZipMember
    sample_rate: int
    fixed_voice_id: str
    supported_languages: tuple[str, ...]
    conditional_smoke_languages: tuple[str, ...]
    espnet_version: str
    checkpoint_espnet_version: str
    parallel_wavegan_version: str
    checkpoint_parallel_wavegan_version: str
    threshold: float
    min_length_ratio: float
    max_length_ratio: float
    use_attention_constraint: bool
    backward_window: int
    forward_window: int
    speed_control_alpha: float


@dataclass(frozen=True, slots=True)
class ExtractedKazakhTtsRuntime:
    """Trusted local paths published from the two verified checkpoint archives."""

    acoustic_meta: Path
    acoustic_config: Path
    acoustic_checkpoint: Path
    acoustic_stats: Path
    vocoder_config: Path
    vocoder_checkpoint: Path


class _ByteReader(Protocol):
    def read(self, size: int = -1) -> bytes: ...


class _ByteWriter(Protocol):
    def write(self, data: bytes) -> int: ...


def load_kazakhtts_runtime(model: ResearchTtsModel) -> KazakhTtsRuntime:
    """Parse the exact no-reference, one-voice runtime contract."""

    if model.generator_family != KAZAKHTTS_GENERATOR_FAMILY:
        raise ResearchTtsError(
            f"KazakhTTS model {model.model_id!r} must use generator family "
            f"{KAZAKHTTS_GENERATOR_FAMILY!r}."
        )
    runtime = model.runtime
    expected = {
        "kind",
        "source_archive_path",
        "source_archive_root",
        "acoustic_archive_path",
        "acoustic_meta_member",
        "acoustic_meta_size_bytes",
        "acoustic_meta_sha256",
        "acoustic_config_member",
        "acoustic_config_size_bytes",
        "acoustic_config_sha256",
        "acoustic_checkpoint_member",
        "acoustic_checkpoint_size_bytes",
        "acoustic_checkpoint_sha256",
        "acoustic_stats_member",
        "acoustic_stats_size_bytes",
        "acoustic_stats_sha256",
        "vocoder_archive_path",
        "vocoder_config_member",
        "vocoder_config_size_bytes",
        "vocoder_config_sha256",
        "vocoder_checkpoint_member",
        "vocoder_checkpoint_size_bytes",
        "vocoder_checkpoint_sha256",
        "sample_rate",
        "fixed_voice_id",
        "supported_languages",
        "conditional_smoke_languages",
        "reference_audio_policy",
        "voice_cloning",
        "espnet_version",
        "checkpoint_espnet_version",
        "parallel_wavegan_version",
        "checkpoint_parallel_wavegan_version",
        "threshold",
        "min_length_ratio",
        "max_length_ratio",
        "use_attention_constraint",
        "backward_window",
        "forward_window",
        "speed_control_alpha",
        "scipy_kaiser_compatibility_shim",
    }
    _expect_exact_keys(runtime, expected, model.model_id)
    if _string(runtime, "kind", model.model_id) != KAZAKHTTS_RUNTIME_KIND:
        raise ResearchTtsError(
            f"KazakhTTS model {model.model_id!r} must use runtime {KAZAKHTTS_RUNTIME_KIND!r}."
        )
    if runtime["reference_audio_policy"] != "forbidden" or runtime["voice_cloning"] is not False:
        raise ResearchTtsError(
            f"KazakhTTS model {model.model_id!r} must forbid reference audio and voice cloning."
        )
    if runtime["scipy_kaiser_compatibility_shim"] is not True:
        raise ResearchTtsError(
            f"KazakhTTS model {model.model_id!r} must declare the tested SciPy compatibility shim."
        )
    artifact_paths = {artifact.relative_path for artifact in model.artifacts}
    source_archive_path = _artifact_path(runtime, "source_archive_path", model, artifact_paths)
    acoustic_archive_path = _artifact_path(
        runtime, "acoustic_archive_path", model, artifact_paths
    )
    vocoder_archive_path = _artifact_path(runtime, "vocoder_archive_path", model, artifact_paths)
    source_archive_root = _single_directory(
        _string(runtime, "source_archive_root", model.model_id), model.model_id
    )
    supported_languages = _language_list(runtime, "supported_languages", model.model_id)
    conditional_languages = _language_list(
        runtime, "conditional_smoke_languages", model.model_id
    )
    if supported_languages != ("kk",) or conditional_languages != ("mixed", "ru"):
        raise ResearchTtsError(
            f"KazakhTTS model {model.model_id!r} must keep kk supported and ru/mixed conditional."
        )
    espnet_version = _string(runtime, "espnet_version", model.model_id)
    parallel_wavegan_version = _string(runtime, "parallel_wavegan_version", model.model_id)
    if espnet_version != "0.10.6" or parallel_wavegan_version != "0.6.1":
        raise ResearchTtsError(
            f"KazakhTTS model {model.model_id!r} uses an untested inference dependency version."
        )
    sample_rate = _positive_int(runtime, "sample_rate", model.model_id)
    if sample_rate != 22050:
        raise ResearchTtsError(f"KazakhTTS model {model.model_id!r} must use 22050 Hz.")
    min_length_ratio = _non_negative_float(runtime, "min_length_ratio", model.model_id)
    max_length_ratio = _positive_float(runtime, "max_length_ratio", model.model_id)
    if min_length_ratio >= max_length_ratio:
        raise ResearchTtsError(
            f"KazakhTTS model {model.model_id!r} length ratios are inconsistent."
        )
    return KazakhTtsRuntime(
        source_archive_path=source_archive_path,
        source_archive_root=source_archive_root,
        acoustic_archive_path=acoustic_archive_path,
        acoustic_meta=_member(runtime, "acoustic_meta", model.model_id),
        acoustic_config=_member(runtime, "acoustic_config", model.model_id),
        acoustic_checkpoint=_member(runtime, "acoustic_checkpoint", model.model_id),
        acoustic_stats=_member(runtime, "acoustic_stats", model.model_id),
        vocoder_archive_path=vocoder_archive_path,
        vocoder_config=_member(runtime, "vocoder_config", model.model_id),
        vocoder_checkpoint=_member(runtime, "vocoder_checkpoint", model.model_id),
        sample_rate=sample_rate,
        fixed_voice_id=_string(runtime, "fixed_voice_id", model.model_id),
        supported_languages=supported_languages,
        conditional_smoke_languages=conditional_languages,
        espnet_version=espnet_version,
        checkpoint_espnet_version=_string(
            runtime, "checkpoint_espnet_version", model.model_id
        ),
        parallel_wavegan_version=parallel_wavegan_version,
        checkpoint_parallel_wavegan_version=_string(
            runtime, "checkpoint_parallel_wavegan_version", model.model_id
        ),
        threshold=_positive_float(runtime, "threshold", model.model_id),
        min_length_ratio=min_length_ratio,
        max_length_ratio=max_length_ratio,
        use_attention_constraint=_boolean(
            runtime, "use_attention_constraint", model.model_id
        ),
        backward_window=_positive_int(runtime, "backward_window", model.model_id),
        forward_window=_positive_int(runtime, "forward_window", model.model_id),
        speed_control_alpha=_positive_float(runtime, "speed_control_alpha", model.model_id),
    )


def extract_verified_kazakhtts_runtime(
    *,
    verified_paths: Mapping[str, Path],
    runtime: KazakhTtsRuntime,
    destination: Path,
) -> ExtractedKazakhTtsRuntime:
    """CRC-check both archives and atomically extract only the pinned inference members."""

    if destination.exists():
        raise ResearchTtsError(f"Refusing to overwrite KazakhTTS runtime: {destination}")
    stage = destination.with_name(f".{destination.name}.stage")
    if stage.exists():
        raise ResearchTtsError(f"KazakhTTS runtime staging path already exists: {stage}")
    stage.mkdir(parents=True, exist_ok=False)
    try:
        acoustic_paths = _extract_zip_members(
            verified_paths[runtime.acoustic_archive_path],
            (
                runtime.acoustic_meta,
                runtime.acoustic_config,
                runtime.acoustic_checkpoint,
                runtime.acoustic_stats,
            ),
            stage,
        )
        vocoder_paths = _extract_zip_members(
            verified_paths[runtime.vocoder_archive_path],
            (runtime.vocoder_config, runtime.vocoder_checkpoint),
            stage,
        )
        extracted = ExtractedKazakhTtsRuntime(
            acoustic_meta=acoustic_paths[runtime.acoustic_meta.member_name],
            acoustic_config=acoustic_paths[runtime.acoustic_config.member_name],
            acoustic_checkpoint=acoustic_paths[runtime.acoustic_checkpoint.member_name],
            acoustic_stats=acoustic_paths[runtime.acoustic_stats.member_name],
            vocoder_config=vocoder_paths[runtime.vocoder_config.member_name],
            vocoder_checkpoint=vocoder_paths[runtime.vocoder_checkpoint.member_name],
        )
        validate_kazakhtts_configs(extracted, runtime)
        stage.replace(destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return ExtractedKazakhTtsRuntime(
        acoustic_meta=destination / runtime.acoustic_meta.member_name,
        acoustic_config=destination / runtime.acoustic_config.member_name,
        acoustic_checkpoint=destination / runtime.acoustic_checkpoint.member_name,
        acoustic_stats=destination / runtime.acoustic_stats.member_name,
        vocoder_config=destination / runtime.vocoder_config.member_name,
        vocoder_checkpoint=destination / runtime.vocoder_checkpoint.member_name,
    )


def validate_kazakhtts_configs(
    extracted: ExtractedKazakhTtsRuntime, runtime: KazakhTtsRuntime
) -> None:
    """Check that pinned configs still describe the accepted Tacotron2/PWG route."""

    acoustic = _yaml_mapping(extracted.acoustic_config, "acoustic config")
    vocoder = _yaml_mapping(extracted.vocoder_config, "vocoder config")
    meta = _yaml_mapping(extracted.acoustic_meta, "acoustic metadata")
    feats = _mapping(acoustic.get("feats_extract_conf"), "acoustic feats_extract_conf")
    if (
        acoustic.get("tts") != "tacotron2"
        or acoustic.get("token_type") != "char"
        or acoustic.get("g2p") is not None
        or feats.get("fs") != runtime.sample_rate
        or meta.get("espnet") != runtime.checkpoint_espnet_version
    ):
        raise ResearchTtsError("KazakhTTS acoustic config does not match the locked route.")
    generator = _mapping(vocoder.get("generator_params"), "vocoder generator_params")
    if (
        vocoder.get("version") != runtime.checkpoint_parallel_wavegan_version
        or vocoder.get("sampling_rate") != runtime.sample_rate
        or vocoder.get("num_mels") != 80
        or generator.get("aux_channels") != 80
        or generator.get("out_channels") != 1
    ):
        raise ResearchTtsError("KazakhTTS vocoder config does not match the locked route.")


def validate_kazakhtts_text(text: str, extracted: ExtractedKazakhTtsRuntime) -> str:
    """Lowercase text and reject characters absent from the checkpoint token list."""

    normalized = " ".join(text.lower().strip().split())
    if not normalized:
        raise ResearchTtsError("KazakhTTS input text is empty after normalization.")
    acoustic = _yaml_mapping(extracted.acoustic_config, "acoustic config")
    token_list = acoustic.get("token_list")
    if not isinstance(token_list, list) or not all(isinstance(token, str) for token in token_list):
        raise ResearchTtsError("KazakhTTS acoustic config has no valid token list.")
    allowed = set(cast(list[str], token_list)).difference(
        {"<blank>", "<unk>", "<space>", "<sos/eos>"}
    )
    unsupported = sorted(set(normalized).difference(allowed).difference({" "}))
    if unsupported:
        raise ResearchTtsError(
            "KazakhTTS input contains unsupported characters: " + " ".join(unsupported)
        )
    return normalized


def _extract_zip_members(
    archive_path: Path,
    members: tuple[VerifiedZipMember, ...],
    destination: Path,
) -> dict[str, Path]:
    expected_names = {member.member_name for member in members}
    if len(expected_names) != len(members):
        raise ResearchTtsError("KazakhTTS runtime repeats a ZIP member declaration.")
    extracted: dict[str, Path] = {}
    try:
        with zipfile.ZipFile(archive_path) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ResearchTtsError(f"KazakhTTS ZIP CRC failed at member: {bad_member}")
            infos_by_name: dict[str, list[zipfile.ZipInfo]] = {}
            for info in archive.infolist():
                infos_by_name.setdefault(info.filename, []).append(info)
            for member in members:
                matches = infos_by_name.get(member.member_name, [])
                if len(matches) != 1:
                    raise ResearchTtsError(
                        f"KazakhTTS ZIP needs exactly one {member.member_name!r}, "
                        f"found {len(matches)}."
                    )
                info = matches[0]
                if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                    raise ResearchTtsError(
                        f"KazakhTTS ZIP member is not regular: {member.member_name}"
                    )
                if info.file_size != member.size_bytes:
                    raise ResearchTtsError(
                        f"KazakhTTS ZIP member size mismatch for {member.member_name}: "
                        f"expected {member.size_bytes}, got {info.file_size}."
                    )
                output = _resolve_below(destination, member.member_name)
                output.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info, "r") as source, output.open("xb") as handle:
                    _copy_and_verify(source, handle, member)
                extracted[member.member_name] = output
    except (OSError, zipfile.BadZipFile) as error:
        raise ResearchTtsError(
            f"Cannot safely read KazakhTTS ZIP {archive_path}: {error}"
        ) from error
    return extracted


def _copy_and_verify(
    source: _ByteReader, output: _ByteWriter, member: VerifiedZipMember
) -> None:
    digest = hashlib.sha256()
    received = 0
    while chunk := source.read(1024 * 1024):
        received += len(chunk)
        if received > member.size_bytes:
            raise ResearchTtsError(
                f"KazakhTTS extracted member exceeds expected size: {member.member_name}"
            )
        digest.update(chunk)
        output.write(chunk)
    if received != member.size_bytes or digest.hexdigest() != member.sha256:
        raise ResearchTtsError(
            f"KazakhTTS extracted member size/SHA-256 mismatch: {member.member_name}"
        )


def _member(runtime: Mapping[str, object], prefix: str, model_id: str) -> VerifiedZipMember:
    return VerifiedZipMember(
        member_name=_safe_member_path(_string(runtime, f"{prefix}_member", model_id), model_id),
        size_bytes=_positive_int(runtime, f"{prefix}_size_bytes", model_id),
        sha256=_sha256(runtime, f"{prefix}_sha256", model_id),
    )


def _yaml_mapping(path: Path, label: str) -> Mapping[str, object]:
    try:
        value: object = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        raise ResearchTtsError(f"Cannot read KazakhTTS {label}: {error}") from error
    return _mapping(value, label)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise ResearchTtsError(f"KazakhTTS {label} must be a mapping.")
    return cast(Mapping[str, object], value)


def _expect_exact_keys(runtime: Mapping[str, object], expected: set[str], model_id: str) -> None:
    unknown = sorted(set(runtime).difference(expected))
    missing = sorted(expected.difference(runtime))
    if unknown or missing:
        details: list[str] = []
        if missing:
            details.append("missing runtime fields: " + ", ".join(missing))
        if unknown:
            details.append("unknown runtime fields: " + ", ".join(unknown))
        raise ResearchTtsError(f"KazakhTTS model {model_id!r} has " + "; ".join(details))


def _artifact_path(
    runtime: Mapping[str, object], name: str, model: ResearchTtsModel, artifact_paths: set[str]
) -> str:
    value = _safe_member_path(_string(runtime, name, model.model_id), model.model_id)
    if value not in artifact_paths:
        raise ResearchTtsError(
            f"KazakhTTS model {model.model_id!r} runtime {name!r} is not a locked artifact."
        )
    return value


def _string(runtime: Mapping[str, object], name: str, model_id: str) -> str:
    value = runtime.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ResearchTtsError(
            f"KazakhTTS model {model_id!r} needs non-empty runtime {name!r}."
        )
    return value.strip()


def _positive_int(runtime: Mapping[str, object], name: str, model_id: str) -> int:
    value = runtime.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ResearchTtsError(
            f"KazakhTTS model {model_id!r} runtime {name!r} must be a positive int."
        )
    return value


def _positive_float(runtime: Mapping[str, object], name: str, model_id: str) -> float:
    value = runtime.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ResearchTtsError(
            f"KazakhTTS model {model_id!r} runtime {name!r} must be positive."
        )
    return float(value)


def _non_negative_float(runtime: Mapping[str, object], name: str, model_id: str) -> float:
    value = runtime.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
        raise ResearchTtsError(
            f"KazakhTTS model {model_id!r} runtime {name!r} must be non-negative."
        )
    return float(value)


def _boolean(runtime: Mapping[str, object], name: str, model_id: str) -> bool:
    value = runtime.get(name)
    if not isinstance(value, bool):
        raise ResearchTtsError(
            f"KazakhTTS model {model_id!r} runtime {name!r} must be boolean."
        )
    return value


def _sha256(runtime: Mapping[str, object], name: str, model_id: str) -> str:
    value = _string(runtime, name, model_id).lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ResearchTtsError(
            f"KazakhTTS model {model_id!r} runtime {name!r} must be SHA-256."
        )
    return value


def _language_list(runtime: Mapping[str, object], name: str, model_id: str) -> tuple[str, ...]:
    value = runtime.get(name)
    if not isinstance(value, list) or not value:
        raise ResearchTtsError(
            f"KazakhTTS model {model_id!r} runtime {name!r} must be a non-empty list."
        )
    languages: list[str] = []
    for item in value:
        if not isinstance(item, str) or item not in {"ru", "kk", "mixed"}:
            raise ResearchTtsError(
                f"KazakhTTS model {model_id!r} runtime {name!r} has invalid language."
            )
        languages.append(item)
    if len(languages) != len(set(languages)):
        raise ResearchTtsError(
            f"KazakhTTS model {model_id!r} runtime {name!r} repeats a language."
        )
    return tuple(sorted(languages))


def _single_directory(value: str, model_id: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 1 or path.name != value or "\\" in value:
        raise ResearchTtsError(
            f"KazakhTTS model {model_id!r} source_archive_root must be one directory."
        )
    return value


def _safe_member_path(value: str, model_id: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or value in {"", "."}:
        raise ResearchTtsError(f"KazakhTTS model {model_id!r} has unsafe archive member path.")
    return path.as_posix()


def _resolve_below(root: Path, relative_path: str) -> Path:
    resolved_root = root.resolve(strict=True)
    candidate = (resolved_root / relative_path).resolve(strict=False)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ResearchTtsError(
            f"KazakhTTS archive member escapes extraction root: {relative_path}"
        ) from error
    return candidate
