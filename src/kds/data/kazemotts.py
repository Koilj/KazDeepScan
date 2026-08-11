"""Safe material handling and fixed controls for the KazEmoTTS research generator.

The upstream repository predates Python 3.13 and bundles an obsolete compiled alignment
extension.  Alignment is used only during training, not inference.  This module therefore
extracts a small allowlist of pinned source files into a temporary directory and provides a
local fail-closed shim for that unused training-only import.  No upstream source tree or
unverified archive member is executed from the repository worktree.
"""

from __future__ import annotations

import hashlib
import stat
import tarfile
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol, cast

from kds.data.manifest import ManifestRow
from kds.data.research_tts import ResearchTtsError, ResearchTtsModel

KAZEMOTTS_SOURCE_ID = "ksc_derived_kk_v2_kazemotts"
KAZEMOTTS_SOURCE_LICENSE = "KSC text CC-BY-4.0; KazEmoTTS model CC-BY-4.0"
KAZEMOTTS_RUNTIME_KIND = "kazemotts_gradtts_hifigan"
MAX_SOURCE_UNPACKED_BYTES = 20 * 1024 * 1024

_SOURCE_FILES = frozenset(
    {
        "configs/hifigan-config.json",
        "configs/train_grad.json",
        "model/__init__.py",
        "model/base.py",
        "model/classifier.py",
        "model/diffusion.py",
        "model/monotonic_align/LICENCE",
        "model/monotonic_align/__init__.py",
        "model/monotonic_align/core.pyx",
        "model/monotonic_align/setup.py",
        "model/text_encoder.py",
        "model/tts.py",
        "model/utils.py",
        "models.py",
        "text/LICENSE",
        "text/__init__.py",
        "text/cleaners.py",
        "text/cmudict.py",
        "text/symbols.py",
        "xutils.py",
    }
)


@dataclass(frozen=True, slots=True)
class KazEmoTtsProfile:
    """One declared fixed speaker/emotion control; it is not a human identity claim."""

    voice_id: str
    speaker_id: int
    emotion_id: int


@dataclass(frozen=True, slots=True)
class KazEmoTtsRuntime:
    """All non-model parameters required for a reproducible KazEmoTTS inference run."""

    source_archive_path: str
    source_archive_root: str
    tts_archive_path: str
    tts_checkpoint_member: str
    tts_checkpoint_size_bytes: int
    tts_checkpoint_sha256: str
    vocoder_archive_path: str
    vocoder_checkpoint_member: str
    vocoder_checkpoint_size_bytes: int
    vocoder_checkpoint_sha256: str
    sample_rate: int
    n_timesteps: int
    temperature: float
    classifier_free_guidance: float
    profiles: tuple[KazEmoTtsProfile, ...]


class _ByteReader(Protocol):
    def read(self, size: int = -1) -> bytes: ...


class _ByteWriter(Protocol):
    def write(self, data: bytes) -> int: ...


def load_kazemotts_runtime(model: ResearchTtsModel) -> KazEmoTtsRuntime:
    """Parse the exact runtime contract, rejecting a partially specified generator."""

    runtime = model.runtime
    expected = {
        "kind",
        "source_archive_path",
        "source_archive_root",
        "tts_archive_path",
        "tts_checkpoint_member",
        "tts_checkpoint_size_bytes",
        "tts_checkpoint_sha256",
        "vocoder_archive_path",
        "vocoder_checkpoint_member",
        "vocoder_checkpoint_size_bytes",
        "vocoder_checkpoint_sha256",
        "sample_rate",
        "n_timesteps",
        "temperature",
        "classifier_free_guidance",
        "profiles",
    }
    unknown = sorted(set(runtime).difference(expected))
    missing = sorted(expected.difference(runtime))
    if unknown or missing:
        details: list[str] = []
        if missing:
            details.append("missing runtime fields: " + ", ".join(missing))
        if unknown:
            details.append("unknown runtime fields: " + ", ".join(unknown))
        raise ResearchTtsError(f"KazEmoTTS model {model.model_id!r} has " + "; ".join(details))
    if _runtime_string(runtime, "kind", model.model_id) != KAZEMOTTS_RUNTIME_KIND:
        raise ResearchTtsError(
            f"KazEmoTTS model {model.model_id!r} must use {KAZEMOTTS_RUNTIME_KIND!r}."
        )
    artifact_paths = {artifact.relative_path for artifact in model.artifacts}
    source_archive_path = _artifact_path(runtime, "source_archive_path", model, artifact_paths)
    tts_archive_path = _artifact_path(runtime, "tts_archive_path", model, artifact_paths)
    vocoder_archive_path = _artifact_path(runtime, "vocoder_archive_path", model, artifact_paths)
    source_archive_root = _single_directory(
        _runtime_string(runtime, "source_archive_root", model.model_id), model.model_id
    )
    profiles_raw = runtime["profiles"]
    if not isinstance(profiles_raw, list) or not profiles_raw:
        raise ResearchTtsError(f"KazEmoTTS model {model.model_id!r} needs non-empty profiles.")
    profiles = tuple(_parse_profile(value, model.model_id) for value in profiles_raw)
    voice_ids = [profile.voice_id for profile in profiles]
    if len(voice_ids) != len(set(voice_ids)):
        raise ResearchTtsError(
            f"KazEmoTTS model {model.model_id!r} has duplicate profile voice_id."
        )
    return KazEmoTtsRuntime(
        source_archive_path=source_archive_path,
        source_archive_root=source_archive_root,
        tts_archive_path=tts_archive_path,
        tts_checkpoint_member=_safe_member_path(
            _runtime_string(runtime, "tts_checkpoint_member", model.model_id), model.model_id
        ),
        tts_checkpoint_size_bytes=_positive_int(
            runtime, "tts_checkpoint_size_bytes", model.model_id
        ),
        tts_checkpoint_sha256=_sha256(runtime, "tts_checkpoint_sha256", model.model_id),
        vocoder_archive_path=vocoder_archive_path,
        vocoder_checkpoint_member=_safe_member_path(
            _runtime_string(runtime, "vocoder_checkpoint_member", model.model_id), model.model_id
        ),
        vocoder_checkpoint_size_bytes=_positive_int(
            runtime, "vocoder_checkpoint_size_bytes", model.model_id
        ),
        vocoder_checkpoint_sha256=_sha256(runtime, "vocoder_checkpoint_sha256", model.model_id),
        sample_rate=_positive_int(runtime, "sample_rate", model.model_id),
        n_timesteps=_positive_int(runtime, "n_timesteps", model.model_id),
        temperature=_positive_float(runtime, "temperature", model.model_id),
        classifier_free_guidance=_positive_float(
            runtime, "classifier_free_guidance", model.model_id
        ),
        profiles=profiles,
    )


def assign_kazemotts_profiles(
    rows: Iterable[ManifestRow], runtime: KazEmoTtsRuntime
) -> list[tuple[ManifestRow, KazEmoTtsProfile]]:
    """Assign declared profiles round-robin, so no profile silently dominates the source."""

    return [
        (row, runtime.profiles[index % len(runtime.profiles)]) for index, row in enumerate(rows)
    ]


def extract_verified_zip_member(
    archive_path: Path,
    member_name: str,
    expected_size_bytes: int,
    expected_sha256: str,
    destination: Path,
) -> None:
    """CRC-check an already SHA-pinned ZIP and publish exactly one regular member."""

    if destination.exists():
        raise ResearchTtsError(f"Refusing to overwrite extracted KazEmoTTS file: {destination}")
    try:
        with zipfile.ZipFile(archive_path) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ResearchTtsError(f"KazEmoTTS ZIP CRC failed at member: {bad_member}")
            matches = [info for info in archive.infolist() if info.filename == member_name]
            if len(matches) != 1:
                raise ResearchTtsError(
                    f"KazEmoTTS ZIP needs exactly one {member_name!r}, found {len(matches)}."
                )
            info = matches[0]
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16):
                raise ResearchTtsError(f"KazEmoTTS ZIP member is not a regular file: {member_name}")
            if info.file_size != expected_size_bytes:
                raise ResearchTtsError(
                    f"KazEmoTTS ZIP member size mismatch for {member_name}: "
                    f"expected {expected_size_bytes}, got {info.file_size}."
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, destination.open("xb") as output:
                _copy_and_verify(source, output, expected_size_bytes, expected_sha256, member_name)
    except (OSError, zipfile.BadZipFile) as error:
        raise ResearchTtsError(
            f"Cannot safely read KazEmoTTS ZIP {archive_path}: {error}"
        ) from error


def extract_verified_kazemotts_source(
    archive_path: Path, runtime: KazEmoTtsRuntime, destination_root: Path
) -> Path:
    """Extract only source files required for inference, never upstream binaries or datasets."""

    if destination_root.exists():
        raise ResearchTtsError(
            f"Refusing to overwrite KazEmoTTS source extraction: {destination_root}"
        )
    destination_root.mkdir(parents=True, exist_ok=False)
    resolved_root = destination_root.resolve(strict=True)
    prefix = f"{runtime.source_archive_root}/"
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
                        f"KazEmoTTS source member is not a regular file: {member.name}"
                    )
                if relative in found:
                    raise ResearchTtsError(f"KazEmoTTS source has duplicate member: {relative}")
                total_size += member.size
                if total_size > MAX_SOURCE_UNPACKED_BYTES:
                    raise ResearchTtsError("KazEmoTTS source exceeds its safe unpacked-size limit.")
                output_path = _resolve_below(resolved_root, relative)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise ResearchTtsError(f"Cannot read KazEmoTTS source member: {member.name}")
                with source, output_path.open("xb") as output:
                    _copy_and_verify(source, output, member.size, None, member.name)
                found.add(relative)
    except (OSError, tarfile.TarError) as error:
        raise ResearchTtsError(
            f"Cannot safely read KazEmoTTS source archive {archive_path}: {error}"
        ) from error
    missing = sorted(_SOURCE_FILES.difference(found))
    if missing:
        raise ResearchTtsError("KazEmoTTS source is missing required files: " + ", ".join(missing))
    _write_training_only_alignment_shim(resolved_root)
    return resolved_root


def _write_training_only_alignment_shim(source_root: Path) -> None:
    """Provide the unused training import without building the upstream Python-3.9 extension."""

    shim_path = source_root / "model" / "monotonic_align" / "model" / "monotonic_align" / "core.py"
    shim_path.parent.mkdir(parents=True, exist_ok=True)
    shim_path.write_text(
        "def maximum_path_c(*_args: object, **_kwargs: object) -> None:\n"
        "    raise RuntimeError(\n"
        "        'KazEmoTTS alignment is training-only and unavailable in inference.'\n"
        "    )\n",
        encoding="utf-8",
    )


def _copy_and_verify(
    source: _ByteReader,
    output: _ByteWriter,
    expected_size_bytes: int,
    expected_sha256: str | None,
    label: str,
) -> None:
    digest = hashlib.sha256()
    received = 0
    while True:
        chunk = source.read(1024 * 1024)
        if not chunk:
            break
        received += len(chunk)
        if received > expected_size_bytes:
            raise ResearchTtsError(f"KazEmoTTS extracted member exceeds expected size: {label}")
        digest.update(chunk)
        output.write(chunk)
    if received != expected_size_bytes:
        raise ResearchTtsError(
            f"KazEmoTTS extracted member size mismatch for {label}: "
            f"expected {expected_size_bytes}, got {received}."
        )
    if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
        raise ResearchTtsError(f"KazEmoTTS extracted member SHA-256 mismatch: {label}")


def _parse_profile(value: object, model_id: str) -> KazEmoTtsProfile:
    if not isinstance(value, dict):
        raise ResearchTtsError(f"KazEmoTTS model {model_id!r} profile must be an object.")
    raw = cast(dict[str, object], value)
    expected = {"voice_id", "speaker_id", "emotion_id"}
    if set(raw) != expected:
        raise ResearchTtsError(f"KazEmoTTS model {model_id!r} profile has invalid fields.")
    speaker_id = raw["speaker_id"]
    emotion_id = raw["emotion_id"]
    if (
        not isinstance(speaker_id, int)
        or isinstance(speaker_id, bool)
        or speaker_id < 0
        or not isinstance(emotion_id, int)
        or isinstance(emotion_id, bool)
        or emotion_id < 0
    ):
        raise ResearchTtsError(
            f"KazEmoTTS model {model_id!r} profile IDs must be non-negative ints."
        )
    return KazEmoTtsProfile(
        voice_id=_runtime_string(raw, "voice_id", model_id),
        speaker_id=speaker_id,
        emotion_id=emotion_id,
    )


def _artifact_path(
    runtime: Mapping[str, object], name: str, model: ResearchTtsModel, artifact_paths: set[str]
) -> str:
    value = _safe_member_path(_runtime_string(runtime, name, model.model_id), model.model_id)
    if value not in artifact_paths:
        raise ResearchTtsError(
            f"KazEmoTTS model {model.model_id!r} runtime {name!r} is not a locked artifact."
        )
    return value


def _runtime_string(runtime: Mapping[str, object], name: str, model_id: str) -> str:
    value = runtime.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ResearchTtsError(f"KazEmoTTS model {model_id!r} needs non-empty runtime {name!r}.")
    return value.strip()


def _positive_int(runtime: Mapping[str, object], name: str, model_id: str) -> int:
    value = runtime.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ResearchTtsError(
            f"KazEmoTTS model {model_id!r} runtime {name!r} must be positive int."
        )
    return value


def _positive_float(runtime: Mapping[str, object], name: str, model_id: str) -> float:
    value = runtime.get(name)
    if not isinstance(value, (float, int)) or isinstance(value, bool) or value <= 0:
        raise ResearchTtsError(
            f"KazEmoTTS model {model_id!r} runtime {name!r} must be positive number."
        )
    return float(value)


def _sha256(runtime: Mapping[str, object], name: str, model_id: str) -> str:
    value = _runtime_string(runtime, name, model_id).lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ResearchTtsError(f"KazEmoTTS model {model_id!r} runtime {name!r} must be SHA-256.")
    return value


def _single_directory(value: str, model_id: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or len(path.parts) != 1 or path.name != value or "\\" in value:
        raise ResearchTtsError(
            f"KazEmoTTS model {model_id!r} source_archive_root must be one directory name."
        )
    return value


def _safe_member_path(value: str, model_id: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or value in {"", "."}:
        raise ResearchTtsError(f"KazEmoTTS model {model_id!r} has unsafe archive member path.")
    return path.as_posix()


def _resolve_below(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ResearchTtsError(
            f"KazEmoTTS source path escapes extraction root: {relative_path}"
        ) from error
    return candidate
