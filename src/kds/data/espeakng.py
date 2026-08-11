"""Pinned local eSpeak NG formant synthesis for Russian/Kazakh research data.

This adapter is deliberately limited to text passed over standard input and the bundled
``ru`` or ``kk`` language data. It neither accepts reference audio nor exposes an API that can
clone a speaker. Ubuntu's small, pinned runtime packages are extracted into a temporary directory
with path, type and unpacked-size checks before the verified executable is invoked.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tarfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from kds.data.manifest import ManifestRow
from kds.data.research_tts import ResearchTtsError, ResearchTtsModel

ESPEAKNG_SOURCE_ID = "ksc_derived_kk_v4_espeakng"
ESPEAKNG_SOURCE_LICENSE = "KSC text CC-BY-4.0; eSpeak NG GPL-3.0-or-later"
ESPEAKNG_RUNTIME_KIND = "espeakng_formant_cli"
MAX_DEB_UNPACKED_BYTES = 64 * 1024 * 1024
_BINARY_RELATIVE_PATH = PurePosixPath("usr/bin/espeak-ng")
_LIBRARY_RELATIVE_DIRECTORY = PurePosixPath("usr/lib/x86_64-linux-gnu")
_DATA_RELATIVE_DIRECTORY = _LIBRARY_RELATIVE_DIRECTORY / "espeak-ng-data"
_VOICE_RUNTIME_FILES: Mapping[str, tuple[PurePosixPath, PurePosixPath]] = {
    "kk": (PurePosixPath("lang/trk/kk"), PurePosixPath("kk_dict")),
    "ru": (PurePosixPath("lang/zle/ru"), PurePosixPath("ru_dict")),
}


@dataclass(frozen=True, slots=True)
class EspeakNgProfile:
    """A deterministic synthesis control, not an identity or a claimed human voice."""

    voice_id: str
    speed_wpm: int
    pitch: int
    amplitude: int


@dataclass(frozen=True, slots=True)
class EspeakNgRuntime:
    """Exact pinned components and safe text-only controls for eSpeak NG."""

    source_archive_path: str
    binary_deb_path: str
    library_deb_paths: tuple[str, ...]
    data_deb_path: str
    voice: str
    sample_rate: int
    profiles: tuple[EspeakNgProfile, ...]


@dataclass(frozen=True, slots=True)
class EspeakNgPaths:
    """Trusted paths inside one temporary, package-extracted runtime directory."""

    binary: Path
    library_directory: Path
    data_directory: Path


def load_espeakng_runtime(model: ResearchTtsModel) -> EspeakNgRuntime:
    """Parse the full formant-TTS contract and reject unpinned/unsafe controls."""

    runtime = model.runtime
    expected = {
        "kind",
        "source_archive_path",
        "binary_deb_path",
        "library_deb_paths",
        "data_deb_path",
        "voice",
        "sample_rate",
        "profiles",
    }
    _expect_exact_keys(runtime, expected, model.model_id)
    if _runtime_string(runtime, "kind", model.model_id) != ESPEAKNG_RUNTIME_KIND:
        raise ResearchTtsError(
            f"eSpeak NG model {model.model_id!r} must use {ESPEAKNG_RUNTIME_KIND!r}."
        )
    artifact_paths = {artifact.relative_path for artifact in model.artifacts}
    source_archive_path = _artifact_path(runtime, "source_archive_path", model, artifact_paths)
    binary_deb_path = _artifact_path(runtime, "binary_deb_path", model, artifact_paths)
    data_deb_path = _artifact_path(runtime, "data_deb_path", model, artifact_paths)
    library_deb_paths = _library_deb_paths(runtime, model, artifact_paths)
    if artifact_paths != {
        source_archive_path,
        binary_deb_path,
        data_deb_path,
        *library_deb_paths,
    }:
        raise ResearchTtsError(
            f"eSpeak NG model {model.model_id!r} must pin only its source and runtime packages."
        )
    voice = _runtime_string(runtime, "voice", model.model_id)
    if voice not in _VOICE_RUNTIME_FILES:
        raise ResearchTtsError(
            f"eSpeak NG model {model.model_id!r} must pin one of {sorted(_VOICE_RUNTIME_FILES)}."
        )
    sample_rate = _positive_int(runtime, "sample_rate", model.model_id)
    if sample_rate != 22_050:
        raise ResearchTtsError(
            f"eSpeak NG model {model.model_id!r} must use its native 22050 Hz WAV output."
        )
    profiles_value = runtime["profiles"]
    if not isinstance(profiles_value, list) or not profiles_value:
        raise ResearchTtsError(f"eSpeak NG model {model.model_id!r} needs non-empty profiles.")
    profiles = tuple(_parse_profile(value, model.model_id) for value in profiles_value)
    profile_ids = [profile.voice_id for profile in profiles]
    if len(profile_ids) != len(set(profile_ids)):
        raise ResearchTtsError(
            f"eSpeak NG model {model.model_id!r} has duplicate profile voice_id."
        )
    return EspeakNgRuntime(
        source_archive_path=source_archive_path,
        binary_deb_path=binary_deb_path,
        library_deb_paths=library_deb_paths,
        data_deb_path=data_deb_path,
        voice=voice,
        sample_rate=sample_rate,
        profiles=profiles,
    )


def assign_espeakng_profiles(
    rows: Iterable[ManifestRow], runtime: EspeakNgRuntime
) -> list[tuple[ManifestRow, EspeakNgProfile]]:
    """Assign declared non-identity controls round-robin without changing KSC provenance."""

    return [
        (row, runtime.profiles[index % len(runtime.profiles)]) for index, row in enumerate(rows)
    ]


def extract_verified_espeakng_runtime(
    verified_paths: Mapping[str, Path], runtime: EspeakNgRuntime, destination: Path
) -> EspeakNgPaths:
    """Safely extract only already-SHA-verified Debian runtime payloads into ``destination``."""

    if destination.exists():
        raise ResearchTtsError(f"eSpeak NG runtime destination already exists: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    extracted_bytes = 0
    for artifact_path in (
        runtime.binary_deb_path,
        *runtime.library_deb_paths,
        runtime.data_deb_path,
    ):
        try:
            archive = verified_paths[artifact_path]
        except KeyError as error:
            raise ResearchTtsError(
                f"Missing verified eSpeak NG package: {artifact_path}"
            ) from error
        extracted_bytes = _extract_deb_payload(archive, destination, extracted_bytes)
    binary = _destination_path(destination, _BINARY_RELATIVE_PATH)
    library_directory = _destination_path(destination, _LIBRARY_RELATIVE_DIRECTORY)
    data_directory = _destination_path(destination, _DATA_RELATIVE_DIRECTORY)
    language_path, dictionary_path = _VOICE_RUNTIME_FILES[runtime.voice]
    expected = (
        binary,
        library_directory / "libespeak-ng.so.1",
        library_directory / "libpcaudio.so.0",
        library_directory / "libsonic.so.0",
        data_directory / language_path,
        data_directory / dictionary_path,
    )
    missing = [str(path.relative_to(destination)) for path in expected if not path.exists()]
    if missing:
        raise ResearchTtsError(
            f"eSpeak NG runtime packages do not provide expected {runtime.voice} runtime files: "
            + ", ".join(missing)
        )
    if not os.access(binary, os.X_OK):
        raise ResearchTtsError("eSpeak NG runtime binary is not executable after safe extraction.")
    return EspeakNgPaths(
        binary=binary, library_directory=library_directory, data_directory=data_directory
    )


def synthesize_espeakng(
    *,
    runtime_paths: EspeakNgPaths,
    runtime: EspeakNgRuntime,
    profile: EspeakNgProfile,
    text: str,
    output: Path,
) -> None:
    """Generate one WAV from text over stdin; no reference-audio input exists in this API."""

    environment = os.environ.copy()
    existing_library_path = environment.get("LD_LIBRARY_PATH", "")
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        filter(None, (str(runtime_paths.library_directory), existing_library_path))
    )
    environment["ESPEAK_DATA_PATH"] = str(runtime_paths.data_directory)
    result = subprocess.run(
        [
            str(runtime_paths.binary),
            "-v",
            runtime.voice,
            "-s",
            str(profile.speed_wpm),
            "-p",
            str(profile.pitch),
            "-a",
            str(profile.amplitude),
            "-w",
            str(output),
            "--stdin",
        ],
        input=text.encode("utf-8"),
        check=False,
        capture_output=True,
        env=environment,
    )
    if result.returncode != 0:
        details = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"eSpeak NG synthesis failed for profile {profile.voice_id}: {details}")
    if not output.is_file():
        raise RuntimeError("eSpeak NG reported success but did not create its WAV output.")


def _extract_deb_payload(archive: Path, destination: Path, extracted_bytes: int) -> int:
    """Run the system Debian reader, then perform our own path/type-safe tar extraction."""

    try:
        result = subprocess.run(
            ["dpkg-deb", "--fsys-tarfile", str(archive)],
            check=False,
            capture_output=True,
        )
    except OSError as error:
        raise ResearchTtsError(
            f"Cannot read eSpeak NG Debian package {archive}: {error}"
        ) from error
    if result.returncode != 0:
        details = result.stderr.decode("utf-8", errors="replace").strip()
        raise ResearchTtsError(f"Debian package reader rejected {archive.name}: {details}")
    try:
        with tarfile.open(fileobj=io.BytesIO(result.stdout), mode="r:") as handle:
            links: list[tuple[PurePosixPath, str]] = []
            for member in handle:
                if member.name in {"", ".", "./"}:
                    if member.isdir():
                        continue
                    raise ResearchTtsError("eSpeak NG package has a non-directory root member.")
                relative = _safe_member_path(member.name)
                if member.isdir():
                    _destination_path(destination, relative).mkdir(parents=True, exist_ok=True)
                    continue
                if member.isreg():
                    if member.size < 0 or extracted_bytes + member.size > MAX_DEB_UNPACKED_BYTES:
                        raise ResearchTtsError(
                            "eSpeak NG Debian payload exceeds safe unpacked limit."
                        )
                    output = _destination_path(destination, relative)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    if output.exists() or output.is_symlink():
                        raise ResearchTtsError(f"Duplicate eSpeak NG package member: {relative}")
                    content = handle.extractfile(member)
                    if content is None:
                        raise ResearchTtsError(f"Cannot read eSpeak NG package member: {relative}")
                    with output.open("xb") as output_handle:
                        shutil.copyfileobj(content, output_handle)
                    if output.stat().st_size != member.size:
                        raise ResearchTtsError(f"Truncated eSpeak NG package member: {relative}")
                    output.chmod(member.mode & 0o777)
                    extracted_bytes += member.size
                    continue
                if member.issym():
                    links.append((relative, member.linkname))
                    continue
                raise ResearchTtsError(
                    f"Unsafe non-file eSpeak NG package member {relative!s} ({member.type!r})."
                )
    except (OSError, tarfile.TarError) as error:
        raise ResearchTtsError(
            f"Cannot safely extract eSpeak NG package {archive.name}: {error}"
        ) from error
    for relative, target in links:
        output = _destination_path(destination, relative)
        _safe_link_target(relative, target)
        if output.exists() or output.is_symlink():
            raise ResearchTtsError(f"Duplicate eSpeak NG package symlink: {relative}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.symlink_to(target)
    return extracted_bytes


def _safe_member_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ResearchTtsError(f"eSpeak NG package member escapes runtime root: {value!r}")
    parts = tuple(part for part in path.parts if part not in {"", "."})
    if not parts:
        raise ResearchTtsError("eSpeak NG package contains an empty member path.")
    return PurePosixPath(*parts)


def _safe_link_target(link_path: PurePosixPath, target: str) -> PurePosixPath:
    target_path = PurePosixPath(target)
    if target_path.is_absolute() or "\\" in target or not target:
        raise ResearchTtsError(f"eSpeak NG package symlink has unsafe target: {target!r}")
    parts = list(link_path.parent.parts)
    for part in target_path.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                raise ResearchTtsError(
                    f"eSpeak NG package symlink escapes runtime root: {target!r}"
                )
            parts.pop()
            continue
        parts.append(part)
    if not parts:
        raise ResearchTtsError(f"eSpeak NG package symlink has empty target: {target!r}")
    return PurePosixPath(*parts)


def _destination_path(root: Path, relative: PurePosixPath) -> Path:
    candidate = root
    for part in relative.parts:
        if candidate.is_symlink():
            raise ResearchTtsError(f"eSpeak NG package path traverses symlink: {relative}")
        candidate = candidate / part
    return candidate


def _expect_exact_keys(raw: Mapping[str, object], expected: set[str], model_id: str) -> None:
    unknown = sorted(set(raw).difference(expected))
    missing = sorted(expected.difference(raw))
    if unknown or missing:
        details: list[str] = []
        if missing:
            details.append("missing runtime fields: " + ", ".join(missing))
        if unknown:
            details.append("unknown runtime fields: " + ", ".join(unknown))
        raise ResearchTtsError(f"eSpeak NG model {model_id!r} has " + "; ".join(details))


def _runtime_string(runtime: Mapping[str, object], name: str, model_id: str) -> str:
    value = runtime.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ResearchTtsError(f"eSpeak NG model {model_id!r} needs non-empty {name!r}.")
    return value.strip()


def _positive_int(runtime: Mapping[str, object], name: str, model_id: str) -> int:
    value = runtime.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ResearchTtsError(f"eSpeak NG model {model_id!r} needs positive integer {name!r}.")
    return value


def _artifact_path(
    runtime: Mapping[str, object], name: str, model: ResearchTtsModel, artifact_paths: set[str]
) -> str:
    value = _runtime_string(runtime, name, model.model_id)
    if value not in artifact_paths:
        raise ResearchTtsError(
            f"eSpeak NG model {model.model_id!r} runtime {name!r} is not a locked artifact."
        )
    return value


def _library_deb_paths(
    runtime: Mapping[str, object], model: ResearchTtsModel, artifact_paths: set[str]
) -> tuple[str, ...]:
    value = runtime.get("library_deb_paths")
    if not isinstance(value, list) or len(value) != 3:
        raise ResearchTtsError(
            f"eSpeak NG model {model.model_id!r} needs exactly three library Debian packages."
        )
    paths: list[str] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, str) or not item.strip() or item not in artifact_paths:
            raise ResearchTtsError(
                f"eSpeak NG model {model.model_id!r} library package {index} is not locked."
            )
        paths.append(item)
    if len(paths) != len(set(paths)):
        raise ResearchTtsError(f"eSpeak NG model {model.model_id!r} repeats a library package.")
    return tuple(paths)


def _parse_profile(value: object, model_id: str) -> EspeakNgProfile:
    if not isinstance(value, dict):
        raise ResearchTtsError(f"eSpeak NG model {model_id!r} profile must be an object.")
    raw = cast(dict[str, object], value)
    _expect_exact_keys(raw, {"voice_id", "speed_wpm", "pitch", "amplitude"}, model_id)
    voice_id = _runtime_string(raw, "voice_id", model_id)
    speed_wpm = _positive_int(raw, "speed_wpm", model_id)
    pitch = _positive_int(raw, "pitch", model_id)
    amplitude = _positive_int(raw, "amplitude", model_id)
    if not 80 <= speed_wpm <= 450:
        raise ResearchTtsError(f"eSpeak NG model {model_id!r} profile speed must be 80..450 WPM.")
    if not 0 < pitch <= 99 or not 0 < amplitude <= 200:
        raise ResearchTtsError(
            f"eSpeak NG model {model_id!r} profile pitch/amplitude out of range."
        )
    return EspeakNgProfile(voice_id=voice_id, speed_wpm=speed_wpm, pitch=pitch, amplitude=amplitude)
