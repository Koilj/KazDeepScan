"""Fail-closed Qwen3-TTS CustomVoice GGUF route for VoxForge RU.

The public CustomVoice checkpoint has fixed speaker tokens.  This wrapper loads
only the pinned local GGUF talker, GGUF codec and a hash-checked CrispASR CUDA
release.  It never exposes the Base-model reference-audio path, VoiceDesign,
runtime auto-download, an external normalizer or an instruction prompt.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

import soundfile as sf  # type: ignore[import-untyped]

from kds.data.assets import sha256_file
from kds.data.research_tts import ResearchTtsModel, verify_research_tts_model_bundle

QWEN3_TTS_CUSTOMVOICE_RUNTIME_KIND = "crispasr_qwen3_tts_customvoice_gguf_fixed_aiden_cuda"
QWEN3_TTS_CUSTOMVOICE_MODEL_ID = "qwen3_tts_0_6b_customvoice_aiden_q8_0"


class Qwen3TtsCustomVoiceError(ValueError):
    """Raised when the fixed Qwen3-TTS CustomVoice route is not safe to run."""


@dataclass(frozen=True, slots=True)
class Qwen3TtsCustomVoiceText:
    """One literal source transcript and its pre-committed deterministic seed."""

    source_text: str
    seed: int


@dataclass(frozen=True, slots=True)
class _RuntimeMember:
    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class Qwen3TtsCustomVoice:
    """Verified local fixed-speaker Qwen3-TTS route.

    The only public synthesis operation accepts literal source text and an empty
    output path.  The command line is fully constructed here so callers cannot
    inject a reference WAV, a voice-design instruction, a different speaker or
    model path.
    """

    executable: Path
    talker_path: Path
    codec_path: Path
    cuda_library_dirs: tuple[Path, ...]
    fixed_speaker_name: str
    sample_rate: int
    target_language: str
    temperature: float
    max_new_tokens: int

    def prepare_text(self, source_text: str) -> Qwen3TtsCustomVoiceText:
        """Accept one literal non-empty UTF-8 source transcript without rewriting it."""

        if not isinstance(source_text, str) or not source_text.strip():
            raise Qwen3TtsCustomVoiceError("Qwen3-TTS synthesis requires non-empty source text.")
        if "\x00" in source_text:
            raise Qwen3TtsCustomVoiceError("Qwen3-TTS source text must not contain NUL bytes.")
        if len(source_text.encode("utf-8")) > 4096:
            raise Qwen3TtsCustomVoiceError(
                "Qwen3-TTS source text exceeds the fixed 4096-byte limit."
            )
        seed = int.from_bytes(hashlib.sha256(source_text.encode("utf-8")).digest()[:4], "big")
        return Qwen3TtsCustomVoiceText(source_text=source_text, seed=seed)

    def command_for(self, prepared: Qwen3TtsCustomVoiceText, output_path: Path) -> tuple[str, ...]:
        """Return the complete no-reference-audio command for one literal text."""

        if output_path.suffix.casefold() != ".wav":
            raise Qwen3TtsCustomVoiceError("Qwen3-TTS output must use the .wav extension.")
        return (
            str(self.executable),
            "--backend",
            "qwen3-tts-customvoice",
            "--model",
            str(self.talker_path),
            "--codec-model",
            str(self.codec_path),
            "--voice",
            self.fixed_speaker_name,
            "--target-lang",
            self.target_language,
            "--seed",
            str(prepared.seed),
            "--temperature",
            str(self.temperature),
            "--max-new-tokens",
            str(self.max_new_tokens),
            "--tts",
            prepared.source_text,
            "--tts-output",
            str(output_path),
            "--gpu-backend",
            "cuda",
            "--no-prints",
        )

    def synthesize_to_file(self, prepared: Qwen3TtsCustomVoiceText, output_path: Path) -> None:
        """Generate exactly one 24 kHz mono WAV with the fixed local route."""

        if output_path.exists() or not output_path.parent.is_dir():
            raise Qwen3TtsCustomVoiceError(
                "Qwen3-TTS output must be a new file under an existing directory."
            )
        completed = subprocess.run(
            self.command_for(prepared, output_path),
            cwd=self.executable.parent,
            env=_cuda_environment(self.cuda_library_dirs),
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip()[-1200:]
            raise Qwen3TtsCustomVoiceError(
                f"Pinned Qwen3-TTS runtime failed with exit code {completed.returncode}: {stderr}"
            )
        if not output_path.is_file():
            raise Qwen3TtsCustomVoiceError("Pinned Qwen3-TTS runtime produced no WAV output.")
        try:
            info = sf.info(output_path)
        except RuntimeError as error:
            raise Qwen3TtsCustomVoiceError(
                "Pinned Qwen3-TTS output is not readable audio."
            ) from error
        if info.samplerate != self.sample_rate or info.channels != 1 or info.frames <= 0:
            raise Qwen3TtsCustomVoiceError(
                "Pinned Qwen3-TTS output must be non-empty 24 kHz mono audio."
            )


def load_qwen3_tts_customvoice(model_root: Path, model: ResearchTtsModel) -> Qwen3TtsCustomVoice:
    """Verify, safely extract and health-check the one pinned CustomVoice route."""

    if model.model_id != QWEN3_TTS_CUSTOMVOICE_MODEL_ID:
        raise Qwen3TtsCustomVoiceError(f"Unexpected Qwen3-TTS model id: {model.model_id!r}.")
    runtime = _validated_runtime(model.runtime)
    if sha256_file(Path(__file__).resolve(strict=True)) != _required_string(
        runtime, "wrapper_sha256"
    ):
        raise Qwen3TtsCustomVoiceError("Qwen3-TTS wrapper SHA-256 differs from the model lock.")
    verified = verify_research_tts_model_bundle(model_root, model)
    bundle_root = (model_root / model.destination).resolve(strict=True)
    talker_path = verified[_required_string(runtime, "talker_path")]
    codec_path = verified[_required_string(runtime, "codec_path")]
    archive_path = verified[_required_string(runtime, "runtime_archive_path")]
    executable_path = _extract_runtime(bundle_root, archive_path, runtime)
    if executable_path != _resolve_verified_path(
        bundle_root, _required_string(runtime, "runtime_executable_path")
    ):
        raise Qwen3TtsCustomVoiceError("Qwen3-TTS runtime executable path does not match the lock.")
    cuda_library_dirs = _project_cuda_library_dirs()
    _verify_runtime_health(executable_path, cuda_library_dirs, runtime)
    return Qwen3TtsCustomVoice(
        executable=executable_path,
        talker_path=talker_path,
        codec_path=codec_path,
        cuda_library_dirs=cuda_library_dirs,
        fixed_speaker_name=_required_string(runtime, "fixed_speaker_name"),
        sample_rate=_required_int(runtime, "sample_rate"),
        target_language=_required_string(runtime, "target_language"),
        temperature=_required_float(runtime, "temperature"),
        max_new_tokens=_required_int(runtime, "max_new_tokens"),
    )


def _validated_runtime(runtime: Mapping[str, object]) -> Mapping[str, object]:
    if _required_string(runtime, "kind") != QWEN3_TTS_CUSTOMVOICE_RUNTIME_KIND:
        raise Qwen3TtsCustomVoiceError("Qwen3-TTS lock has an unsupported runtime kind.")
    expected = {
        "runtime_backend": "qwen3-tts-customvoice",
        "fixed_speaker_name": "aiden",
        "fixed_voice_id": "qwen3_tts_customvoice:aiden",
        "target_language": "ru",
        "reference_audio_policy": "forbidden",
        "voice_design": "forbidden",
        "device": "cuda",
        "cuda_runtime_policy": "require_project_venv_cuda12_runtime_and_cublas12",
        "literal_text_policy": "pass_literal_utf8_source_text_as_one_argv_value",
    }
    for key, value in expected.items():
        if _required_string(runtime, key) != value:
            raise Qwen3TtsCustomVoiceError(f"Qwen3-TTS runtime must set {key}={value!r}.")
    if _required_bool(runtime, "voice_cloning", "Qwen3-TTS runtime"):
        raise Qwen3TtsCustomVoiceError("Qwen3-TTS runtime must forbid voice cloning.")
    if not _required_bool(runtime, "text_input_only", "Qwen3-TTS runtime"):
        raise Qwen3TtsCustomVoiceError("Qwen3-TTS runtime must be text-input-only.")
    for key in (
        "talker_path",
        "codec_path",
        "runtime_archive_path",
        "runtime_directory",
        "runtime_executable_path",
        "runtime_version",
        "runtime_source_revision",
        "runtime_release_asset",
        "wrapper_module",
        "wrapper_sha256",
    ):
        _required_string(runtime, key)
    if _required_string(runtime, "runtime_version") != "0.8.28":
        raise Qwen3TtsCustomVoiceError("Qwen3-TTS runtime must pin CrispASR v0.8.28.")
    if (
        _required_string(runtime, "runtime_source_revision")
        != "99e990aeeb3bd281b836cf2813ec85f24d4e408e"
    ):
        raise Qwen3TtsCustomVoiceError("Qwen3-TTS runtime has an unexpected CrispASR revision.")
    if _required_int(runtime, "sample_rate") != 24000:
        raise Qwen3TtsCustomVoiceError("Qwen3-TTS runtime sample rate must be 24000 Hz.")
    if _required_float(runtime, "temperature") != 0.9:
        raise Qwen3TtsCustomVoiceError("Qwen3-TTS runtime temperature must be fixed at 0.9.")
    if _required_int(runtime, "max_new_tokens") != 512:
        raise Qwen3TtsCustomVoiceError("Qwen3-TTS runtime max_new_tokens must be fixed at 512.")
    if not _runtime_members(runtime):
        raise Qwen3TtsCustomVoiceError("Qwen3-TTS runtime archive member lock is empty.")
    return runtime


def _extract_runtime(bundle_root: Path, archive_path: Path, runtime: Mapping[str, object]) -> Path:
    runtime_directory = _required_string(runtime, "runtime_directory")
    destination = _resolve_verified_path(bundle_root, runtime_directory)
    members = _runtime_members(runtime)
    if destination.exists():
        _verify_runtime_tree(destination, members)
    else:
        stage_parent = destination.parent
        stage_parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix="kds-qwen3-tts-", dir=stage_parent))
        try:
            _extract_archive_to_stage(archive_path, stage, members)
            staged_destination = _resolve_verified_path(
                stage, PurePosixPath(runtime_directory).name
            )
            _verify_runtime_tree(staged_destination, members)
            staged_destination.replace(destination)
        except (OSError, tarfile.TarError) as error:
            raise Qwen3TtsCustomVoiceError(
                f"Cannot safely extract pinned CrispASR runtime: {error}"
            ) from error
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    executable = _resolve_verified_path(
        bundle_root, _required_string(runtime, "runtime_executable_path")
    )
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise Qwen3TtsCustomVoiceError("Pinned CrispASR executable is missing or not executable.")
    return executable


def _extract_archive_to_stage(
    archive_path: Path, stage: Path, expected_members: Mapping[str, _RuntimeMember]
) -> None:
    seen: set[str] = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            path = _safe_archive_path(member.name)
            if member.isdir():
                continue
            if not member.isfile() or path not in expected_members or path in seen:
                raise Qwen3TtsCustomVoiceError(
                    "Pinned CrispASR archive has an unsafe or unknown member."
                )
            expected = expected_members[path]
            if member.size != expected.size_bytes:
                raise Qwen3TtsCustomVoiceError(
                    f"Pinned CrispASR archive member size mismatch: {path}."
                )
            target = _resolve_verified_path(stage, path)
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise Qwen3TtsCustomVoiceError(
                    f"Cannot read pinned CrispASR archive member: {path}."
                )
            with source, target.open("xb") as handle:
                shutil.copyfileobj(source, handle)
            if sha256_file(target) != expected.sha256:
                raise Qwen3TtsCustomVoiceError(
                    f"Pinned CrispASR archive member SHA-256 mismatch: {path}."
                )
            if target.name.startswith("crispasr"):
                target.chmod(0o700)
            seen.add(path)
    if set(expected_members) != seen:
        raise Qwen3TtsCustomVoiceError("Pinned CrispASR archive is missing required members.")


def _verify_runtime_tree(destination: Path, expected_members: Mapping[str, _RuntimeMember]) -> None:
    if not destination.is_dir() or destination.is_symlink():
        raise Qwen3TtsCustomVoiceError("Pinned CrispASR runtime directory is unsafe or missing.")
    observed: set[str] = set()
    for path in destination.rglob("*"):
        if path.is_symlink():
            raise Qwen3TtsCustomVoiceError("Pinned CrispASR runtime must not contain symlinks.")
        if not path.is_file():
            continue
        relative = path.relative_to(destination.parent).as_posix()
        expected = expected_members.get(relative)
        if expected is None or path.stat().st_size != expected.size_bytes:
            raise Qwen3TtsCustomVoiceError("Pinned CrispASR runtime tree differs from its lock.")
        if sha256_file(path) != expected.sha256:
            raise Qwen3TtsCustomVoiceError(
                "Pinned CrispASR runtime member hash differs from its lock."
            )
        observed.add(relative)
    if observed != set(expected_members):
        raise Qwen3TtsCustomVoiceError("Pinned CrispASR runtime tree is incomplete.")


def _project_cuda_library_dirs() -> tuple[Path, ...]:
    site_packages = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    directories = (
        site_packages / "nvidia" / "cuda_runtime" / "lib",
        site_packages / "nvidia" / "cublas" / "lib",
    )
    requirements = (
        (directories[0], "libcudart.so.12"),
        (directories[1], "libcublas.so.12"),
        (directories[1], "libcublasLt.so.12"),
    )
    if any(
        not directory.is_dir() or not (directory / filename).is_file()
        for directory, filename in requirements
    ):
        raise Qwen3TtsCustomVoiceError(
            "Pinned CrispASR CUDA runtime requires the project's CUDA-12 torch runtime libraries."
        )
    return directories


def _verify_runtime_health(
    executable: Path, cuda_library_dirs: tuple[Path, ...], runtime: Mapping[str, object]
) -> None:
    environment = _cuda_environment(cuda_library_dirs)
    try:
        version = subprocess.run(
            [str(executable), "--version"],
            cwd=executable.parent,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        backends = subprocess.run(
            [str(executable), "--list-backends-json"],
            cwd=executable.parent,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except OSError as error:
        raise Qwen3TtsCustomVoiceError(
            f"Cannot execute pinned CrispASR runtime: {error}"
        ) from error
    if version.returncode != 0 or "ggml backends : cpu,cuda" not in version.stdout:
        raise Qwen3TtsCustomVoiceError("Pinned CrispASR runtime is not the expected CUDA build.")
    if f"version       : {_required_string(runtime, 'runtime_version')}" not in version.stdout:
        raise Qwen3TtsCustomVoiceError("Pinned CrispASR runtime version differs from the lock.")
    if _required_string(runtime, "runtime_source_revision")[:8] not in version.stdout:
        raise Qwen3TtsCustomVoiceError(
            "Pinned CrispASR runtime source revision differs from the lock."
        )
    try:
        backends_json: object = json.loads(backends.stdout)
    except json.JSONDecodeError as error:
        raise Qwen3TtsCustomVoiceError(
            "Pinned CrispASR backend inventory is not valid JSON."
        ) from error
    if backends.returncode != 0 or not _contains_customvoice_backend(backends_json):
        raise Qwen3TtsCustomVoiceError("Pinned CrispASR runtime lacks qwen3-tts-customvoice.")


def _contains_customvoice_backend(payload: object) -> bool:
    if not isinstance(payload, Mapping):
        return False
    backends = payload.get("backends")
    if not isinstance(backends, list):
        return False
    return any(
        isinstance(value, Mapping) and value.get("name") == "qwen3-tts-customvoice"
        for value in backends
    )


def _cuda_environment(cuda_library_dirs: tuple[Path, ...]) -> dict[str, str]:
    environment = {"LC_ALL": "C", "PATH": os.defpath}
    environment["LD_LIBRARY_PATH"] = ":".join(str(path) for path in cuda_library_dirs)
    return environment


def _runtime_members(runtime: Mapping[str, object]) -> dict[str, _RuntimeMember]:
    raw_members = _required_mapping(runtime, "runtime_archive_members")
    members: dict[str, _RuntimeMember] = {}
    for path, raw_member in raw_members.items():
        if not isinstance(path, str):
            raise Qwen3TtsCustomVoiceError("Qwen3-TTS runtime member path must be text.")
        member = _required_mapping_value(raw_member, f"runtime member {path}")
        members[path] = _RuntimeMember(
            relative_path=path,
            size_bytes=_required_int(member, "size_bytes"),
            sha256=_required_string(member, "sha256"),
        )
    return members


def _safe_archive_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or path.name in {"", "."}:
        raise Qwen3TtsCustomVoiceError("Pinned CrispASR archive path is unsafe.")
    return path.as_posix()


def _resolve_verified_path(root: Path, value: str) -> Path:
    relative = _safe_archive_path(value)
    candidate = (root / relative).resolve(strict=False)
    try:
        candidate.relative_to(root.resolve(strict=True))
    except ValueError as error:
        raise Qwen3TtsCustomVoiceError("Pinned Qwen3-TTS path escapes its model bundle.") from error
    return candidate


def _required_mapping(runtime: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = runtime.get(key)
    if not isinstance(value, Mapping):
        raise Qwen3TtsCustomVoiceError(f"Qwen3-TTS runtime {key} must be an object.")
    return cast(Mapping[str, object], value)


def _required_mapping_value(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise Qwen3TtsCustomVoiceError(f"Qwen3-TTS {label} must be an object.")
    return cast(Mapping[str, object], value)


def _required_string(runtime: Mapping[str, object], key: str) -> str:
    value = runtime.get(key)
    if not isinstance(value, str) or not value:
        raise Qwen3TtsCustomVoiceError(f"Qwen3-TTS runtime {key} must be non-empty text.")
    return value


def _required_bool(runtime: Mapping[str, object], key: str, label: str) -> bool:
    value = runtime.get(key)
    if not isinstance(value, bool):
        raise Qwen3TtsCustomVoiceError(f"{label} {key} must be boolean.")
    return value


def _required_int(runtime: Mapping[str, object], key: str) -> int:
    value = runtime.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise Qwen3TtsCustomVoiceError(f"Qwen3-TTS runtime {key} must be an integer.")
    return value


def _required_float(runtime: Mapping[str, object], key: str) -> float:
    value = runtime.get(key)
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        raise Qwen3TtsCustomVoiceError(f"Qwen3-TTS runtime {key} must be numeric.")
    return float(value)
