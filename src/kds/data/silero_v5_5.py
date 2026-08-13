"""Fail-closed text-only adapter for the pinned Silero V5.5 RU ``eugene`` package.

The upstream package exposes SSML, random voices and a ``voice_path`` parameter.
None of those interfaces are representable here.  This adapter accepts literal Russian
text only and always invokes the exact pinned package with its built-in ``eugene``
profile at 48 kHz.
"""

from __future__ import annotations

import hashlib
import io
import pickletools
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
import torch
from torch.package import PackageImporter  # type: ignore[attr-defined]

from kds.data.manifest import ManifestRow
from kds.data.research_tts import ResearchTtsError, ResearchTtsModel

SILERO_V5_5_SOURCE_ID = "silero_v5_5_ru_eugene_v1"
SILERO_V5_5_SOURCE_LICENSE = "Silero V5.5 RU model CC-BY-NC-SA-4.0"
SILERO_V5_5_RUNTIME_KIND = "silero_v5_5_ru_fixed_eugene_torchpackage"
SILERO_V5_5_PACKAGE_ROOT = "questions_public_eugene_int_intensity"
SILERO_V5_5_WRAPPER_MEMBER = f"{SILERO_V5_5_PACKAGE_ROOT}/multi_acc_v3_package.py"
SILERO_V5_5_MODEL_MEMBER = f"{SILERO_V5_5_PACKAGE_ROOT}/tts_models/model"
SILERO_V5_5_WRAPPER_SHA256 = "393e53dd0e732b7a800605d0d477b6c8dd0b5b64917ad9cd27db06da46f6aa51"
SILERO_V5_5_MODEL_PICKLE_SHA256 = "208b2f257714ce47671bc4853764ba3345b7ebbe255a25b54926699df8c9c13c"
SILERO_V5_5_MAX_ARCHIVE_MEMBERS = 600
SILERO_V5_5_MAX_UNCOMPRESSED_BYTES = 160 * 1024 * 1024
SILERO_V5_5_FIXED_SPEAKER = "eugene"
SILERO_V5_5_SAMPLE_RATE = 48_000
SILERO_V5_5_TEXT_NORMALIZER_ID = "silero_v5_5_ru_literal_whitespace_only_v1"
SILERO_V5_5_ALLOWED_LETTERS = frozenset("абвгдежзийклмнопрстуфхцчшщъыьэюяё")
SILERO_V5_5_ALLOWED_PUNCTUATION = frozenset("!,-.:;?–—…")
SILERO_V5_5_FORBIDDEN_CONTROL_CHARACTERS = frozenset("+_|~^*$<>")


class SileroV55Error(ValueError):
    """Raised when the locked package or its restricted text-only path is unsafe."""


@dataclass(frozen=True, slots=True)
class SileroV55Runtime:
    """The entire callable V5.5 contract: one package, one profile, one sample rate."""

    package_path: str
    source_archive_path: str
    sample_rate: int
    fixed_speaker: str


def load_silero_v5_5_runtime(model: ResearchTtsModel) -> SileroV55Runtime:
    """Parse only the fixed ``v5_5_ru`` / ``eugene`` runtime contract."""

    runtime = model.runtime
    if runtime.get("kind") != SILERO_V5_5_RUNTIME_KIND:
        raise ResearchTtsError(
            f"Silero model {model.model_id!r} needs kind={SILERO_V5_5_RUNTIME_KIND!r}."
        )
    package_path = _runtime_path(runtime, "package_path", model.model_id)
    source_archive_path = _runtime_path(runtime, "source_archive_path", model.model_id)
    if runtime.get("sample_rate") != SILERO_V5_5_SAMPLE_RATE:
        raise ResearchTtsError("Silero V5.5 runtime sample_rate must be exactly 48000.")
    if runtime.get("fixed_speaker") != SILERO_V5_5_FIXED_SPEAKER:
        raise ResearchTtsError("Silero V5.5 runtime must fix speaker='eugene'.")
    required_policy = {
        "reference_audio_policy": "forbidden",
        "voice_cloning": False,
        "text_input_only": True,
        "ssml": "forbidden",
        "voice_path": "forbidden",
        "symbol_durs": "forbidden",
        "return_timestamps": "forbidden",
        "external_text_normalizer": "forbidden",
    }
    for key, expected in required_policy.items():
        if runtime.get(key) != expected:
            raise ResearchTtsError(
                f"Silero V5.5 runtime {key!r} must be pinned to {expected!r}."
            )
    return SileroV55Runtime(
        package_path=package_path,
        source_archive_path=source_archive_path,
        sample_rate=SILERO_V5_5_SAMPLE_RATE,
        fixed_speaker=SILERO_V5_5_FIXED_SPEAKER,
    )


def normalize_silero_v5_5_text(text: str) -> str:
    """Keep source content literal apart from whitespace collapse; reject control markup.

    V5.5 internally lowercases and normalizes em dashes.  The adapter makes neither a
    lexical replacement nor a stress/SSML decision, and blocks upstream control symbols
    before the package can interpret them.
    """

    if not isinstance(text, str):
        raise SileroV55Error("Silero V5.5 input must be a string.")
    normalized = " ".join(text.split())
    if not normalized:
        raise SileroV55Error("Silero V5.5 input is empty after whitespace normalization.")
    unsupported = sorted(
        {
            character
            for character in normalized
            if character.casefold() not in SILERO_V5_5_ALLOWED_LETTERS
            and character not in SILERO_V5_5_ALLOWED_PUNCTUATION
            and not character.isspace()
        }
    )
    controls = sorted(
        set(normalized).intersection(SILERO_V5_5_FORBIDDEN_CONTROL_CHARACTERS)
    )
    if unsupported or controls:
        rendered = ", ".join(repr(character) for character in sorted(set(unsupported + controls)))
        raise SileroV55Error(f"Silero V5.5 text has unsupported characters: {rendered}.")
    if not any(character.casefold() in SILERO_V5_5_ALLOWED_LETTERS for character in normalized):
        raise SileroV55Error("Silero V5.5 input contains no Russian letters.")
    return normalized


def inspect_silero_v5_5_package(package_path: Path) -> None:
    """Validate ZIP safety and the hash-pinned dispatcher before package loading."""

    if not package_path.is_file():
        raise SileroV55Error(f"Silero V5.5 package does not exist: {package_path}")
    try:
        with zipfile.ZipFile(package_path) as archive:
            members = archive.infolist()
            if not members or len(members) > SILERO_V5_5_MAX_ARCHIVE_MEMBERS:
                raise SileroV55Error("Silero V5.5 package has an invalid number of ZIP members.")
            uncompressed_bytes = 0
            for member in members:
                _validate_package_member(member)
                uncompressed_bytes += member.file_size
            if uncompressed_bytes > SILERO_V5_5_MAX_UNCOMPRESSED_BYTES:
                raise SileroV55Error("Silero V5.5 package exceeds the uncompressed safety limit.")
            names = {member.filename for member in members}
            required = {
                SILERO_V5_5_WRAPPER_MEMBER,
                SILERO_V5_5_MODEL_MEMBER,
                f"{SILERO_V5_5_PACKAGE_ROOT}/.data/ts_code/0/data.pkl",
                f"{SILERO_V5_5_PACKAGE_ROOT}/.data/ts_code/0/constants.pkl",
                f"{SILERO_V5_5_PACKAGE_ROOT}/.data/python_version",
                f"{SILERO_V5_5_PACKAGE_ROOT}/.data/version",
            }
            missing = sorted(required.difference(names))
            if missing:
                raise SileroV55Error(
                    "Silero V5.5 package misses required members: " + ", ".join(missing)
                )
            if archive.testzip() is not None:
                raise SileroV55Error("Silero V5.5 package ZIP CRC validation failed.")
            wrapper = archive.read(SILERO_V5_5_WRAPPER_MEMBER)
            if hashlib.sha256(wrapper).hexdigest() != SILERO_V5_5_WRAPPER_SHA256:
                raise SileroV55Error("Silero V5.5 wrapper has an unexpected SHA-256.")
            model_pickle = archive.read(SILERO_V5_5_MODEL_MEMBER)
            if hashlib.sha256(model_pickle).hexdigest() != SILERO_V5_5_MODEL_PICKLE_SHA256:
                raise SileroV55Error("Silero V5.5 model dispatcher has an unexpected SHA-256.")
            _validate_model_dispatcher_pickle(model_pickle)
    except (OSError, zipfile.BadZipFile) as error:
        raise SileroV55Error(f"Cannot safely inspect Silero V5.5 package: {error}") from error


def load_silero_v5_5_model(
    package_path: Path, runtime: SileroV55Runtime, device: torch.device
) -> Any:
    """Load a byte-pinned package only after the static fixed-profile checks pass."""

    inspect_silero_v5_5_package(package_path)
    if device.type not in {"cpu", "cuda"}:
        raise SileroV55Error("Silero V5.5 device must be CPU or CUDA.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SileroV55Error("Silero V5.5 CUDA was requested but is unavailable.")
    try:
        importer = PackageImporter(str(package_path))
        model = importer.load_pickle("tts_models", "model")
        model.to(device)
    except (ImportError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
        raise SileroV55Error(
            f"Cannot load audited Silero V5.5 package: {error}"
        ) from error
    speakers = getattr(model, "speakers", None)
    if not isinstance(speakers, list) or runtime.fixed_speaker not in speakers:
        raise SileroV55Error("Loaded Silero V5.5 package lacks fixed profile 'eugene'.")
    return model


def synthesize_silero_v5_5(
    *, model: Any, text: str, runtime: SileroV55Runtime, output: Path
) -> None:
    """Generate one WAV through the only permitted V5.5 call shape.

    There is deliberately no argument for a reference clip, a random profile, SSML,
    prosody symbols, a voice path, timestamps, language override or intensity override.
    """

    literal_text = normalize_silero_v5_5_text(text)
    try:
        with torch.inference_mode():
            waveform = model.apply_tts(
                text=literal_text,
                ssml_text=None,
                speaker=runtime.fixed_speaker,
                sample_rate=runtime.sample_rate,
                put_accent=True,
                put_stress_homo=True,
                put_yo=True,
                put_yo_homo=True,
                stress_single_vowel=True,
                voice_path=None,
                symbol_durs=None,
                return_ts=False,
                lang=None,
                type_str=None,
                intensity=3,
            )
    except (AssertionError, RuntimeError, ValueError) as error:
        raise SileroV55Error(f"Silero V5.5 fixed-eugene synthesis failed: {error}") from error
    if not isinstance(waveform, torch.Tensor):
        raise SileroV55Error("Silero V5.5 did not return a torch Tensor waveform.")
    audio = waveform.detach().to("cpu", torch.float32).numpy()
    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise SileroV55Error("Silero V5.5 produced an empty or non-finite waveform.")
    sf.write(output, audio, runtime.sample_rate, subtype="PCM_16")


def silero_v5_5_spoof_row(
    *,
    base_row: ManifestRow,
    model: ResearchTtsModel,
    relative_path: str,
    sha256: str,
    duration_s: float,
    original_sr: int,
    created_at: str,
    device: str,
) -> ManifestRow:
    """Create a paired RU spoof row without altering its bona-fide text provenance."""

    if (
        base_row.split != "test"
        or base_row.label != "bonafide"
        or base_row.language != "ru"
        or base_row.source_name != "common_voice_ru_v24"
    ):
        raise SileroV55Error("Silero V5.5 base row must be a Common Voice RU test bona-fide row.")
    sample_key = hashlib.sha256(
        f"{base_row.sample_id}:{model.model_id}:{SILERO_V5_5_FIXED_SPEAKER}".encode()
    ).hexdigest()[:16]
    profile_id = f"{model.model_id}:{SILERO_V5_5_FIXED_SPEAKER}"
    return ManifestRow(
        sample_id=f"{SILERO_V5_5_SOURCE_ID}:{sample_key}",
        relative_path=relative_path,
        sha256=sha256,
        split="test",
        label="spoof",
        language="ru",
        code_switch=base_row.code_switch,
        parent_group_id=f"{SILERO_V5_5_SOURCE_ID}:profile:{profile_id}",
        source_name=SILERO_V5_5_SOURCE_ID,
        source_license=SILERO_V5_5_SOURCE_LICENSE,
        rights_basis=(
            "Offline derivative for personal research from source transcript "
            f"{base_row.text_id}; {model.license}; fixed eugene profile; "
            "no reference audio or voice cloning"
        ),
        speaker_pseudo_id=f"{SILERO_V5_5_SOURCE_ID}:synthetic-profile:{profile_id}",
        text_id=base_row.text_id,
        text_hash=base_row.text_hash,
        duration_s=duration_s,
        generator_family=model.generator_family,
        generator_name=model.generator_name,
        generator_version=model.generator_version,
        voice_id=profile_id,
        clone_consent_id="not_applicable:fixed-pretrained-tts-no-reference-audio",
        device=device,
        capture_route="offline_text_only_silero_v5_5_ru_fixed_eugene",
        original_sr=original_sr,
        codec="wav",
        augmentation_chain=f"text_normalization={SILERO_V5_5_TEXT_NORMALIZER_ID}",
        augmentation_seed="",
        created_at=created_at,
    )


def _runtime_path(runtime: Mapping[str, object], key: str, model_id: str) -> str:
    value = runtime.get(key)
    if not isinstance(value, str) or not value:
        raise ResearchTtsError(f"Silero model {model_id!r} needs non-empty runtime {key!r}.")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or value in {".", ".."}:
        raise ResearchTtsError(f"Silero model {model_id!r} has unsafe runtime {key!r}.")
    return value


def _validate_package_member(member: zipfile.ZipInfo) -> None:
    path = PurePosixPath(member.filename)
    if (
        not member.filename
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in member.filename
        or member.is_dir()
        or not member.filename.startswith(f"{SILERO_V5_5_PACKAGE_ROOT}/")
    ):
        raise SileroV55Error(f"Unsafe Silero V5.5 package member: {member.filename!r}")
    mode = member.external_attr >> 16
    if mode and (mode & 0o170000) not in {0, 0o100000}:
        raise SileroV55Error(f"Silero V5.5 package has non-regular member: {member.filename!r}")


def _validate_model_dispatcher_pickle(payload: bytes) -> None:
    """Allow exactly the globals in the pinned package dispatcher pickle."""

    expected_globals = {
        "builtins set",
        "collections OrderedDict",
        "custom_tokenizers.bert_tokenizer BasicTokenizer",
        "custom_tokenizers.bert_tokenizer SimpleBertTokenizer",
        "custom_tokenizers.bert_tokenizer WordpieceTokenizer",
        "models.accentor AccentorNgram",
        "models.homosolver HomoSolver",
        "models.model SileroStress",
        "multi_acc_v3_package PartTTSModelMultiAcc_v3",
        "multi_acc_v3_package TTSModelMultiAcc_v3",
        "re _compile",
        "torch device",
        "torch.jit._script unpackage_script_module",
    }
    globals_found: set[str] = set()
    try:
        for opcode, argument, _position in pickletools.genops(io.BytesIO(payload)):
            if opcode.name == "GLOBAL":
                if not isinstance(argument, str):
                    raise SileroV55Error("Silero V5.5 dispatcher contains a non-string GLOBAL.")
                globals_found.add(argument)
    except (ValueError, IndexError) as error:
        raise SileroV55Error(f"Silero V5.5 dispatcher is not a valid pickle: {error}") from error
    if globals_found != expected_globals:
        raise SileroV55Error(
            "Silero V5.5 dispatcher GLOBAL allowlist mismatch: "
            f"expected {sorted(expected_globals)!r}, got {sorted(globals_found)!r}."
        )
