"""Fail-closed local adapter primitives for the pinned Silero V4 Cyrillic TTS package.

The upstream package exposes a ``random`` profile which can load a tensor from a
caller-supplied path.  That path is intentionally outside this project: only the
fixed upstream Russian and Kazakh profiles are representable here.  The adapter
also refuses text that the upstream wrapper would silently discard.
"""

from __future__ import annotations

import hashlib
import pickletools
import zipfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
import torch
from torch.package import PackageImporter  # type: ignore[attr-defined]

from kds.data.manifest import ManifestRow
from kds.data.research_tts import ResearchTtsError, ResearchTtsModel

SILERO_V4_SOURCE_ID = "fleurs_ru_kk_v1_silero_v4"
SILERO_V4_SOURCE_LICENSE = "FLEURS CC-BY-4.0; Silero V4 model CC-BY-NC-SA-4.0"
SILERO_V4_PACKAGE_ROOT = "v4_cyrillic"
SILERO_V4_WRAPPER_MEMBER = f"{SILERO_V4_PACKAGE_ROOT}/multi_acc_v3_package.py"
SILERO_V4_MODEL_MEMBER = f"{SILERO_V4_PACKAGE_ROOT}/tts_models/model"
SILERO_V4_WRAPPER_SHA256 = "2a5f1d8317a534cc8b5bec5d095a846a8e5c98c34949b27756ef9252d184c347"
SILERO_V4_MODEL_PICKLE_SHA256 = "34d9d874c49c7f088e9481fcd2b1516b2ab7719702d76cb915ca7ae5a57836f2"
SILERO_V4_MAX_ARCHIVE_MEMBERS = 600
SILERO_V4_MAX_UNCOMPRESSED_BYTES = 64 * 1024 * 1024

# The package translates this explicit, finite Cyrillic set into its internal
# symbols.  Reject rather than silently removing digits, Latin abbreviations,
# fractions, currencies or other lexical content.
SILERO_V4_SUPPORTED_LETTERS = frozenset(
    "абвгдежзийклмнопрстуфхцчшщъыьэюяёђѓєіјњћќўѳғҕҗҙқҡңҥҫүұҳҷһӏӑӓӕӗәӝӟӥӧөӱӳӵӹ"
)
SILERO_V4_SUPPORTED_PUNCTUATION = frozenset("!,-.:?")
SILERO_V4_TEXT_NORMALIZER_ID = "silero_v4_cyrillic_safe_v1"
SILERO_V4_SAFE_CHARACTER_REPLACEMENTS = str.maketrans(
    {
        "—": "-",
        "–": "-",
        "‑": "-",
        ";": ",",
        "«": "",
        "»": "",
        '"': "",
        "\u200b": "",
    }
)


class SileroV4Error(ValueError):
    """Raised when the locked Silero package or its controlled input is unsafe."""


@dataclass(frozen=True, slots=True)
class SileroV4Profile:
    language: str
    voice_id: str


@dataclass(frozen=True, slots=True)
class SileroV4Runtime:
    package_path: str
    source_archive_path: str
    sample_rate: int
    profiles_by_language: Mapping[str, tuple[SileroV4Profile, ...]]


def load_silero_v4_runtime(model: ResearchTtsModel) -> SileroV4Runtime:
    """Parse the narrow runtime contract accepted for the fixed-profile model."""

    runtime = model.runtime
    if runtime.get("kind") != "silero_v4_cyrillic_torchpackage":
        raise ResearchTtsError(
            f"Silero model {model.model_id!r} needs kind='silero_v4_cyrillic_torchpackage'."
        )
    package_path = _runtime_path(runtime, "package_path", model.model_id)
    source_archive_path = _runtime_path(runtime, "source_archive_path", model.model_id)
    sample_rate = runtime.get("sample_rate")
    if sample_rate != 48_000:
        raise ResearchTtsError("Silero V4 runtime sample_rate must be exactly 48000.")
    raw_profiles = runtime.get("profiles")
    if not isinstance(raw_profiles, dict) or set(raw_profiles) != {"ru", "kk"}:
        raise ResearchTtsError("Silero V4 runtime profiles must define exactly ru and kk.")
    profiles_by_language: dict[str, tuple[SileroV4Profile, ...]] = {}
    for language in ("ru", "kk"):
        voices = raw_profiles[language]
        if not isinstance(voices, list) or not voices:
            raise ResearchTtsError(f"Silero V4 runtime {language} profiles must be non-empty.")
        parsed: list[SileroV4Profile] = []
        for voice_id in voices:
            if not isinstance(voice_id, str) or not voice_id.strip() or voice_id == "random":
                raise ResearchTtsError(
                    f"Silero V4 runtime has an invalid fixed {language} profile: {voice_id!r}."
                )
            parsed.append(SileroV4Profile(language=language, voice_id=voice_id))
        if len({profile.voice_id for profile in parsed}) != len(parsed):
            raise ResearchTtsError(f"Silero V4 runtime has duplicate {language} profiles.")
        profiles_by_language[language] = tuple(parsed)
    return SileroV4Runtime(
        package_path=package_path,
        source_archive_path=source_archive_path,
        sample_rate=sample_rate,
        profiles_by_language=profiles_by_language,
    )


def normalize_silero_v4_text(text: str) -> str:
    """Apply only documented punctuation cleanup and reject unsupported lexical input."""

    normalized = " ".join(text.translate(SILERO_V4_SAFE_CHARACTER_REPLACEMENTS).lower().split())
    if not normalized:
        raise SileroV4Error("Silero V4 input is empty after whitespace normalization.")
    unsupported = sorted(
        {
            character
            for character in normalized
            if character not in SILERO_V4_SUPPORTED_LETTERS
            and character not in SILERO_V4_SUPPORTED_PUNCTUATION
            and not character.isspace()
        }
    )
    if unsupported:
        rendered = ", ".join(repr(character) for character in unsupported)
        raise SileroV4Error(f"Silero V4 text has unsupported characters: {rendered}.")
    if not any(character in SILERO_V4_SUPPORTED_LETTERS for character in normalized):
        raise SileroV4Error("Silero V4 input contains no supported letters.")
    return normalized


def assign_silero_v4_profiles(
    rows: Iterable[ManifestRow], runtime: SileroV4Runtime
) -> list[tuple[ManifestRow, SileroV4Profile]]:
    """Assign fixed profiles deterministically within each language, without a random profile."""

    rows = list(rows)
    by_language: dict[str, list[ManifestRow]] = {"ru": [], "kk": []}
    for row in rows:
        if (
            row.split != "test"
            or row.label != "bonafide"
            or row.code_switch != "false"
            or row.language not in by_language
        ):
            raise SileroV4Error(
                "Silero V4 base rows must be non-code-switched ru/kk test bona-fide assets."
            )
        by_language[row.language].append(row)
    assignments: list[tuple[ManifestRow, SileroV4Profile]] = []
    for language in ("ru", "kk"):
        profiles = runtime.profiles_by_language[language]
        for index, row in enumerate(sorted(by_language[language], key=lambda item: item.sample_id)):
            assignments.append((row, profiles[index % len(profiles)]))
    return assignments


def inspect_silero_v4_package(package_path: Path) -> None:
    """Validate ZIP safety, CRCs and the tiny Python wrapper before package loading.

    The outer artifact is SHA-pinned by ``research_tts``.  These checks make the
    remaining executable surface explicit: the package may contain one audited
    wrapper and TorchScript data, not arbitrary paths or an unbounded ZIP bomb.
    """

    if not package_path.is_file():
        raise SileroV4Error(f"Silero V4 package does not exist: {package_path}")
    try:
        with zipfile.ZipFile(package_path) as archive:
            members = archive.infolist()
            if not members or len(members) > SILERO_V4_MAX_ARCHIVE_MEMBERS:
                raise SileroV4Error("Silero V4 package has an invalid number of ZIP members.")
            uncompressed_bytes = 0
            for member in members:
                _validate_package_member(member)
                uncompressed_bytes += member.file_size
            if uncompressed_bytes > SILERO_V4_MAX_UNCOMPRESSED_BYTES:
                raise SileroV4Error("Silero V4 package exceeds the uncompressed safety limit.")
            names = {member.filename for member in members}
            required = {
                SILERO_V4_WRAPPER_MEMBER,
                SILERO_V4_MODEL_MEMBER,
                f"{SILERO_V4_PACKAGE_ROOT}/.data/ts_code/0/data.pkl",
                f"{SILERO_V4_PACKAGE_ROOT}/.data/ts_code/0/constants.pkl",
                f"{SILERO_V4_PACKAGE_ROOT}/.data/python_version",
                f"{SILERO_V4_PACKAGE_ROOT}/.data/version",
            }
            missing = sorted(required.difference(names))
            if missing:
                raise SileroV4Error(
                    "Silero V4 package misses required members: " + ", ".join(missing)
                )
            if archive.testzip() is not None:
                raise SileroV4Error("Silero V4 package ZIP CRC validation failed.")
            wrapper = archive.read(SILERO_V4_WRAPPER_MEMBER)
            if hashlib.sha256(wrapper).hexdigest() != SILERO_V4_WRAPPER_SHA256:
                raise SileroV4Error("Silero V4 package wrapper has an unexpected SHA-256.")
            model_pickle = archive.read(SILERO_V4_MODEL_MEMBER)
            if hashlib.sha256(model_pickle).hexdigest() != SILERO_V4_MODEL_PICKLE_SHA256:
                raise SileroV4Error("Silero V4 model dispatcher has an unexpected SHA-256.")
            _validate_model_dispatcher_pickle(model_pickle)
    except (OSError, zipfile.BadZipFile) as error:
        raise SileroV4Error(f"Cannot safely inspect Silero V4 package: {error}") from error


def load_silero_v4_model(
    package_path: Path, runtime: SileroV4Runtime, device: torch.device
) -> Any:
    """Load the audited TorchScript package after all byte and ZIP checks have passed."""

    inspect_silero_v4_package(package_path)
    if device.type not in {"cpu", "cuda"}:
        raise SileroV4Error("Silero V4 device must be CPU or CUDA.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SileroV4Error("Silero V4 CUDA was requested but is unavailable.")
    try:
        importer = PackageImporter(str(package_path))
        model = importer.load_pickle("tts_models", "model")
        model.to(device)
    except (ImportError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
        raise SileroV4Error(
            f"Cannot load audited Silero V4 TorchScript package: {error}"
        ) from error
    speakers = getattr(model, "speakers", None)
    if not isinstance(speakers, list):
        raise SileroV4Error("Loaded Silero V4 package has no fixed speaker list.")
    required_profiles = {
        profile.voice_id
        for profiles in runtime.profiles_by_language.values()
        for profile in profiles
    }
    missing_profiles = sorted(required_profiles.difference(speakers))
    if missing_profiles:
        raise SileroV4Error(
            "Loaded Silero V4 package lacks fixed profiles: " + ", ".join(missing_profiles)
        )
    return model


def synthesize_silero_v4(
    *,
    model: Any,
    profile: SileroV4Profile,
    text: str,
    runtime: SileroV4Runtime,
    output: Path,
) -> None:
    """Generate one fixed-profile WAV without a reference-audio or external voice path."""

    normalized = normalize_silero_v4_text(text)
    try:
        with torch.inference_mode():
            waveform = model.apply_tts(
                text=normalized,
                ssml_text=None,
                speaker=profile.voice_id,
                sample_rate=runtime.sample_rate,
                put_accent=profile.language == "ru",
                put_yo=profile.language == "ru",
                voice_path=None,
                symbol_durs=None,
                return_ts=False,
            )
    except (AssertionError, RuntimeError, ValueError) as error:
        raise SileroV4Error(
            f"Silero V4 synthesis failed for fixed profile {profile.voice_id!r}: {error}"
        ) from error
    if not isinstance(waveform, torch.Tensor):
        raise SileroV4Error("Silero V4 did not return a torch Tensor waveform.")
    audio = waveform.detach().to("cpu", torch.float32).numpy()
    if audio.ndim != 1 or audio.size == 0 or not np.isfinite(audio).all():
        raise SileroV4Error("Silero V4 produced an empty or non-finite waveform.")
    sf.write(output, audio, runtime.sample_rate, subtype="PCM_16")


def silero_v4_spoof_row(
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
    """Create one FLEURS-derived spoof row while retaining its original transcript provenance."""

    if base_row.language != profile.language or base_row.code_switch != "false":
        raise SileroV4Error("Silero V4 profile does not match its non-code-switched base row.")
    sample_key = hashlib.sha256(
        f"{base_row.sample_id}:{model.model_id}:{profile.voice_id}".encode()
    ).hexdigest()[:16]
    profile_id = f"{model.model_id}:{profile.voice_id}"
    return ManifestRow(
        sample_id=f"{SILERO_V4_SOURCE_ID}:{sample_key}",
        relative_path=relative_path,
        sha256=sha256,
        split="test",
        label="spoof",
        language=base_row.language,
        code_switch="false",
        parent_group_id=f"{SILERO_V4_SOURCE_ID}:profile:{profile_id}",
        source_name=SILERO_V4_SOURCE_ID,
        source_license=SILERO_V4_SOURCE_LICENSE,
        rights_basis=(
            "Offline derivative for personal research from Google FLEURS transcript "
            f"{base_row.text_id}; {model.license}; fixed pretrained profile; "
            "no reference audio or voice cloning"
        ),
        speaker_pseudo_id=f"{SILERO_V4_SOURCE_ID}:synthetic-profile:{profile_id}",
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
        augmentation_chain=f"text_normalization={SILERO_V4_TEXT_NORMALIZER_ID}",
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
        or not member.filename.startswith(f"{SILERO_V4_PACKAGE_ROOT}/")
    ):
        raise SileroV4Error(f"Unsafe Silero V4 package member: {member.filename!r}")
    mode = member.external_attr >> 16
    if mode and (mode & 0o170000) not in {0, 0o100000}:
        raise SileroV4Error(f"Silero V4 package has non-regular member: {member.filename!r}")


def _validate_model_dispatcher_pickle(payload: bytes) -> None:
    """Allow only the three globals in the non-executed package dispatcher pickle."""

    expected_globals = {
        "multi_acc_v3_package TTSModelMultiAcc_v3",
        "torch.jit._script unpackage_script_module",
        "torch device",
    }
    globals_found: set[str] = set()
    try:
        for opcode, argument, _position in pickletools.genops(payload):
            if opcode.name == "GLOBAL":
                if not isinstance(argument, str):
                    raise SileroV4Error("Silero V4 dispatcher contains a non-string GLOBAL.")
                globals_found.add(argument)
    except (ValueError, IndexError) as error:
        raise SileroV4Error(f"Silero V4 dispatcher is not a valid pickle: {error}") from error
    if globals_found != expected_globals:
        raise SileroV4Error(
            "Silero V4 dispatcher GLOBAL allowlist mismatch: "
            f"expected {sorted(expected_globals)!r}, got {sorted(globals_found)!r}."
        )
