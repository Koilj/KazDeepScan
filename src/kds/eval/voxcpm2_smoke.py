"""Helpers for the one permitted non-candidate VoxCPM2 CUDA smoke process."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import socket
import tarfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kds.data.voxcpm2_text_only import BoundText, bind_text, collapse_whitespace


class VoxCPM2SmokeError(ValueError):
    """Raised when the one-shot smoke route leaves its frozen contract."""


@dataclass(frozen=True, slots=True)
class InstalledDistributionAudit:
    distributions: tuple[tuple[str, str], ...]
    fingerprint: str


@dataclass(frozen=True, slots=True)
class WaveformAudit:
    frames: int
    sample_rate_hz: int
    duration_seconds: str
    peak_abs: str
    rms: str
    finite: bool


def installed_distribution_audit() -> InstalledDistributionAudit:
    """Bind normalized installed distribution names and exact versions."""

    rows = tuple(
        sorted(
            {
                (
                    str(distribution.metadata["Name"]).casefold().replace("_", "-"),
                    distribution.version,
                )
                for distribution in importlib.metadata.distributions()
                if distribution.metadata["Name"]
            }
        )
    )
    material = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode()
    return InstalledDistributionAudit(rows, hashlib.sha256(material).hexdigest())


def install_python_network_guard() -> list[str]:
    """Deny Python socket egress before any upstream TTS module is imported."""

    attempts: list[str] = []
    class DeniedSocket(socket.socket):
        def connect(self, address: object) -> None:
            attempts.append(repr(address))
            raise VoxCPM2SmokeError("Network access is forbidden in the VoxCPM2 smoke process.")

        def connect_ex(self, address: object) -> int:
            attempts.append(repr(address))
            raise VoxCPM2SmokeError("Network access is forbidden in the VoxCPM2 smoke process.")

    def denied_create_connection(*args: object, **kwargs: object) -> None:
        attempts.append(repr((args, kwargs)))
        raise VoxCPM2SmokeError("Network access is forbidden in the VoxCPM2 smoke process.")

    def denied_getaddrinfo(*args: object, **kwargs: object) -> None:
        attempts.append(repr((args, kwargs)))
        raise VoxCPM2SmokeError("DNS access is forbidden in the VoxCPM2 smoke process.")

    socket.socket = DeniedSocket  # type: ignore[misc]
    socket.create_connection = denied_create_connection  # type: ignore[assignment]
    socket.getaddrinfo = denied_getaddrinfo  # type: ignore[assignment]
    return attempts


def screen_smoke_text_against_denis(archive_path: Path, smoke_text: str) -> dict[str, object]:
    """Prove that the fixed smoke sentence is not a Denis candidate transcript."""

    smoke_binding = bind_text(smoke_text)
    literal_matches = canonical_matches = 0
    transcript_members = 0
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith(".txt"):
                continue
            handle = archive.extractfile(member)
            if handle is None:
                raise VoxCPM2SmokeError(f"Cannot read Denis transcript member: {member.name}")
            try:
                literal = handle.read().decode("utf-8")
            except UnicodeDecodeError as error:
                raise VoxCPM2SmokeError(
                    f"Denis transcript is not UTF-8: {member.name}"
                ) from error
            transcript_members += 1
            binding = bind_text(literal)
            literal_matches += binding.literal_sha256 == smoke_binding.literal_sha256
            canonical_matches += (
                binding.collapse_whitespace_sha256
                == smoke_binding.collapse_whitespace_sha256
            )
    if transcript_members != 1_150:
        raise VoxCPM2SmokeError(
            f"Expected 1150 Denis transcripts, found {transcript_members}."
        )
    if literal_matches or canonical_matches:
        raise VoxCPM2SmokeError("Fixed smoke text collides with a Denis transcript.")
    return {
        "transcript_members": transcript_members,
        "literal_matches": literal_matches,
        "collapse_whitespace_matches": canonical_matches,
        "smoke_text_binding": {
            "literal_sha256": smoke_binding.literal_sha256,
            "collapse_whitespace_sha256": smoke_binding.collapse_whitespace_sha256,
        },
    }


def audit_waveform(waveform: Sequence[float] | np.ndarray, sample_rate_hz: int) -> WaveformAudit:
    """Apply the narrow technical checks required of the smoke waveform."""

    array = np.asarray(waveform, dtype=np.float32)
    if array.ndim != 1 or array.size == 0:
        raise VoxCPM2SmokeError("Smoke waveform must be a non-empty mono vector.")
    if sample_rate_hz != 48_000:
        raise VoxCPM2SmokeError(f"Smoke output must be 48000 Hz, got {sample_rate_hz}.")
    finite = bool(np.isfinite(array).all())
    if not finite:
        raise VoxCPM2SmokeError("Smoke waveform contains NaN or infinity.")
    peak = float(np.max(np.abs(array)))
    rms = math.sqrt(float(np.mean(np.square(array, dtype=np.float64))))
    duration = array.size / sample_rate_hz
    if peak <= 0.0 or rms <= 0.0 or not 0.1 <= duration <= 60.0:
        raise VoxCPM2SmokeError(
            f"Smoke waveform has implausible peak/rms/duration: {peak}/{rms}/{duration}."
        )
    return WaveformAudit(
        frames=int(array.size),
        sample_rate_hz=sample_rate_hz,
        duration_seconds=f"{duration:.6f}",
        peak_abs=f"{peak:.9f}",
        rms=f"{rms:.9f}",
        finite=finite,
    )


class OneCallModel:
    """Capture the exact kwargs and prohibit a second generation call."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.calls = 0
        self.kwargs: dict[str, object] | None = None

    def generate(self, **kwargs: object) -> Any:
        self.calls += 1
        if self.calls != 1:
            raise VoxCPM2SmokeError("The smoke process attempted more than one generation call.")
        self.kwargs = dict(kwargs)
        return self.model.generate(**kwargs)


def require_binding(text: str, expected: BoundText) -> str:
    """Return canonical text only when both predeclared hashes still match."""

    if bind_text(text) != expected:
        raise VoxCPM2SmokeError("Smoke text binding changed.")
    return collapse_whitespace(text)
