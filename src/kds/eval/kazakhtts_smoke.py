"""Frozen pre-detector acoustic smoke plan for the Stage-C KazakhTTS route."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from kds.data.assets import sha256_file

KAZAKHTTS_SMOKE_SCHEMA_VERSION = 1


class KazakhTtsSmokeError(ValueError):
    """Raised when the smoke plan or its immutable bindings are invalid."""


@dataclass(frozen=True, slots=True)
class SmokeBinding:
    path: Path
    sha256: str


@dataclass(frozen=True, slots=True)
class KazakhTtsSmokeCase:
    case_id: str
    language: str
    status: str
    text: str


@dataclass(frozen=True, slots=True)
class KazakhTtsSmokePlan:
    protocol_id: str
    model_lock: SmokeBinding
    generator_route_gate: SmokeBinding
    seed: int
    cases: tuple[KazakhTtsSmokeCase, ...]


def load_kazakhtts_smoke_plan(path: Path) -> KazakhTtsSmokePlan:
    """Load a strict smoke plan and verify both bound Stage-C inputs immediately."""

    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise KazakhTtsSmokeError(f"Cannot read KazakhTTS smoke plan {path}: {error}") from error
    raw = _mapping(value, "KazakhTTS smoke plan")
    _exact_keys(
        raw,
        {
            "schema_version",
            "protocol_id",
            "model_lock",
            "generator_route_gate",
            "seed",
            "cases",
            "decision_rule",
        },
        "KazakhTTS smoke plan",
    )
    if raw["schema_version"] != KAZAKHTTS_SMOKE_SCHEMA_VERSION:
        raise KazakhTtsSmokeError(
            f"KazakhTTS smoke schema_version must be {KAZAKHTTS_SMOKE_SCHEMA_VERSION}."
        )
    decision = _mapping(raw["decision_rule"], "KazakhTTS smoke decision rule")
    _exact_keys(
        decision,
        {"detector_inference", "reference_audio", "voice_cloning", "kk", "ru_mixed"},
        "KazakhTTS smoke decision rule",
    )
    if any(decision.get(field) != "forbidden" for field in (
        "detector_inference",
        "reference_audio",
        "voice_cloning",
    )):
        raise KazakhTtsSmokeError("KazakhTTS smoke must forbid detector, reference and cloning.")
    seed = raw["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise KazakhTtsSmokeError("KazakhTTS smoke seed must be a non-negative integer.")
    cases_value = raw["cases"]
    if not isinstance(cases_value, list):
        raise KazakhTtsSmokeError("KazakhTTS smoke cases must be an array.")
    cases = tuple(_case(item, index) for index, item in enumerate(cases_value, start=1))
    if {case.language for case in cases} != {"ru", "kk", "mixed"} or len(cases) != 3:
        raise KazakhTtsSmokeError("KazakhTTS smoke must define exactly one RU, KK and mixed case.")
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise KazakhTtsSmokeError("KazakhTTS smoke case IDs must be unique.")
    plan = KazakhTtsSmokePlan(
        protocol_id=_string(raw, "protocol_id", "KazakhTTS smoke plan"),
        model_lock=_binding(raw["model_lock"], path.parent, "model lock"),
        generator_route_gate=_binding(
            raw["generator_route_gate"], path.parent, "generator route gate"
        ),
        seed=seed,
        cases=cases,
    )
    for label, binding in (
        ("model lock", plan.model_lock),
        ("generator route gate", plan.generator_route_gate),
    ):
        if not binding.path.is_file():
            raise KazakhTtsSmokeError(f"KazakhTTS smoke {label} is missing: {binding.path}")
        actual = sha256_file(binding.path)
        if actual != binding.sha256:
            raise KazakhTtsSmokeError(
                f"KazakhTTS smoke {label} SHA-256 mismatch: "
                f"expected {binding.sha256}, got {actual}."
            )
    return plan


def _case(value: object, index: int) -> KazakhTtsSmokeCase:
    label = f"KazakhTTS smoke case {index}"
    raw = _mapping(value, label)
    _exact_keys(raw, {"id", "language", "status", "text"}, label)
    language = _string(raw, "language", label)
    status = _string(raw, "status", label)
    expected_status = (
        "officially_supported" if language == "kk" else "conditional_acoustic_smoke_only"
    )
    if language not in {"ru", "kk", "mixed"} or status != expected_status:
        raise KazakhTtsSmokeError(f"{label} has an invalid language/status contract.")
    return KazakhTtsSmokeCase(
        case_id=_string(raw, "id", label),
        language=language,
        status=status,
        text=_string(raw, "text", label),
    )


def _binding(value: object, base: Path, label: str) -> SmokeBinding:
    raw = _mapping(value, f"KazakhTTS smoke {label}")
    _exact_keys(raw, {"path", "sha256"}, f"KazakhTTS smoke {label}")
    relative = Path(_string(raw, "path", f"KazakhTTS smoke {label}"))
    if relative.is_absolute():
        raise KazakhTtsSmokeError(f"KazakhTTS smoke {label} path must be relative.")
    candidate = (base / relative).resolve(strict=False)
    project_root = base.parents[1].resolve(strict=True)
    try:
        candidate.relative_to(project_root)
    except ValueError as error:
        raise KazakhTtsSmokeError(
            f"KazakhTTS smoke {label} path escapes the project."
        ) from error
    return SmokeBinding(
        path=candidate,
        sha256=_sha256(raw, "sha256", f"KazakhTTS smoke {label}"),
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise KazakhTtsSmokeError(f"{label} must be an object.")
    return cast(Mapping[str, object], value)


def _exact_keys(raw: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(raw) != expected:
        raise KazakhTtsSmokeError(f"{label} has missing or unknown fields.")


def _string(raw: Mapping[str, object], name: str, label: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value.strip():
        raise KazakhTtsSmokeError(f"{label} field {name!r} must be a non-empty string.")
    return value.strip()


def _sha256(raw: Mapping[str, object], name: str, label: str) -> str:
    value = _string(raw, name, label).lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise KazakhTtsSmokeError(f"{label} field {name!r} must be SHA-256.")
    return value
