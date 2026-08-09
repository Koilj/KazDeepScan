"""Verified local model bundles for personal-research TTS generation.

The model lock is deliberately separate from the Python dependency lock.  It pins model
revisions and every downloaded byte, so a command cannot silently fetch a newer voice model.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from kds.data.assets import sha256_file

MODEL_LOCK_SCHEMA_VERSION = 1
DEFAULT_DOWNLOAD_LIMIT_BYTES = 2 * 1024 * 1024 * 1024


class ResearchTtsError(ValueError):
    """Raised when a research TTS model bundle is unpinned or fails verification."""


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    relative_path: str
    url: str
    expected_size_bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ResearchTtsModel:
    model_id: str
    destination: str
    generator_family: str
    generator_name: str
    generator_version: str
    license: str
    source_url: str
    runtime: Mapping[str, object]
    artifacts: tuple[ModelArtifact, ...]


@dataclass(frozen=True, slots=True)
class ResearchTtsModelLock:
    protocol_id: str
    models: tuple[ResearchTtsModel, ...]


def load_research_tts_model_lock(path: Path) -> ResearchTtsModelLock:
    """Load a strict, versioned model lock without accepting arbitrary download URLs."""

    if not path.is_file():
        raise ResearchTtsError(f"Research TTS model lock does not exist: {path}")
    try:
        raw_value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResearchTtsError(f"Cannot read model lock {path}: {error}") from error
    if not isinstance(raw_value, dict):
        raise ResearchTtsError("Research TTS model lock root must be a JSON object.")
    raw = cast(dict[str, object], raw_value)
    _expect_keys(raw, {"schema_version", "protocol_id", "models"}, "Model lock")
    if raw["schema_version"] != MODEL_LOCK_SCHEMA_VERSION:
        raise ResearchTtsError(
            "Model lock schema_version must be "
            f"{MODEL_LOCK_SCHEMA_VERSION!r}, got {raw['schema_version']!r}."
        )
    protocol_id = _string(raw, "protocol_id", "Model lock")
    raw_models = raw["models"]
    if not isinstance(raw_models, list) or not raw_models:
        raise ResearchTtsError("Model lock models must be a non-empty JSON array.")
    models = tuple(_parse_model(value, index) for index, value in enumerate(raw_models, start=1))
    model_ids = [model.model_id for model in models]
    destinations = [model.destination for model in models]
    if len(model_ids) != len(set(model_ids)):
        raise ResearchTtsError("Model lock contains duplicate model_id values.")
    if len(destinations) != len(set(destinations)):
        raise ResearchTtsError("Model lock contains duplicate destination values.")
    return ResearchTtsModelLock(protocol_id=protocol_id, models=models)


def verify_research_tts_model_bundle(
    model_root: Path, model: ResearchTtsModel
) -> dict[str, Path]:
    """Verify every local artifact before a generator is permitted to read it."""

    root = _resolve_below(model_root, model.destination)
    if not root.is_dir():
        raise ResearchTtsError(f"Missing model bundle directory for {model.model_id!r}: {root}")
    verified: dict[str, Path] = {}
    for artifact in model.artifacts:
        path = _resolve_below(root, artifact.relative_path)
        if not path.is_file():
            raise ResearchTtsError(
                f"Missing model artifact for {model.model_id!r}: {artifact.relative_path}"
            )
        actual_size = path.stat().st_size
        if actual_size != artifact.expected_size_bytes:
            raise ResearchTtsError(
                f"Model artifact size mismatch for {model.model_id!r} {artifact.relative_path}: "
                f"expected {artifact.expected_size_bytes}, got {actual_size}."
            )
        actual_hash = sha256_file(path)
        if actual_hash != artifact.sha256:
            raise ResearchTtsError(
                f"Model artifact SHA-256 mismatch for {model.model_id!r} {artifact.relative_path}: "
                f"expected {artifact.sha256}, got {actual_hash}."
            )
        verified[artifact.relative_path] = path
    return verified


def verify_research_tts_model_lock(
    model_root: Path, lock: ResearchTtsModelLock
) -> dict[str, dict[str, Path]]:
    """Verify the complete locked model set and return trusted local paths by model id."""

    return {
        model.model_id: verify_research_tts_model_bundle(model_root, model)
        for model in lock.models
    }


def download_research_tts_model_lock(
    model_root: Path,
    lock: ResearchTtsModelLock,
    *,
    max_download_bytes: int = DEFAULT_DOWNLOAD_LIMIT_BYTES,
) -> None:
    """Download a complete model lock atomically, enforcing size and SHA-256 for every file."""

    if max_download_bytes <= 0:
        raise ResearchTtsError("max_download_bytes must be positive.")
    total_expected = sum(
        artifact.expected_size_bytes for model in lock.models for artifact in model.artifacts
    )
    if total_expected > max_download_bytes:
        raise ResearchTtsError(
            f"Locked model download is {total_expected} bytes, above the allowed "
            f"{max_download_bytes} bytes."
        )
    model_root.mkdir(parents=True, exist_ok=True)
    resolved_root = model_root.resolve(strict=True)
    destinations = [
        (_resolve_below(resolved_root, model.destination), model) for model in lock.models
    ]
    existing = [str(destination) for destination, _model in destinations if destination.exists()]
    if existing:
        raise ResearchTtsError(
            "Refusing to replace existing model bundle directories: " + ", ".join(existing)
        )

    stage_path = Path(tempfile.mkdtemp(prefix="kds-research-tts-", dir=resolved_root))
    try:
        for _destination, model in destinations:
            stage_bundle = _resolve_below(stage_path, model.destination)
            stage_bundle.mkdir(parents=True, exist_ok=False)
            for artifact in model.artifacts:
                destination = _resolve_below(stage_bundle, artifact.relative_path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                _download_artifact(artifact, destination)
        for destination, model in destinations:
            staged_bundle = _resolve_below(stage_path, model.destination)
            verify_research_tts_model_bundle(stage_path, model)
            staged_bundle.replace(destination)
    except OSError as error:
        raise ResearchTtsError(f"Research TTS model download failed safely: {error}") from error
    finally:
        shutil.rmtree(stage_path, ignore_errors=True)


def _parse_model(value: object, index: int) -> ResearchTtsModel:
    label = f"Model lock model {index}"
    if not isinstance(value, dict):
        raise ResearchTtsError(f"{label} must be a JSON object.")
    raw = cast(dict[str, object], value)
    _expect_keys(
        raw,
        {
            "model_id",
            "destination",
            "generator_family",
            "generator_name",
            "generator_version",
            "license",
            "source_url",
            "runtime",
            "artifacts",
        },
        label,
    )
    destination = _safe_relative_path(_string(raw, "destination", label), label)
    if len(PurePosixPath(destination).parts) != 1:
        raise ResearchTtsError(f"{label} destination must name exactly one directory.")
    source_url = _https_url(_string(raw, "source_url", label), label)
    runtime = raw["runtime"]
    if not isinstance(runtime, dict):
        raise ResearchTtsError(f"{label} runtime must be a JSON object.")
    artifacts_value = raw["artifacts"]
    if not isinstance(artifacts_value, list) or not artifacts_value:
        raise ResearchTtsError(f"{label} artifacts must be a non-empty JSON array.")
    artifacts = tuple(
        _parse_artifact(artifact, f"{label} artifact {artifact_index}")
        for artifact_index, artifact in enumerate(artifacts_value, start=1)
    )
    artifact_paths = [artifact.relative_path for artifact in artifacts]
    if len(artifact_paths) != len(set(artifact_paths)):
        raise ResearchTtsError(f"{label} has duplicate artifact relative_path values.")
    return ResearchTtsModel(
        model_id=_string(raw, "model_id", label),
        destination=destination,
        generator_family=_string(raw, "generator_family", label),
        generator_name=_string(raw, "generator_name", label),
        generator_version=_string(raw, "generator_version", label),
        license=_string(raw, "license", label),
        source_url=source_url,
        runtime=cast(Mapping[str, object], runtime),
        artifacts=artifacts,
    )


def _parse_artifact(value: object, label: str) -> ModelArtifact:
    if not isinstance(value, dict):
        raise ResearchTtsError(f"{label} must be a JSON object.")
    raw = cast(dict[str, object], value)
    _expect_keys(raw, {"relative_path", "url", "expected_size_bytes", "sha256"}, label)
    size = raw["expected_size_bytes"]
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ResearchTtsError(f"{label} expected_size_bytes must be a positive integer.")
    sha256 = _string(raw, "sha256", label).lower()
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
        raise ResearchTtsError(f"{label} sha256 must be lowercase hexadecimal SHA-256.")
    return ModelArtifact(
        relative_path=_safe_relative_path(_string(raw, "relative_path", label), label),
        url=_https_url(_string(raw, "url", label), label),
        expected_size_bytes=size,
        sha256=sha256,
    )


def _download_artifact(artifact: ModelArtifact, destination: Path) -> None:
    digest = hashlib.sha256()
    received = 0
    request = Request(artifact.url, headers={"User-Agent": "KazDeepScan research model intake"})
    try:
        with urlopen(request, timeout=60) as response, destination.open("xb") as handle:
            while chunk := response.read(1024 * 1024):
                received += len(chunk)
                if received > artifact.expected_size_bytes:
                    raise ResearchTtsError(
                        f"Model artifact exceeds expected size: {artifact.relative_path}"
                    )
                digest.update(chunk)
                handle.write(chunk)
    except OSError as error:
        raise ResearchTtsError(
            f"Could not download model artifact {artifact.relative_path}: {error}"
        ) from error
    if received != artifact.expected_size_bytes:
        raise ResearchTtsError(
            f"Model artifact size mismatch for {artifact.relative_path}: "
            f"expected {artifact.expected_size_bytes}, got {received}."
        )
    actual_hash = digest.hexdigest()
    if actual_hash != artifact.sha256:
        raise ResearchTtsError(
            f"Model artifact SHA-256 mismatch for {artifact.relative_path}: "
            f"expected {artifact.sha256}, got {actual_hash}."
        )


def _expect_keys(raw: Mapping[str, object], expected: set[str], label: str) -> None:
    unknown = sorted(set(raw).difference(expected))
    missing = sorted(expected.difference(raw))
    if not unknown and not missing:
        return
    details: list[str] = []
    if missing:
        details.append("missing fields: " + ", ".join(missing))
    if unknown:
        details.append("unknown fields: " + ", ".join(unknown))
    raise ResearchTtsError(f"{label} has " + "; ".join(details) + ".")


def _string(raw: Mapping[str, object], name: str, label: str) -> str:
    value = raw[name]
    if not isinstance(value, str) or not value.strip():
        raise ResearchTtsError(f"{label} field {name!r} must be a non-empty string.")
    return value.strip()


def _safe_relative_path(value: str, label: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or value in {"", "."}:
        raise ResearchTtsError(f"{label} path must be a portable relative path.")
    return path.as_posix()


def _https_url(value: str, label: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ResearchTtsError(f"{label} URL must be an absolute HTTPS URL.")
    return value


def _resolve_below(root: Path, relative_path: str) -> Path:
    resolved_root = root.resolve(strict=True)
    candidate = (resolved_root / relative_path).resolve(strict=False)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as error:
        raise ResearchTtsError(f"Path escapes research model root: {relative_path!r}") from error
    return candidate
