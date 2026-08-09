from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from kds.data.research_tts import (
    ResearchTtsError,
    load_research_tts_model_lock,
    verify_research_tts_model_bundle,
)


def _lock_payload(*, relative_path: str = "model.bin") -> dict[str, object]:
    payload = b"test-model"
    return {
        "schema_version": 1,
        "protocol_id": "test-research-tts",
        "models": [
            {
                "model_id": "test-model",
                "destination": "test-model",
                "generator_family": "test-family",
                "generator_name": "test-name",
                "generator_version": "1",
                "license": "test-license",
                "source_url": "https://example.test/model",
                "runtime": {"kind": "test"},
                "artifacts": [
                    {
                        "relative_path": relative_path,
                        "url": "https://example.test/model.bin",
                        "expected_size_bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            }
        ],
    }


def _write_lock(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_research_tts_lock_rejects_model_artifact_path_escape(tmp_path: Path) -> None:
    path = tmp_path / "lock.json"
    _write_lock(path, _lock_payload(relative_path="../outside.bin"))

    with pytest.raises(ResearchTtsError, match="portable relative path"):
        load_research_tts_model_lock(path)


def test_research_tts_bundle_verification_requires_exact_artifact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "lock.json"
    _write_lock(path, _lock_payload())
    lock = load_research_tts_model_lock(path)
    bundle_root = tmp_path / "models" / "test-model"
    bundle_root.mkdir(parents=True)
    (bundle_root / "model.bin").write_bytes(b"test-model")

    verified = verify_research_tts_model_bundle(tmp_path / "models", lock.models[0])

    assert verified["model.bin"] == bundle_root / "model.bin"
    (bundle_root / "model.bin").write_bytes(b"changed")
    with pytest.raises(ResearchTtsError, match="size mismatch"):
        verify_research_tts_model_bundle(tmp_path / "models", lock.models[0])


def test_research_tts_lock_rejects_unknown_model_fields(tmp_path: Path) -> None:
    path = tmp_path / "lock.json"
    payload = _lock_payload()
    models = payload["models"]
    assert isinstance(models, list)
    model = models[0]
    assert isinstance(model, dict)
    model["unexpected"] = "value"
    _write_lock(path, payload)

    with pytest.raises(ResearchTtsError, match="unknown fields"):
        load_research_tts_model_lock(path)
