from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from kds.data.research_tts import load_research_tts_model_lock
from kds.data.voxcpm2 import (
    VoxCPM2AuditError,
    _audit_safetensors,
    _tokenizer_python_policy,
)


def _write_safetensors(path: Path, offsets: list[int]) -> None:
    header = json.dumps(
        {"tensor": {"dtype": "F32", "shape": [1], "data_offsets": offsets}},
        separators=(",", ":"),
    ).encode()
    payload_bytes = max(offsets)
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * payload_bytes)


def test_safetensors_audit_accepts_one_contiguous_tensor(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, [0, 4])

    header_bytes, tensors, dtypes, payload_bytes = _audit_safetensors(path)

    assert header_bytes > 0
    assert tensors == 1
    assert dtypes == {"F32": 1}
    assert payload_bytes == 4


def test_safetensors_audit_rejects_payload_gap(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    _write_safetensors(path, [1, 4])

    with pytest.raises(VoxCPM2AuditError, match="gaps"):
        _audit_safetensors(path)


def test_tokenizer_python_policy_rejects_network_call() -> None:
    imports, forbidden = _tokenizer_python_policy(b"import requests\nrequests.get('x')\n")

    assert imports == ("requests",)
    assert forbidden == ("requests.get",)


def test_current_voxcpm2_model_lock_is_strict_and_complete() -> None:
    lock = load_research_tts_model_lock(
        Path("configs/research/voxcpm2_official_text_only_v1_models.json")
    )
    model = lock.models[0]

    assert model.generator_family == "openbmb_voxcpm2_official_text_only"
    assert len(model.artifacts) == 10
    assert sum(artifact.expected_size_bytes for artifact in model.artifacts) == 4_964_839_611
    assert model.runtime["reference_audio_policy"] == "forbidden_null_only"
    assert model.runtime["runtime_environment_materialized"] is False
