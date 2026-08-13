from __future__ import annotations

import json
import socket
from pathlib import Path

import numpy as np
import pytest

from kds.data.assets import sha256_file
from kds.eval.voxcpm2_smoke import (
    OneCallModel,
    VoxCPM2SmokeError,
    audit_waveform,
    install_python_network_guard,
)


class _Model:
    def generate(self, **kwargs: object) -> list[float]:
        return [float(len(kwargs))]


def test_one_call_model_refuses_second_attempt() -> None:
    model = OneCallModel(_Model())

    assert model.generate(text="one") == [1.0]
    with pytest.raises(VoxCPM2SmokeError, match="more than one"):
        model.generate(text="two")


def test_waveform_audit_requires_finite_nonempty_48khz() -> None:
    report = audit_waveform(np.ones(4_800, dtype=np.float32) * 0.1, 48_000)

    assert report.frames == 4_800
    assert report.duration_seconds == "0.100000"
    with pytest.raises(VoxCPM2SmokeError, match="48000"):
        audit_waveform(np.ones(4_800, dtype=np.float32), 16_000)
    with pytest.raises(VoxCPM2SmokeError, match="NaN"):
        audit_waveform(np.array([float("nan")], dtype=np.float32), 48_000)


def test_python_network_guard_records_and_blocks_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    original_socket = socket.socket
    original_create = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo
    attempts = install_python_network_guard()
    try:
        with pytest.raises(VoxCPM2SmokeError, match="DNS"):
            socket.getaddrinfo("example.test", 443)
        assert len(attempts) == 1
    finally:
        monkeypatch.setattr(socket, "socket", original_socket)
        monkeypatch.setattr(socket, "create_connection", original_create)
        monkeypatch.setattr(socket, "getaddrinfo", original_getaddrinfo)


def test_current_smoke_and_pre_inference_failure_receipts_are_consistent() -> None:
    smoke_path = Path("data/licenses/voxcpm2_official_v1_cuda_smoke_v1.json")
    failure_path = Path(
        "data/licenses/voxcpm2_official_v1_cuda_smoke_pre_inference_failure_v1.json"
    )
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    failure = json.loads(failure_path.read_text(encoding="utf-8"))

    assert sha256_file(smoke_path) == (
        "0c24a8325d5c1159b2ac2885ebb46d8e38386c984d6908822170a442ea3d6982"
    )
    assert sha256_file(failure_path) == (
        "d803afad53782aeb38be2b29ea6182c44652cea5692cd715dd094b3efc98ff41"
    )
    assert smoke["generation"]["call_count"] == 1
    assert smoke["network_policy"]["observed_upstream_network_attempts"] == 0
    assert smoke["output"]["sample_rate_hz"] == 48_000
    assert smoke["claims"]["detector_inference_performed"] is False
    assert smoke["smoke_runner_sha256"] == sha256_file(
        Path("scripts/smoke_voxcpm2_official.py")
    )
    assert smoke["wrapper_sha256"] == sha256_file(
        Path("src/kds/data/voxcpm2_text_only.py")
    )
    assert failure["failure"]["stochastic_generation_started"] is False
    assert failure["failure"]["wav_created"] is False
