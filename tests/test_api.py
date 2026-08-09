from __future__ import annotations

from array import array
from pathlib import Path

import anyio
import httpx

from kds.audio.contracts import AudioQuality, MediaInfo, PreparationStatus, SpeechSegment
from kds.audio.pipeline import PreparedAudio
from kds.audio.waveform import Waveform
from kds.audio.windows import WindowConfig, build_inference_windows
from kds.serving.api import AnalysisService, create_app


class StubPipeline:
    def prepare(self, source: Path, declared_mime: str | None) -> PreparedAudio:
        assert source.is_file()
        assert declared_mime == "audio/wav"
        waveform = Waveform(array("h", [1_000] * 48_000), 16_000)
        segment = SpeechSegment(0, 48_000, 16_000)
        return PreparedAudio(
            media=MediaInfo(source, ("wav",), 3.0, 1),
            waveform=waveform,
            quality=AudioQuality(peak=0.1, rms_dbfs=-20.0, clipped_fraction=0.0, dc_offset=0.0),
            speech_segments=(segment,),
            speech_seconds=3.0,
            windows=tuple(build_inference_windows([segment], WindowConfig())),
            status=PreparationStatus.READY,
            quality_flags=(),
        )


def _request(app: object, method: str, url: str, **kwargs: object) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.request(method, url, **kwargs)

    return anyio.run(send)


def test_health_and_unready_endpoints_are_explicit() -> None:
    app = create_app(AnalysisService(pipeline=StubPipeline()))

    health = _request(app, "GET", "/healthz")
    readiness = _request(app, "GET", "/readyz")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert readiness.status_code == 503
    assert readiness.json()["detail"]["status"] == "model_unavailable"


def test_ready_audio_never_receives_untrained_risk_score() -> None:
    app = create_app(AnalysisService(pipeline=StubPipeline()))

    response = _request(
        app,
        "POST",
        "/v1/analyze",
        files={"audio": ("voice.wav", b"not-a-real-wav", "audio/wav")},
    )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["status"] == "model_unavailable"
    assert detail["risk_score"] is None
    assert detail["speech_seconds"] == 3.0
