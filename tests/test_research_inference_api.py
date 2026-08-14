from __future__ import annotations

from array import array
from pathlib import Path

import anyio
import httpx
import pytest

from kds.audio.contracts import AudioQuality, MediaInfo, PreparationStatus, SpeechSegment
from kds.audio.pipeline import PreparedAudio
from kds.audio.waveform import Waveform
from kds.audio.windows import WindowConfig, build_inference_windows
from kds.inference import (
    RESEARCH_ONLY_WARNING,
    ResearchInferenceResult,
    ResearchWindowResult,
    load_research_inference_contract,
)
from kds.serving.research_api import (
    ResearchAnalysisService,
    create_research_app,
    create_research_app_from_environment,
)


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


class StubScorer:
    def __init__(self) -> None:
        self.contract = load_research_inference_contract(
            Path("configs/inference/b0_user_audio_local_research_v1.json")
        )

    @property
    def model_version(self) -> str:
        return "stub-research-model"

    def score(self, prepared: PreparedAudio) -> ResearchInferenceResult:
        window = prepared.windows[0]
        return ResearchInferenceResult(
            raw_spoof_logit=0.25,
            uncalibrated_spoof_score=0.5621765,
            interpretation="spoof_like",
            windows=(
                ResearchWindowResult(
                    start_s=window.start_seconds,
                    end_s=window.end_seconds,
                    real_samples=window.real_samples,
                    raw_spoof_logit=0.25,
                    uncalibrated_spoof_score=0.5621765,
                    interpretation="spoof_like",
                ),
            ),
        )


def _request(app: object, method: str, url: str, **kwargs: object) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            return await client.request(method, url, **kwargs)

    return anyio.run(send)


def test_research_api_is_fail_closed_without_a_scorer() -> None:
    app = create_research_app()

    readiness = _request(app, "GET", "/readyz")

    assert readiness.status_code == 503
    assert readiness.json()["detail"]["status"] == "model_unavailable"
    assert readiness.json()["detail"]["research_only"] is True


def test_research_api_factory_requires_an_explicit_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KDS_RESEARCH_INFERENCE_CONTRACT", raising=False)

    with pytest.raises(RuntimeError, match="KDS_RESEARCH_INFERENCE_CONTRACT is required"):
        create_research_app_from_environment()


def test_research_api_requires_explicit_acknowledgement() -> None:
    app = create_research_app(ResearchAnalysisService(scorer=StubScorer(), pipeline=StubPipeline()))

    response = _request(
        app,
        "POST",
        "/v1/research/analyze",
        files={"audio": ("voice.wav", b"audio", "audio/wav")},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["status"] == "research_acknowledgement_required"
    assert response.json()["detail"]["warning"] == RESEARCH_ONLY_WARNING


def test_research_api_requires_external_user_audio_confirmation() -> None:
    app = create_research_app(ResearchAnalysisService(scorer=StubScorer(), pipeline=StubPipeline()))

    response = _request(
        app,
        "POST",
        "/v1/research/analyze",
        files={"audio": ("voice.wav", b"audio", "audio/wav")},
        data={"acknowledge_research_only": "true"},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["status"] == "external_user_audio_confirmation_required"


def test_research_api_returns_uncalibrated_score_without_product_claims() -> None:
    app = create_research_app(ResearchAnalysisService(scorer=StubScorer(), pipeline=StubPipeline()))

    response = _request(
        app,
        "POST",
        "/v1/research/analyze",
        files={"audio": ("voice.wav", b"audio", "audio/wav")},
        data={
            "acknowledge_research_only": "true",
            "confirm_external_user_audio": "true",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["research_only"] is True
    assert payload["uncalibrated_spoof_score"] == 0.5621765
    assert payload["calibrated"] is False
    assert payload["probability_claim"] is False
    assert payload["fraud_claim"] is False
    assert payload["product_grade"] is False
    assert payload["warning"] == RESEARCH_ONLY_WARNING
    assert "risk_score" not in payload
