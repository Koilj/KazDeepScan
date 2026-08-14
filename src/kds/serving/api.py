from __future__ import annotations

import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Protocol
from uuid import UUID, uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from kds import __release_version__
from kds.audio.contracts import AudioLimits, PreparationStatus
from kds.audio.pipeline import AudioPreparationPipeline, PreparedAudio

LanguageHint = Literal["ru", "kk", "mixed", "auto"]
AnalysisMode = Literal["file", "call"]
DISCLAIMER = (
    "Оценка риска не подтверждает личность собеседника и не является доказательством мошенничества."
)
UploadReader = Callable[[int], Awaitable[bytes]]


class RiskScorer(Protocol):
    model_version: str

    def score(self, prepared: PreparedAudio) -> ScoredAnalysis:
        """Score ready audio using a trained, calibrated model."""


@dataclass(frozen=True, slots=True)
class ScoredAnalysis:
    risk_score: float
    risk_band: Literal["low", "uncertain", "high"]
    segments: tuple[RiskSegment, ...]


class RiskSegment(BaseModel):
    start_s: float = Field(ge=0.0)
    end_s: float = Field(gt=0.0)
    risk_score: float = Field(ge=0.0, le=1.0)


class AnalyzeResponse(BaseModel):
    analysis_id: UUID
    status: Literal["ok", "insufficient_speech", "rejected_quality", "model_unavailable"]
    risk_score: float | None = Field(default=None, ge=0.0, le=1.0)
    risk_band: Literal["low", "uncertain", "high"] | None = None
    model_version: str | None = None
    speech_seconds: float = Field(ge=0.0)
    quality_flags: list[str]
    segments: list[RiskSegment]
    disclaimer: str = DISCLAIMER


class HealthResponse(BaseModel):
    status: Literal["ok"]


class ReadinessResponse(BaseModel):
    status: Literal["ready", "model_unavailable"]


class AnalysisService:
    def __init__(
        self,
        pipeline: AudioPreparationPipeline | None = None,
        scorer: RiskScorer | None = None,
        limits: AudioLimits | None = None,
    ) -> None:
        self._pipeline = pipeline or AudioPreparationPipeline()
        self._scorer = scorer
        self._limits = limits or AudioLimits()

    @property
    def is_ready(self) -> bool:
        return self._scorer is not None

    def analyze(self, source: Path, declared_mime: str | None) -> AnalyzeResponse:
        prepared = self._pipeline.prepare(source, declared_mime)
        analysis_id = uuid4()
        if prepared.status is PreparationStatus.INSUFFICIENT_SPEECH:
            return _response_from_prepared(analysis_id, prepared, "insufficient_speech")
        if prepared.status is PreparationStatus.REJECTED_QUALITY:
            return _response_from_prepared(analysis_id, prepared, "rejected_quality")
        if self._scorer is None:
            return _response_from_prepared(analysis_id, prepared, "model_unavailable")

        scored = self._scorer.score(prepared)
        return AnalyzeResponse(
            analysis_id=analysis_id,
            status="ok",
            risk_score=scored.risk_score,
            risk_band=scored.risk_band,
            model_version=self._scorer.model_version,
            speech_seconds=prepared.speech_seconds,
            quality_flags=list(prepared.quality_flags),
            segments=list(scored.segments),
        )


def _response_from_prepared(
    analysis_id: UUID,
    prepared: PreparedAudio,
    response_status: Literal["insufficient_speech", "rejected_quality", "model_unavailable"],
) -> AnalyzeResponse:
    return AnalyzeResponse(
        analysis_id=analysis_id,
        status=response_status,
        speech_seconds=prepared.speech_seconds,
        quality_flags=list(prepared.quality_flags),
        segments=[],
    )


async def persist_upload(upload: UploadFile, destination: Path, maximum_bytes: int) -> int:
    """Store an upload in an empty private temporary path while enforcing a streaming size limit."""

    written = 0
    chunk_size = 1024 * 1024
    with destination.open("xb") as file_handle:
        while chunk := await upload.read(chunk_size):
            written += len(chunk)
            if written > maximum_bytes:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Audio exceeds the {maximum_bytes}-byte limit.",
                )
            file_handle.write(chunk)
    return written


def create_app(service: AnalysisService | None = None) -> FastAPI:
    service = service or AnalysisService()
    app = FastAPI(title="KazDeepScan Research API", version=__release_version__)

    @app.get("/healthz", response_model=HealthResponse)
    async def healthz() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/readyz", response_model=ReadinessResponse)
    async def readyz() -> ReadinessResponse:
        if not service.is_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=ReadinessResponse(status="model_unavailable").model_dump(),
            )
        return ReadinessResponse(status="ready")

    @app.post("/v1/analyze", response_model=AnalyzeResponse)
    async def analyze(
        audio: Annotated[UploadFile, File()],
        mode: Annotated[AnalysisMode, Form()] = "file",
        language_hint: Annotated[LanguageHint, Form()] = "auto",
    ) -> AnalyzeResponse:
        del mode, language_hint
        with tempfile.TemporaryDirectory(prefix="kds-upload-") as temporary_directory:
            source = Path(temporary_directory) / f"{uuid4().hex}.upload"
            await persist_upload(audio, source, service._limits.max_upload_bytes)
            response = service.analyze(source, audio.content_type)
        if response.status == "model_unavailable":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=response.model_dump(mode="json"),
            )
        return response

    return app
