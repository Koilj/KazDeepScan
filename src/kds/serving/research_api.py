"""Separate opt-in API for local, uncalibrated research inference."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Annotated, Literal, Protocol
from uuid import UUID, uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from kds.audio.contracts import AudioLimits, AudioPipelineError, PreparationStatus
from kds.audio.pipeline import AudioPreparationPipeline, PreparedAudio
from kds.audio.windows import WindowConfig
from kds.inference.research import (
    RESEARCH_ONLY_WARNING,
    ResearchInferenceContract,
    ResearchInferenceContractError,
    ResearchInferenceEngine,
    ResearchInferenceError,
    ResearchInferenceResult,
    file_sha256,
    load_research_inference_engine,
)
from kds.serving.api import persist_upload

ResearchStatus = Literal[
    "ok", "insufficient_speech", "rejected_quality", "model_unavailable"
]
ResearchInterpretation = Literal["bonafide_like", "spoof_like"]


class ResearchScorer(Protocol):
    contract: ResearchInferenceContract

    @property
    def model_version(self) -> str: ...

    def score(self, prepared: PreparedAudio) -> ResearchInferenceResult: ...


class ResearchWindowResponse(BaseModel):
    start_s: float = Field(ge=0.0)
    end_s: float = Field(gt=0.0)
    real_samples: int = Field(gt=0)
    raw_spoof_logit: float
    uncalibrated_spoof_score: float = Field(ge=0.0, le=1.0)
    interpretation: ResearchInterpretation


class ResearchAnalyzeResponse(BaseModel):
    analysis_id: UUID
    status: ResearchStatus
    research_only: Literal[True] = True
    contract_id: str
    contract_sha256: str
    model_version: str
    input_sha256: str
    speech_seconds: float = Field(ge=0.0)
    quality_flags: list[str]
    raw_spoof_logit: float | None = None
    uncalibrated_spoof_score: float | None = Field(default=None, ge=0.0, le=1.0)
    interpretation: ResearchInterpretation | None = None
    calibrated: Literal[False] = False
    probability_claim: Literal[False] = False
    fraud_claim: Literal[False] = False
    product_grade: Literal[False] = False
    warning: str = RESEARCH_ONLY_WARNING
    limitations: list[str]
    windows: list[ResearchWindowResponse]


class ResearchHealthResponse(BaseModel):
    status: Literal["ok"]
    research_only: Literal[True] = True


class ResearchReadinessResponse(BaseModel):
    status: Literal["ready", "model_unavailable"]
    research_only: Literal[True] = True
    contract_id: str | None = None
    model_version: str | None = None
    warning: str = RESEARCH_ONLY_WARNING


class ResearchAnalysisService:
    def __init__(
        self,
        scorer: ResearchScorer | None = None,
        pipeline: AudioPreparationPipeline | None = None,
        limits: AudioLimits | None = None,
    ) -> None:
        self._scorer = scorer
        if limits is not None:
            self._limits = limits
        elif scorer is not None:
            preprocessing = scorer.contract.preprocessing
            self._limits = AudioLimits(
                target_sample_rate=preprocessing.target_sample_rate,
                minimum_speech_seconds=preprocessing.minimum_speech_seconds,
            )
        else:
            self._limits = AudioLimits()
        if pipeline is not None:
            self._pipeline = pipeline
        elif scorer is not None:
            preprocessing = scorer.contract.preprocessing
            self._pipeline = AudioPreparationPipeline(
                limits=self._limits,
                window_config=WindowConfig(
                    samples=preprocessing.window_samples,
                    hop_samples=preprocessing.hop_samples,
                ),
            )
        else:
            self._pipeline = AudioPreparationPipeline(limits=self._limits)

    @property
    def is_ready(self) -> bool:
        return self._scorer is not None

    @property
    def maximum_upload_bytes(self) -> int:
        return self._limits.max_upload_bytes

    @property
    def readiness(self) -> ResearchReadinessResponse:
        if self._scorer is None:
            return ResearchReadinessResponse(status="model_unavailable")
        return ResearchReadinessResponse(
            status="ready",
            contract_id=self._scorer.contract.contract_id,
            model_version=self._scorer.model_version,
        )

    def analyze(self, source: Path, declared_mime: str | None) -> ResearchAnalyzeResponse:
        if self._scorer is None:
            raise ResearchInferenceError("Research scorer is unavailable.")
        input_sha256 = file_sha256(source)
        prepared = self._pipeline.prepare(source, declared_mime)
        if prepared.status is not PreparationStatus.READY:
            return self._prepared_response(prepared, input_sha256)
        result = self._scorer.score(prepared)
        contract = self._scorer.contract
        return ResearchAnalyzeResponse(
            analysis_id=uuid4(),
            status="ok",
            contract_id=contract.contract_id,
            contract_sha256=contract.sha256,
            model_version=self._scorer.model_version,
            input_sha256=input_sha256,
            speech_seconds=prepared.speech_seconds,
            quality_flags=list(prepared.quality_flags),
            raw_spoof_logit=result.raw_spoof_logit,
            uncalibrated_spoof_score=result.uncalibrated_spoof_score,
            interpretation=result.interpretation,
            limitations=list(contract.limitations),
            windows=[
                ResearchWindowResponse(
                    start_s=window.start_s,
                    end_s=window.end_s,
                    real_samples=window.real_samples,
                    raw_spoof_logit=window.raw_spoof_logit,
                    uncalibrated_spoof_score=window.uncalibrated_spoof_score,
                    interpretation=window.interpretation,
                )
                for window in result.windows
            ],
        )

    def _prepared_response(
        self, prepared: PreparedAudio, input_sha256: str
    ) -> ResearchAnalyzeResponse:
        if self._scorer is None:
            raise ResearchInferenceError("Research scorer is unavailable.")
        status_value: Literal["insufficient_speech", "rejected_quality"]
        if prepared.status is PreparationStatus.INSUFFICIENT_SPEECH:
            status_value = "insufficient_speech"
        elif prepared.status is PreparationStatus.REJECTED_QUALITY:
            status_value = "rejected_quality"
        else:
            raise ResearchInferenceError("Unexpected prepared-audio status.")
        contract = self._scorer.contract
        return ResearchAnalyzeResponse(
            analysis_id=uuid4(),
            status=status_value,
            contract_id=contract.contract_id,
            contract_sha256=contract.sha256,
            model_version=self._scorer.model_version,
            input_sha256=input_sha256,
            speech_seconds=prepared.speech_seconds,
            quality_flags=list(prepared.quality_flags),
            limitations=list(contract.limitations),
            windows=[],
        )


def create_research_app(service: ResearchAnalysisService | None = None) -> FastAPI:
    """Create the separate research API; the existing v1 API remains unchanged."""

    service = service or ResearchAnalysisService()
    app = FastAPI(
        title="KazDeepScan Local Research Inference API",
        version="1.0.0-research+user-inference-v1",
        description=RESEARCH_ONLY_WARNING,
    )

    @app.get("/healthz", response_model=ResearchHealthResponse)
    async def healthz() -> ResearchHealthResponse:
        return ResearchHealthResponse(status="ok")

    @app.get("/readyz", response_model=ResearchReadinessResponse)
    async def readyz() -> ResearchReadinessResponse:
        readiness = service.readiness
        if readiness.status != "ready":
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=readiness.model_dump(),
            )
        return readiness

    @app.post("/v1/research/analyze", response_model=ResearchAnalyzeResponse)
    async def analyze(
        audio: Annotated[UploadFile, File()],
        acknowledge_research_only: Annotated[bool, Form()] = False,
        confirm_external_user_audio: Annotated[bool, Form()] = False,
    ) -> ResearchAnalyzeResponse:
        if not acknowledge_research_only:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "status": "research_acknowledgement_required",
                    "warning": RESEARCH_ONLY_WARNING,
                },
            )
        if not confirm_external_user_audio:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    "status": "external_user_audio_confirmation_required",
                    "detail": "Frozen project/evaluation assets are prohibited on this route.",
                },
            )
        if not service.is_ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=service.readiness.model_dump(),
            )
        try:
            with tempfile.TemporaryDirectory(prefix="kds-research-upload-") as directory:
                source = Path(directory) / f"{uuid4().hex}.upload"
                await persist_upload(audio, source, service.maximum_upload_bytes)
                return service.analyze(source, audio.content_type)
        except AudioPipelineError as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "status": "audio_rejected",
                    "code": error.code.value,
                    "detail": error.detail,
                },
            ) from error
        except (ResearchInferenceContractError, ResearchInferenceError) as error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"status": "research_inference_failed", "detail": str(error)},
            ) from error

    return app


def create_research_app_from_environment() -> FastAPI:
    """Load the opt-in contract from the environment and fail closed during startup."""

    contract_value = os.environ.get("KDS_RESEARCH_INFERENCE_CONTRACT")
    if contract_value is None or not contract_value.strip():
        raise RuntimeError(
            "KDS_RESEARCH_INFERENCE_CONTRACT is required for the research inference API."
        )
    engine: ResearchInferenceEngine = load_research_inference_engine(Path(contract_value))
    return create_research_app(ResearchAnalysisService(scorer=engine))
