"""Opt-in local inference that is isolated from frozen evaluation runners."""

from kds.inference.research import (
    RESEARCH_ONLY_WARNING,
    ResearchInferenceContract,
    ResearchInferenceContractError,
    ResearchInferenceEngine,
    ResearchInferenceError,
    ResearchInferenceResult,
    ResearchWindowResult,
    assert_user_audio_path_allowed,
    file_sha256,
    load_research_inference_contract,
    load_research_inference_engine,
)

__all__ = [
    "RESEARCH_ONLY_WARNING",
    "ResearchInferenceContract",
    "ResearchInferenceContractError",
    "ResearchInferenceEngine",
    "ResearchInferenceError",
    "ResearchInferenceResult",
    "ResearchWindowResult",
    "assert_user_audio_path_allowed",
    "file_sha256",
    "load_research_inference_contract",
    "load_research_inference_engine",
]
