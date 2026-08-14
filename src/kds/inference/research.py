"""Strict research-only inference for external user-supplied audio.

This module never imports or calls a frozen evaluation runner. It reads one explicitly pinned
research checkpoint, scores only the caller's prepared audio in memory, and does not write logits,
predictions, manifests, execution locks, or model artifacts.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

import torch
from torch import Tensor

from kds.audio.contracts import PreparationStatus, WindowDescriptor
from kds.audio.pipeline import PreparedAudio
from kds.models import B0Config, B0LogMelCnn
from kds.training.frozen_b0 import state_dict_sha256

RESEARCH_INFERENCE_SCHEMA_VERSION = 1
RESEARCH_ONLY_WARNING = (
    "Результат является некалиброванным исследовательским сигналом сходства. Он не является "
    "вероятностью, идентификацией говорящего, доказательством мошенничества или product-grade "
    "оценкой и не должен использоваться для автоматических решений."
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CONTRACT_ID = re.compile(r"[a-z0-9][a-z0-9._-]*")
_REQUIRED_LIMITATIONS = {
    "uncalibrated_score_not_probability",
    "training_data_overlap_unverified",
    "not_speaker_independent",
    "not_fraud_determination",
    "not_product_grade",
}
_CHECKPOINT_KEYS = {
    "model_name",
    "model_config",
    "training_seed",
    "training_purpose",
    "source_mixed_research_matrix",
    "best_dev_loss",
    "final_test_metrics",
    "state_dict",
}
ResearchInterpretation = Literal["bonafide_like", "spoof_like"]


class ResearchInferenceContractError(ValueError):
    """Raised when the user-inference contract or pinned checkpoint is not trustworthy."""

    def __init__(self, issues: str | list[str]) -> None:
        self.issues = (issues,) if isinstance(issues, str) else tuple(issues)
        super().__init__("\n".join(self.issues))


class ResearchInferenceError(RuntimeError):
    """Raised when ready audio cannot be scored under the validated contract."""


@dataclass(frozen=True, slots=True)
class ResearchCheckpointContract:
    path: Path
    sha256: str
    state_dict_sha256: str
    model_id: str
    architecture: str
    training_purpose: str
    model_config: B0Config


@dataclass(frozen=True, slots=True)
class ResearchInputScope:
    allowed: str
    prohibited_roots: tuple[Path, ...]
    training_data_overlap: str


@dataclass(frozen=True, slots=True)
class ResearchPreprocessingContract:
    target_sample_rate: int
    minimum_speech_seconds: float
    window_samples: int
    hop_samples: int
    short_window_policy: str
    vad_scope: str


@dataclass(frozen=True, slots=True)
class ResearchInferenceSettings:
    device: str
    batch_size: int
    aggregation: str
    score_transform: str
    raw_logit_boundary: float
    repeat_completed_evaluation_prohibited: bool


@dataclass(frozen=True, slots=True)
class ResearchInferenceContract:
    path: Path
    sha256: str
    contract_id: str
    purpose: str
    input_scope: ResearchInputScope
    checkpoint: ResearchCheckpointContract
    preprocessing: ResearchPreprocessingContract
    inference: ResearchInferenceSettings
    score_name: str
    calibrated: bool
    probability_claim: bool
    fraud_claim: bool
    product_grade: bool
    warning: str
    limitations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResearchWindowResult:
    start_s: float
    end_s: float
    real_samples: int
    raw_spoof_logit: float
    uncalibrated_spoof_score: float
    interpretation: ResearchInterpretation


@dataclass(frozen=True, slots=True)
class ResearchInferenceResult:
    raw_spoof_logit: float
    uncalibrated_spoof_score: float
    interpretation: ResearchInterpretation
    windows: tuple[ResearchWindowResult, ...]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ResearchInferenceContractError(f"Cannot hash file {path}: {error}") from error
    return digest.hexdigest()


def load_research_inference_contract(path: Path) -> ResearchInferenceContract:
    """Load a strict, side-effect-free contract for external user audio."""

    try:
        contract_bytes = path.read_bytes()
        value: object = json.loads(contract_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResearchInferenceContractError(
            f"Cannot read research contract {path}: {error}"
        ) from error
    raw = _object(value, "research inference contract")
    _exact_keys(
        raw,
        {
            "schema_version",
            "contract_id",
            "purpose",
            "input_scope",
            "checkpoint",
            "preprocessing",
            "inference",
            "output_semantics",
            "limitations",
        },
        "research inference contract",
    )
    if raw["schema_version"] != RESEARCH_INFERENCE_SCHEMA_VERSION:
        raise ResearchInferenceContractError("Research inference schema_version must be 1.")
    contract_id = _string(raw, "contract_id", "research inference contract")
    if _CONTRACT_ID.fullmatch(contract_id) is None:
        raise ResearchInferenceContractError("contract_id contains unsupported characters.")
    purpose = _string(raw, "purpose", "research inference contract")
    if purpose != "research_user_audio_only":
        raise ResearchInferenceContractError("purpose must be research_user_audio_only.")

    resolved_path = path.resolve()
    base = resolved_path.parent
    input_scope = _parse_input_scope(raw["input_scope"], base)
    checkpoint = _parse_checkpoint(raw["checkpoint"], base)
    preprocessing = _parse_preprocessing(raw["preprocessing"])
    inference = _parse_inference(raw["inference"])
    output = _parse_output_semantics(raw["output_semantics"])
    limitations = _string_tuple(raw["limitations"], "limitations")
    missing_limitations = sorted(_REQUIRED_LIMITATIONS.difference(limitations))
    if missing_limitations:
        raise ResearchInferenceContractError(
            "Research contract lacks mandatory limitations: " + ", ".join(missing_limitations)
        )
    if preprocessing.target_sample_rate != checkpoint.model_config.sample_rate:
        raise ResearchInferenceContractError(
            "Preprocessing sample rate differs from the checkpoint model config."
        )

    return ResearchInferenceContract(
        path=resolved_path,
        sha256=hashlib.sha256(contract_bytes).hexdigest(),
        contract_id=contract_id,
        purpose=purpose,
        input_scope=input_scope,
        checkpoint=checkpoint,
        preprocessing=preprocessing,
        inference=inference,
        score_name=output[0],
        calibrated=output[1],
        probability_claim=output[2],
        fraud_claim=output[3],
        product_grade=output[4],
        warning=output[5],
        limitations=limitations,
    )


def assert_user_audio_path_allowed(contract: ResearchInferenceContract, source: Path) -> Path:
    """Reject project datasets/models so this route cannot be used to repeat frozen evaluation."""

    try:
        resolved = source.resolve(strict=True)
    except OSError as error:
        raise ResearchInferenceContractError(f"User audio does not exist: {source}") from error
    if not resolved.is_file():
        raise ResearchInferenceContractError(f"User audio must be a regular file: {resolved}")
    for prohibited_root in contract.input_scope.prohibited_roots:
        if resolved.is_relative_to(prohibited_root):
            raise ResearchInferenceContractError(
                "Research user inference refuses project data/model roots: "
                f"{resolved} is below {prohibited_root}."
            )
    return resolved


def load_research_inference_engine(
    contract_or_path: ResearchInferenceContract | Path,
) -> ResearchInferenceEngine:
    contract = (
        contract_or_path
        if isinstance(contract_or_path, ResearchInferenceContract)
        else load_research_inference_contract(contract_or_path)
    )
    checkpoint = contract.checkpoint
    actual_checkpoint_sha256 = file_sha256(checkpoint.path)
    if actual_checkpoint_sha256 != checkpoint.sha256:
        raise ResearchInferenceContractError(
            "Research checkpoint SHA-256 mismatch: "
            f"expected {checkpoint.sha256}, got {actual_checkpoint_sha256}."
        )
    try:
        payload_value: object = torch.load(
            checkpoint.path, map_location="cpu", weights_only=True
        )
    except (OSError, RuntimeError, ValueError) as error:
        raise ResearchInferenceContractError(
            f"Cannot safely load pinned research checkpoint {checkpoint.path}: {error}"
        ) from error
    payload = _object(payload_value, "research checkpoint")
    _exact_keys(payload, _CHECKPOINT_KEYS, "research checkpoint")
    if payload["model_name"] != checkpoint.architecture:
        raise ResearchInferenceContractError("Checkpoint architecture differs from the contract.")
    if payload["training_purpose"] != checkpoint.training_purpose:
        raise ResearchInferenceContractError(
            "Checkpoint training purpose differs from the contract."
        )
    if payload["model_config"] != asdict(checkpoint.model_config):
        raise ResearchInferenceContractError("Checkpoint model config differs from the contract.")
    state_value = payload["state_dict"]
    if not isinstance(state_value, dict) or not state_value:
        raise ResearchInferenceContractError("Checkpoint state_dict must be a non-empty object.")
    if not all(
        isinstance(name, str) and isinstance(tensor, Tensor)
        for name, tensor in state_value.items()
    ):
        raise ResearchInferenceContractError("Checkpoint state_dict has non-tensor entries.")
    state_dict = cast(dict[str, Tensor], state_value)
    actual_state_sha256 = state_dict_sha256(state_dict)
    if actual_state_sha256 != checkpoint.state_dict_sha256:
        raise ResearchInferenceContractError(
            "Checkpoint state_dict SHA-256 mismatch: "
            f"expected {checkpoint.state_dict_sha256}, got {actual_state_sha256}."
        )
    if not all(bool(torch.isfinite(tensor).all()) for tensor in state_dict.values()):
        raise ResearchInferenceContractError("Checkpoint state_dict contains non-finite values.")

    model = B0LogMelCnn(checkpoint.model_config)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ResearchInferenceContractError(
            f"Checkpoint state_dict is incompatible with B0: {error}"
        ) from error
    model.eval()
    return ResearchInferenceEngine(contract, model)


class ResearchInferenceEngine:
    """In-memory scorer for one validated research contract and read-only checkpoint."""

    def __init__(self, contract: ResearchInferenceContract, model: B0LogMelCnn) -> None:
        self.contract = contract
        self._model = model.to(torch.device(contract.inference.device))

    @property
    def model_version(self) -> str:
        return f"{self.contract.checkpoint.model_id}@sha256:{self.contract.checkpoint.sha256[:12]}"

    def score(self, prepared: PreparedAudio) -> ResearchInferenceResult:
        if prepared.status is not PreparationStatus.READY:
            raise ResearchInferenceError("Only READY audio may be passed to the research model.")
        if prepared.waveform.sample_rate != self.contract.preprocessing.target_sample_rate:
            raise ResearchInferenceError("Prepared waveform sample rate violates the contract.")
        if prepared.speech_seconds < self.contract.preprocessing.minimum_speech_seconds:
            raise ResearchInferenceError("Prepared speech duration violates the contract.")
        if not prepared.windows:
            raise ResearchInferenceError("READY audio must contain at least one inference window.")

        waveforms = tuple(self._window_tensor(prepared, window) for window in prepared.windows)
        logits: list[Tensor] = []
        batch_size = self.contract.inference.batch_size
        device = torch.device(self.contract.inference.device)
        with torch.inference_mode():
            for start in range(0, len(waveforms), batch_size):
                batch = torch.stack(waveforms[start : start + batch_size]).to(device)
                batch_logits = self._model(batch).detach().cpu()
                if batch_logits.ndim != 1 or batch_logits.numel() != batch.shape[0]:
                    raise ResearchInferenceError("B0 returned an unexpected logit shape.")
                logits.append(batch_logits)
        window_logits = torch.cat(logits)
        if not bool(torch.isfinite(window_logits).all()):
            raise ResearchInferenceError("B0 returned a non-finite logit.")
        real_samples = torch.tensor(
            [window.real_samples for window in prepared.windows], dtype=window_logits.dtype
        )
        aggregate_logit = (window_logits * real_samples).sum() / real_samples.sum()
        aggregate_score = torch.sigmoid(aggregate_logit)
        windows = tuple(
            ResearchWindowResult(
                start_s=window.start_seconds,
                end_s=window.end_seconds,
                real_samples=window.real_samples,
                raw_spoof_logit=float(logit),
                uncalibrated_spoof_score=float(torch.sigmoid(logit)),
                interpretation=_interpretation(float(logit)),
            )
            for window, logit in zip(prepared.windows, window_logits, strict=True)
        )
        return ResearchInferenceResult(
            raw_spoof_logit=float(aggregate_logit),
            uncalibrated_spoof_score=float(aggregate_score),
            interpretation=_interpretation(float(aggregate_logit)),
            windows=windows,
        )

    def _window_tensor(self, prepared: PreparedAudio, window: WindowDescriptor) -> Tensor:
        expected = self.contract.preprocessing.window_samples
        if window.sample_rate != self.contract.preprocessing.target_sample_rate:
            raise ResearchInferenceError("Window sample rate violates the contract.")
        if window.target_samples != expected or window.real_samples <= 0:
            raise ResearchInferenceError("Window geometry violates the contract.")
        if window.start_sample < 0 or window.end_sample > len(prepared.waveform.samples):
            raise ResearchInferenceError("Window lies outside the prepared waveform.")
        samples = prepared.waveform.samples[window.start_sample : window.end_sample]
        waveform = torch.tensor(samples, dtype=torch.float32).div(32_768.0)
        if waveform.numel() < expected:
            repeats = math.ceil(expected / waveform.numel())
            waveform = waveform.repeat(repeats)
        return waveform[:expected]


def _interpretation(raw_logit: float) -> ResearchInterpretation:
    return "spoof_like" if raw_logit >= 0.0 else "bonafide_like"


def _parse_input_scope(value: object, base: Path) -> ResearchInputScope:
    raw = _object(value, "input_scope")
    _exact_keys(
        raw,
        {"allowed", "prohibited_project_roots", "training_data_overlap"},
        "input_scope",
    )
    allowed = _string(raw, "allowed", "input_scope")
    if allowed != "user_supplied_external_audio_only":
        raise ResearchInferenceContractError(
            "input_scope.allowed must be user_supplied_external_audio_only."
        )
    overlap = _string(raw, "training_data_overlap", "input_scope")
    if overlap != "unverified":
        raise ResearchInferenceContractError("training_data_overlap must remain unverified.")
    roots_value = raw["prohibited_project_roots"]
    if not isinstance(roots_value, list) or len(roots_value) < 3:
        raise ResearchInferenceContractError(
            "prohibited_project_roots must contain at least three relative roots."
        )
    roots = tuple(_relative_path(item, base, "prohibited_project_roots") for item in roots_value)
    if len(set(roots)) != len(roots):
        raise ResearchInferenceContractError("prohibited_project_roots contains duplicates.")
    return ResearchInputScope(
        allowed=allowed,
        prohibited_roots=roots,
        training_data_overlap=overlap,
    )


def _parse_checkpoint(value: object, base: Path) -> ResearchCheckpointContract:
    raw = _object(value, "checkpoint")
    _exact_keys(
        raw,
        {
            "path",
            "sha256",
            "state_dict_sha256",
            "model_id",
            "architecture",
            "training_purpose",
            "model_config",
        },
        "checkpoint",
    )
    architecture = _string(raw, "architecture", "checkpoint")
    if architecture != "b0_logmel_cnn":
        raise ResearchInferenceContractError("Only b0_logmel_cnn is allowed in contract v1.")
    training_purpose = _string(raw, "training_purpose", "checkpoint")
    if training_purpose != "research":
        raise ResearchInferenceContractError("Checkpoint training_purpose must be research.")
    return ResearchCheckpointContract(
        path=_relative_path(raw["path"], base, "checkpoint.path"),
        sha256=_sha256(raw, "sha256", "checkpoint"),
        state_dict_sha256=_sha256(raw, "state_dict_sha256", "checkpoint"),
        model_id=_string(raw, "model_id", "checkpoint"),
        architecture=architecture,
        training_purpose=training_purpose,
        model_config=_parse_model_config(raw["model_config"]),
    )


def _parse_model_config(value: object) -> B0Config:
    raw = _object(value, "checkpoint.model_config")
    _exact_keys(
        raw, {"sample_rate", "n_fft", "hop_length", "n_mels", "dropout"}, "model_config"
    )
    return B0Config(
        sample_rate=_positive_int(raw["sample_rate"], "model_config.sample_rate"),
        n_fft=_positive_int(raw["n_fft"], "model_config.n_fft"),
        hop_length=_positive_int(raw["hop_length"], "model_config.hop_length"),
        n_mels=_positive_int(raw["n_mels"], "model_config.n_mels"),
        dropout=_finite_float(raw["dropout"], "model_config.dropout"),
    )


def _parse_preprocessing(value: object) -> ResearchPreprocessingContract:
    raw = _object(value, "preprocessing")
    _exact_keys(
        raw,
        {
            "target_sample_rate",
            "minimum_speech_seconds",
            "window_samples",
            "hop_samples",
            "short_window_policy",
            "vad_scope",
        },
        "preprocessing",
    )
    short_window_policy = _string(raw, "short_window_policy", "preprocessing")
    vad_scope = _string(raw, "vad_scope", "preprocessing")
    if short_window_policy != "repeat_to_window" or vad_scope != "speech_segments_only":
        raise ResearchInferenceContractError("Unsupported preprocessing policy.")
    return ResearchPreprocessingContract(
        target_sample_rate=_positive_int(
            raw["target_sample_rate"], "preprocessing.target_sample_rate"
        ),
        minimum_speech_seconds=_positive_float(
            raw["minimum_speech_seconds"], "preprocessing.minimum_speech_seconds"
        ),
        window_samples=_positive_int(raw["window_samples"], "preprocessing.window_samples"),
        hop_samples=_positive_int(raw["hop_samples"], "preprocessing.hop_samples"),
        short_window_policy=short_window_policy,
        vad_scope=vad_scope,
    )


def _parse_inference(value: object) -> ResearchInferenceSettings:
    raw = _object(value, "inference")
    _exact_keys(
        raw,
        {
            "device",
            "batch_size",
            "aggregation",
            "score_transform",
            "raw_logit_boundary",
            "repeat_completed_evaluation_prohibited",
        },
        "inference",
    )
    device = _string(raw, "device", "inference")
    aggregation = _string(raw, "aggregation", "inference")
    score_transform = _string(raw, "score_transform", "inference")
    if device != "cpu":
        raise ResearchInferenceContractError(
            "Contract v1 requires deterministic local CPU inference."
        )
    if aggregation != "duration_weighted_mean_raw_logit":
        raise ResearchInferenceContractError("Unsupported research aggregation.")
    if score_transform != "sigmoid_uncalibrated":
        raise ResearchInferenceContractError("Unsupported research score transform.")
    boundary = _finite_float(raw["raw_logit_boundary"], "inference.raw_logit_boundary")
    if boundary != 0.0:
        raise ResearchInferenceContractError(
            "raw_logit_boundary must remain the trained zero boundary."
        )
    repeat_prohibited = raw["repeat_completed_evaluation_prohibited"]
    if repeat_prohibited is not True:
        raise ResearchInferenceContractError(
            "repeat_completed_evaluation_prohibited must be true."
        )
    batch_size = _positive_int(raw["batch_size"], "inference.batch_size")
    if batch_size > 64:
        raise ResearchInferenceContractError("inference.batch_size must not exceed 64.")
    return ResearchInferenceSettings(
        device=device,
        batch_size=batch_size,
        aggregation=aggregation,
        score_transform=score_transform,
        raw_logit_boundary=boundary,
        repeat_completed_evaluation_prohibited=repeat_prohibited,
    )


def _parse_output_semantics(value: object) -> tuple[str, bool, bool, bool, bool, str]:
    raw = _object(value, "output_semantics")
    _exact_keys(
        raw,
        {
            "score_name",
            "calibrated",
            "probability_claim",
            "fraud_claim",
            "product_grade",
            "warning",
        },
        "output_semantics",
    )
    score_name = _string(raw, "score_name", "output_semantics")
    warning = _string(raw, "warning", "output_semantics")
    flags = tuple(
        raw[name]
        for name in ("calibrated", "probability_claim", "fraud_claim", "product_grade")
    )
    if score_name != "uncalibrated_spoof_score" or flags != (False, False, False, False):
        raise ResearchInferenceContractError(
            "Research output semantics must retain all false claims."
        )
    if warning != RESEARCH_ONLY_WARNING:
        raise ResearchInferenceContractError("Research-only warning was changed or weakened.")
    return score_name, False, False, False, False, warning


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ResearchInferenceContractError(f"{label} must be a JSON object with string keys.")
    return cast(dict[str, object], value)


def _exact_keys(raw: Mapping[str, object], expected: set[str], label: str) -> None:
    actual = set(raw)
    if actual != expected:
        raise ResearchInferenceContractError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}."
        )


def _string(raw: Mapping[str, object], name: str, label: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ResearchInferenceContractError(f"{label}.{name} must be a non-empty trimmed string.")
    return value


def _sha256(raw: Mapping[str, object], name: str, label: str) -> str:
    value = _string(raw, name, label)
    if _SHA256.fullmatch(value) is None:
        raise ResearchInferenceContractError(f"{label}.{name} must be a lowercase SHA-256.")
    return value


def _relative_path(value: object, base: Path, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ResearchInferenceContractError(f"{label} must be a non-empty relative path.")
    return (base / value).resolve()


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ResearchInferenceContractError(f"{label} must be a positive integer.")
    return value


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ResearchInferenceContractError(f"{label} must be a finite number.")
    return float(value)


def _positive_float(value: object, label: str) -> float:
    result = _finite_float(value, label)
    if result <= 0.0:
        raise ResearchInferenceContractError(f"{label} must be positive.")
    return result


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ResearchInferenceContractError(f"{label} must be a non-empty string array.")
    if not all(isinstance(item, str) and item and item.strip() == item for item in value):
        raise ResearchInferenceContractError(f"{label} must contain non-empty trimmed strings.")
    result = tuple(cast(list[str], value))
    if len(set(result)) != len(result):
        raise ResearchInferenceContractError(f"{label} contains duplicates.")
    return result
