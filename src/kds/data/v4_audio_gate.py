"""Canonical decode, technical QA and perceptual-audio primitives for model v4."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from kds.audio.contracts import (
    AudioLimits,
    AudioPipelineError,
    PreparationStatus,
    SpeechSegment,
)
from kds.audio.media import FFmpegClient
from kds.audio.pipeline import QualityPolicy
from kds.audio.vad import WebRtcVadDetector
from kds.audio.waveform import Waveform, measure_quality, read_pcm16_mono_wav
from kds.data.assets import sha256_file

V4_AUDIO_FINGERPRINT_VERSION = "kds-speech-spectral-phash-v1"
V4_AUDIO_FINGERPRINT_HEX_LENGTH = 64
V4_NEAR_AUDIO_HAMMING_THRESHOLD = 12
V4_NEAR_AUDIO_SPEECH_DURATION_RATIO = 0.15


class V4AudioGateError(ValueError):
    """Raised when v4 audio state is incomplete, inconsistent or unsafe to publish."""


@dataclass(frozen=True, slots=True)
class V4DecodeTask:
    sample_id: str
    raw_relative_path: str
    raw_sha256: str
    source_path: str
    decoded_relative_path: str
    destination_path: str


@dataclass(frozen=True, slots=True)
class V4DecodeResult:
    sample_id: str
    raw_relative_path: str
    raw_sha256: str
    decoded_relative_path: str
    decoded_audio_sha256: str
    decoded_size_bytes: int
    duration_s: float
    peak: float
    rms_dbfs: float
    clipped_fraction: float
    dc_offset: float
    speech_seconds: float
    speech_segment_count: int
    audio_fingerprint_v1: str
    preparation_status: str
    rejection_reason: str

    def as_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class V4AudioSignature:
    identity: str
    audio_sha256: str
    fingerprint: str
    speech_seconds: float


@dataclass(frozen=True, slots=True)
class V4HistoryFingerprintTask:
    identity: str
    manifest_audio_sha256: str
    relative_path: str
    source_path: str


@dataclass(frozen=True, slots=True)
class V4HistoryFingerprintResult:
    identity: str
    manifest_audio_sha256: str
    relative_path: str
    canonical_audio_sha256: str
    duration_s: float
    speech_seconds: float
    audio_fingerprint_v1: str
    status: str
    detail: str

    def as_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class _CanonicalMetrics:
    decoded_audio_sha256: str
    decoded_size_bytes: int
    duration_s: float
    peak: float
    rms_dbfs: float
    clipped_fraction: float
    dc_offset: float
    speech_seconds: float
    speech_segment_count: int
    audio_fingerprint_v1: str
    preparation_status: str
    rejection_reason: str


@dataclass(frozen=True, slots=True)
class V4NearAudioMatch:
    candidate_identity: str
    reference_identity: str
    hamming_distance: int
    speech_duration_ratio: float


@dataclass(frozen=True, slots=True)
class V4DecodedCandidate:
    selection_rank: int
    language: str
    label: str
    result: V4DecodeResult

    @property
    def cell(self) -> str:
        return f"{self.language}/{self.label}"


@dataclass(frozen=True, slots=True)
class V4DecodedDecision:
    candidate: V4DecodedCandidate
    eligibility_status: str
    rejection_reason: str
    exact_duplicate_of_candidate_id: str
    historical_exact_matches: tuple[str, ...]
    historical_near_matches: tuple[V4NearAudioMatch, ...]
    within_pool_near_matches: tuple[V4NearAudioMatch, ...]


def decoded_relative_path(raw_sha256: str) -> str:
    _require_sha256(raw_sha256, "raw SHA-256")
    return (
        "processed/v4/xlsr_sls_model_v4_source_decode_qa_v1/"
        f"{raw_sha256[:2]}/{raw_sha256}.wav"
    )


def canonical_audio_fingerprint(
    waveform: Waveform,
    speech_segments: Sequence[SpeechSegment],
) -> str:
    """Return a gain/padding-tolerant 256-bit speech spectral perceptual hash.

    This is a duplicate-candidate screen, not an identity or speaker embedding. It pools a
    log-power spectrum into 16 frequency bands and 16 normalized speech-time bins, then compares
    every bin with the median of its band. Exact decoded hashes remain the authoritative exact
    duplicate check; low Hamming-distance hits are held for review rather than silently merged.
    """

    if waveform.sample_rate != 16_000 or not waveform.samples:
        raise V4AudioGateError("Fingerprint input must be non-empty canonical 16-kHz audio.")
    speech_parts = [
        np.asarray(waveform.samples[item.start_sample : item.end_sample], dtype=np.float64)
        for item in speech_segments
        if item.sample_rate == waveform.sample_rate and item.end_sample > item.start_sample
    ]
    active = (
        np.concatenate(speech_parts)
        if speech_parts and sum(part.size for part in speech_parts) >= 400
        else np.asarray(waveform.samples, dtype=np.float64)
    )
    active /= 32_768.0
    frame_samples = 400
    hop_samples = 160
    if active.size < frame_samples:
        active = np.pad(active, (0, frame_samples - active.size))
    frame_count = 1 + (active.size - frame_samples) // hop_samples
    frame_starts = np.arange(frame_count, dtype=np.int64) * hop_samples
    sample_offsets = np.arange(frame_samples, dtype=np.int64)
    frames = active[frame_starts[:, None] + sample_offsets[None, :]]
    frames *= np.hanning(frame_samples)[None, :]
    power = np.abs(np.fft.rfft(frames, n=512, axis=1)) ** 2
    frequencies = np.fft.rfftfreq(512, d=1.0 / waveform.sample_rate)
    band_edges = np.geomspace(80.0, 7_600.0, num=17)
    bands = []
    for low, high in zip(band_edges[:-1], band_edges[1:], strict=True):
        mask = (frequencies >= low) & (frequencies < high)
        bands.append(np.log1p(power[:, mask].sum(axis=1)))
    spectral = np.stack(bands, axis=1)
    boundaries = np.linspace(0.0, float(spectral.shape[0]), num=17)
    pooled_parts = []
    for index in range(16):
        start = min(int(np.floor(boundaries[index])), spectral.shape[0] - 1)
        end = min(
            max(start + 1, int(np.ceil(boundaries[index + 1]))),
            spectral.shape[0],
        )
        pooled_parts.append(spectral[start:end].mean(axis=0))
    pooled = np.stack(pooled_parts, axis=0)
    medians = np.median(pooled, axis=0, keepdims=True)
    bits = (pooled > medians).astype(np.uint8).reshape(-1)
    fingerprint = np.packbits(bits, bitorder="big").tobytes().hex()
    if len(fingerprint) != V4_AUDIO_FINGERPRINT_HEX_LENGTH:
        raise V4AudioGateError("Internal v4 audio fingerprint length is invalid.")
    return fingerprint


def fingerprint_hamming_distance(left: str, right: str) -> int:
    _require_fingerprint(left)
    _require_fingerprint(right)
    return (int(left, 16) ^ int(right, 16)).bit_count()


def find_near_audio_matches(
    candidates: Sequence[V4AudioSignature],
    references: Sequence[V4AudioSignature],
    *,
    hamming_threshold: int = V4_NEAR_AUDIO_HAMMING_THRESHOLD,
    speech_duration_ratio: float = V4_NEAR_AUDIO_SPEECH_DURATION_RATIO,
) -> tuple[V4NearAudioMatch, ...]:
    """Find conservative near-duplicate candidates using exact-block LSH and Hamming distance."""

    if hamming_threshold not in range(0, 16) or not 0 <= speech_duration_ratio <= 1:
        raise V4AudioGateError("Invalid v4 near-audio matching thresholds.")
    reference_by_identity: dict[str, V4AudioSignature] = {}
    for reference in references:
        _validate_signature(reference)
        if reference.identity in reference_by_identity:
            raise V4AudioGateError("Duplicate near-audio reference identity.")
        reference_by_identity[reference.identity] = reference
    reference_list = list(reference_by_identity.values())
    reference_blocks = (
        np.stack(
            [
                np.frombuffer(bytes.fromhex(reference.fingerprint), dtype=">u2")
                for reference in reference_list
            ],
            axis=0,
        )
        if reference_list
        else np.empty((0, 16), dtype=">u2")
    )
    block_orders = [np.argsort(reference_blocks[:, block]) for block in range(16)]
    sorted_blocks = [
        reference_blocks[block_orders[block], block] for block in range(16)
    ]
    matches: list[V4NearAudioMatch] = []
    seen_candidates: set[str] = set()
    for candidate in candidates:
        _validate_signature(candidate)
        if candidate.identity in seen_candidates:
            raise V4AudioGateError("Duplicate near-audio candidate identity.")
        seen_candidates.add(candidate.identity)
        candidate_blocks = np.frombuffer(bytes.fromhex(candidate.fingerprint), dtype=">u2")
        possible: set[int] = set()
        for block in range(16):
            value = candidate_blocks[block]
            left = int(np.searchsorted(sorted_blocks[block], value, side="left"))
            right = int(np.searchsorted(sorted_blocks[block], value, side="right"))
            possible.update(int(index) for index in block_orders[block][left:right])
        for reference_index in sorted(possible):
            reference = reference_list[reference_index]
            if candidate.identity == reference.identity:
                continue
            duration_delta = abs(candidate.speech_seconds - reference.speech_seconds) / max(
                candidate.speech_seconds, reference.speech_seconds, 1e-9
            )
            if duration_delta > speech_duration_ratio:
                continue
            distance = fingerprint_hamming_distance(candidate.fingerprint, reference.fingerprint)
            if distance <= hamming_threshold:
                matches.append(
                    V4NearAudioMatch(
                        candidate_identity=candidate.identity,
                        reference_identity=reference.identity,
                        hamming_distance=distance,
                        speech_duration_ratio=duration_delta,
                    )
                )
    return tuple(matches)


def decide_v4_decoded_audio_eligibility(
    candidates: Sequence[V4DecodedCandidate],
    historical_exact_references: Mapping[str, Sequence[str]],
    historical_signatures: Sequence[V4AudioSignature],
) -> tuple[V4DecodedDecision, ...]:
    """Apply QA, decoded-exact and conservative near-audio gates in frozen rank order."""

    ordered = sorted(
        candidates,
        key=lambda item: (
            item.language,
            item.label,
            item.selection_rank,
            item.result.sample_id,
        ),
    )
    if len({item.result.sample_id for item in ordered}) != len(ordered):
        raise V4AudioGateError("Decoded v4 candidates contain duplicate sample IDs.")
    ready = [item for item in ordered if item.result.preparation_status == "ready"]
    signatures = [
        V4AudioSignature(
            identity=item.result.sample_id,
            audio_sha256=item.result.decoded_audio_sha256,
            fingerprint=item.result.audio_fingerprint_v1,
            speech_seconds=item.result.speech_seconds,
        )
        for item in ready
    ]
    historical_near = find_near_audio_matches(signatures, historical_signatures)
    within_near = find_near_audio_matches(signatures, signatures)
    current_sha_by_id = {item.identity: item.audio_sha256 for item in signatures}
    history_sha_by_id = {item.identity: item.audio_sha256 for item in historical_signatures}
    historical_near_by_id: dict[str, list[V4NearAudioMatch]] = {}
    within_near_by_id: dict[str, list[V4NearAudioMatch]] = {}
    for match in historical_near:
        if current_sha_by_id[match.candidate_identity] == history_sha_by_id[
            match.reference_identity
        ]:
            continue
        historical_near_by_id.setdefault(match.candidate_identity, []).append(match)
    for match in within_near:
        if current_sha_by_id[match.candidate_identity] == current_sha_by_id[
            match.reference_identity
        ]:
            continue
        within_near_by_id.setdefault(match.candidate_identity, []).append(match)
    exact_owner: dict[str, V4DecodedCandidate] = {}
    exact_duplicate_of: dict[str, str] = {}
    for item in ready:
        prior = exact_owner.get(item.result.decoded_audio_sha256)
        if prior is None:
            exact_owner[item.result.decoded_audio_sha256] = item
            continue
        if prior.cell != item.cell:
            raise V4AudioGateError(
                "Exact decoded audio has conflicting v4 language/label assignments."
            )
        exact_duplicate_of[item.result.sample_id] = prior.result.sample_id
    decisions: list[V4DecodedDecision] = []
    for item in ordered:
        result = item.result
        historical_exact = tuple(
            sorted(historical_exact_references.get(result.decoded_audio_sha256, ()))
        )
        candidate_history_near = tuple(
            sorted(
                historical_near_by_id.get(result.sample_id, ()),
                key=lambda match: (match.hamming_distance, match.reference_identity),
            )
        )
        candidate_within_near = tuple(
            sorted(
                within_near_by_id.get(result.sample_id, ()),
                key=lambda match: (match.hamming_distance, match.reference_identity),
            )
        )
        duplicate_of = exact_duplicate_of.get(result.sample_id, "")
        if result.preparation_status != "ready":
            status = "rejected"
            reason = result.rejection_reason or result.preparation_status
        elif historical_exact:
            status = "rejected"
            reason = "historical_exact_decoded_audio"
        elif duplicate_of:
            status = "rejected"
            reason = "within_pool_exact_decoded_audio"
        elif candidate_history_near or candidate_within_near:
            status = "pending_near_audio_review"
            reason = "near_audio_review_required"
        else:
            status = "eligible"
            reason = ""
        decisions.append(
            V4DecodedDecision(
                candidate=item,
                eligibility_status=status,
                rejection_reason=reason,
                exact_duplicate_of_candidate_id=duplicate_of,
                historical_exact_matches=historical_exact,
                historical_near_matches=candidate_history_near,
                within_pool_near_matches=candidate_within_near,
            )
        )
    return tuple(decisions)


def load_v4_decode_journal(
    path: Path,
    tasks: Mapping[str, V4DecodeTask],
) -> dict[str, V4DecodeResult]:
    """Load an append-only local journal and bind every row to the current exact task."""

    if not path.exists():
        return {}
    results: dict[str, V4DecodeResult] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise V4AudioGateError(f"Cannot read v4 decode journal: {path}") from error
    for line_number, line in enumerate(lines, start=1):
        try:
            payload: object = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError
            result = V4DecodeResult(**payload)
        except (json.JSONDecodeError, TypeError) as error:
            raise V4AudioGateError(
                f"Invalid v4 decode journal row {line_number}."
            ) from error
        task = tasks.get(result.sample_id)
        if task is None or result.sample_id in results or not _result_matches_task(result, task):
            raise V4AudioGateError(
                f"v4 decode journal row {line_number} does not match a unique current task."
            )
        if result.decoded_audio_sha256:
            destination = Path(task.destination_path)
            if not destination.is_file() or sha256_file(destination) != result.decoded_audio_sha256:
                raise V4AudioGateError(
                    f"v4 decode journal asset binding failed for {result.sample_id!r}."
                )
        results[result.sample_id] = result
    return results


def append_v4_decode_journal(path: Path, result: V4DecodeResult) -> None:
    if not path.parent.is_dir():
        raise V4AudioGateError("v4 decode journal parent does not exist.")
    try:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(result.as_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        raise V4AudioGateError("Cannot append v4 decode journal.") from error


def load_v4_history_fingerprint_journal(
    path: Path,
    tasks: Mapping[str, V4HistoryFingerprintTask],
) -> dict[str, V4HistoryFingerprintResult]:
    if not path.exists():
        return {}
    results: dict[str, V4HistoryFingerprintResult] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise V4AudioGateError(f"Cannot read v4 history fingerprint journal: {path}") from error
    for line_number, line in enumerate(lines, start=1):
        try:
            payload: object = json.loads(line)
            if not isinstance(payload, dict):
                raise TypeError
            result = V4HistoryFingerprintResult(**payload)
        except (json.JSONDecodeError, TypeError) as error:
            raise V4AudioGateError(
                f"Invalid v4 history fingerprint journal row {line_number}."
            ) from error
        task = tasks.get(result.identity)
        if (
            task is None
            or result.identity in results
            or result.manifest_audio_sha256 != task.manifest_audio_sha256
            or result.relative_path != task.relative_path
        ):
            raise V4AudioGateError(
                f"v4 history fingerprint journal row {line_number} is not task-bound."
            )
        if result.status == "fingerprinted":
            _require_sha256(result.canonical_audio_sha256, "history canonical SHA-256")
            _require_fingerprint(result.audio_fingerprint_v1)
        source = Path(task.source_path)
        if not source.is_file() or sha256_file(source) != task.manifest_audio_sha256:
            raise V4AudioGateError(
                f"v4 history source binding failed for {result.identity!r}."
            )
        results[result.identity] = result
    return results


def append_v4_history_fingerprint_journal(
    path: Path,
    result: V4HistoryFingerprintResult,
) -> None:
    if not path.parent.is_dir():
        raise V4AudioGateError("v4 history fingerprint journal parent does not exist.")
    try:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(result.as_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    except OSError as error:
        raise V4AudioGateError("Cannot append v4 history fingerprint journal.") from error


def run_v4_decode_task(task: V4DecodeTask) -> V4DecodeResult:
    """Worker-safe canonical decode and QA operation with no destination overwrite."""

    source = Path(task.source_path)
    destination = Path(task.destination_path)
    try:
        if not source.is_file() or sha256_file(source) != task.raw_sha256:
            raise V4AudioGateError("Raw source asset binding failed.")
        if destination.exists():
            return _analyze_canonical_destination(task, destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{destination.stem}-", dir=destination.parent
        ) as stage_name:
            staged = Path(stage_name) / destination.name
            client = FFmpegClient()
            client.probe(source, _v4_audio_limits())
            client.normalize_to_wav(source, staged, 16_000)
            result = _analyze_canonical_destination(task, staged)
            try:
                os.link(staged, destination)
            except FileExistsError:
                existing = _analyze_canonical_destination(task, destination)
                if existing.decoded_audio_sha256 != result.decoded_audio_sha256:
                    raise V4AudioGateError(
                        "Conflicting v4 decoded destination appeared."
                    ) from None
                return existing
            return result
    except V4AudioGateError:
        raise
    except (AudioPipelineError, OSError, RuntimeError, ValueError) as error:
        reason = error.code.value if isinstance(error, AudioPipelineError) else "decode_or_qa_error"
        return V4DecodeResult(
            sample_id=task.sample_id,
            raw_relative_path=task.raw_relative_path,
            raw_sha256=task.raw_sha256,
            decoded_relative_path="",
            decoded_audio_sha256="",
            decoded_size_bytes=0,
            duration_s=0.0,
            peak=0.0,
            rms_dbfs=0.0,
            clipped_fraction=0.0,
            dc_offset=0.0,
            speech_seconds=0.0,
            speech_segment_count=0,
            audio_fingerprint_v1="",
            preparation_status="decode_error",
            rejection_reason=f"{reason}: {error}",
        )


def run_v4_history_fingerprint_task(
    task: V4HistoryFingerprintTask,
) -> V4HistoryFingerprintResult:
    """Fingerprint one available historical asset without publishing normalized history bytes."""

    source = Path(task.source_path)
    if not source.is_file() or sha256_file(source) != task.manifest_audio_sha256:
        raise V4AudioGateError("Historical manifest asset binding failed.")
    try:
        try:
            metrics = _canonical_metrics(source)
        except AudioPipelineError:
            with tempfile.TemporaryDirectory(prefix="kds-v4-history-audio-") as stage_name:
                normalized = Path(stage_name) / "normalized.wav"
                client = FFmpegClient()
                client.probe(source, _v4_audio_limits())
                client.normalize_to_wav(source, normalized, 16_000)
                metrics = _canonical_metrics(normalized)
        return V4HistoryFingerprintResult(
            identity=task.identity,
            manifest_audio_sha256=task.manifest_audio_sha256,
            relative_path=task.relative_path,
            canonical_audio_sha256=metrics.decoded_audio_sha256,
            duration_s=metrics.duration_s,
            speech_seconds=metrics.speech_seconds,
            audio_fingerprint_v1=metrics.audio_fingerprint_v1,
            status="fingerprinted",
            detail="",
        )
    except (AudioPipelineError, OSError, RuntimeError, ValueError) as error:
        reason = error.code.value if isinstance(error, AudioPipelineError) else "decode_error"
        return V4HistoryFingerprintResult(
            identity=task.identity,
            manifest_audio_sha256=task.manifest_audio_sha256,
            relative_path=task.relative_path,
            canonical_audio_sha256="",
            duration_s=0.0,
            speech_seconds=0.0,
            audio_fingerprint_v1="",
            status="unavailable_for_fingerprint",
            detail=f"{reason}: {error}",
        )


def _analyze_canonical_destination(task: V4DecodeTask, path: Path) -> V4DecodeResult:
    metrics = _canonical_metrics(path)
    return V4DecodeResult(
        sample_id=task.sample_id,
        raw_relative_path=task.raw_relative_path,
        raw_sha256=task.raw_sha256,
        decoded_relative_path=task.decoded_relative_path,
        decoded_audio_sha256=metrics.decoded_audio_sha256,
        decoded_size_bytes=metrics.decoded_size_bytes,
        duration_s=metrics.duration_s,
        peak=metrics.peak,
        rms_dbfs=metrics.rms_dbfs,
        clipped_fraction=metrics.clipped_fraction,
        dc_offset=metrics.dc_offset,
        speech_seconds=metrics.speech_seconds,
        speech_segment_count=metrics.speech_segment_count,
        audio_fingerprint_v1=metrics.audio_fingerprint_v1,
        preparation_status=metrics.preparation_status,
        rejection_reason=metrics.rejection_reason,
    )


def _canonical_metrics(path: Path) -> _CanonicalMetrics:
    waveform = read_pcm16_mono_wav(path, 16_000)
    quality = measure_quality(waveform)
    segments = tuple(WebRtcVadDetector().detect(waveform))
    speech_seconds = sum(item.duration_seconds for item in segments)
    policy = QualityPolicy()
    if quality.rms_dbfs < policy.min_rms_dbfs:
        status = PreparationStatus.REJECTED_QUALITY.value
        rejection = "signal_too_quiet"
    elif quality.clipped_fraction > policy.max_clipped_fraction:
        status = PreparationStatus.REJECTED_QUALITY.value
        rejection = "excessive_clipping"
    elif speech_seconds < _v4_audio_limits().minimum_speech_seconds:
        status = PreparationStatus.INSUFFICIENT_SPEECH.value
        rejection = "insufficient_speech"
    else:
        status = PreparationStatus.READY.value
        rejection = ""
    return _CanonicalMetrics(
        decoded_audio_sha256=sha256_file(path),
        decoded_size_bytes=path.stat().st_size,
        duration_s=waveform.duration_seconds,
        peak=quality.peak,
        rms_dbfs=quality.rms_dbfs,
        clipped_fraction=quality.clipped_fraction,
        dc_offset=quality.dc_offset,
        speech_seconds=speech_seconds,
        speech_segment_count=len(segments),
        audio_fingerprint_v1=canonical_audio_fingerprint(waveform, segments),
        preparation_status=status,
        rejection_reason=rejection,
    )


def _v4_audio_limits() -> AudioLimits:
    return AudioLimits()


def _result_matches_task(result: V4DecodeResult, task: V4DecodeTask) -> bool:
    if (
        result.raw_relative_path != task.raw_relative_path
        or result.raw_sha256 != task.raw_sha256
        or result.decoded_relative_path not in {"", task.decoded_relative_path}
    ):
        return False
    if result.decoded_audio_sha256:
        try:
            _require_sha256(result.decoded_audio_sha256, "decoded SHA-256")
            _require_fingerprint(result.audio_fingerprint_v1)
        except V4AudioGateError:
            return False
    return True


def _validate_signature(signature: V4AudioSignature) -> None:
    if not signature.identity or signature.speech_seconds < 0:
        raise V4AudioGateError("Invalid near-audio signature metadata.")
    _require_sha256(signature.audio_sha256, "signature audio SHA-256")
    _require_fingerprint(signature.fingerprint)


def _require_fingerprint(value: str) -> None:
    if len(value) != V4_AUDIO_FINGERPRINT_HEX_LENGTH:
        raise V4AudioGateError("Audio fingerprint must contain 256 bits.")
    try:
        bytes.fromhex(value)
    except ValueError as error:
        raise V4AudioGateError("Audio fingerprint must be lowercase hexadecimal.") from error
    if value != value.lower():
        raise V4AudioGateError("Audio fingerprint must be lowercase hexadecimal.")


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise V4AudioGateError(f"{label} is invalid.")


def decode_tasks_by_id(tasks: Iterable[V4DecodeTask]) -> dict[str, V4DecodeTask]:
    result: dict[str, V4DecodeTask] = {}
    for task in tasks:
        _require_sha256(task.raw_sha256, "task raw SHA-256")
        if (
            not task.sample_id
            or task.sample_id in result
            or task.decoded_relative_path != decoded_relative_path(task.raw_sha256)
        ):
            raise V4AudioGateError("Invalid or duplicate v4 decode task.")
        result[task.sample_id] = task
    return result
