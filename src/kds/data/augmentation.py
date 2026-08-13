"""Deterministic, label-agnostic waveform perturbations for research training only."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor


@dataclass(frozen=True, slots=True)
class SymmetricTrainAugmentation:
    """Pinned simulation of channel, codec and replay effects.

    The random-looking parameters are derived only from the policy namespace, epoch and
    sample ID.  A label is deliberately not an input, so bona-fide and spoof assets use the
    same policy and distribution.  This is a waveform simulation, not a claim that a real
    transmission codec or physical room impulse response was applied.
    """

    policy_id: str
    seed_namespace: str
    channel_gain_db_min: float
    channel_gain_db_max: float
    codec_resample_rates_hz: tuple[int, ...]
    codec_quantization_bits: int
    replay_delay_ms_min: float
    replay_delay_ms_max: float
    replay_attenuation_min: float
    replay_attenuation_max: float

    def __post_init__(self) -> None:
        if not self.policy_id or not self.seed_namespace:
            raise ValueError("Augmentation policy_id and seed_namespace must not be empty.")
        if self.channel_gain_db_min > self.channel_gain_db_max:
            raise ValueError("channel_gain_db_min must not exceed channel_gain_db_max.")
        if not self.codec_resample_rates_hz or any(
            rate <= 0 for rate in self.codec_resample_rates_hz
        ):
            raise ValueError("codec_resample_rates_hz must contain positive sample rates.")
        if tuple(sorted(set(self.codec_resample_rates_hz))) != self.codec_resample_rates_hz:
            raise ValueError("codec_resample_rates_hz must be sorted and unique.")
        if not 2 <= self.codec_quantization_bits <= 16:
            raise ValueError("codec_quantization_bits must be between 2 and 16.")
        if self.replay_delay_ms_min <= 0 or self.replay_delay_ms_min > self.replay_delay_ms_max:
            raise ValueError("Replay delay bounds must be positive and ordered.")
        if not 0 <= self.replay_attenuation_min <= self.replay_attenuation_max < 1:
            raise ValueError("Replay attenuation must be ordered in [0, 1).")


def apply_symmetric_train_augmentation(
    waveform: Tensor,
    *,
    sample_id: str,
    epoch: int,
    sample_rate: int,
    config: SymmetricTrainAugmentation,
) -> Tensor:
    """Return one deterministic full channel/codec/replay simulation chain.

    The caller must provide a mono one-dimensional waveform.  The operation is CPU/GPU agnostic
    and has no mutable RNG state, which makes a locked plan reproducible across data-loader
    workers.  It preserves length and clamps the final float waveform to the normalized range.
    """

    if waveform.ndim != 1 or waveform.numel() == 0:
        raise ValueError("Augmentation requires a non-empty one-dimensional waveform.")
    if not sample_id:
        raise ValueError("Augmentation requires a non-empty sample_id.")
    if epoch < 0 or sample_rate <= 0:
        raise ValueError("Augmentation epoch and sample_rate must be positive/non-negative.")

    gain_db = _uniform(
        config,
        sample_id,
        epoch,
        "channel_gain_db",
        config.channel_gain_db_min,
        config.channel_gain_db_max,
    )
    signal = waveform * math.pow(10.0, gain_db / 20.0)

    rate_index = _index(
        config,
        sample_id,
        epoch,
        "codec_resample_rate",
        len(config.codec_resample_rates_hz),
    )
    signal = _codec_simulation(
        signal,
        source_sample_rate=sample_rate,
        reduced_sample_rate=config.codec_resample_rates_hz[rate_index],
        quantization_bits=config.codec_quantization_bits,
    )

    delay_ms = _uniform(
        config,
        sample_id,
        epoch,
        "replay_delay_ms",
        config.replay_delay_ms_min,
        config.replay_delay_ms_max,
    )
    attenuation = _uniform(
        config,
        sample_id,
        epoch,
        "replay_attenuation",
        config.replay_attenuation_min,
        config.replay_attenuation_max,
    )
    return _replay_simulation(
        signal,
        delay_samples=max(1, round(delay_ms * sample_rate / 1_000.0)),
        attenuation=attenuation,
    ).clamp(-1.0, 1.0)


def _uniform(
    config: SymmetricTrainAugmentation,
    sample_id: str,
    epoch: int,
    component: str,
    lower: float,
    upper: float,
) -> float:
    if lower == upper:
        return lower
    fraction = _fraction(config, sample_id, epoch, component)
    return lower + (upper - lower) * fraction


def _index(
    config: SymmetricTrainAugmentation,
    sample_id: str,
    epoch: int,
    component: str,
    length: int,
) -> int:
    if length <= 0:
        raise ValueError("Augmentation index length must be positive.")
    return min(length - 1, int(_fraction(config, sample_id, epoch, component) * length))


def _fraction(
    config: SymmetricTrainAugmentation,
    sample_id: str,
    epoch: int,
    component: str,
) -> float:
    identity = (
        f"{config.seed_namespace}:{config.policy_id}:{epoch}:{sample_id}:{component}"
    ).encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "big") / float(2**64)


def _codec_simulation(
    waveform: Tensor,
    *,
    source_sample_rate: int,
    reduced_sample_rate: int,
    quantization_bits: int,
) -> Tensor:
    length = waveform.numel()
    reduced_length = max(1, round(length * reduced_sample_rate / source_sample_rate))
    signal = waveform.reshape(1, 1, length)
    reduced = functional.interpolate(
        signal, size=reduced_length, mode="linear", align_corners=False
    )
    restored = functional.interpolate(
        reduced, size=length, mode="linear", align_corners=False
    ).reshape(-1)
    levels = (1 << quantization_bits) - 1
    limited = restored.clamp(-1.0, 1.0)
    return (torch.round((limited + 1.0) * levels / 2.0) * 2.0 / levels) - 1.0


def _replay_simulation(waveform: Tensor, *, delay_samples: int, attenuation: float) -> Tensor:
    if delay_samples >= waveform.numel():
        return waveform
    delayed = torch.zeros_like(waveform)
    delayed[delay_samples:] = waveform[:-delay_samples]
    return waveform + attenuation * delayed
