from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from kds.audio.contracts import SpeechSegment, WindowDescriptor


@dataclass(frozen=True, slots=True)
class WindowConfig:
    samples: int = 64_600
    hop_samples: int = 32_000

    def __post_init__(self) -> None:
        if self.samples <= 0 or self.hop_samples <= 0:
            raise ValueError("Window and hop sizes must be positive.")


def build_inference_windows(
    speech_segments: Iterable[SpeechSegment], config: WindowConfig | None = None
) -> list[WindowDescriptor]:
    """Cover each speech span using full windows plus one right-aligned final window."""

    config = config or WindowConfig()
    windows: list[WindowDescriptor] = []
    for segment in speech_segments:
        length = segment.end_sample - segment.start_sample
        if length <= 0:
            continue
        if length <= config.samples:
            windows.append(
                WindowDescriptor(
                    start_sample=segment.start_sample,
                    end_sample=segment.end_sample,
                    target_samples=config.samples,
                    sample_rate=segment.sample_rate,
                )
            )
            continue

        starts = list(
            range(
                segment.start_sample,
                segment.end_sample - config.samples + 1,
                config.hop_samples,
            )
        )
        final_start = segment.end_sample - config.samples
        if starts[-1] != final_start:
            starts.append(final_start)
        windows.extend(
            WindowDescriptor(
                start_sample=start,
                end_sample=start + config.samples,
                target_samples=config.samples,
                sample_rate=segment.sample_rate,
            )
            for start in starts
        )

    return sorted(set(windows), key=lambda window: (window.start_sample, window.end_sample))
