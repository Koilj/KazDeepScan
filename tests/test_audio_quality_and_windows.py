from __future__ import annotations

import unittest
from array import array

from kds.audio.contracts import SpeechSegment
from kds.audio.vad import EnergyVadDetector
from kds.audio.waveform import Waveform, measure_quality
from kds.audio.windows import WindowConfig, build_inference_windows


class AudioQualityAndWindowsTests(unittest.TestCase):
    def test_measures_signal_and_detects_three_seconds_of_speech(self) -> None:
        waveform = Waveform(array("h", [8_000] * (16_000 * 3)), sample_rate=16_000)

        quality = measure_quality(waveform)
        segments = EnergyVadDetector().detect(waveform)

        self.assertAlmostEqual(quality.peak, 8_000 / 32_768)
        self.assertLess(quality.rms_dbfs, -11.0)
        self.assertGreater(quality.rms_dbfs, -13.0)
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].duration_seconds, 3.0)

    def test_short_speech_window_retains_real_length_for_model_mask(self) -> None:
        segment = SpeechSegment(0, 48_000, 16_000)

        windows = build_inference_windows([segment])

        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0].real_samples, 48_000)
        self.assertEqual(windows[0].target_samples, 64_600)

    def test_long_segment_has_no_duplicate_final_window(self) -> None:
        config = WindowConfig(samples=10, hop_samples=5)
        segment = SpeechSegment(0, 20, 16_000)

        windows = build_inference_windows([segment], config)

        bounds = [(window.start_sample, window.end_sample) for window in windows]
        self.assertEqual(bounds, [(0, 10), (5, 15), (10, 20)])


if __name__ == "__main__":
    unittest.main()
