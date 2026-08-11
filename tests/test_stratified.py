from __future__ import annotations

from pathlib import Path
from typing import cast

import soundfile as sf  # type: ignore[import-untyped]
import torch

from kds.data.manifest import ManifestRow
from kds.eval.stratified import evaluate_b0_with_strata, stratum_keys
from kds.models import B0LogMelCnn
from tests.factories import manifest_mapping


class _FixedLogitModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        self.forward_calls += 1
        return torch.full((waveforms.shape[0],), 0.1, device=waveforms.device)


def test_stratum_keys_keep_bonafide_source_separate_from_spoof_provenance() -> None:
    bonafide = ManifestRow.from_mapping(
        manifest_mapping(label="bonafide", source_name="recording-corpus"), row_number=2
    )
    spoof = ManifestRow.from_mapping(
        manifest_mapping(
            label="spoof",
            source_name="synthesis-corpus",
            generator_family="formant",
            generator_name="eSpeak NG",
            generator_version="1.52",
            voice_id="fixed-control",
        ),
        row_number=2,
    )

    assert stratum_keys(bonafide) == ("bonafide_source:recording-corpus",)
    assert stratum_keys(spoof) == (
        "spoof_generator_family:formant",
        "spoof_voice_id:fixed-control",
    )


def test_evaluate_b0_with_strata_uses_one_loader_traversal(tmp_path: Path) -> None:
    for name in ("bonafide.wav", "spoof-1.wav", "spoof-2.wav"):
        sf.write(tmp_path / name, [0.01] * 1_600, 16_000)
    rows = [
        ManifestRow.from_mapping(
            manifest_mapping(
                sample_id="bonafide-1",
                relative_path="bonafide.wav",
                label="bonafide",
                source_name="recording-corpus",
            ),
            row_number=2,
        ),
        ManifestRow.from_mapping(
            manifest_mapping(
                sample_id="spoof-1",
                relative_path="spoof-1.wav",
                label="spoof",
                source_name="synthesis-corpus",
                generator_family="formant",
                generator_name="eSpeak NG",
                generator_version="1.52",
                voice_id="fixed-control",
            ),
            row_number=3,
        ),
        ManifestRow.from_mapping(
            manifest_mapping(
                sample_id="spoof-2",
                relative_path="spoof-2.wav",
                label="spoof",
                source_name="synthesis-corpus",
                generator_family="formant",
                generator_name="eSpeak NG",
                generator_version="1.52",
                voice_id="fixed-control",
            ),
            row_number=4,
        ),
    ]
    model = _FixedLogitModel()

    result, metrics = evaluate_b0_with_strata(
        cast(B0LogMelCnn, model),
        rows,
        audio_root=tmp_path,
        batch_size=1,
        seed="test",
        device=torch.device("cpu"),
        num_workers=0,
    )

    assert model.forward_calls == 3
    assert result.correct == 2
    assert metrics["bonafide_source:recording-corpus"]["correct"] == 0
    assert metrics["spoof_generator_family:formant"]["correct"] == 2
    assert metrics["spoof_voice_id:fixed-control"]["examples"] == 2
