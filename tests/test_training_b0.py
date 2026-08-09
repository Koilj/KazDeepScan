from __future__ import annotations

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from kds.data.dataset import AudioSample
from kds.models import B0Config, B0LogMelCnn
from kds.training import collate_audio_samples, evaluate_b0, train_b0_epoch


def _sample(sample_id: str, label: float) -> AudioSample:
    return AudioSample(
        waveform=torch.randn(1_600, dtype=torch.float32) * 0.01,
        label=torch.tensor(label, dtype=torch.float32),
        sample_id=sample_id,
        parent_group_id=f"parent-{sample_id}",
        language="ru",
    )


def test_b0_train_and_eval_epoch_return_aggregate_metrics() -> None:
    samples = [_sample("a", 0.0), _sample("b", 1.0), _sample("c", 0.0), _sample("d", 1.0)]
    loader = DataLoader(samples, batch_size=2, collate_fn=collate_audio_samples)
    model = B0LogMelCnn(B0Config(n_mels=32))
    optimizer = AdamW(model.parameters(), lr=1e-4)

    train_result = train_b0_epoch(model, loader, optimizer, torch.device("cpu"))
    evaluation = evaluate_b0(model, loader, torch.device("cpu"))

    assert train_result.examples == 4
    assert evaluation.examples == 4
    assert 0.0 <= evaluation.accuracy <= 1.0
    assert evaluation.loss >= 0.0
    assert evaluation.bonafide_examples == 2
    assert evaluation.spoof_examples == 2
    assert evaluation.balanced_accuracy is not None


def test_b0_evaluation_marks_balanced_accuracy_unavailable_for_one_class() -> None:
    samples = [_sample("spoof-a", 1.0), _sample("spoof-b", 1.0)]
    loader = DataLoader(samples, batch_size=2, collate_fn=collate_audio_samples)
    model = B0LogMelCnn(B0Config(n_mels=32))

    evaluation = evaluate_b0(model, loader, torch.device("cpu"))

    assert evaluation.bonafide_examples == 0
    assert evaluation.bonafide_accuracy is None
    assert evaluation.spoof_examples == 2
    assert evaluation.spoof_accuracy is not None
    assert evaluation.balanced_accuracy is None
