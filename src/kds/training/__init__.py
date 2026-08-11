"""Reproducible training loops for approved, manifest-backed data."""

from kds.training.b0 import (
    AudioBatch,
    EpochResult,
    collate_audio_samples,
    evaluate_b0,
    make_audio_loader,
    train_b0_epoch,
)
from kds.training.xlsr_sls import (
    XlsrStageBConfigurationReport,
    configure_xlsr_stage_b,
    evaluate_xlsr_sls,
    freeze_xlsr_encoder,
    train_xlsr_sls_head_epoch,
    train_xlsr_sls_stage_b_epoch,
)

__all__ = [
    "AudioBatch",
    "EpochResult",
    "XlsrStageBConfigurationReport",
    "collate_audio_samples",
    "configure_xlsr_stage_b",
    "evaluate_b0",
    "evaluate_xlsr_sls",
    "freeze_xlsr_encoder",
    "make_audio_loader",
    "train_b0_epoch",
    "train_xlsr_sls_head_epoch",
    "train_xlsr_sls_stage_b_epoch",
]
