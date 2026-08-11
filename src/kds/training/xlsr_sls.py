"""Stage-A/Stage-B training primitives for XLS-R and the SLS head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from kds.models import XlsrSlsClassifier
from kds.training.b0 import AudioBatch, EpochResult

XlsrPrecision = Literal["fp32", "bf16"]


@dataclass(frozen=True, slots=True)
class XlsrStageBConfigurationReport:
    encoder_blocks: int
    trainable_encoder_blocks: tuple[int, ...]
    trainable_encoder_parameters: int
    trainable_head_parameters: int
    gradient_checkpointing: bool


def freeze_xlsr_encoder(model: XlsrSlsClassifier) -> None:
    """Freeze the SSL encoder; Stage A trains only layer-mix/pooling/classifier parameters."""

    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    for parameter in model.head.parameters():
        parameter.requires_grad_(True)
    model.encoder.eval()


def configure_xlsr_stage_b(
    model: XlsrSlsClassifier,
    *,
    last_encoder_blocks: int,
    gradient_checkpointing: bool,
) -> XlsrStageBConfigurationReport:
    """Freeze the XLS-R prefix and unfreeze exactly the requested final transformer blocks."""

    blocks = _xlsr_encoder_blocks(model)
    if last_encoder_blocks <= 0 or last_encoder_blocks > len(blocks):
        raise ValueError(
            f"last_encoder_blocks must be in [1, {len(blocks)}], got {last_encoder_blocks}."
        )
    for parameter in model.encoder.parameters():
        parameter.requires_grad_(False)
    first_trainable = len(blocks) - last_encoder_blocks
    for block in blocks[first_trainable:]:
        for parameter in block.parameters():
            parameter.requires_grad_(True)
    for parameter in model.head.parameters():
        parameter.requires_grad_(True)

    checkpointing_method = getattr(
        model.encoder,
        "gradient_checkpointing_enable"
        if gradient_checkpointing
        else "gradient_checkpointing_disable",
        None,
    )
    if checkpointing_method is None or not callable(checkpointing_method):
        raise ValueError("XLS-R encoder does not expose gradient checkpointing controls.")
    checkpointing_method()
    _set_stage_b_modes(model, blocks, first_trainable)
    return XlsrStageBConfigurationReport(
        encoder_blocks=len(blocks),
        trainable_encoder_blocks=tuple(range(first_trainable, len(blocks))),
        trainable_encoder_parameters=sum(
            parameter.numel() for parameter in model.encoder.parameters() if parameter.requires_grad
        ),
        trainable_head_parameters=sum(parameter.numel() for parameter in model.head.parameters()),
        gradient_checkpointing=gradient_checkpointing,
    )


def train_xlsr_sls_head_epoch(
    model: XlsrSlsClassifier,
    data_loader: DataLoader[AudioBatch],
    optimizer: Optimizer,
    device: torch.device,
    *,
    precision: XlsrPrecision,
    gradient_accumulation_steps: int,
    gradient_clip_norm: float,
    max_batches: int | None = None,
) -> EpochResult:
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    if gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive.")
    _validate_precision(precision, device)
    freeze_xlsr_encoder(model)
    model.train()
    # model.train() recursively changes the frozen encoder; keep its dropout/statistics fixed.
    model.encoder.eval()
    optimizer.zero_grad(set_to_none=True)
    return _run_epoch(
        model,
        data_loader,
        device,
        precision=precision,
        optimizer=optimizer,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_clip_norm=gradient_clip_norm,
        parameters_to_clip=tuple(model.head.parameters()),
        max_batches=max_batches,
    )


def train_xlsr_sls_stage_b_epoch(
    model: XlsrSlsClassifier,
    data_loader: DataLoader[AudioBatch],
    optimizer: Optimizer,
    device: torch.device,
    *,
    precision: XlsrPrecision,
    gradient_accumulation_steps: int,
    gradient_clip_norm: float,
    last_encoder_blocks: int,
    gradient_checkpointing: bool,
    max_batches: int | None = None,
) -> EpochResult:
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    if gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive.")
    _validate_precision(precision, device)
    configure_xlsr_stage_b(
        model,
        last_encoder_blocks=last_encoder_blocks,
        gradient_checkpointing=gradient_checkpointing,
    )
    optimizer.zero_grad(set_to_none=True)
    trainable_parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    return _run_epoch(
        model,
        data_loader,
        device,
        precision=precision,
        optimizer=optimizer,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_clip_norm=gradient_clip_norm,
        parameters_to_clip=trainable_parameters,
        max_batches=max_batches,
    )


def evaluate_xlsr_sls(
    model: XlsrSlsClassifier,
    data_loader: DataLoader[AudioBatch],
    device: torch.device,
    *,
    precision: XlsrPrecision,
    max_batches: int | None = None,
) -> EpochResult:
    _validate_precision(precision, device)
    model.eval()
    with torch.inference_mode():
        return _run_epoch(
            model,
            data_loader,
            device,
            precision=precision,
            optimizer=None,
            gradient_accumulation_steps=1,
            gradient_clip_norm=1.0,
            parameters_to_clip=None,
            max_batches=max_batches,
        )


def _run_epoch(
    model: XlsrSlsClassifier,
    data_loader: DataLoader[AudioBatch],
    device: torch.device,
    *,
    precision: XlsrPrecision,
    optimizer: Optimizer | None,
    gradient_accumulation_steps: int,
    gradient_clip_norm: float,
    parameters_to_clip: tuple[nn.Parameter, ...] | None,
    max_batches: int | None,
) -> EpochResult:
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided.")
    available_batches = len(data_loader)
    batch_limit = min(available_batches, max_batches or available_batches)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    total_loss = 0.0
    correct = 0
    examples = 0
    bonafide_examples = 0
    bonafide_correct = 0
    spoof_examples = 0
    spoof_correct = 0
    accumulated_examples = 0

    for batch_index, batch in enumerate(data_loader):
        if batch_index >= batch_limit:
            break
        waveforms = batch.waveforms.to(device, non_blocking=True)
        labels = batch.labels.to(device, non_blocking=True)
        current_batch_size = labels.numel()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=precision == "bf16",
        ):
            logits = model(waveforms)
            loss_sum = criterion(logits, labels)
        if optimizer is not None:
            loss_sum.backward()
            accumulated_examples += current_batch_size
            update_due = (batch_index + 1) % gradient_accumulation_steps == 0
            last_batch = batch_index + 1 == batch_limit
            if update_due or last_batch:
                if not parameters_to_clip:
                    raise RuntimeError("Training requires parameters to clip.")
                if accumulated_examples <= 0:
                    raise RuntimeError("Gradient accumulation contains no examples.")
                for parameter in parameters_to_clip:
                    if parameter.grad is not None:
                        parameter.grad.div_(accumulated_examples)
                torch.nn.utils.clip_grad_norm_(parameters_to_clip, gradient_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accumulated_examples = 0

        total_loss += float(loss_sum.detach())
        predictions = logits.detach() >= 0.0
        expected_spoof = labels >= 0.5
        correct += int((predictions == expected_spoof).sum())
        bonafide_mask = ~expected_spoof
        spoof_mask = expected_spoof
        bonafide_examples += int(bonafide_mask.sum())
        bonafide_correct += int((predictions[bonafide_mask] == expected_spoof[bonafide_mask]).sum())
        spoof_examples += int(spoof_mask.sum())
        spoof_correct += int((predictions[spoof_mask] == expected_spoof[spoof_mask]).sum())
        examples += current_batch_size

    if examples == 0:
        raise ValueError("Data loader yielded no XLS-R batches.")
    bonafide_accuracy = bonafide_correct / bonafide_examples if bonafide_examples else None
    spoof_accuracy = spoof_correct / spoof_examples if spoof_examples else None
    balanced_accuracy = (
        (bonafide_accuracy + spoof_accuracy) / 2
        if bonafide_accuracy is not None and spoof_accuracy is not None
        else None
    )
    return EpochResult(
        loss=total_loss / examples,
        accuracy=correct / examples,
        correct=correct,
        examples=examples,
        bonafide_examples=bonafide_examples,
        spoof_examples=spoof_examples,
        bonafide_correct=bonafide_correct,
        spoof_correct=spoof_correct,
        bonafide_accuracy=bonafide_accuracy,
        spoof_accuracy=spoof_accuracy,
        balanced_accuracy=balanced_accuracy,
    )


def _validate_precision(precision: XlsrPrecision, device: torch.device) -> None:
    if precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be 'fp32' or 'bf16'.")
    if precision == "bf16" and device.type != "cuda":
        raise ValueError("bf16 XLS-R training requires CUDA in this project.")


def _xlsr_encoder_blocks(model: XlsrSlsClassifier) -> tuple[nn.Module, ...]:
    encoder_stack = getattr(model.encoder, "encoder", None)
    blocks = getattr(encoder_stack, "layers", None)
    if not isinstance(blocks, nn.ModuleList) or not blocks:
        raise ValueError("XLS-R encoder must expose a non-empty encoder.layers ModuleList.")
    return tuple(blocks)


def _set_stage_b_modes(
    model: XlsrSlsClassifier,
    blocks: tuple[nn.Module, ...],
    first_trainable: int,
) -> None:
    # Frozen prefix dropout must stay disabled; only the head and unfrozen tail train.
    model.eval()
    model.head.train()
    for block in blocks[first_trainable:]:
        block.train()
