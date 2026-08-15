"""Run the write-once XLS-R+SLS model-v4 bilingual research training contract."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import tempfile
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import torch
import transformers
from torch.optim import AdamW
from torch.utils.data import DataLoader

from kds.data.assets import require_valid_assets
from kds.data.dataset import DatasetConfig, ManifestAudioDataset
from kds.data.licenses import load_license_ledger
from kds.models import XlsrSlsClassifier
from kds.training import (
    collate_audio_samples,
    configure_xlsr_stage_b,
    evaluate_xlsr_sls,
    freeze_xlsr_encoder,
    train_xlsr_sls_head_epoch,
    train_xlsr_sls_stage_b_epoch,
)
from kds.training.b0 import AudioBatch
from kds.training.frozen_b0 import epoch_metrics, state_dict_sha256
from kds.training.v4_training_plan import (
    SelectedV4Rows,
    V4BalancedCellBatchSampler,
    V4TrainingPlan,
    load_v4_training_plan,
    v4_training_plan_record,
    validate_and_select_v4_training,
)


def _cuda_device(plan: V4TrainingPlan) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("v4 XLS-R+SLS training requires CUDA, but CUDA is unavailable.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("v4 XLS-R+SLS training requires CUDA BF16 support.")
    if torch.__version__ != plan.runtime.torch_version:
        raise RuntimeError(
            f"v4 runtime lock mismatch for torch: expected {plan.runtime.torch_version}, "
            f"got {torch.__version__}."
        )
    if torch.version.cuda != plan.runtime.cuda_runtime:
        raise RuntimeError(
            f"v4 runtime lock mismatch for CUDA: expected {plan.runtime.cuda_runtime}, "
            f"got {torch.version.cuda}."
        )
    if transformers.__version__ != plan.runtime.transformers_version:
        raise RuntimeError(
            "v4 runtime lock mismatch for transformers: "
            f"expected {plan.runtime.transformers_version}, got {transformers.__version__}."
        )
    if platform.python_version() != plan.runtime.python_version:
        raise RuntimeError(
            f"v4 runtime lock mismatch for Python: expected {plan.runtime.python_version}, "
            f"got {platform.python_version()}."
        )
    return torch.device("cuda")


def _require_unexecuted_outputs(plan: V4TrainingPlan) -> None:
    for label, path in (
        ("checkpoint", plan.outputs.checkpoint),
        ("report", plan.outputs.report),
        ("execution lock", plan.outputs.execution_lock),
    ):
        if os.path.lexists(path):
            raise ValueError(f"Refusing to repeat or overwrite v4 training; {label} exists: {path}")
    plan.outputs.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    plan.outputs.execution_lock.parent.mkdir(parents=True, exist_ok=True)
    if not plan.outputs.report.parent.is_dir():
        raise ValueError(f"v4 report directory does not exist: {plan.outputs.report.parent}")


def _write_json_new(path: Path, value: dict[str, object]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError(f"Refusing to overwrite v4 output: {path}") from error
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_publish(
    checkpoint_path: Path,
    report_path: Path,
    checkpoint: dict[str, object],
    report: dict[str, object],
) -> None:
    checkpoint_temporary: Path | None = None
    report_temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=checkpoint_path.parent,
            prefix=f".{checkpoint_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            checkpoint_temporary = Path(handle.name)
            torch.save(checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=report_path.parent,
            prefix=f".{report_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            report_temporary = Path(handle.name)
            json.dump(report, handle, ensure_ascii=False, allow_nan=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        for temporary, destination in (
            (checkpoint_temporary, checkpoint_path),
            (report_temporary, report_path),
        ):
            try:
                os.link(temporary, destination)
            except FileExistsError as error:
                raise ValueError(f"Refusing to overwrite v4 output: {destination}") from error
        checkpoint_temporary.unlink()
        checkpoint_temporary = None
        report_temporary.unlink()
        report_temporary = None
    finally:
        if checkpoint_temporary is not None:
            checkpoint_temporary.unlink(missing_ok=True)
        if report_temporary is not None:
            report_temporary.unlink(missing_ok=True)


def _environment_record(device: torch.device) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_name": properties.name,
        "device_total_memory_bytes": properties.total_memory,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }


def _seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_model(plan: V4TrainingPlan, device: torch.device) -> XlsrSlsClassifier:
    model = XlsrSlsClassifier.from_pretrained(
        str(plan.encoder.checkpoint_dir),
        attention_size=plan.head.attention_size,
        classifier_size=plan.head.classifier_size,
        dropout=plan.head.dropout,
        local_files_only=True,
    )
    return model.to(device)


def _loaders(
    plan: V4TrainingPlan, selected: SelectedV4Rows, audio_root: Path
) -> tuple[
    ManifestAudioDataset,
    V4BalancedCellBatchSampler,
    DataLoader[AudioBatch],
    DataLoader[AudioBatch],
    DataLoader[AudioBatch],
]:
    seed = str(plan.training.seed)
    train_dataset = ManifestAudioDataset(
        list(selected.train),
        DatasetConfig(
            audio_root=audio_root,
            sample_rate=plan.training.sample_rate,
            window_samples=plan.training.window_samples,
            mode="train",
            seed=seed,
            augmentation=plan.training.augmentation,
        ),
    )
    sampler = V4BalancedCellBatchSampler(
        selected.train, batch_size=plan.training.batch_size, seed=plan.training.seed
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=plan.training.num_workers,
        collate_fn=collate_audio_samples,
        pin_memory=plan.training.pin_memory,
    )
    ru_loader = DataLoader(
        ManifestAudioDataset(
            list(selected.dev_ru),
            DatasetConfig(
                audio_root=audio_root,
                sample_rate=plan.training.sample_rate,
                window_samples=plan.training.window_samples,
                mode="eval",
                seed=seed,
            ),
        ),
        batch_size=plan.training.batch_size,
        shuffle=False,
        num_workers=plan.training.num_workers,
        collate_fn=collate_audio_samples,
        pin_memory=plan.training.pin_memory,
    )
    kk_loader = DataLoader(
        ManifestAudioDataset(
            list(selected.dev_kk),
            DatasetConfig(
                audio_root=audio_root,
                sample_rate=plan.training.sample_rate,
                window_samples=plan.training.window_samples,
                mode="eval",
                seed=seed,
            ),
        ),
        batch_size=plan.training.batch_size,
        shuffle=False,
        num_workers=plan.training.num_workers,
        collate_fn=collate_audio_samples,
        pin_memory=plan.training.pin_memory,
    )
    return (
        train_dataset,
        sampler,
        cast(DataLoader[AudioBatch], train_loader),
        cast(DataLoader[AudioBatch], ru_loader),
        cast(DataLoader[AudioBatch], kk_loader),
    )


def _macro_dev(
    model: XlsrSlsClassifier,
    ru_loader: DataLoader[AudioBatch],
    kk_loader: DataLoader[AudioBatch],
    device: torch.device,
    *,
    max_batches: int | None = None,
) -> tuple[float, dict[str, dict[str, object]]]:
    ru = evaluate_xlsr_sls(model, ru_loader, device, precision="bf16", max_batches=max_batches)
    kk = evaluate_xlsr_sls(model, kk_loader, device, precision="bf16", max_batches=max_batches)
    return (ru.loss + kk.loss) / 2.0, {"ru": epoch_metrics(ru), "kk": epoch_metrics(kk)}


def _preflight(
    plan: V4TrainingPlan, selected: SelectedV4Rows, device: torch.device
) -> dict[str, object]:
    return {
        "status": "validated",
        "mode": "validate_only",
        "run_plan": v4_training_plan_record(plan),
        "assets_validated": len(selected.train) + len(selected.dev_ru) + len(selected.dev_kk),
        "environment": _environment_record(device),
        "frozen_final_evaluation_performed": False,
        "calibration_performed": False,
        "final_inference_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the write-once, hash-pinned v4 XLS-R+SLS training contract."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--validate-only", action="store_true", help="Validate only; no forward pass or outputs."
    )
    mode.add_argument(
        "--profile-only",
        action="store_true",
        help="One non-selecting tail-unfreeze train/dev batch; no outputs.",
    )
    arguments = parser.parse_args()

    plan = load_v4_training_plan(arguments.plan)
    _require_unexecuted_outputs(plan)
    device = _cuda_device(plan)
    ledger = load_license_ledger(plan.license_ledger.path)
    protocol, selected = validate_and_select_v4_training(plan, ledger)
    require_valid_assets(
        [*selected.train, *selected.dev_ru, *selected.dev_kk], arguments.audio_root
    )
    preflight = _preflight(plan, selected, device)
    preflight["protocol"] = asdict(protocol)
    if arguments.validate_only:
        print(json.dumps(preflight, ensure_ascii=False, allow_nan=False))
        return 0

    _seed(plan.training.seed)
    train_dataset, sampler, train_loader, ru_loader, kk_loader = _loaders(
        plan, selected, arguments.audio_root
    )
    model = _build_model(plan, device)
    if arguments.profile_only:
        configuration = configure_xlsr_stage_b(
            model,
            last_encoder_blocks=plan.training.last_encoder_blocks,
            gradient_checkpointing=True,
        )
        optimizer = AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=plan.training.encoder_learning_rate,
            weight_decay=plan.training.weight_decay,
        )
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        train_dataset.set_epoch(0)
        sampler.set_epoch(0)
        train_result = train_xlsr_sls_stage_b_epoch(
            model,
            train_loader,
            optimizer,
            device,
            precision="bf16",
            gradient_accumulation_steps=plan.training.gradient_accumulation_steps,
            gradient_clip_norm=plan.training.gradient_clip_norm,
            last_encoder_blocks=plan.training.last_encoder_blocks,
            gradient_checkpointing=True,
            max_batches=1,
        )
        macro_loss, dev_by_language = _macro_dev(model, ru_loader, kk_loader, device, max_batches=1)
        torch.cuda.synchronize(device)
        print(
            json.dumps(
                {
                    **preflight,
                    "status": "profiled",
                    "mode": "profile_only",
                    "elapsed_seconds": time.monotonic() - started,
                    "stage_b_configuration": asdict(configuration),
                    "train_batch": epoch_metrics(train_result),
                    "dev_batch_by_language": dev_by_language,
                    "profile_macro_language_dev_loss": macro_loss,
                    "cuda_memory": {
                        "allocated_bytes": torch.cuda.memory_allocated(device),
                        "reserved_bytes": torch.cuda.memory_reserved(device),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                    },
                    "artifacts_published": False,
                    "checkpoint_selection_performed": False,
                },
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        return 0

    started_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    _write_json_new(
        plan.outputs.execution_lock,
        {
            "run_id": plan.run_id,
            "plan_sha256": plan.plan_sha256,
            "status": "started",
            "started_at": started_at,
            "checkpoint": str(plan.outputs.checkpoint),
            "report": str(plan.outputs.report),
        },
    )
    started = time.monotonic()
    history: list[dict[str, object]] = []
    freeze_xlsr_encoder(model)
    warmup_optimizer = AdamW(
        model.head.parameters(),
        lr=plan.training.head_learning_rate,
        weight_decay=plan.training.weight_decay,
    )
    for epoch_index in range(plan.training.warmup_epochs):
        train_dataset.set_epoch(epoch_index)
        sampler.set_epoch(epoch_index)
        train_result = train_xlsr_sls_head_epoch(
            model,
            train_loader,
            warmup_optimizer,
            device,
            precision="bf16",
            gradient_accumulation_steps=plan.training.gradient_accumulation_steps,
            gradient_clip_norm=plan.training.gradient_clip_norm,
        )
        record = {
            "phase": "warmup",
            "epoch": epoch_index + 1,
            "train": epoch_metrics(train_result),
            "checkpoint_selected": False,
        }
        history.append(record)
        print(json.dumps({"progress": record}, ensure_ascii=False), flush=True)
    configuration = configure_xlsr_stage_b(
        model, last_encoder_blocks=plan.training.last_encoder_blocks, gradient_checkpointing=True
    )
    stage_b_optimizer = AdamW(
        [
            {
                "params": [
                    parameter for parameter in model.encoder.parameters() if parameter.requires_grad
                ],
                "lr": plan.training.encoder_learning_rate,
            },
            {"params": list(model.head.parameters()), "lr": plan.training.head_learning_rate},
        ],
        weight_decay=plan.training.weight_decay,
    )
    best_macro_loss = float("inf")
    best_epoch: int | None = None
    best_state: dict[str, torch.Tensor] | None = None
    for epoch_index in range(plan.training.unfreeze_epochs):
        absolute_epoch = plan.training.warmup_epochs + epoch_index
        train_dataset.set_epoch(absolute_epoch)
        sampler.set_epoch(absolute_epoch)
        train_result = train_xlsr_sls_stage_b_epoch(
            model,
            train_loader,
            stage_b_optimizer,
            device,
            precision="bf16",
            gradient_accumulation_steps=plan.training.gradient_accumulation_steps,
            gradient_clip_norm=plan.training.gradient_clip_norm,
            last_encoder_blocks=plan.training.last_encoder_blocks,
            gradient_checkpointing=True,
        )
        macro_loss, dev_by_language = _macro_dev(model, ru_loader, kk_loader, device)
        epoch = epoch_index + 1
        selected_checkpoint = macro_loss < best_macro_loss
        record = {
            "phase": "tail_unfreeze",
            "epoch": epoch,
            "train": epoch_metrics(train_result),
            "dev_by_language": dev_by_language,
            "macro_language_dev_loss": macro_loss,
            "checkpoint_selected": selected_checkpoint,
        }
        history.append(record)
        print(json.dumps({"progress": record}, ensure_ascii=False), flush=True)
        if selected_checkpoint:
            best_macro_loss = macro_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
    if best_state is None or best_epoch is None:
        raise RuntimeError("v4 training did not select a tail-unfreeze checkpoint.")
    selected_state_sha256 = state_dict_sha256(best_state)
    model.load_state_dict(best_state)
    torch.cuda.synchronize(device)
    report: dict[str, object] = {
        **preflight,
        "status": "ok",
        "mode": "train",
        "started_at": started_at,
        "elapsed_seconds": time.monotonic() - started,
        "stage_b_configuration": asdict(configuration),
        "selection_metric": plan.training.selection_metric,
        "best_tail_unfreeze_epoch": best_epoch,
        "best_macro_language_dev_loss": best_macro_loss,
        "selected_model_state_sha256": selected_state_sha256,
        "history": history,
        "cuda_memory": {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "checkpoint_scope": "sls_head_and_final_xlsr_blocks",
        "calibrated": False,
        "frozen_final_evaluation_performed": False,
        "final_inference_performed": False,
    }
    checkpoint: dict[str, object] = {
        "model_name": "xlsr_sls_model_v4",
        "training_purpose": "research",
        "run_id": plan.run_id,
        "encoder": v4_training_plan_record(plan)["encoder"],
        "head_config": asdict(plan.head),
        "selected_model_state_sha256": selected_state_sha256,
        "model_state_dict": best_state,
        "training_run": report,
    }
    _atomic_publish(plan.outputs.checkpoint, plan.outputs.report, checkpoint, report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
