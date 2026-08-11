from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import torch
import transformers
from torch import Tensor
from torch.optim import AdamW

from kds.data.assets import require_valid_assets
from kds.data.dataset import DatasetConfig, ManifestAudioDataset
from kds.data.licenses import load_license_ledger
from kds.models import XlsrSlsClassifier
from kds.training import (
    evaluate_xlsr_sls,
    freeze_xlsr_encoder,
    make_audio_loader,
    train_xlsr_sls_head_epoch,
)
from kds.training.frozen_b0 import epoch_metrics, state_dict_sha256
from kds.training.xlsr_stage_a_plan import (
    XlsrStageAPlan,
    load_xlsr_stage_a_plan,
    validate_and_select_xlsr_stage_a,
    xlsr_stage_a_plan_record,
)


def _cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("XLS-R Stage A requires CUDA, but CUDA is unavailable.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("XLS-R Stage A requires CUDA BF16 support, but it is unavailable.")
    return torch.device("cuda")


def _require_new_outputs(plan: XlsrStageAPlan) -> None:
    for label, path in (("checkpoint", plan.outputs.checkpoint), ("report", plan.outputs.report)):
        if os.path.lexists(path):
            raise ValueError(
                f"Refusing to repeat or overwrite Stage A; {label} already exists: {path}"
            )
        if not path.parent.is_dir():
            raise ValueError(f"{label.capitalize()} output directory does not exist: {path.parent}")


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
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(checkpoint_temporary, checkpoint_path)
        except FileExistsError as error:
            raise ValueError(f"Refusing to overwrite checkpoint: {checkpoint_path}") from error
        checkpoint_temporary.unlink()
        checkpoint_temporary = None
        try:
            os.link(report_temporary, report_path)
        except FileExistsError as error:
            raise ValueError(f"Refusing to overwrite report: {report_path}") from error
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
    }


def _build_model(plan: XlsrStageAPlan, device: torch.device) -> XlsrSlsClassifier:
    model = XlsrSlsClassifier.from_pretrained(
        str(plan.encoder.checkpoint_dir),
        attention_size=plan.head.attention_size,
        classifier_size=plan.head.classifier_size,
        dropout=plan.head.dropout,
        local_files_only=True,
    )
    freeze_xlsr_encoder(model)
    return model.to(device)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train only the SLS head over a frozen, local XLS-R encoder using a hash-pinned "
            "research train/dev plan. This command never loads frozen final manifests."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the plan, CUDA/BF16, protocol and every selected train/dev asset.",
    )
    mode.add_argument(
        "--profile-only",
        action="store_true",
        help="Run one train and one dev batch for CUDA memory profiling; publish no artifact.",
    )
    arguments = parser.parse_args()

    plan = load_xlsr_stage_a_plan(arguments.plan)
    _require_new_outputs(plan)
    device = _cuda_device()
    ledger = load_license_ledger(plan.license_ledger.path)
    protocol_report, selected = validate_and_select_xlsr_stage_a(plan, ledger)
    all_rows = [*selected.train, *selected.dev]
    require_valid_assets(all_rows, arguments.audio_root)
    environment = _environment_record(device)
    preflight: dict[str, object] = {
        "status": "validated" if arguments.validate_only else "running",
        "mode": (
            "validate_only"
            if arguments.validate_only
            else "profile_only"
            if arguments.profile_only
            else "train"
        ),
        "run_plan": xlsr_stage_a_plan_record(plan),
        "protocol": asdict(protocol_report),
        "assets_validated": len(all_rows),
        "environment": environment,
        "frozen_final_evaluation_performed": False,
    }
    if arguments.validate_only:
        print(json.dumps(preflight, ensure_ascii=False, allow_nan=False))
        return 0

    torch.manual_seed(plan.training.seed)
    torch.cuda.manual_seed_all(plan.training.seed)
    seed = str(plan.training.seed)
    train_dataset = ManifestAudioDataset(
        list(selected.train),
        DatasetConfig(
            audio_root=arguments.audio_root,
            sample_rate=plan.training.sample_rate,
            window_samples=plan.training.window_samples,
            mode="train",
            seed=seed,
        ),
    )
    dev_dataset = ManifestAudioDataset(
        list(selected.dev),
        DatasetConfig(
            audio_root=arguments.audio_root,
            sample_rate=plan.training.sample_rate,
            window_samples=plan.training.window_samples,
            mode="eval",
            seed=seed,
        ),
    )
    train_loader = make_audio_loader(
        train_dataset,
        batch_size=plan.training.batch_size,
        shuffle=True,
        num_workers=plan.training.num_workers,
        pin_memory=plan.training.pin_memory,
    )
    dev_loader = make_audio_loader(
        dev_dataset,
        batch_size=plan.training.batch_size,
        shuffle=False,
        num_workers=plan.training.num_workers,
        pin_memory=plan.training.pin_memory,
    )
    model = _build_model(plan, device)
    optimizer = AdamW(
        model.head.parameters(),
        lr=plan.training.learning_rate,
        weight_decay=plan.training.weight_decay,
    )

    if arguments.profile_only:
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        train_result = train_xlsr_sls_head_epoch(
            model,
            train_loader,
            optimizer,
            device,
            precision="bf16",
            gradient_accumulation_steps=plan.training.gradient_accumulation_steps,
            gradient_clip_norm=plan.training.gradient_clip_norm,
            max_batches=1,
        )
        dev_result = evaluate_xlsr_sls(
            model,
            dev_loader,
            device,
            precision="bf16",
            max_batches=1,
        )
        torch.cuda.synchronize(device)
        profile = {
            **preflight,
            "status": "profiled",
            "elapsed_seconds": time.monotonic() - started,
            "train_batch": epoch_metrics(train_result),
            "dev_batch": epoch_metrics(dev_result),
            "cuda_memory": {
                "batch_size": plan.training.batch_size,
                "allocated_bytes": torch.cuda.memory_allocated(device),
                "reserved_bytes": torch.cuda.memory_reserved(device),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            },
            "artifacts_published": False,
        }
        print(json.dumps(profile, ensure_ascii=False, allow_nan=False))
        return 0

    started = time.monotonic()
    history: list[dict[str, object]] = []
    best_dev_loss = float("inf")
    best_epoch: int | None = None
    best_head_state: dict[str, Tensor] | None = None
    for epoch_index in range(plan.training.epochs):
        train_dataset.set_epoch(epoch_index)
        train_result = train_xlsr_sls_head_epoch(
            model,
            train_loader,
            optimizer,
            device,
            precision="bf16",
            gradient_accumulation_steps=plan.training.gradient_accumulation_steps,
            gradient_clip_norm=plan.training.gradient_clip_norm,
        )
        dev_result = evaluate_xlsr_sls(model, dev_loader, device, precision="bf16")
        epoch = epoch_index + 1
        epoch_record: dict[str, object] = {
            "epoch": epoch,
            "train": epoch_metrics(train_result),
            "dev": epoch_metrics(dev_result),
        }
        history.append(epoch_record)
        print(json.dumps({"progress": epoch_record}, ensure_ascii=False), flush=True)
        if dev_result.loss < best_dev_loss:
            best_dev_loss = dev_result.loss
            best_epoch = epoch
            best_head_state = {
                name: value.detach().cpu().clone()
                for name, value in model.head.state_dict().items()
            }

    if best_head_state is None or best_epoch is None:
        raise RuntimeError("No SLS head checkpoint was selected from development loss.")
    selected_state_sha256 = state_dict_sha256(best_head_state)
    model.head.load_state_dict(best_head_state)
    torch.cuda.synchronize(device)
    report: dict[str, object] = {
        **preflight,
        "status": "ok",
        "elapsed_seconds": time.monotonic() - started,
        "best_epoch": best_epoch,
        "best_dev_loss": best_dev_loss,
        "selected_head_state_sha256": selected_state_sha256,
        "history": history,
        "cuda_memory": {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "checkpoint_scope": "sls_head_only",
        "frozen_final_evaluation_performed": False,
        "calibrated": False,
    }
    checkpoint: dict[str, object] = {
        "model_name": "xlsr_sls",
        "stage": "A",
        "training_purpose": "research",
        "encoder": xlsr_stage_a_plan_record(plan)["encoder"],
        "head_config": asdict(plan.head),
        "selected_head_state_sha256": selected_state_sha256,
        "head_state_dict": best_head_state,
        "stage_a_run": report,
    }
    _atomic_publish(plan.outputs.checkpoint, plan.outputs.report, checkpoint, report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
