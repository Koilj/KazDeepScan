from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
from torch import Tensor
from torch.optim import AdamW

from kds.data.assets import require_valid_assets
from kds.data.dataset import DatasetConfig, ManifestAudioDataset
from kds.data.licenses import load_license_ledger
from kds.data.unseen_generator_ood import (
    load_unseen_generator_suite,
    validate_and_select_unseen_generator_suite,
)
from kds.eval import evaluate_b0_with_strata
from kds.models import B0LogMelCnn
from kds.training import evaluate_b0, make_audio_loader, train_b0_epoch
from kds.training.frozen_b0 import (
    FrozenB0RunPlan,
    epoch_metrics,
    frozen_b0_run_plan_record,
    load_frozen_b0_run_plan,
    state_dict_sha256,
    validate_frozen_b0_suite_inputs,
)


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The frozen run requires CUDA, but CUDA is unavailable.")
    return device


def _require_new_outputs(plan: FrozenB0RunPlan) -> None:
    for label, path in (
        ("checkpoint", plan.outputs.checkpoint),
        ("report", plan.outputs.report),
    ):
        if os.path.lexists(path):
            raise ValueError(
                f"Refusing to repeat or overwrite a frozen run; {label} already exists: {path}"
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Execute one pre-registered B0 run and evaluate every frozen unseen-generator "
            "final test exactly once."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the plan, suite, output reservation and all assets without fitting.",
    )
    arguments = parser.parse_args()

    plan = load_frozen_b0_run_plan(arguments.plan)
    _require_new_outputs(plan)
    device = _device(plan.training.device)
    suite = load_unseen_generator_suite(plan.suite_path)
    validate_frozen_b0_suite_inputs(plan, suite)
    suite_report, selected = validate_and_select_unseen_generator_suite(
        suite, load_license_ledger(plan.license_ledger.path)
    )
    all_rows = [
        *selected.train,
        *selected.dev,
        *(row for final_test in selected.final_tests for row in final_test.rows),
    ]
    require_valid_assets(all_rows, arguments.audio_root)

    preflight = {
        "status": "validated" if arguments.validate_only else "running",
        "run_plan": frozen_b0_run_plan_record(plan),
        "suite": asdict(suite_report),
        "assets_validated": len(all_rows),
    }
    if arguments.validate_only:
        print(json.dumps(preflight, ensure_ascii=False))
        return 0

    torch.manual_seed(plan.training.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(plan.training.seed)
    seed = str(plan.training.seed)
    train_rows = list(selected.train)
    dev_rows = list(selected.dev)
    train_dataset = ManifestAudioDataset(
        train_rows,
        DatasetConfig(
            audio_root=arguments.audio_root,
            sample_rate=plan.model_config.sample_rate,
            window_samples=plan.training.window_samples,
            mode="train",
            seed=seed,
        ),
    )
    dev_dataset = ManifestAudioDataset(
        dev_rows,
        DatasetConfig(
            audio_root=arguments.audio_root,
            sample_rate=plan.model_config.sample_rate,
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
    )
    dev_loader = make_audio_loader(
        dev_dataset,
        batch_size=plan.training.batch_size,
        shuffle=False,
        num_workers=plan.training.num_workers,
    )
    model = B0LogMelCnn(plan.model_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=plan.training.learning_rate,
        weight_decay=plan.training.weight_decay,
    )

    history: list[dict[str, object]] = []
    best_dev_loss = float("inf")
    best_epoch: int | None = None
    best_state: dict[str, Tensor] | None = None
    for epoch_index in range(plan.training.epochs):
        train_dataset.set_epoch(epoch_index)
        train_result = train_b0_epoch(model, train_loader, optimizer, device)
        dev_result = evaluate_b0(model, dev_loader, device)
        epoch = epoch_index + 1
        history.append(
            {
                "epoch": epoch,
                "train": epoch_metrics(train_result),
                "dev": epoch_metrics(dev_result),
            }
        )
        if dev_result.loss < best_dev_loss:
            best_dev_loss = dev_result.loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }

    if best_state is None or best_epoch is None:
        raise RuntimeError("No checkpoint was selected from development loss.")
    selected_state_sha256 = state_dict_sha256(best_state)
    model.load_state_dict(best_state)

    final_results: list[dict[str, object]] = []
    for final_test in selected.final_tests:
        # Each frozen final is traversed only here, after the state hash and dev-selected epoch
        # are fixed. Aggregate and strata metrics are collected together in this one traversal.
        result, strata = evaluate_b0_with_strata(
            model,
            list(final_test.rows),
            audio_root=arguments.audio_root,
            batch_size=plan.training.batch_size,
            seed=seed,
            device=device,
            num_workers=plan.training.num_workers,
            sample_rate=plan.model_config.sample_rate,
            window_samples=plan.training.window_samples,
        )
        final_results.append(
            {
                "test_id": final_test.test_id,
                "metrics": epoch_metrics(result),
                "stratified_metrics": strata,
            }
        )

    report: dict[str, object] = {
        "status": "ok",
        "device": str(device),
        "run_plan": frozen_b0_run_plan_record(plan),
        "suite": asdict(suite_report),
        "assets_validated": len(all_rows),
        "best_epoch": best_epoch,
        "best_dev_loss": best_dev_loss,
        "selected_state_sha256": selected_state_sha256,
        "history": history,
        "final_tests": final_results,
        "calibrated": False,
    }
    checkpoint: dict[str, object] = {
        "model_name": "b0_logmel_cnn",
        "model_config": asdict(plan.model_config),
        "training_purpose": "research",
        "frozen_run": report,
        "state_dict": best_state,
    }
    _atomic_publish(
        plan.outputs.checkpoint,
        plan.outputs.report,
        checkpoint,
        report,
    )
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
