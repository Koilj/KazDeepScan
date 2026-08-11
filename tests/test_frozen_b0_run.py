from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from kds.training.frozen_b0 import (
    FrozenB0RunPlanError,
    frozen_b0_run_plan_record,
    load_frozen_b0_run_plan,
    state_dict_sha256,
)


def _write_plan(
    path: Path,
    suite_path: Path,
    *,
    suite_sha256: str | None = None,
    report: str = "result.json",
) -> None:
    actual_sha256 = hashlib.sha256(suite_path.read_bytes()).hexdigest()
    ledger_path = path.parent / "ledger.csv"
    ledger_path.write_text("ledger\n", encoding="utf-8")
    manifest_paths = [path.parent / f"manifest-{index}.csv" for index in range(1, 4)]
    for index, manifest_path in enumerate(manifest_paths, start=1):
        manifest_path.write_text(f"manifest-{index}\n", encoding="utf-8")
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "unseen-b0-test-v2",
                "purpose": "research",
                "suite": {
                    "path": suite_path.name,
                    "sha256": suite_sha256 or actual_sha256,
                },
                "license_ledger": {
                    "path": ledger_path.name,
                    "sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
                },
                "manifests": [
                    {
                        "path": manifest_path.name,
                        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                    }
                    for manifest_path in manifest_paths
                ],
                "model": {
                    "name": "b0_logmel_cnn",
                    "config": {
                        "sample_rate": 16000,
                        "n_fft": 512,
                        "hop_length": 160,
                        "n_mels": 80,
                        "dropout": 0.2,
                    },
                },
                "training": {
                    "seed": 20260818,
                    "epochs": 5,
                    "batch_size": 16,
                    "window_samples": 64600,
                    "learning_rate": 0.0001,
                    "weight_decay": 0.0001,
                    "num_workers": 0,
                    "device": "cuda",
                },
                "outputs": {
                    "checkpoint": "checkpoint.pt",
                    "report": report,
                },
            }
        ),
        encoding="utf-8",
    )


def test_frozen_b0_plan_pins_suite_model_training_and_outputs(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.json"
    suite_path.write_text('{"schema_version": 1}\n', encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, suite_path)

    plan = load_frozen_b0_run_plan(plan_path)
    record = frozen_b0_run_plan_record(plan)

    assert plan.training.seed == 20260818
    assert plan.training.window_samples == 64600
    assert plan.model_config.n_mels == 80
    assert plan.outputs.checkpoint == (tmp_path / "checkpoint.pt").resolve()
    assert record["plan_sha256"] == hashlib.sha256(plan_path.read_bytes()).hexdigest()


def test_frozen_b0_plan_rejects_suite_changed_after_registration(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.json"
    suite_path.write_text('{"schema_version": 1}\n', encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, suite_path)
    suite_path.write_text('{"schema_version": 2}\n', encoding="utf-8")

    with pytest.raises(FrozenB0RunPlanError, match="suite SHA-256 mismatch"):
        load_frozen_b0_run_plan(plan_path)


def test_frozen_b0_plan_rejects_same_checkpoint_and_report_path(tmp_path: Path) -> None:
    suite_path = tmp_path / "suite.json"
    suite_path.write_text("{}\n", encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    _write_plan(plan_path, suite_path, report="checkpoint.pt")

    with pytest.raises(FrozenB0RunPlanError, match="output paths must be different"):
        load_frozen_b0_run_plan(plan_path)


def test_state_dict_hash_is_stable_and_sensitive_to_tensor_values() -> None:
    first = {"weight": torch.tensor([[1.0, 2.0]]), "step": torch.tensor(3)}
    reordered = {"step": torch.tensor(3), "weight": torch.tensor([[1.0, 2.0]])}
    changed = {"weight": torch.tensor([[1.0, 2.5]]), "step": torch.tensor(3)}

    assert state_dict_sha256(first) == state_dict_sha256(reordered)
    assert state_dict_sha256(first) != state_dict_sha256(changed)
