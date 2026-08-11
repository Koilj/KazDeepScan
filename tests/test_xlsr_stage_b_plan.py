from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from kds.data.licenses import load_license_ledger
from kds.data.manifest import load_manifest, write_manifest
from kds.training.xlsr_stage_b_plan import (
    XlsrStageBPlanError,
    load_xlsr_stage_b_plan,
    validate_and_select_xlsr_stage_b,
    xlsr_stage_b_plan_record,
)
from tests.test_xlsr_stage_a_plan import _prepare_plan


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fresh_dev_manifest(root: Path) -> Path:
    old_rows = load_manifest(root / "dev.csv")
    fresh_rows = [
        replace(
            row,
            sample_id=f"fresh-{row.sample_id}",
            relative_path=f"fresh-{row.relative_path}",
            sha256=hashlib.sha256(f"fresh:{row.sha256}".encode()).hexdigest(),
            parent_group_id=f"fresh-{row.parent_group_id}",
            speaker_pseudo_id=f"fresh-{row.speaker_pseudo_id}",
            text_id=f"fresh-{row.text_id}",
            text_hash=hashlib.sha256(f"fresh:{row.text_hash}".encode()).hexdigest(),
        )
        for row in old_rows
    ]
    path = root / "fresh-dev.csv"
    write_manifest(path, fresh_rows)
    return path


def _prepare_stage_b_plan(root: Path, *, overlapping_dev: bool = False) -> Path:
    base_plan = _prepare_plan(root)
    initial_head = root / "result.pt"
    initial_head.write_bytes(b"stage-a-head")
    if overlapping_dev:
        fresh_dev = root / "fresh-dev.csv"
        fresh_dev.write_bytes((root / "dev.csv").read_bytes())
    else:
        fresh_dev = _fresh_dev_manifest(root)
    plan = root / "stage-b.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "xlsr-stage-b-test",
                "purpose": "research",
                "base_stage_a_plan": {
                    "path": base_plan.name,
                    "sha256": _sha256(base_plan),
                },
                "initial_head": {
                    "checkpoint": {
                        "path": initial_head.name,
                        "sha256": _sha256(initial_head),
                    },
                    "state_dict_sha256": "f" * 64,
                },
                "dev": {
                    "manifest": {"path": fresh_dev.name, "sha256": _sha256(fresh_dev)},
                    "source_split": "dev",
                    "expected_source_ids": ["dev-source"],
                    "expected_languages": ["ru"],
                },
                "training": {
                    "seed": 20260820,
                    "epochs": 15,
                    "batch_size": 4,
                    "gradient_accumulation_steps": 8,
                    "window_samples": 64600,
                    "sample_rate": 16000,
                    "encoder_learning_rate": 0.00001,
                    "head_learning_rate": 0.0001,
                    "weight_decay": 0.0001,
                    "gradient_clip_norm": 1.0,
                    "num_workers": 0,
                    "pin_memory": True,
                    "device": "cuda",
                    "precision": "bf16",
                    "last_encoder_blocks": 8,
                    "gradient_checkpointing": True,
                    "selection_metric": "dev_loss",
                },
                "outputs": {"checkpoint": "stage-b.pt", "report": "stage-b-report.json"},
            }
        ),
        encoding="utf-8",
    )
    return plan


def test_stage_b_plan_pins_stage_a_head_and_fresh_dev(tmp_path: Path) -> None:
    plan_path = _prepare_stage_b_plan(tmp_path)

    plan = load_xlsr_stage_b_plan(plan_path)
    report, selected = validate_and_select_xlsr_stage_b(
        plan, load_license_ledger(plan.base_stage_a_plan.license_ledger.path)
    )
    record = xlsr_stage_b_plan_record(plan)

    assert len(selected.train) == 2
    assert len(selected.dev) == 2
    assert report.initial_stage_a_dev_overlap == {
        "sample_id": 0,
        "asset_sha256": 0,
        "text_hash": 0,
        "parent_group_id": 0,
    }
    assert plan.training.last_encoder_blocks == 8
    assert record["plan_sha256"] == _sha256(plan_path)


def test_stage_b_plan_rejects_changed_initial_head(tmp_path: Path) -> None:
    plan_path = _prepare_stage_b_plan(tmp_path)
    (tmp_path / "result.pt").write_bytes(b"changed")

    with pytest.raises(XlsrStageBPlanError, match="Initial Stage-A head SHA-256 mismatch"):
        load_xlsr_stage_b_plan(plan_path)


def test_stage_b_plan_rejects_overlap_with_stage_a_dev(tmp_path: Path) -> None:
    plan_path = _prepare_stage_b_plan(tmp_path, overlapping_dev=True)
    plan = load_xlsr_stage_b_plan(plan_path)

    with pytest.raises(XlsrStageBPlanError, match="overlaps the Stage-A selection dev"):
        validate_and_select_xlsr_stage_b(
            plan, load_license_ledger(plan.base_stage_a_plan.license_ledger.path)
        )
