from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from kds.eval.xlsr_stage_c import (
    XlsrStageCPlanError,
    load_xlsr_stage_c_plan,
    stage_c_plan_record,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pin(path: Path, root: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(root)), "sha256": _sha256(path)}


def _write_plan(root: Path) -> Path:
    encoder = root / "encoder"
    encoder.mkdir()
    files = {
        "checkpoint.pt": b"checkpoint",
        "ledger.csv": b"ledger\n",
        "calibration.csv": b"calibration\n",
        "final.csv": b"final\n",
        "gate.json": b"{}\n",
        "exposure.json": b"{}\n",
        "runner.py": b"# pinned runner\n",
        "contract.py": b"# pinned contract\n",
    }
    for name, content in files.items():
        (root / name).write_bytes(content)
    config = encoder / "config.json"
    config.write_text("{}\n", encoding="utf-8")
    weights = encoder / "weights.bin"
    weights.write_bytes(b"weights")
    stage_b_report = root / "stage-b-report.json"
    stage_b_report.write_text(
        json.dumps(
            {
                "status": "ok",
                "checkpoint_scope": "sls_head_and_final_xlsr_blocks",
                "selected_trainable_state_sha256": "a" * 64,
                "frozen_final_evaluation_performed": False,
                "calibrated": False,
            }
        ),
        encoding="utf-8",
    )
    plan = root / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": "xlsr-sls-stage-b-v2-fresh-suite-stage-c-v1",
                "purpose": "research",
                "protocol": {
                    "kind": "asset_level_blind_multilingual_research_evaluation",
                    "quality_claim": "research_only_not_product_quality",
                    "test_novelty": (
                        "exact_assets_never_inferred_project_wide_not_source_or_"
                        "speaker_independent"
                    ),
                    "calibration": "temperature_only_on_pinned_pyara_role",
                    "decision_boundary": "fixed_calibrated_probability_0.5",
                    "pooled_language_metric": "prohibited",
                },
                "license_ledger": _pin(root / "ledger.csv", root),
                "checkpoint": {
                    **_pin(root / "checkpoint.pt", root),
                    "stage_b_report": _pin(stage_b_report, root),
                    "selected_trainable_state_sha256": "a" * 64,
                },
                "encoder": {
                    "checkpoint_dir": "encoder",
                    "revision": "b" * 40,
                    "config": _pin(config, root),
                    "weights": _pin(weights, root),
                },
                "head": {"attention_size": 128, "classifier_size": 256, "dropout": 0.2},
                "roles": {
                    "calibration": {
                        "manifest": _pin(root / "calibration.csv", root),
                        "selected_split": "dev",
                        "expected_rows": 976,
                    },
                    "final_suite": {
                        "manifest": _pin(root / "final.csv", root),
                        "expected_rows": 334,
                        "expected_pairs_by_language": {"ru": 50, "kk": 60, "mixed": 57},
                        "full_acoustic_gate": _pin(root / "gate.json", root),
                        "project_exposure_audit": _pin(root / "exposure.json", root),
                    },
                },
                "implementation": [
                    _pin(root / "runner.py", root),
                    _pin(root / "contract.py", root),
                ],
                "inference": {
                    "sample_rate": 16000,
                    "window_samples": 64600,
                    "batch_size": 4,
                    "num_workers": 0,
                    "device": "cuda",
                    "precision": "bf16",
                    "calibrated_probability_boundary": 0.5,
                    "temperature_max_iter": 50,
                },
                "outputs": {"execution_lock": "execution.json", "report": "report.json"},
            }
        ),
        encoding="utf-8",
    )
    return plan


def test_stage_c_plan_records_asset_level_blind_limitations(tmp_path: Path) -> None:
    path = _write_plan(tmp_path)

    plan = load_xlsr_stage_c_plan(path)
    record = stage_c_plan_record(plan)

    assert plan.plan_sha256 == _sha256(path)
    assert record["protocol"]["pooled_language_metric"] == "prohibited"  # type: ignore[index]
    assert record["roles"]["final_suite"]["expected_pairs_by_language"] == {  # type: ignore[index]
        "kk": 60,
        "mixed": 57,
        "ru": 50,
    }


def test_stage_c_plan_rejects_changed_pinned_implementation(tmp_path: Path) -> None:
    path = _write_plan(tmp_path)
    (tmp_path / "runner.py").write_text("# changed\n", encoding="utf-8")

    with pytest.raises(XlsrStageCPlanError, match="missing or changed"):
        load_xlsr_stage_c_plan(path)
