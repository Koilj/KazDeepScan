from __future__ import annotations

import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

from kds.data.licenses import load_license_ledger


def _script() -> Any:
    spec = spec_from_file_location(
        "kds_test_voxforge_qwen_evaluation",
        "scripts/evaluate_xlsr_voxforge_ru_mdc_qwen3_tts_customvoice.py",
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_voxforge_qwen_contract_pins_completed_gates_and_disjoint_calibration() -> None:
    script = _script()
    engine = script._engine()
    plan = engine.load_plan(
        Path(
            "configs/research/"
            "xlsr_sls_stage_b_v2_voxforge_ru_mdc_qwen3_tts_customvoice_aiden_v1.json"
        )
    )
    inputs = script._validate_inputs(
        engine, plan, load_license_ledger(plan.license_ledger.path)
    )

    assert plan.run_id == (
        "xlsr-sls-stage-b-v2-voxforge-ru-mdc-qwen3-tts-customvoice-aiden-v1"
    )
    assert len(inputs.calibration) == 976
    assert len(inputs.final_ru) == 158
    for field in engine.STAGE_B_LEAKAGE_FIELDS:
        assert not {getattr(row, field) for row in inputs.calibration}.intersection(
            getattr(row, field) for row in inputs.final_ru
        )


def test_voxforge_qwen_completion_receipt_closes_the_one_time_run() -> None:
    receipt = json.loads(
        Path(
            "data/manifests/"
            "voxforge_ru_mdc_qwen3_tts_customvoice_aiden_evaluation_completion_v1.json"
        ).read_text(encoding="utf-8")
    )

    assert receipt["actual_execution"] == {
        "detector_inference_performed": True,
        "final_sample_results": 158,
        "mode": "one_time_gpu_inference",
        "one_time_execution": True,
        "report_status": "ok",
        "rerun_authorized": False,
    }
    assert receipt["final_ru"]["accuracy"]["correct"] == 146
    assert receipt["final_ru"]["bonafide_recall"]["correct"] == 74
    assert receipt["final_ru"]["spoof_recall"]["correct"] == 72
    assert receipt["final_ru"]["pairs_both_correct"]["correct"] == 67
    assert receipt["report"]["detector_inference_performed"] is True
    assert receipt["write_once_integrity"] == {
        "execution_lock_overwritten": False,
        "report_overwritten": False,
        "report_repaired_in_place": False,
    }
