from __future__ import annotations

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
