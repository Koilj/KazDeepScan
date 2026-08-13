from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

from kds.data.licenses import load_license_ledger


def _script() -> Any:
    spec = spec_from_file_location(
        "kds_test_v5_5_evaluation",
        "scripts/evaluate_xlsr_common_voice_ru_v24_silero_v5_5.py",
    )
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v5_5_contract_pins_the_completed_gate_and_zero_overlap_audit() -> None:
    script = _script()
    engine = script._engine()
    plan = engine.load_plan(
        Path("configs/research/xlsr_sls_stage_b_v2_common_voice_ru_v24_silero_v5_5_eugene_v1.json")
    )
    inputs = script._validate_inputs(engine, plan, load_license_ledger(plan.license_ledger.path))

    assert plan.run_id == "xlsr-sls-stage-b-v2-common-voice-ru-v24-silero-v5-5-eugene-v1"
    assert len(inputs.calibration) == 976
    assert len(inputs.final_ru) == 84
