from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from kds.data.licenses import load_license_ledger
from kds.eval.v4_calibration import (
    V4CalibrationError,
    load_v4_calibration_plan,
    plan_record,
    validate_v4_calibration_inputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = PROJECT_ROOT / "configs/research/v4/xlsr_sls_model_v4_ru_calibration_v1.json"


def test_v4_calibration_plan_binds_only_the_frozen_ru_pairs() -> None:
    plan = load_v4_calibration_plan(PLAN_PATH)

    assert plan.run_id == "xlsr-sls-model-v4-ru-calibration-v1"
    assert plan.calibration.expected_rows == 146
    assert plan.calibration.expected_pairs == 73
    assert plan.protocol["final_inference"] == "prohibited"
    assert plan.protocol["threshold_selection"] == "prohibited"
    assert plan_record(plan)["checkpoint"] == {
        "path": str(plan.checkpoint.path),
        "sha256": "8be73165a4e6f65e966fa6d6a162fbb319d7089d1e8c1597c131e9ccb226852f",
        "selected_model_state_sha256": (
            "3cfca24a3731d3f9e3c259dcea905be07aefc4fbf2fbefa98189696df01fbe4a"
        ),
    }


def test_v4_calibration_rejects_a_pair_count_mutation() -> None:
    plan = load_v4_calibration_plan(PLAN_PATH)
    mutated = replace(plan, calibration=replace(plan.calibration, expected_pairs=72))
    ledger = load_license_ledger(plan.license_ledger.path)

    with pytest.raises(V4CalibrationError, match="complete one-bonafide/one-spoof"):
        validate_v4_calibration_inputs(mutated, ledger)
