from __future__ import annotations

import json
from pathlib import Path

import pytest

from kds.eval.kazakhtts_smoke import KazakhTtsSmokeError, load_kazakhtts_smoke_plan


def test_kazakhtts_smoke_plan_binds_inputs_and_keeps_ru_mixed_conditional() -> None:
    plan = load_kazakhtts_smoke_plan(
        Path("configs/research/fresh_suite_stage_c_kazakhtts_smoke_v1.json")
    )

    assert plan.seed == 20260812
    assert {case.language for case in plan.cases} == {"ru", "kk", "mixed"}
    assert next(case for case in plan.cases if case.language == "kk").status == (
        "officially_supported"
    )
    assert all(
        case.status == "conditional_acoustic_smoke_only"
        for case in plan.cases
        if case.language in {"ru", "mixed"}
    )


def test_kazakhtts_smoke_plan_rejects_changed_bound_input(tmp_path: Path) -> None:
    source_path = Path("configs/research/fresh_suite_stage_c_kazakhtts_smoke_v1.json")
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    payload["model_lock"]["path"] = "model.json"
    payload["generator_route_gate"]["path"] = "gate.json"
    (tmp_path / "model.json").write_text("changed", encoding="utf-8")
    (tmp_path / "gate.json").write_text("gate", encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(KazakhTtsSmokeError, match="SHA-256 mismatch"):
        load_kazakhtts_smoke_plan(plan_path)
