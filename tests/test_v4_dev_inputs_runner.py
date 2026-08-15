from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any


def _module() -> dict[str, Any]:
    return runpy.run_path("scripts/build_v4_isolated_dev_inputs.py")


def test_v4_dev_input_plan_binds_isolated_roles_and_predeclared_reserve() -> None:
    module = _module()
    root = Path(".").resolve()

    plan = module["load_plan"](
        Path("configs/research/v4/xlsr_sls_model_v4_dev_inputs_v1.json"), root
    )
    module["_validate_role_binding"](plan, root)

    assert plan.candidate_pairs == 600
    assert plan.frozen_pairs == 474
    assert plan.candidate_pairs - plan.frozen_pairs == 126
    assert plan.inputs["license_ledger"].rows == 3
