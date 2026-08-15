"""Freeze isolated v4 RU calibration metadata inputs without materializing or scoring audio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kds.data.assets import sha256_file
from kds.data.v4_calibration import (
    V4CalibrationInputError,
    audit_v4_calibration_inputs,
    load_v4_calibration_input_plan,
    write_v4_calibration_input_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--voxforge-archive", type=Path, required=True)
    parser.add_argument("--audited-at", required=True)
    arguments = parser.parse_args()
    try:
        project_root = arguments.project_root.resolve(strict=True)
        plan = load_v4_calibration_input_plan(arguments.plan, project_root)
        selection, receipt = audit_v4_calibration_inputs(
            plan_path=arguments.plan,
            project_root=project_root,
            voxforge_archive=arguments.voxforge_archive,
            audited_at=arguments.audited_at,
        )
        write_v4_calibration_input_outputs(
            plan=plan,
            project_root=project_root,
            selection=selection,
            receipt=receipt,
        )
    except (OSError, V4CalibrationInputError, ValueError) as error:
        print(json.dumps({"status": "error", "issues": [str(error)]}, ensure_ascii=False))
        return 2
    receipt_path = project_root / plan.output_receipt
    print(
        json.dumps(
            {
                "status": "ok",
                "metadata_selection": plan.output_selection,
                "receipt": plan.output_receipt,
                "receipt_sha256": sha256_file(receipt_path),
                "selected_records": len(selection),
                "calibration_performed": False,
                "final_inference_performed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
