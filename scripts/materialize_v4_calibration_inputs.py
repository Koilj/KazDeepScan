"""Materialize and isolate the frozen v4 RU calibration source/eSpeak pairs without scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kds.data.assets import sha256_file
from kds.data.v4_calibration_materialization import (
    V4CalibrationMaterializationError,
    run_calibration_materialization,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--voxforge-archive", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--created-at", required=True)
    arguments = parser.parse_args()
    try:
        root = arguments.project_root.resolve(strict=True)
        plan = run_calibration_materialization(
            plan_path=arguments.plan,
            project_root=root,
            data_root=arguments.data_root,
            voxforge_archive=arguments.voxforge_archive,
            workers=arguments.workers,
            created_at=arguments.created_at,
        )
    except (OSError, V4CalibrationMaterializationError, ValueError) as error:
        print(json.dumps({"status": "error", "issues": [str(error)]}, ensure_ascii=False))
        return 2
    receipt = root / plan.outputs["receipt"]
    print(
        json.dumps(
            {
                "status": "ok",
                "receipt": plan.outputs["receipt"],
                "receipt_sha256": sha256_file(receipt),
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
