#!/usr/bin/env python3
"""Validate or execute the one-time XLS-R+SLS model-v4 RU temperature calibration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kds.data.assets import require_valid_assets, sha256_file
from kds.data.licenses import load_license_ledger
from kds.eval.v4_calibration import (
    V4CalibrationError,
    _cuda_device,
    execute_v4_calibration,
    load_v4_calibration_plan,
    preflight_v4_calibration,
    validate_v4_calibration_checkpoint_file,
    validate_v4_calibration_inputs,
    write_v4_calibration_preflight,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args(argv)
    plan = load_v4_calibration_plan(arguments.plan)
    ledger = load_license_ledger(plan.license_ledger.path)
    rows = validate_v4_calibration_inputs(plan, ledger)
    require_valid_assets(rows, arguments.audio_root)
    validate_v4_calibration_checkpoint_file(plan)
    device = _cuda_device(plan)
    preflight = preflight_v4_calibration(plan, rows, device)
    if arguments.validate_only:
        digest = write_v4_calibration_preflight(plan, preflight)
        print(json.dumps({**preflight, "preflight_sha256": digest}, ensure_ascii=False))
        return 0
    report = execute_v4_calibration(plan, rows, device, arguments.audio_root, preflight)
    print(
        json.dumps(
            {**report, "report_sha256": sha256_file(plan.outputs.report)},
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, V4CalibrationError, ValueError) as error:
        detail = "\n".join(error.issues) if isinstance(error, V4CalibrationError) else str(error)
        print(json.dumps({"status": "error", "detail": detail}, ensure_ascii=False))
        raise SystemExit(2) from error
