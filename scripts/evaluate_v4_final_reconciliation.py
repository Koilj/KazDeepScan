#!/usr/bin/env python3
"""Run the authorized one-time XLS-R+SLS v4 final reconciliation evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kds.data.assets import require_valid_assets, sha256_file
from kds.data.licenses import load_license_ledger
from kds.eval.v4_final_evaluation import (
    V4FinalEvaluationError,
    _cuda_device,
    execute_v4_final_evaluation,
    load_v4_final_evaluation_plan,
    preflight_v4_final_evaluation,
    validate_v4_final_evaluation_checkpoint_file,
    validate_v4_final_evaluation_inputs,
    write_v4_final_evaluation_preflight,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args(argv)
    plan = load_v4_final_evaluation_plan(arguments.plan)
    rows, overlap_counts = validate_v4_final_evaluation_inputs(
        plan, load_license_ledger(plan.license_ledger.path)
    )
    require_valid_assets(rows, arguments.audio_root)
    validate_v4_final_evaluation_checkpoint_file(plan)
    device = _cuda_device(plan)
    preflight = preflight_v4_final_evaluation(plan, rows, overlap_counts, device)
    if arguments.validate_only:
        digest = write_v4_final_evaluation_preflight(plan, preflight)
        print(
            json.dumps(
                {**preflight, "preflight_sha256": digest}, ensure_ascii=False, allow_nan=False
            )
        )
        return 0
    report = execute_v4_final_evaluation(plan, rows, device, arguments.audio_root, preflight)
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
    except (OSError, RuntimeError, ValueError, V4FinalEvaluationError) as error:
        detail = (
            "\n".join(error.issues) if isinstance(error, V4FinalEvaluationError) else str(error)
        )
        print(json.dumps({"status": "error", "detail": detail}, ensure_ascii=False))
        raise SystemExit(2) from error
