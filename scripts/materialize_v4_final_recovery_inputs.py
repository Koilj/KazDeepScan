"""Run or review-finalize the one-shot v4 final recovery contract."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kds.data.assets import sha256_file
from kds.data.v4_final_materialization import V4FinalMaterializationError
from kds.data.v4_final_recovery_materialization import (
    V4FinalRecoveryError,
    finalize_pair_lock,
    preflight_materialization,
    run_materialization,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode in ("preflight", "materialize"):
        current = modes.add_parser(mode)
        current.add_argument("--plan", type=Path, required=True)
        current.add_argument("--project-root", type=Path, default=Path("."))
        current.add_argument("--data-root", type=Path, default=Path("data"))
        current.add_argument("--common-voice-archive", type=Path, required=True)
        current.add_argument("--fleurs-release-root", type=Path, required=True)
        current.add_argument("--created-at", required=True)
        if mode == "materialize":
            current.add_argument("--workers", type=int, default=4)
    finalize = modes.add_parser("finalize-pair-lock")
    finalize.add_argument("--plan", type=Path, required=True)
    finalize.add_argument("--project-root", type=Path, default=Path("."))
    finalize.add_argument("--data-root", type=Path, default=Path("data"))
    finalize.add_argument("--reviewer-a", type=Path, required=True)
    finalize.add_argument("--reviewer-b", type=Path, required=True)
    finalize.add_argument("--created-at", required=True)
    arguments = parser.parse_args()
    try:
        if arguments.mode == "preflight":
            plan = preflight_materialization(
                plan_path=arguments.plan,
                project_root=arguments.project_root,
                data_root=arguments.data_root,
                common_voice_archive=arguments.common_voice_archive,
                fleurs_release_root=arguments.fleurs_release_root,
                created_at=arguments.created_at,
            )
            result = {
                "status": "ok",
                "plan": plan.path,
                "plan_sha256": plan.sha256,
                "writes_performed": False,
                "final_inference_performed": False,
            }
        elif arguments.mode == "materialize":
            plan = run_materialization(
                plan_path=arguments.plan,
                project_root=arguments.project_root,
                data_root=arguments.data_root,
                common_voice_archive=arguments.common_voice_archive,
                fleurs_release_root=arguments.fleurs_release_root,
                workers=arguments.workers,
                created_at=arguments.created_at,
            )
            receipt = Path(arguments.project_root) / plan.outputs["materialization_receipt"]
            result = {
                "status": "ok",
                "receipt": plan.outputs["materialization_receipt"],
                "receipt_sha256": sha256_file(receipt),
                "pair_lock_performed": False,
                "final_inference_performed": False,
            }
        else:
            plan = finalize_pair_lock(
                plan_path=arguments.plan,
                project_root=arguments.project_root,
                data_root=arguments.data_root,
                reviewer_a=arguments.reviewer_a,
                reviewer_b=arguments.reviewer_b,
                created_at=arguments.created_at,
            )
            receipt = Path(arguments.project_root) / plan.outputs["pair_lock_receipt"]
            result = {
                "status": "ok",
                "receipt": plan.outputs["pair_lock_receipt"],
                "receipt_sha256": sha256_file(receipt),
                "final_inference_performed": False,
            }
    except (
        OSError,
        RuntimeError,
        ValueError,
        V4FinalRecoveryError,
        V4FinalMaterializationError,
    ) as error:
        print(json.dumps({"status": "error", "issues": [str(error)]}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
