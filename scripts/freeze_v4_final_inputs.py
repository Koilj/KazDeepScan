#!/usr/bin/env python3
"""Freeze v4 RU/KK final-source metadata without touching audio or model logits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kds.data.v4_final_inputs import V4FinalInputError, run_v4_final_input_selection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--common-voice-archive", type=Path, required=True)
    parser.add_argument("--fleurs-release-root", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        plan, result, receipt_sha256 = run_v4_final_input_selection(
            plan_path=arguments.plan,
            project_root=arguments.project_root,
            common_voice_archive=arguments.common_voice_archive,
            fleurs_release_root=arguments.fleurs_release_root,
        )
    except (OSError, RuntimeError, ValueError, V4FinalInputError) as error:
        print(json.dumps({"status": "error", "issues": [str(error)]}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "selection": plan.output_selection,
                "receipt": plan.output_receipt,
                "receipt_sha256": receipt_sha256,
                "ru_selected_pairs": len(result.ru),
                "kk_selected_pairs": len(result.kk),
                "raw_audio_extraction_performed": False,
                "synthetic_audio_generated": False,
                "final_inference_performed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
