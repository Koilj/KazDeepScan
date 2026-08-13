"""Write a project-wide historical generator screen for official OpenBMB VoxCPM2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from kds.data.assets import sha256_file
from kds.eval.voxcpm2_route_screen import (
    VoxCPM2RouteScreenError,
    screen_voxcpm2_project_history,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--artifact-receipt", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        if arguments.output.exists() or not arguments.output.parent.is_dir():
            raise VoxCPM2RouteScreenError(
                "Receipt output must be new with an existing parent directory."
            )
        payload = screen_voxcpm2_project_history(
            project_root=arguments.project_root,
            manifest_root=arguments.manifest_root,
            created_at=arguments.created_at,
            artifact_receipt_path=arguments.artifact_receipt,
        )
        with arguments.output.open("x", encoding="utf-8") as output:
            json.dump(payload, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "error", "issues": [str(error)]}, ensure_ascii=False))
        return 2
    claims = cast(dict[str, object], payload["claims"])
    scope = cast(dict[str, object], payload["scope"])
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(arguments.output),
                "output_sha256": sha256_file(arguments.output),
                "manifest_files": scope["manifest_files"],
                "manifest_rows": scope["manifest_rows"],
                "new_project_generator_family": claims["new_project_generator_family"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
