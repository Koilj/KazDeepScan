"""Audit a selected evaluation subset against configured roles and manifest inventory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from kds.data.assets import sha256_file
from kds.data.manifest import ManifestError
from kds.eval.candidate_exposure import CandidateExposureError, audit_candidate_project_exposure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidate-split", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--related-source-manifest", action="append", type=Path, default=[])
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        if arguments.output.exists() or not arguments.output.parent.is_dir():
            raise CandidateExposureError("Exposure receipt must be new with an existing parent.")
        payload = audit_candidate_project_exposure(
            candidate_manifest=arguments.candidate_manifest,
            candidate_split=arguments.candidate_split,
            source_manifest=arguments.source_manifest,
            related_source_manifests=arguments.related_source_manifest,
            project_root=arguments.project_root,
            config_root=arguments.config_root,
            manifest_root=arguments.manifest_root,
            created_at=arguments.created_at,
        )
        with arguments.output.open("x", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, ensure_ascii=False, indent=2, sort_keys=True)
            file_handle.write("\n")
    except (CandidateExposureError, ManifestError, OSError, ValueError) as error:
        issues = list(error.issues) if isinstance(error, ManifestError) else [str(error)]
        print(json.dumps({"status": "error", "issues": issues}, ensure_ascii=False))
        return 2
    candidate = cast(dict[str, object], payload["candidate"])
    scope = cast(dict[str, object], payload["scope"])
    print(
        json.dumps(
            {
                "status": "ok",
                "candidate_rows": candidate["rows"],
                "configured_rows": scope["configured_rows"],
                "inventory_rows": scope["non_candidate_inventory_rows"],
                "output": str(arguments.output),
                "sha256": sha256_file(arguments.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
