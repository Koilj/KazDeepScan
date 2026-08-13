"""Select and freeze a bona-fide candidate set absent from all configured project roles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kds.data.assets import sha256_file
from kds.data.manifest import ManifestError, write_manifest
from kds.eval.bonafide_candidate import select_unexposed_bonafide_candidate
from kds.eval.candidate_exposure import CandidateExposureError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-split", required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-receipt", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        if arguments.output_manifest.exists() or not arguments.output_manifest.parent.is_dir():
            raise CandidateExposureError("Output manifest must be new with an existing parent.")
        if arguments.output_receipt.exists() or not arguments.output_receipt.parent.is_dir():
            raise CandidateExposureError("Output receipt must be new with an existing parent.")
        selection = select_unexposed_bonafide_candidate(
            source_manifest=arguments.source_manifest,
            source_split=arguments.source_split,
            project_root=arguments.project_root,
            config_root=arguments.config_root,
            created_at=arguments.created_at,
        )
        write_manifest(arguments.output_manifest, selection.rows)
        receipt = {
            **selection.receipt,
            "output_manifest": {
                "path": arguments.output_manifest.as_posix(),
                "sha256": sha256_file(arguments.output_manifest),
                "rows": len(selection.rows),
            },
        }
        with arguments.output_receipt.open("x", encoding="utf-8") as file_handle:
            json.dump(receipt, file_handle, ensure_ascii=False, indent=2, sort_keys=True)
            file_handle.write("\n")
    except (CandidateExposureError, ManifestError, OSError, ValueError) as error:
        issues = list(error.issues) if isinstance(error, ManifestError) else [str(error)]
        print(json.dumps({"status": "error", "issues": issues}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "selected_rows": len(selection.rows),
                "manifest": str(arguments.output_manifest),
                "receipt": str(arguments.output_receipt),
                "receipt_sha256": sha256_file(arguments.output_receipt),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
