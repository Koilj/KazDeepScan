from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from kds.data.ruasd_catalog import (
    RuAsdCatalogError,
    audit_ruasd_collection,
    load_ruasd_artifact_catalog,
    write_ruasd_audit_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only integrity and metadata audit for a pinned full RuASD release."
    )
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument(
        "--catalog",
        type=Path,
        default=Path("data/licenses/ruasd_v1_artifact_catalog.csv"),
    )
    parser.add_argument(
        "--verify-sha256",
        action="store_true",
        help="Hash every archive before reading metadata; this reads the full local release.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Write one compact progress event per validated archive to stderr.",
    )
    parser.add_argument("--output-report", type=Path, default=None)
    arguments = parser.parse_args()

    try:
        progress_callback = _print_progress if arguments.progress else None
        audit = audit_ruasd_collection(
            arguments.archive_dir,
            load_ruasd_artifact_catalog(arguments.catalog),
            verify_sha256=arguments.verify_sha256,
            progress_callback=progress_callback,
        )
        if arguments.output_report is not None:
            write_ruasd_audit_report(arguments.output_report, audit)
    except RuAsdCatalogError as error:
        print(json.dumps({"status": "error", "issues": [str(error)]}, ensure_ascii=False))
        return 2
    payload = {"status": "ok", **audit.as_mapping()}
    if arguments.output_report is not None:
        payload["output_report"] = str(arguments.output_report)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def _print_progress(completed_archives: int, total_archives: int, archive_name: str) -> None:
    print(
        json.dumps(
            {
                "status": "progress",
                "completed_archives": completed_archives,
                "total_archives": total_archives,
                "percent": round(completed_archives / total_archives * 100, 1),
                "archive": archive_name,
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
