"""Print a read-only, JSON KSC2 multipart archive audit receipt."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from kds.data.ksc2 import Ksc2AuditError, audit_ksc2_archive, write_ksc2_audit_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream-audit all pinned KSC2 archive parts without extraction."
    )
    parser.add_argument("--parts-directory", type=Path, required=True)
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Write one compact progress event per consumed archive part to stderr.",
    )
    parser.add_argument("--output-report", type=Path, default=None)
    arguments = parser.parse_args()
    try:
        progress_callback = _print_progress if arguments.progress else None
        report = audit_ksc2_archive(
            arguments.parts_directory,
            progress_callback=progress_callback,
        )
        if arguments.output_report is not None:
            write_ksc2_audit_report(arguments.output_report, report)
    except Ksc2AuditError as error:
        print(json.dumps({"status": "error", "issues": [str(error)]}, ensure_ascii=False))
        return 2
    payload = {"status": "ok", **asdict(report)}
    if arguments.output_report is not None:
        payload["output_report"] = str(arguments.output_report)
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


def _print_progress(completed_parts: int, total_parts: int, part_name: str) -> None:
    print(
        json.dumps(
            {
                "status": "progress",
                "completed_parts": completed_parts,
                "total_parts": total_parts,
                "percent": round(completed_parts / total_parts * 100, 1),
                "part": part_name,
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
