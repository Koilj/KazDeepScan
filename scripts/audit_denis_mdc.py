"""Publish a write-once read-only source receipt for the pinned Denis 1.0 archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kds.data.assets import sha256_file
from kds.data.denis import DenisArchiveAuditError, audit_denis_archive


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--audited-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        if arguments.output.exists() or not arguments.output.parent.is_dir():
            raise DenisArchiveAuditError(
                "Receipt output must be new with an existing parent directory."
            )
        audit = audit_denis_archive(arguments.archive)
        with arguments.output.open("x", encoding="utf-8") as output:
            json.dump(
                audit.receipt(audited_at=arguments.audited_at),
                output,
                ensure_ascii=False,
                indent=2,
            )
            output.write("\n")
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "error", "issues": [str(error)]}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(arguments.output),
                "output_sha256": sha256_file(arguments.output),
                "intake_status": audit.intake_status,
                "paired_records": audit.paired_records,
                "fully_decoded_audio_files": audit.fully_decoded_audio_files,
                "duration_at_least_2_5_seconds": audit.duration_at_least_2_5_seconds,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
