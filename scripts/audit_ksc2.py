"""Print a read-only, JSON KSC2 multipart archive audit receipt."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from kds.data.ksc2 import Ksc2AuditError, audit_ksc2_archive


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stream-audit all pinned KSC2 archive parts without extraction."
    )
    parser.add_argument("--parts-directory", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        report = audit_ksc2_archive(arguments.parts_directory)
    except Ksc2AuditError as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
