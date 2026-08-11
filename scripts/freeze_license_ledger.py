from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from kds.data.licenses import (
    LicenseLedgerError,
    load_license_ledger,
    write_license_ledger_snapshot,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a minimal write-once license-ledger snapshot for a frozen plan."
    )
    parser.add_argument(
        "--ledger", type=Path, default=Path("data/licenses/license_ledger.csv")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-id", action="append", required=True)
    arguments = parser.parse_args()

    try:
        selected = write_license_ledger_snapshot(
            arguments.output,
            load_license_ledger(arguments.ledger),
            source_ids=arguments.source_id,
        )
    except LicenseLedgerError as error:
        print(json.dumps({"status": "error", "issues": error.issues}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "path": str(arguments.output),
                "sha256": _sha256(arguments.output),
                "source_ids": selected,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
