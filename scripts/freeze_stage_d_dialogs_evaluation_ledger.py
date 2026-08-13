"""Write the minimal immutable license ledger for the Stage-D RU evaluation contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kds.data.assets import sha256_file
from kds.data.licenses import LicenseLedgerError, load_license_ledger, write_license_ledger_snapshot

_SOURCE_IDS = (
    "common_voice_ru_v24",
    "pyara_ru_v7",
    "stage_d_ru_dialogs_vits2_masha_neutral_v1",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--license-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        source_ids = write_license_ledger_snapshot(
            arguments.output,
            load_license_ledger(arguments.license_ledger),
            source_ids=_SOURCE_IDS,
        )
    except LicenseLedgerError as error:
        print(json.dumps({"status": "error", "issues": list(error.issues)}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(arguments.output),
                "sha256": sha256_file(arguments.output),
                "source_ids": source_ids,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
