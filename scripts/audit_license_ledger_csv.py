#!/usr/bin/env python3
"""Fail-closed structural audit for the mutable license-ledger CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from kds.data.licenses import LICENSE_LEDGER_FIELD_ORDER, load_license_ledger


class LicenseLedgerCsvShapeError(ValueError):
    """Raised when CSV structure could truncate or shift ledger fields."""


def audit_license_ledger_csv(path: Path) -> int:
    """Require the exact header and field count before semantic ledger loading."""

    if not path.is_file():
        raise LicenseLedgerCsvShapeError(f"License ledger does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            header = next(reader, None)
            if tuple(header or ()) != LICENSE_LEDGER_FIELD_ORDER:
                raise LicenseLedgerCsvShapeError(
                    "License ledger header must exactly match the required ordered columns."
                )
            rows = 0
            for row_number, row in enumerate(reader, start=2):
                if len(row) != len(LICENSE_LEDGER_FIELD_ORDER):
                    raise LicenseLedgerCsvShapeError(
                        f"Ledger row {row_number}: expected "
                        f"{len(LICENSE_LEDGER_FIELD_ORDER)} fields, got {len(row)}; "
                        "quote values containing commas."
                    )
                rows += 1
    except csv.Error as error:
        raise LicenseLedgerCsvShapeError(f"Malformed license-ledger CSV: {error}") from error
    except OSError as error:
        raise LicenseLedgerCsvShapeError(f"Cannot read license ledger: {path}") from error
    if rows == 0:
        raise LicenseLedgerCsvShapeError("License ledger contains no data rows.")
    entries = load_license_ledger(path)
    if len(entries) != rows:
        raise LicenseLedgerCsvShapeError(
            f"Semantic loader returned {len(entries)} entries for {rows} CSV rows."
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "ledger",
        nargs="?",
        type=Path,
        default=Path("data/licenses/license_ledger.csv"),
    )
    args = parser.parse_args()
    rows = audit_license_ledger_csv(args.ledger)
    print(f"license ledger CSV shape and semantics: OK ({rows} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
