"""Freeze the four-source rights ledger required by the XLS-R v3 governance contract."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

REQUIRED_SOURCE_IDS = (
    "common_voice_ru_v24",
    "pyara_ru_v7",
    "ruasd_ru_v1_full",
    "stage_d_ru_dialogs_vits2_masha_neutral_v1",
)


def _read_rows(path: Path) -> tuple[tuple[str, ...], dict[str, dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            fieldnames = tuple(next(reader))
        except StopIteration as error:
            raise ValueError(f"License ledger has no header: {path}") from error
        if not fieldnames:
            raise ValueError(f"License ledger has no header: {path}")
        rows: list[dict[str, str]] = []
        for row_number, values in enumerate(reader, start=2):
            if len(values) < len(fieldnames):
                raise ValueError(f"License ledger row {row_number} has too few columns.")
            if len(values) > len(fieldnames):
                values = [*values[: len(fieldnames) - 1], ",".join(values[len(fieldnames) - 1 :])]
            rows.append(dict(zip(fieldnames, values, strict=True)))
    by_source = {row.get("source_id", ""): row for row in rows}
    if len(by_source) != len(rows):
        raise ValueError("License ledger contains duplicate or empty source_id values.")
    missing = sorted(set(REQUIRED_SOURCE_IDS).difference(by_source))
    if missing:
        raise ValueError("Ledger lacks required v3 sources: " + ", ".join(missing))
    for source_id in REQUIRED_SOURCE_IDS:
        row = by_source[source_id]
        if row.get("status") not in {"verified", "owner_authorized_personal_research"}:
            raise ValueError(f"Required source {source_id!r} is not rights-verified.")
        if source_id == "stage_d_ru_dialogs_vits2_masha_neutral_v1":
            permitted = row.get("ood_evaluation_use") == "research_only"
        else:
            permitted = row.get("train_dev_test_use") == "research_only"
        if not permitted:
            raise ValueError(
                f"Required source {source_id!r} is not research-permitted for its role."
            )
    return fieldnames, by_source


def _write_new(path: Path, fields: tuple[str, ...], rows: dict[str, dict[str, str]]) -> None:
    if os.path.lexists(path):
        raise ValueError(f"Refusing to overwrite frozen v3 ledger: {path}")
    if not path.parent.is_dir():
        raise ValueError(f"Output directory does not exist: {path.parent}")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows[source_id] for source_id in REQUIRED_SOURCE_IDS)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    fields, rows = _read_rows(arguments.ledger)
    _write_new(arguments.output, fields, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
