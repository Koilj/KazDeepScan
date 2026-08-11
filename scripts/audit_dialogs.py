#!/usr/bin/env python3
"""Audit the hash-pinned Dialogs Russian conversation release."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from kds.data.dialogs import (
    DialogsAuditError,
    audit_dialogs_release,
    require_dialogs_bonafide_final,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--output-receipt",
        type=Path,
        help="Optional new JSON receipt path. Existing files are never overwritten.",
    )
    parser.add_argument(
        "--require-bonafide-final",
        action="store_true",
        help="Exit non-zero unless every published metadata path has a supplied WAV.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.output_receipt is not None and arguments.output_receipt.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {arguments.output_receipt}")
    audit = audit_dialogs_release(arguments.artifact_root)
    rendered = json.dumps(asdict(audit), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if arguments.output_receipt is not None:
        arguments.output_receipt.parent.mkdir(parents=True, exist_ok=True)
        arguments.output_receipt.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if arguments.require_bonafide_final:
        try:
            require_dialogs_bonafide_final(audit)
        except DialogsAuditError as error:
            raise SystemExit(f"ERROR: {error}") from error


if __name__ == "__main__":
    main()
