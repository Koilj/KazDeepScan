#!/usr/bin/env python3
"""Audit the downloaded OpenSTT RHVoice archive without extracting it."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from kds.data.openstt_rhvoice import audit_openstt_rhvoice_archive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--output-receipt",
        type=Path,
        help="Optional new JSON receipt path. Existing files are never overwritten.",
    )
    parser.add_argument(
        "--diagnostic-output",
        type=Path,
        help="Optional new JSON path that records a failed audit's exception details.",
    )
    return parser.parse_args()


def _write_new_json(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    path.write_text(rendered, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.output_receipt is not None and args.diagnostic_output == args.output_receipt:
        raise ValueError("--diagnostic-output and --output-receipt must be different paths.")
    if args.diagnostic_output is not None and args.diagnostic_output.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing diagnostic: {args.diagnostic_output}"
        )
    try:
        audit = audit_openstt_rhvoice_archive(args.archive, args.manifest)
    except Exception as error:
        if args.diagnostic_output is not None:
            _write_new_json(
                args.diagnostic_output,
                {"error_type": type(error).__name__, "message": str(error), "status": "failed"},
            )
        raise
    payload = asdict(audit)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output_receipt is not None:
        _write_new_json(args.output_receipt, payload)
    print(rendered, end="")


if __name__ == "__main__":
    main()
