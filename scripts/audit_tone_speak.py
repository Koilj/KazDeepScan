#!/usr/bin/env python3
"""Audit a complete hash-pinned ToneSpeak download without extracting MP3 files."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from kds.data.tone_speak import audit_tone_speak_release


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--output-receipt",
        type=Path,
        help="Optional new JSON receipt path. Existing files are never overwritten.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    if arguments.output_receipt is not None and arguments.output_receipt.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {arguments.output_receipt}")
    audit = audit_tone_speak_release(arguments.artifact_root)
    rendered = json.dumps(asdict(audit), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if arguments.output_receipt is not None:
        arguments.output_receipt.parent.mkdir(parents=True, exist_ok=True)
        arguments.output_receipt.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
