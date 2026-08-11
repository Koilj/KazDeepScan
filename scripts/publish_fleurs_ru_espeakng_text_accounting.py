"""Publish explicit zero-text-rejection accounting for the fixed FLEURS RU eSpeak base."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from kds.data.assets import sha256_file
from kds.data.manifest import ManifestError, load_manifest, validate_manifest


def _object(path: Path, label: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return cast(dict[str, object], value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        if arguments.output.exists() or not arguments.output.parent.is_dir():
            raise ValueError("Text-accounting output must be new with an existing parent.")
        base = load_manifest(arguments.base_manifest)
        raw = load_manifest(arguments.raw_manifest)
        validate_manifest(base)
        validate_manifest(raw)
        receipt = _object(arguments.selection_receipt, "FLEURS RU eSpeak selection receipt")
        if (
            receipt.get("selected_manifest") != str(arguments.base_manifest)
            or receipt.get("selected_manifest_sha256") != sha256_file(arguments.base_manifest)
            or receipt.get("selected_rows") != len(base)
        ):
            raise ValueError("Selection receipt does not pin the exact FLEURS RU eSpeak base.")
        raw_texts = {row.text_hash for row in raw}
        base_texts = {row.text_hash for row in base}
        if (
            len(raw) != len(base)
            or len(raw_texts) != len(raw)
            or raw_texts != base_texts
            or any(row.label != "spoof" or row.language != "ru" for row in raw)
        ):
            raise ValueError("Raw eSpeak manifest is not an exact RU spoof rendering of the base.")
        payload = {
            "schema_version": 1,
            "base_manifest_sha256": {
                str(arguments.base_manifest): sha256_file(arguments.base_manifest)
            },
            "selection_receipt": str(arguments.selection_receipt),
            "selection_receipt_sha256": sha256_file(arguments.selection_receipt),
            "raw_manifest": str(arguments.raw_manifest),
            "raw_manifest_sha256": sha256_file(arguments.raw_manifest),
            "published_rows": len(raw),
            "rejected_rows": [],
            "rule": "No text normalization, filtering, or rejection occurred after base selection.",
        }
        with arguments.output.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except (ManifestError, OSError, ValueError) as error:
        issues = list(error.issues) if isinstance(error, ManifestError) else [str(error)]
        print(json.dumps({"status": "error", "issues": issues}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {"status": "ok", "published_rows": len(raw), "output": str(arguments.output)},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
