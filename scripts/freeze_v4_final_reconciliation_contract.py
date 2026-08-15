"""Freeze the publication-only v4 final reconciliation contract."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import cast

from kds.data.assets import sha256_file
from kds.data.v4_final_reconciliation import _PROHIBITIONS, OUTPUTS, PROTOCOL_ID

SALVAGE_PLAN = "configs/research/v4/xlsr_sls_model_v4_final_salvage_materialization_v1.json"
SALVAGE_FAILURE = "docs/artifacts/v4/final_salvage_materialization_attempt_failure_2026-08-16.md"
SALVAGE_KK_JOURNAL = (
    "artifacts/v4/xlsr_sls_model_v4_final_salvage_materialization_v1/kk_kazakhtts_one_shot.jsonl"
)
RUNTIME = "artifacts/v4/xlsr_sls_model_v4_final_salvage_materialization_v1"
PLAN = "configs/research/v4/xlsr_sls_model_v4_final_reconciliation_v1.json"


def _path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Path escapes project root: {relative}") from error
    return path


def _binding(root: Path, relative: str, rows: int | None = None) -> dict[str, object]:
    path = _path(root, relative)
    if not path.is_file():
        raise ValueError(f"Missing reconciliation input: {relative}")
    return {"path": relative, "sha256": sha256_file(path), "rows": rows}


def freeze(*, project_root: Path, created_at: str) -> str:
    root = project_root.resolve(strict=True)
    output = _path(root, PLAN)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("Reconciliation plan is write-once and its parent must exist.")
    raw: object = json.loads(_path(root, SALVAGE_PLAN).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not isinstance(raw.get("inputs"), dict):
        raise ValueError("Salvage plan inputs are unreadable.")
    salvage_inputs = cast(dict[str, object], raw["inputs"])
    inputs: dict[str, dict[str, object]] = {
        "salvage_plan": _binding(root, SALVAGE_PLAN),
        "salvage_failure_receipt": _binding(root, SALVAGE_FAILURE),
        "salvage_kk_journal": _binding(root, SALVAGE_KK_JOURNAL),
        "ru_source_decode_journal": _binding(root, f"{RUNTIME}/ru_source_decode_qa.jsonl"),
        "kk_source_decode_journal": _binding(root, f"{RUNTIME}/kk_source_decode_qa.jsonl"),
        "ru_spoof_decode_journal": _binding(root, f"{RUNTIME}/ru_spoof_decode_qa.jsonl"),
        "kk_spoof_decode_journal": _binding(root, f"{RUNTIME}/kk_spoof_decode_qa.jsonl"),
        "reconciliation_module": _binding(root, "src/kds/data/v4_final_reconciliation.py"),
        "runner_script": _binding(root, "scripts/reconcile_v4_final_inputs.py"),
    }
    for name in (
        "salvage_selection",
        "materialization_ledger",
        "qwen_model_lock",
        "kazakhtts_model_lock",
        "recovery_qwen_journal",
        "recovery_kk_journal",
        "historical_fingerprint_inventory",
        "source_decode_inventory",
        "kk_spoof_decode_inventory",
        "dev_source_decode_journal",
        "dev_spoof_decode_journal",
        "calibration_source_decode_journal",
        "calibration_spoof_decode_journal",
    ):
        value = salvage_inputs.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"Salvage plan lacks input {name}.")
        inputs[name] = dict(value)
    expected = {
        "salvage_plan",
        "salvage_failure_receipt",
        "salvage_selection",
        "materialization_ledger",
        "qwen_model_lock",
        "kazakhtts_model_lock",
        "recovery_qwen_journal",
        "recovery_kk_journal",
        "salvage_kk_journal",
        "ru_source_decode_journal",
        "kk_source_decode_journal",
        "ru_spoof_decode_journal",
        "kk_spoof_decode_journal",
        "historical_fingerprint_inventory",
        "source_decode_inventory",
        "kk_spoof_decode_inventory",
        "dev_source_decode_journal",
        "dev_spoof_decode_journal",
        "calibration_source_decode_journal",
        "calibration_spoof_decode_journal",
        "reconciliation_module",
        "runner_script",
    }
    if set(inputs) != expected:
        raise ValueError("Reconciliation input set drifted.")
    plan = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "created_at": created_at,
        "inputs": inputs,
        "outputs": OUTPUTS,
        "prohibitions": {name: True for name in sorted(_PROHIBITIONS)},
    }
    with tempfile.TemporaryDirectory(
        prefix=".kds-v4-reconciliation-", dir=root / "configs"
    ) as temporary:
        staged = Path(temporary) / output.name
        staged.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        staged.replace(output)
    return sha256_file(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--created-at", required=True)
    args = parser.parse_args()
    try:
        result = freeze(project_root=args.project_root, created_at=args.created_at)
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "error", "issues": [str(error)]}, ensure_ascii=False))
        return 2
    print(json.dumps({"status": "ok", "plan": PLAN, "sha256": result}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
