"""Freeze the finite v4 salvage boundary after the recovery TTS failure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
from pathlib import Path
from typing import cast

from kds.data.assets import sha256_file
from kds.data.v4_final_inputs import V4_FINAL_SELECTION_FIELDS
from kds.data.v4_final_salvage_materialization import (
    _PROHIBITIONS,
    MODEL_ROOTS,
    OUTPUTS,
    PARTIAL_RAW_ROOTS,
    PROCESSED_ROOT,
    PROTOCOL_ID,
    RAW_ROOTS,
    RUNTIME_ROOT,
)

RECOVERY_PLAN = (
    "configs/research/v4/xlsr_sls_model_v4_final_recovery_materialization_v1_revalidated.json"
)
RECOVERY_FAILURE = "docs/artifacts/v4/final_recovery_materialization_attempt_failure_2026-08-15.md"
RECOVERY_QWEN_JOURNAL = (
    "artifacts/v4/xlsr_sls_model_v4_final_recovery_materialization_v1/ru_qwen_one_shot.jsonl"
)
RECOVERY_KK_JOURNAL = (
    "artifacts/v4/xlsr_sls_model_v4_final_recovery_materialization_v1/kk_kazakhtts_one_shot.jsonl"
)
SALVAGE_SELECTION = "data/manifests/v4/xlsr_sls_model_v4_final_salvage_metadata_v1.csv"
PERMANENT_REJECTS = "docs/artifacts/v4/v4_final_salvage_permanent_rejects_2026-08-15.json"
SALVAGE_AUTHORIZATION = "docs/artifacts/v4/v4_final_salvage_authorization_2026-08-15.json"
SALVAGE_LEDGER = "data/licenses/frozen/xlsr_sls_model_v4_final_salvage_materialization_v1.csv"
SALVAGE_PLAN = "configs/research/v4/xlsr_sls_model_v4_final_salvage_materialization_v1.json"


class SalvageFreezeError(ValueError):
    """Raised when salvage metadata would rewrite or extend the exhausted scope."""


def _path(root: Path, relative: str) -> Path:
    result = (root / relative).resolve()
    try:
        result.relative_to(root)
    except ValueError as error:
        raise SalvageFreezeError(f"Path escapes project root: {relative}") from error
    return result


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SalvageFreezeError(f"Cannot read {label}.") from error
    if not isinstance(value, dict):
        raise SalvageFreezeError(f"{label} must be an object.")
    return cast(dict[str, object], value)


def _rows(path: Path) -> int | None:
    if path.suffix != ".csv":
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _binding(root: Path, relative: str, rows: int | None = None) -> dict[str, object]:
    path = _path(root, relative)
    if not path.is_file():
        raise SalvageFreezeError(f"Missing salvage input: {relative}")
    return {"path": relative, "sha256": sha256_file(path), "rows": rows}


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=V4_FINAL_SELECTION_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _aggregate(directory: Path) -> dict[str, object]:
    if not directory.is_dir():
        raise SalvageFreezeError(f"Partial recovery directory is missing: {directory}")
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    total = 0
    for path in files:
        size = path.stat().st_size
        digest.update(
            f"{path.relative_to(directory).as_posix()}\0{sha256_file(path)}\0{size}\n".encode()
        )
        total += size
    return {
        "file_count": len(files),
        "bytes": total,
        "aggregate_sha256": digest.hexdigest(),
    }


def _ledger(root: Path) -> list[dict[str, str]]:
    recovery = _path(
        root, "data/licenses/frozen/xlsr_sls_model_v4_final_recovery_materialization_v1.csv"
    )
    with recovery.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    mapping = {
        "common_voice_ru_v24_v4_final_recovery": "common_voice_ru_v24_v4_final_salvage",
        "google_fleurs_kk_v1_v4_final_recovery": "google_fleurs_kk_v1_v4_final_salvage",
        (
            "qwen3_tts_customvoice_aiden_v4_final_recovery"
        ): "qwen3_tts_customvoice_aiden_v4_final_salvage",
        (
            "issai_kazakhtts2_male2_tacotron2_pwg_v4_final_recovery"
        ): "issai_kazakhtts2_male2_tacotron2_pwg_v4_final_salvage",
    }
    if len(rows) != 4 or {row.get("source_id") for row in rows} != set(mapping):
        raise SalvageFreezeError("Recovery ledger source set changed.")
    converted: list[dict[str, str]] = []
    for row in rows:
        item = dict(row)
        item["source_id"] = mapping[cast(str, row["source_id"])]
        item["notes"] = (
            "v4 salvage only: reuse byte-verified partial recovery audio without resynthesis; "
            "generate exactly 227 remaining prevalidated KK rows once; two locked-token rejects "
            "and all replacement/backfill are forbidden."
        )
        converted.append(item)
    if fields != tuple(converted[0]):
        raise SalvageFreezeError("Recovery ledger schema changed.")
    return converted


def freeze(*, project_root: Path, created_at: str) -> dict[str, str]:
    root = project_root.resolve(strict=True)
    recovery_plan = _read_json(_path(root, RECOVERY_PLAN), "recovery plan")
    recovery_inputs = cast(dict[str, object], recovery_plan.get("inputs"))
    recovery_selection_binding = cast(dict[str, object], recovery_inputs.get("recovery_selection"))
    selection_path = _path(root, cast(str, recovery_selection_binding["path"]))
    with selection_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != V4_FINAL_SELECTION_FIELDS:
            raise SalvageFreezeError("Recovery selection schema changed.")
        recovery_rows = list(reader)
    rejects = [
        row
        for row in recovery_rows
        if row["language"] == "kk" and row["selection_rank"] in {"272", "310"}
    ]
    salvage = [row for row in recovery_rows if row not in rejects]
    if len(recovery_rows) != 999 or len(rejects) != 2 or len(salvage) != 997:
        raise SalvageFreezeError("Salvage must remove exactly KK ranks 272 and 310 from recovery.")
    journal_paths = (_path(root, RECOVERY_QWEN_JOURNAL), _path(root, RECOVERY_KK_JOURNAL))
    try:
        qwen_events = [
            json.loads(line) for line in journal_paths[0].read_text(encoding="utf-8").splitlines()
        ]
        kk_events = [
            json.loads(line) for line in journal_paths[1].read_text(encoding="utf-8").splitlines()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SalvageFreezeError("Recovery one-shot journals are unreadable.") from error
    if len(qwen_events) != 998 or len(kk_events) != 542:
        raise SalvageFreezeError(
            "Recovery journals do not prove 499 RU plus 271 KK generated rows."
        )
    expected = (
        SALVAGE_SELECTION,
        PERMANENT_REJECTS,
        SALVAGE_AUTHORIZATION,
        SALVAGE_LEDGER,
        SALVAGE_PLAN,
    )
    if any(_path(root, value).exists() for value in expected):
        raise SalvageFreezeError("Salvage contract outputs are write-once and already exist.")
    if any(not _path(root, value).parent.is_dir() for value in expected):
        raise SalvageFreezeError("A salvage output parent is missing.")
    with tempfile.TemporaryDirectory(
        prefix=".kds-v4-final-salvage-freeze-", dir=root / "configs"
    ) as temporary:
        stage = Path(temporary)
        staged = {value: stage / Path(value).name for value in expected}
        _write_csv(staged[SALVAGE_SELECTION], salvage)
        rejection_items = []
        for row in sorted(rejects, key=lambda item: int(item["selection_rank"])):
            rejection_items.append(
                {
                    "language": "kk",
                    "selection_rank": int(row["selection_rank"]),
                    "sample_id": row["sample_id"],
                    "text_id": row["text_id"],
                    "text_hash": row["text_hash"],
                    "synthesis_text_sha256": row["synthesis_text_sha256"],
                    "reason": "unsupported_character_in_locked_kazakhtts_token_list",
                    "resynthesis_forbidden": True,
                    "replacement_or_backfill_forbidden": True,
                }
            )
        permanent = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "rejections": rejection_items}
        staged[PERMANENT_REJECTS].write_text(
            json.dumps(permanent, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        partial = {
            name: {"relative_path": relative, **_aggregate(root / "data" / relative)}
            for name, relative in PARTIAL_RAW_ROOTS.items()
        }
        authorization = {
            "schema_version": 1,
            "protocol_id": "xlsr-sls-model-v4-final-salvage-authorization-v1",
            "created_at": created_at,
            "status": "partial_recovery_salvage_authorized",
            "recovery_attempt": {
                "plan_path": RECOVERY_PLAN,
                "plan_sha256": sha256_file(_path(root, RECOVERY_PLAN)),
                "failure_receipt_path": RECOVERY_FAILURE,
                "failure_receipt_sha256": sha256_file(_path(root, RECOVERY_FAILURE)),
                "qwen_journal_path": RECOVERY_QWEN_JOURNAL,
                "qwen_journal_sha256": sha256_file(journal_paths[0]),
                "kk_journal_path": RECOVERY_KK_JOURNAL,
                "kk_journal_sha256": sha256_file(journal_paths[1]),
            },
            "partial_artifacts": partial,
            "claims": {
                "partial_outputs_reused_without_resynthesis": True,
                "remaining_kk_one_shot_rows": 227,
                "permanent_kk_rejects": 2,
                "replacement_or_backfill_authorized": False,
                "detector_inference_authorized": False,
                "final_inference_authorized": False,
            },
        }
        staged[SALVAGE_AUTHORIZATION].write_text(
            json.dumps(authorization, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        ledger = _ledger(root)
        with staged[SALVAGE_LEDGER].open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(ledger[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(ledger)
        inputs = dict(recovery_inputs)
        inputs.pop("recovery_authorization", None)
        inputs.pop("failed_materialization_plan", None)
        inputs.pop("failed_materialization_failure_receipt", None)
        inputs.pop("failed_qwen_journal", None)
        inputs["salvage_selection"] = _binding(stage, staged[SALVAGE_SELECTION].name, 997)
        inputs["permanent_rejects"] = _binding(stage, staged[PERMANENT_REJECTS].name)
        inputs["salvage_authorization"] = _binding(stage, staged[SALVAGE_AUTHORIZATION].name)
        inputs["recovery_materialization_plan"] = _binding(root, RECOVERY_PLAN)
        inputs["recovery_failure_receipt"] = _binding(root, RECOVERY_FAILURE)
        inputs["recovery_qwen_journal"] = _binding(root, RECOVERY_QWEN_JOURNAL)
        inputs["recovery_kk_journal"] = _binding(root, RECOVERY_KK_JOURNAL)
        inputs["materialization_ledger"] = _binding(stage, staged[SALVAGE_LEDGER].name, 4)
        inputs["salvage_materialization_module"] = _binding(
            root, "src/kds/data/v4_final_salvage_materialization.py"
        )
        inputs["runner_script"] = _binding(root, "scripts/materialize_v4_final_salvage_inputs.py")
        if set(inputs) != {
            "metadata_plan",
            "metadata_receipt",
            "metadata_selection",
            "recovery_selection",
            "salvage_selection",
            "permanent_rejects",
            "salvage_authorization",
            "recovery_materialization_plan",
            "recovery_failure_receipt",
            "recovery_qwen_journal",
            "recovery_kk_journal",
            "materialization_ledger",
            "fleurs_artifact_lock",
            "qwen_model_lock",
            "kazakhtts_model_lock",
            "train_manifest",
            "dev_manifest",
            "historical_fingerprint_inventory",
            "source_decode_inventory",
            "kk_spoof_decode_inventory",
            "dev_source_decode_journal",
            "dev_spoof_decode_journal",
            "calibration_source_decode_journal",
            "calibration_spoof_decode_journal",
            "final_inputs_module",
            "base_materialization_module",
            "recovery_materialization_module",
            "salvage_materialization_module",
            "audio_gate_module",
            "common_voice_module",
            "fleurs_module",
            "qwen_module",
            "qwen_recovery_module",
            "kazakhtts_module",
            "kazakhtts_inference_module",
            "runner_script",
        }:
            raise SalvageFreezeError("Salvage plan input set drifted.")
        for name in (
            "salvage_selection",
            "permanent_rejects",
            "salvage_authorization",
            "materialization_ledger",
        ):
            bound = cast(dict[str, object], inputs[name])
            bound["path"] = {
                "salvage_selection": SALVAGE_SELECTION,
                "permanent_rejects": PERMANENT_REJECTS,
                "salvage_authorization": SALVAGE_AUTHORIZATION,
                "materialization_ledger": SALVAGE_LEDGER,
            }[name]
        plan = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "created_at": created_at,
            "inputs": inputs,
            "working": {
                "raw_roots": RAW_ROOTS,
                "processed_root": PROCESSED_ROOT,
                "runtime_root": RUNTIME_ROOT,
                "model_roots": MODEL_ROOTS,
            },
            "outputs": OUTPUTS,
            "prohibitions": {name: True for name in sorted(_PROHIBITIONS)},
        }
        staged[SALVAGE_PLAN].write_text(
            json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for relative in expected:
            staged[relative].replace(_path(root, relative))
    return {relative: sha256_file(_path(root, relative)) for relative in expected}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--created-at", required=True)
    arguments = parser.parse_args()
    try:
        result = freeze(project_root=arguments.project_root, created_at=arguments.created_at)
    except (OSError, ValueError) as error:
        print(json.dumps({"status": "error", "issues": [str(error)]}, ensure_ascii=False))
        return 2
    print(json.dumps({"status": "ok", "outputs": result}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
