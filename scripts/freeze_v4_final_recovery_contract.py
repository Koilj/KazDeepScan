"""Freeze the metadata-only v4 final recovery contract after failed Qwen rank 1."""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from pathlib import Path
from typing import cast

from kds.data.assets import sha256_file
from kds.data.v4_final_inputs import V4_FINAL_SELECTION_FIELDS
from kds.data.v4_final_recovery_materialization import (
    _PROHIBITIONS,
    KK_SOURCE_ID,
    KK_SPOOF_ID,
    MODEL_ROOTS,
    OUTPUTS,
    PROCESSED_ROOT,
    PROTOCOL_ID,
    RAW_ROOTS,
    RU_SOURCE_ID,
    RU_SPOOF_ID,
    RUNTIME_ROOT,
)

OLD_PLAN = "configs/research/v4/xlsr_sls_model_v4_final_materialization_v1.json"
FAILURE_RECEIPT = "docs/artifacts/v4/final_materialization_attempt_failure_2026-08-15.md"
FAILED_QWEN_JOURNAL = (
    "artifacts/v4/xlsr_sls_model_v4_final_materialization_v1/ru_qwen_one_shot.jsonl"
)
RECOVERY_SELECTION = "data/manifests/v4/xlsr_sls_model_v4_final_recovery_metadata_v1.csv"
RECOVERY_AUTHORIZATION = "docs/artifacts/v4/v4_final_recovery_authorization_2026-08-15.json"
RECOVERY_LEDGER = "data/licenses/frozen/xlsr_sls_model_v4_final_recovery_materialization_v1.csv"
RECOVERY_PLAN = "configs/research/v4/xlsr_sls_model_v4_final_recovery_materialization_v1.json"
REVALIDATED_RECOVERY_PLAN = (
    "configs/research/v4/xlsr_sls_model_v4_final_recovery_materialization_v1_revalidated.json"
)


class RecoveryFreezeError(ValueError):
    """Raised when recovery metadata would enlarge or rewrite the failed scope."""


def _project_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RecoveryFreezeError(f"Path escapes the project root: {relative}") from error
    return path


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RecoveryFreezeError(f"Cannot read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise RecoveryFreezeError(f"{label} must be a JSON object.")
    return cast(dict[str, object], payload)


def _row_count(path: Path) -> int | None:
    if path.suffix != ".csv":
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _binding(root: Path, relative: str, rows: int | None = None) -> dict[str, object]:
    path = _project_path(root, relative)
    if not path.is_file():
        raise RecoveryFreezeError(f"Missing frozen recovery input: {relative}")
    return {"path": relative, "sha256": sha256_file(path), "rows": rows}


def _staged_binding(path: Path, relative: str, rows: int | None = None) -> dict[str, object]:
    if not path.is_file():
        raise RecoveryFreezeError(f"Missing staged recovery input: {relative}")
    return {"path": relative, "sha256": sha256_file(path), "rows": rows}


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _recovery_ledger(root: Path) -> list[dict[str, str]]:
    original = _project_path(
        root, "data/licenses/frozen/xlsr_sls_model_v4_final_materialization_v1.csv"
    )
    with original.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        rows = list(reader)
    expected = {
        "common_voice_ru_v24_v4_final",
        "google_fleurs_kk_v1_v4_final",
        "qwen3_tts_customvoice_aiden_v4_final",
        "issai_kazakhtts2_male2_tacotron2_pwg_v4_final",
    }
    if not rows or {row.get("source_id") for row in rows} != expected:
        raise RecoveryFreezeError("Historical final materialization ledger changed.")
    source_ids = {
        "common_voice_ru_v24_v4_final": RU_SOURCE_ID,
        "google_fleurs_kk_v1_v4_final": KK_SOURCE_ID,
        "qwen3_tts_customvoice_aiden_v4_final": RU_SPOOF_ID,
        "issai_kazakhtts2_male2_tacotron2_pwg_v4_final": KK_SPOOF_ID,
    }
    scope = {
        RU_SOURCE_ID: (
            "Permits recovery extraction/QA/isolation/review/pair lock only for 499 "
            "unattempted RU rows; original rank 1 is irrecoverably rejected. "
            "No replacement or backfill."
        ),
        KK_SOURCE_ID: (
            "Permits recovery extraction/QA/isolation/review/pair lock only for 500 "
            "unattempted KK rows; no replacement or backfill."
        ),
        RU_SPOOF_ID: (
            "Permits one text-only recovery synthesis per 499 unattempted RU row. "
            "Rank 1 is forbidden; no reference audio, resynthesis, replacement or backfill."
        ),
        KK_SPOOF_ID: (
            "Permits one text-only recovery synthesis per 500 unattempted KK row; "
            "no reference audio, resynthesis, replacement or backfill."
        ),
    }
    converted: list[dict[str, str]] = []
    for row in rows:
        source_id = source_ids[cast(str, row["source_id"])]
        item = dict(row)
        item["source_id"] = source_id
        item["notes"] = scope[source_id]
        converted.append(item)
    if fields != tuple(converted[0]):
        raise RecoveryFreezeError("Historical ledger columns changed.")
    return converted


def freeze(*, project_root: Path, created_at: str) -> dict[str, str]:
    root = project_root.resolve(strict=True)
    original_plan = _read_json(_project_path(root, OLD_PLAN), "failed materialization plan")
    original_inputs = cast(dict[str, object], original_plan.get("inputs"))
    original_selection_path = cast(dict[str, object], original_inputs["metadata_selection"])["path"]
    if not isinstance(original_selection_path, str):
        raise RecoveryFreezeError("Failed plan metadata selection is invalid.")
    selection_source = _project_path(root, original_selection_path)
    with selection_source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != V4_FINAL_SELECTION_FIELDS:
            raise RecoveryFreezeError("Original final metadata selection schema changed.")
        rows = list(reader)
    rejected = [row for row in rows if row["language"] == "ru" and row["selection_rank"] == "1"]
    recovery = [row for row in rows if row not in rejected]
    if len(rows) != 1000 or len(rejected) != 1 or len(recovery) != 999:
        raise RecoveryFreezeError("Recovery must exclude exactly one RU rank-1 row.")
    journal_path = _project_path(root, FAILED_QWEN_JOURNAL)
    try:
        events = [
            json.loads(line) for line in journal_path.read_text(encoding="utf-8").splitlines()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RecoveryFreezeError("Failed Qwen journal is unreadable.") from error
    if len(events) != 1 or events[0] != {
        "event": "planned",
        "output": "ru_qwen_001_3c6fd2b8ea62.wav",
        "sample_id": rejected[0]["sample_id"],
    }:
        raise RecoveryFreezeError("Failed Qwen journal does not bind the irrecoverable rank-1 row.")
    targets = (RECOVERY_SELECTION, RECOVERY_AUTHORIZATION, RECOVERY_LEDGER, RECOVERY_PLAN)
    if any(_project_path(root, relative).exists() for relative in targets):
        raise RecoveryFreezeError("Recovery contract outputs are write-once and already exist.")
    if any(not _project_path(root, relative).parent.is_dir() for relative in targets):
        raise RecoveryFreezeError("Recovery contract output parent is missing.")
    with tempfile.TemporaryDirectory(
        prefix=".kds-v4-final-recovery-freeze-", dir=root / "configs"
    ) as stage_name:
        stage = Path(stage_name)
        staged = {relative: stage / Path(relative).name for relative in targets}
        _write_csv(staged[RECOVERY_SELECTION], V4_FINAL_SELECTION_FIELDS, recovery)
        failure_path = _project_path(root, FAILURE_RECEIPT)
        authorization = {
            "schema_version": 1,
            "protocol_id": "xlsr-sls-model-v4-final-recovery-authorization-v1",
            "created_at": created_at,
            "status": "rank_one_irrecoverable_reject_recovery_authorized",
            "failed_attempt": {
                "plan_path": OLD_PLAN,
                "plan_sha256": sha256_file(_project_path(root, OLD_PLAN)),
                "failure_receipt_path": FAILURE_RECEIPT,
                "failure_receipt_sha256": sha256_file(failure_path),
                "qwen_journal_path": FAILED_QWEN_JOURNAL,
                "qwen_journal_sha256": sha256_file(journal_path),
            },
            "irrecoverable_reject": {
                "language": "ru",
                "selection_rank": 1,
                "sample_id": rejected[0]["sample_id"],
                "text_id": rejected[0]["text_id"],
                "text_hash": rejected[0]["text_hash"],
                "synthesis_text_sha256": rejected[0]["synthesis_text_sha256"],
                "resynthesis_forbidden": True,
                "replacement_or_backfill_forbidden": True,
            },
            "claims": {
                "original_contract_remains_immutable": True,
                "only_previously_unattempted_rows_authorized": True,
                "recovery_selected_rows": {"ru": 499, "kk": 500},
                "detector_checkpoint_loading_authorized": False,
                "detector_inference_authorized": False,
                "final_inference_authorized": False,
            },
        }
        staged[RECOVERY_AUTHORIZATION].write_text(
            json.dumps(authorization, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        ledger_rows = _recovery_ledger(root)
        _write_csv(staged[RECOVERY_LEDGER], tuple(ledger_rows[0]), ledger_rows)
        retained = (
            "metadata_plan",
            "metadata_receipt",
            "metadata_selection",
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
            "audio_gate_module",
            "common_voice_module",
            "fleurs_module",
            "qwen_module",
            "kazakhtts_module",
            "kazakhtts_inference_module",
        )
        inputs: dict[str, dict[str, object]] = {}
        for name in retained:
            source = cast(dict[str, object], original_inputs[name])
            relative = cast(str, source["path"])
            inputs[name] = _binding(root, relative, cast(int | None, source["rows"]))
        inputs.update(
            {
                "recovery_selection": _staged_binding(
                    staged[RECOVERY_SELECTION], RECOVERY_SELECTION, 999
                ),
                "recovery_authorization": _staged_binding(
                    staged[RECOVERY_AUTHORIZATION], RECOVERY_AUTHORIZATION
                ),
                "failed_materialization_plan": _binding(root, OLD_PLAN),
                "failed_materialization_failure_receipt": _binding(root, FAILURE_RECEIPT),
                "failed_qwen_journal": _binding(root, FAILED_QWEN_JOURNAL),
                "materialization_ledger": _staged_binding(
                    staged[RECOVERY_LEDGER], RECOVERY_LEDGER, 4
                ),
                "base_materialization_module": _binding(
                    root, "src/kds/data/v4_final_materialization.py"
                ),
                "recovery_materialization_module": _binding(
                    root, "src/kds/data/v4_final_recovery_materialization.py"
                ),
                "qwen_recovery_module": _binding(
                    root, "src/kds/data/qwen3_tts_customvoice_recovery.py"
                ),
                "runner_script": _binding(root, "scripts/materialize_v4_final_recovery_inputs.py"),
            }
        )
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
        staged[RECOVERY_PLAN].write_text(
            json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        for relative in targets:
            staged[relative].replace(_project_path(root, relative))
    return {relative: sha256_file(_project_path(root, relative)) for relative in targets}


def revalidate(*, project_root: Path, created_at: str) -> dict[str, str]:
    """Publish a new unused-plan binding after a code-only validation correction."""

    root = project_root.resolve(strict=True)
    source = _project_path(root, RECOVERY_PLAN)
    target = _project_path(root, REVALIDATED_RECOVERY_PLAN)
    if target.exists() or not target.parent.is_dir():
        raise RecoveryFreezeError("Revalidated recovery plan output is not new.")
    plan = _read_json(source, "original recovery plan")
    inputs = cast(dict[str, object], plan.get("inputs"))
    rebound: dict[str, dict[str, object]] = {}
    for name, raw_binding in inputs.items():
        binding = cast(dict[str, object], raw_binding)
        relative = binding.get("path")
        rows = binding.get("rows")
        if not isinstance(relative, str) or (rows is not None and not isinstance(rows, int)):
            raise RecoveryFreezeError("Original recovery input binding is invalid.")
        rebound[name] = _binding(root, relative, rows)
    plan["created_at"] = created_at
    plan["inputs"] = rebound
    target.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {REVALIDATED_RECOVERY_PLAN: sha256_file(target)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--revalidate", action="store_true")
    arguments = parser.parse_args()
    try:
        operation = revalidate if arguments.revalidate else freeze
        result = operation(project_root=arguments.project_root, created_at=arguments.created_at)
    except (OSError, ValueError, RecoveryFreezeError) as error:
        print(json.dumps({"status": "error", "issues": [str(error)]}, ensure_ascii=False))
        return 2
    print(json.dumps({"status": "ok", "outputs": result}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
