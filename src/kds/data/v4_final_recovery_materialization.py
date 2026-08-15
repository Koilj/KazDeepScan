"""One-shot recovery materialization for the unattempted v4 final rows only.

The preceding v4 final contract is immutable and exhausted after Qwen rank 1
failed before writing its WAV.  This adapter never reuses that namespace.  It
accepts exactly the metadata subset that excludes the irrecoverable RU rank 1,
then reuses the proven extraction/QA/isolation/review machinery under a fresh
hash-pinned namespace.  Detector and final inference are intentionally absent.
"""

# ruff: noqa: E501

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath
from typing import cast

import soundfile as sf  # type: ignore[import-untyped]

from kds.data import v4_final_materialization as base
from kds.data.assets import require_valid_assets, resolve_asset_path, sha256_file
from kds.data.licenses import load_license_ledger, validate_manifest_licenses
from kds.data.manifest import ManifestRow, validate_manifest, write_manifest
from kds.data.qwen3_tts_customvoice_recovery import (
    load_recovery_qwen3_tts_customvoice,
    synthesize_to_absolute_file,
)
from kds.data.research_tts import (
    load_research_tts_model_lock,
    verify_research_tts_model_lock,
)
from kds.data.v4_audio_gate import (
    V4DecodeResult,
    V4DecodeTask,
    append_v4_decode_journal,
    load_v4_decode_journal,
    run_v4_decode_task,
)
from kds.data.v4_final_inputs import V4_FINAL_SELECTION_FIELDS

PROTOCOL_ID = "xlsr-sls-model-v4-final-recovery-materialization-v1"
RECOVERY_AUTHORIZATION_PROTOCOL_ID = "xlsr-sls-model-v4-final-recovery-authorization-v1"
RU_SOURCE_ID = "common_voice_ru_v24_v4_final_recovery"
KK_SOURCE_ID = "google_fleurs_kk_v1_v4_final_recovery"
RU_SPOOF_ID = "qwen3_tts_customvoice_aiden_v4_final_recovery"
KK_SPOOF_ID = "issai_kazakhtts2_male2_tacotron2_pwg_v4_final_recovery"

RAW_ROOTS = {
    "ru_source": "raw/v4/xlsr_sls_model_v4_final_recovery_materialization_v1/ru_source",
    "kk_source": "raw/v4/xlsr_sls_model_v4_final_recovery_materialization_v1/kk_source",
    "ru_spoof": "raw/v4/xlsr_sls_model_v4_final_recovery_materialization_v1/ru_qwen",
    "kk_spoof": "raw/v4/xlsr_sls_model_v4_final_recovery_materialization_v1/kk_kazakhtts",
}
PROCESSED_ROOT = "processed/v4/xlsr_sls_model_v4_final_recovery_materialization_v1"
RUNTIME_ROOT = "artifacts/v4/xlsr_sls_model_v4_final_recovery_materialization_v1"
MODEL_ROOTS = {
    "qwen": "models/research/voxforge_ru_mdc_qwen3_tts_customvoice_aiden",
    "kazakhtts": "models/research/kazakhtts_tacotron2_pwg_v1",
}
OUTPUTS = {
    "ru_source_raw_manifest": "data/manifests/v4/xlsr_sls_model_v4_final_recovery_ru_source_raw_v1.csv",
    "kk_source_raw_manifest": "data/manifests/v4/xlsr_sls_model_v4_final_recovery_kk_source_raw_v1.csv",
    "ru_spoof_raw_manifest": "data/manifests/v4/xlsr_sls_model_v4_final_recovery_ru_qwen_raw_v1.csv",
    "kk_spoof_raw_manifest": "data/manifests/v4/xlsr_sls_model_v4_final_recovery_kk_kazakhtts_raw_v1.csv",
    "ru_source_ready_manifest": "data/manifests/v4/xlsr_sls_model_v4_final_recovery_ru_source_ready_v1.csv",
    "kk_source_ready_manifest": "data/manifests/v4/xlsr_sls_model_v4_final_recovery_kk_source_ready_v1.csv",
    "ru_spoof_ready_manifest": "data/manifests/v4/xlsr_sls_model_v4_final_recovery_ru_qwen_ready_v1.csv",
    "kk_spoof_ready_manifest": "data/manifests/v4/xlsr_sls_model_v4_final_recovery_kk_kazakhtts_ready_v1.csv",
    "audio_inventory": "data/manifests/v4/xlsr_sls_model_v4_final_recovery_audio_gate_inventory_v1.csv",
    "review_packet": "data/manifests/v4/xlsr_sls_model_v4_final_recovery_acoustic_language_packet_v1.csv",
    "reviewer_a_template": "data/manifests/v4/xlsr_sls_model_v4_final_recovery_acoustic_language_reviewer_a_v1.csv",
    "reviewer_b_template": "data/manifests/v4/xlsr_sls_model_v4_final_recovery_acoustic_language_reviewer_b_v1.csv",
    "materialization_receipt": "docs/artifacts/v4/xlsr_sls_model_v4_final_recovery_materialization_v1.json",
    "pair_lock_manifest": "data/manifests/v4/xlsr_sls_model_v4_final_recovery_pairs_frozen_v1.csv",
    "pair_lock_receipt": "docs/artifacts/v4/xlsr_sls_model_v4_final_recovery_pair_lock_v1.json",
}
_REQUIRED_INPUTS = {
    "metadata_plan",
    "metadata_receipt",
    "metadata_selection",
    "recovery_selection",
    "recovery_authorization",
    "failed_materialization_plan",
    "failed_materialization_failure_receipt",
    "failed_qwen_journal",
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
    "audio_gate_module",
    "common_voice_module",
    "fleurs_module",
    "qwen_module",
    "qwen_recovery_module",
    "kazakhtts_module",
    "kazakhtts_inference_module",
    "runner_script",
}
_PROHIBITIONS = {
    "network_downloads",
    "detector_checkpoint_loading",
    "calibration",
    "temperature_fitting",
    "final_inference",
    "detector_inference",
    "detector_feedback",
    "output_overwrite",
    "resynthesis",
    "replacement_or_backfill",
    "reuse_failed_rank_one_audio_or_text",
    "pair_lock_before_two_reviews",
}


class V4FinalRecoveryError(ValueError):
    """Raised when the separate recovery boundary cannot be proven."""


def _object(path: Path, label: str) -> dict[str, object]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V4FinalRecoveryError(f"Cannot read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise V4FinalRecoveryError(f"{label} must be a JSON object.")
    return cast(dict[str, object], payload)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise V4FinalRecoveryError(f"{label} must be a JSON object.")
    return cast(dict[str, object], value)


def _safe_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise V4FinalRecoveryError(f"{label} must be a non-empty project-relative path.")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or ".." in parsed.parts or "\\" in value or value == ".":
        raise V4FinalRecoveryError(f"{label} is not a safe project-relative path.")
    return parsed.as_posix()


def _project_path(root: Path, relative: str, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise V4FinalRecoveryError(f"{label} escapes the project root.") from error
    return path


def _verify(binding: base.Binding, root: Path, label: str) -> Path:
    try:
        return base._verify(binding, root, label)
    except base.V4FinalMaterializationError as error:
        raise V4FinalRecoveryError(str(error)) from error


def _read_recovery_selection(path: Path) -> tuple[base.SelectedRow, ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != V4_FINAL_SELECTION_FIELDS:
                raise V4FinalRecoveryError("Recovery selection schema changed.")
            source = tuple(reader)
    except OSError as error:
        raise V4FinalRecoveryError("Cannot read recovery selection.") from error
    rows: list[base.SelectedRow] = []
    for number, item in enumerate(source, start=2):
        try:
            rows.append(
                base.SelectedRow(
                    language=(item.get("language") or "").strip(),
                    selection_rank=int(item.get("selection_rank") or ""),
                    sample_id=(item.get("sample_id") or "").strip(),
                    source_member=(item.get("source_member") or "").strip(),
                    source_split=(item.get("source_split") or "").strip(),
                    parent_group_id=(item.get("parent_group_id") or "").strip(),
                    speaker_pseudo_id=(item.get("speaker_pseudo_id") or "").strip(),
                    text_id=(item.get("text_id") or "").strip(),
                    text_hash=base._sha(item.get("text_hash"), "recovery text hash"),
                    synthesis_text_sha256=base._sha(
                        item.get("synthesis_text_sha256"), "recovery synthesis text hash"
                    ),
                    synthesis_seed=(item.get("synthesis_seed") or "").strip(),
                    normalization_operations=(item.get("normalization_operations") or "").strip(),
                )
            )
        except (TypeError, ValueError) as error:
            raise V4FinalRecoveryError(f"Recovery selection row {number} is invalid.") from error
    return tuple(rows)


def _original_plan(plan: base.Plan) -> base.Plan:
    return base.Plan(
        path=plan.path,
        sha256=plan.sha256,
        created_at=plan.created_at,
        inputs=plan.inputs,
        raw_roots=RAW_ROOTS,
        processed_root=PROCESSED_ROOT,
        runtime_root=RUNTIME_ROOT,
        model_roots=MODEL_ROOTS,
        outputs=OUTPUTS,
    )


def _validate_authorization(plan: base.Plan, root: Path) -> None:
    authorization = _object(
        _verify(plan.inputs["recovery_authorization"], root, "recovery authorization"),
        "recovery authorization",
    )
    expected = {
        "schema_version",
        "protocol_id",
        "created_at",
        "status",
        "failed_attempt",
        "irrecoverable_reject",
        "claims",
    }
    if (
        set(authorization) != expected
        or authorization["schema_version"] != 1
        or authorization["protocol_id"] != RECOVERY_AUTHORIZATION_PROTOCOL_ID
        or authorization["status"] != "rank_one_irrecoverable_reject_recovery_authorized"
    ):
        raise V4FinalRecoveryError("Recovery authorization schema/status is invalid.")
    failed = _mapping(authorization["failed_attempt"], "failed attempt")
    reject = _mapping(authorization["irrecoverable_reject"], "irrecoverable reject")
    claims = _mapping(authorization["claims"], "recovery claims")
    if (
        failed.get("plan_path") != plan.inputs["failed_materialization_plan"].path
        or failed.get("plan_sha256") != plan.inputs["failed_materialization_plan"].sha256
        or failed.get("failure_receipt_path")
        != plan.inputs["failed_materialization_failure_receipt"].path
        or failed.get("failure_receipt_sha256")
        != plan.inputs["failed_materialization_failure_receipt"].sha256
        or failed.get("qwen_journal_path") != plan.inputs["failed_qwen_journal"].path
        or failed.get("qwen_journal_sha256") != plan.inputs["failed_qwen_journal"].sha256
        or reject.get("language") != "ru"
        or reject.get("selection_rank") != 1
        or reject.get("resynthesis_forbidden") is not True
        or reject.get("replacement_or_backfill_forbidden") is not True
        or claims.get("only_previously_unattempted_rows_authorized") is not True
        or claims.get("final_inference_authorized") is not False
    ):
        raise V4FinalRecoveryError(
            "Recovery authorization does not bind the failed rank-1 attempt."
        )
    journal = _verify(plan.inputs["failed_qwen_journal"], root, "failed Qwen journal")
    try:
        events = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V4FinalRecoveryError("Failed Qwen journal is unreadable.") from error
    if len(events) != 1 or events[0].get("event") != "planned":
        raise V4FinalRecoveryError("Failed Qwen journal must contain exactly one planned event.")
    if reject.get("sample_id") != events[0].get("sample_id"):
        raise V4FinalRecoveryError("Recovery reject does not identify the journaled Qwen attempt.")
    failure = _verify(
        plan.inputs["failed_materialization_failure_receipt"], root, "failure receipt"
    ).read_text(encoding="utf-8")
    if "rank `1`" not in failure or "resynthesis" not in failure:
        raise V4FinalRecoveryError("Failure receipt does not forbid rank-1 resynthesis.")


def load_plan(path: Path, project_root: Path) -> base.Plan:
    """Load the recovery contract and prove it cannot expand the failed scope."""

    root = project_root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise V4FinalRecoveryError("Recovery plan must be below the project root.") from error
    raw = _object(resolved, "v4 final recovery plan")
    expected = {
        "schema_version",
        "protocol_id",
        "created_at",
        "inputs",
        "working",
        "outputs",
        "prohibitions",
    }
    if set(raw) != expected or raw["schema_version"] != 1 or raw["protocol_id"] != PROTOCOL_ID:
        raise V4FinalRecoveryError("Recovery plan schema/protocol is invalid.")
    raw_inputs = _mapping(raw["inputs"], "recovery inputs")
    if set(raw_inputs) != _REQUIRED_INPUTS:
        raise V4FinalRecoveryError("Recovery plan inputs are incomplete.")
    inputs = {name: base._binding(value, f"inputs.{name}") for name, value in raw_inputs.items()}
    working = _mapping(raw["working"], "recovery working")
    if set(working) != {"raw_roots", "processed_root", "runtime_root", "model_roots"}:
        raise V4FinalRecoveryError("Recovery working section is invalid.")
    raw_roots = {
        name: _safe_path(value, f"raw root {name}")
        for name, value in _mapping(working["raw_roots"], "raw roots").items()
    }
    model_roots = {
        name: _safe_path(value, f"model root {name}")
        for name, value in _mapping(working["model_roots"], "model roots").items()
    }
    if (
        raw_roots != RAW_ROOTS
        or _safe_path(working["processed_root"], "processed root") != PROCESSED_ROOT
        or _safe_path(working["runtime_root"], "runtime root") != RUNTIME_ROOT
        or model_roots != MODEL_ROOTS
    ):
        raise V4FinalRecoveryError("Recovery namespaces or model roots changed.")
    raw_outputs = _mapping(raw["outputs"], "recovery outputs")
    outputs = {name: _safe_path(value, f"output {name}") for name, value in raw_outputs.items()}
    if outputs != OUTPUTS:
        raise V4FinalRecoveryError("Recovery output paths changed.")
    prohibitions = _mapping(raw["prohibitions"], "recovery prohibitions")
    if set(prohibitions) != _PROHIBITIONS or any(
        value is not True for value in prohibitions.values()
    ):
        raise V4FinalRecoveryError("Recovery prohibitions are not fail-closed.")
    for name, binding in inputs.items():
        _verify(binding, root, f"inputs.{name}")
    plan = base.Plan(
        path=relative,
        sha256=sha256_file(resolved),
        created_at=base._timestamp(raw["created_at"], "created_at"),
        inputs=inputs,
        raw_roots=raw_roots,
        processed_root=PROCESSED_ROOT,
        runtime_root=RUNTIME_ROOT,
        model_roots=model_roots,
        outputs=outputs,
    )
    _validate_authorization(plan, root)
    _load_selection(plan, root)
    return plan


def _load_selection(plan: base.Plan, root: Path) -> tuple[base.SelectedRow, ...]:
    """Return exactly the original frozen selection minus attempted RU rank 1."""

    try:
        original = base._load_selection(_original_plan(plan), root)
    except base.V4FinalMaterializationError as error:
        raise V4FinalRecoveryError(str(error)) from error
    recovery = _read_recovery_selection(
        _verify(plan.inputs["recovery_selection"], root, "recovery selection")
    )
    expected = tuple(
        row for row in original if not (row.language == "ru" and row.selection_rank == 1)
    )
    if recovery != expected or len(recovery) != 999:
        raise V4FinalRecoveryError(
            "Recovery selection must be exactly the frozen 1000-row selection minus RU rank 1."
        )
    if any(row.language == "ru" and row.selection_rank == 1 for row in recovery):
        raise V4FinalRecoveryError("The attempted RU rank 1 is not eligible for recovery.")
    if (
        sum(row.language == "ru" for row in recovery) != 499
        or sum(row.language == "kk" for row in recovery) != 500
    ):
        raise V4FinalRecoveryError("Recovery selection language counts changed.")
    return recovery


@contextmanager
def _base_contract_context() -> Iterator[None]:
    """Use proven helpers with recovery-only identities, restored after every command."""

    replacements = {
        "PROTOCOL_ID": PROTOCOL_ID,
        "RU_SOURCE_ID": RU_SOURCE_ID,
        "KK_SOURCE_ID": KK_SOURCE_ID,
        "RU_SPOOF_ID": RU_SPOOF_ID,
        "KK_SPOOF_ID": KK_SPOOF_ID,
    }
    previous = {name: getattr(base, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(base, name, value)


def _preflight_tts_routes(plan: base.Plan, root: Path) -> None:
    with _base_contract_context():
        base._preflight_tts_routes(plan, root)
    lock = load_research_tts_model_lock(_verify(plan.inputs["qwen_model_lock"], root, "Qwen lock"))
    if len(lock.models) != 1:
        raise V4FinalRecoveryError("Recovery Qwen lock must contain exactly one route.")
    load_recovery_qwen3_tts_customvoice(
        _project_path(root, plan.model_roots["qwen"], "Qwen model root"), lock.models[0]
    )


def preflight_materialization(
    *,
    plan_path: Path,
    project_root: Path,
    data_root: Path,
    common_voice_archive: Path,
    fleurs_release_root: Path,
    created_at: str,
) -> base.Plan:
    """Validate every non-audio dependency before the recovery one-shot pass."""

    base._timestamp(created_at, "created_at")
    root = project_root.resolve(strict=True)
    plan = load_plan(plan_path, root)
    with _base_contract_context():
        base._require_ledger(plan, root)
        if data_root.resolve(strict=True) != _project_path(root, "data", "data root"):
            raise V4FinalRecoveryError("data_root must be the project data directory.")
        base._verify_common_voice_archive(common_voice_archive)
        selected = _load_selection(plan, root)
        base._validate_selected_source_metadata(selected, common_voice_archive, fleurs_release_root)
        base._history(plan, root)
    runtime = _project_path(root, plan.runtime_root, "recovery runtime root")
    if runtime.exists():
        raise V4FinalRecoveryError("Recovery runtime namespace already exists.")
    for value in plan.raw_roots.values():
        if (data_root / value).exists():
            raise V4FinalRecoveryError("Recovery raw materialization destination is not new.")
    for relative in plan.outputs.values():
        target = _project_path(root, relative, "recovery output")
        if target.exists() or not target.parent.is_dir():
            raise V4FinalRecoveryError("Every recovery output must be new with an existing parent.")
    _preflight_tts_routes(plan, root)
    return plan


def _synthesize_ru(
    rows: Sequence[ManifestRow],
    text_by_id: Mapping[str, str],
    selected_by_id: Mapping[str, base.SelectedRow],
    plan: base.Plan,
    root: Path,
    data_root: Path,
    created_at: str,
) -> tuple[ManifestRow, ...]:
    lock = load_research_tts_model_lock(_verify(plan.inputs["qwen_model_lock"], root, "Qwen lock"))
    if len(lock.models) != 1:
        raise V4FinalRecoveryError("Recovery Qwen lock must contain exactly one route.")
    model = lock.models[0]
    model_root = _project_path(root, plan.model_roots["qwen"], "Qwen model root")
    verify_research_tts_model_lock(model_root, lock)
    runtime = load_recovery_qwen3_tts_customvoice(model_root, model)
    destination = data_root / plan.raw_roots["ru_spoof"]
    journal = (
        _project_path(root, plan.runtime_root, "recovery runtime root") / "ru_qwen_one_shot.jsonl"
    )
    if destination.exists() or journal.exists():
        raise V4FinalRecoveryError(
            "Recovery Qwen namespace already exists; resynthesis is forbidden."
        )
    destination.mkdir(parents=True)
    journal.parent.mkdir(parents=True, exist_ok=True)
    output: list[ManifestRow] = []
    with _base_contract_context():
        for item in sorted(rows, key=lambda row: selected_by_id[row.sample_id].selection_rank):
            selected = selected_by_id[item.sample_id]
            prepared = runtime.prepare_text(text_by_id[item.sample_id])
            if (
                str(prepared.seed) != selected.synthesis_seed
                or hashlib.sha256(prepared.source_text.encode("utf-8")).hexdigest()
                != selected.synthesis_text_sha256
            ):
                raise V4FinalRecoveryError(
                    "Recovery Qwen text or seed diverges from frozen metadata."
                )
            name = f"ru_qwen_{selected.selection_rank:03d}_{item.text_hash[:12]}.wav"
            path = destination / name
            base._append_jsonl(
                journal, {"event": "planned", "sample_id": item.sample_id, "output": name}
            )
            synthesize_to_absolute_file(runtime, prepared, path)
            info = sf.info(path)
            if info.samplerate != runtime.sample_rate or info.channels != 1 or info.duration <= 0:
                raise V4FinalRecoveryError("Recovery Qwen produced invalid locked WAV.")
            row = base._spoof_row(
                item,
                model,
                RU_SPOOF_ID,
                path.relative_to(data_root).as_posix(),
                sha256_file(path),
                float(info.duration),
                int(info.samplerate),
                created_at,
                str(prepared.seed),
                selected.synthesis_text_sha256,
                "cuda:0",
            )
            output.append(replace(row, voice_id="qwen3_tts_customvoice:aiden"))
            base._append_jsonl(
                journal,
                {
                    "event": "generated",
                    "sample_id": row.sample_id,
                    "base_sample_id": item.sample_id,
                    "audio_sha256": row.sha256,
                },
            )
    validate_manifest(output)
    return tuple(output)


def _decode(
    rows: Sequence[ManifestRow],
    namespace: str,
    plan: base.Plan,
    data_root: Path,
    runtime_root: Path,
    workers: int,
) -> dict[str, V4DecodeResult]:
    journal = runtime_root / f"{namespace}_decode_qa.jsonl"
    tasks = {
        row.sample_id: V4DecodeTask(
            row.sample_id,
            row.relative_path,
            row.sha256,
            str(resolve_asset_path(data_root, row.relative_path)),
            f"{plan.processed_root}/{namespace}/{row.sha256[:2]}/{row.sha256}.wav",
            str(data_root / plan.processed_root / namespace / row.sha256[:2] / f"{row.sha256}.wav"),
        )
        for row in rows
    }
    completed = load_v4_decode_journal(journal, tasks)
    missing = [task for sample_id, task in tasks.items() if sample_id not in completed]
    if not missing:
        return completed
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures: dict[Future[V4DecodeResult], V4DecodeTask] = {
            pool.submit(run_v4_decode_task, task): task for task in missing
        }
        for number, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            append_v4_decode_journal(journal, result)
            completed[result.sample_id] = result
            if number % 50 == 0 or number == len(missing):
                print(
                    json.dumps(
                        {
                            "status": "progress",
                            "stage": namespace,
                            "completed": len(completed),
                            "total": len(tasks),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    return completed


def _stage_outputs(root: Path, outputs: Mapping[str, str]) -> Path:
    targets = [_project_path(root, value, f"output {name}") for name, value in outputs.items()]
    if any(path.exists() or not path.parent.is_dir() for path in targets):
        raise V4FinalRecoveryError("Recovery outputs must be new with existing parents.")
    return Path(tempfile.mkdtemp(prefix=".kds-v4-final-recovery-", dir=targets[0].parent))


def run_materialization(
    *,
    plan_path: Path,
    project_root: Path,
    data_root: Path,
    common_voice_archive: Path,
    fleurs_release_root: Path,
    workers: int,
    created_at: str,
) -> base.Plan:
    """Run the sole recovery extraction/synthesis pass; it never locks pairs."""

    if workers <= 0:
        raise V4FinalRecoveryError("workers must be positive.")
    root = project_root.resolve(strict=True)
    plan = preflight_materialization(
        plan_path=plan_path,
        project_root=root,
        data_root=data_root,
        common_voice_archive=common_voice_archive,
        fleurs_release_root=fleurs_release_root,
        created_at=created_at,
    )
    selected = _load_selection(plan, root)
    runtime_root = _project_path(root, plan.runtime_root, "recovery runtime root")
    runtime_root.mkdir(parents=True)
    with _base_contract_context():
        ru_source, kk_source, source_text, synthesis_text = base._source_rows(
            selected, common_voice_archive, fleurs_release_root, data_root, plan, created_at
        )
    selected_by_id = {row.sample_id: row for row in selected}
    ru_spoof = _synthesize_ru(
        ru_source, source_text, selected_by_id, plan, root, data_root, created_at
    )
    with _base_contract_context():
        kk_spoof = base._synthesize_kk(
            kk_source, source_text, selected_by_id, plan, root, data_root, created_at
        )
    groups = {
        "ru_source": ru_source,
        "kk_source": kk_source,
        "ru_spoof": ru_spoof,
        "kk_spoof": kk_spoof,
    }
    ledger = load_license_ledger(
        _verify(plan.inputs["materialization_ledger"], root, "recovery ledger")
    )
    for rows in groups.values():
        validate_manifest(rows)
        validate_manifest_licenses(rows, ledger)
        require_valid_assets(rows, data_root)
    decoded = {
        name: _decode(rows, name, plan, data_root, runtime_root, workers)
        for name, rows in groups.items()
    }
    with _base_contract_context():
        historical_exact, historical_signatures, history = base._history(plan, root)
        ranks = {row.sample_id: row.selection_rank for row in selected}
        pair_by_sample = {row.sample_id: row.pair_key for row in selected}
        for source_rows, spoof_rows in ((ru_source, ru_spoof), (kk_source, kk_spoof)):
            by_text = {row.text_id: row for row in source_rows}
            for spoof in spoof_rows:
                source = by_text.get(spoof.text_id)
                if source is None or source.text_hash != spoof.text_hash:
                    raise V4FinalRecoveryError("Recovery spoof route broke source text pairing.")
                pair_by_sample[spoof.sample_id] = pair_by_sample[source.sample_id]
                ranks[spoof.sample_id] = ranks[source.sample_id]
        decisions = {
            name: base._decisions(
                rows, decoded[name], ranks, historical_exact, historical_signatures
            )
            for name, rows in groups.items()
        }
        ready = {
            name: base._ready({row.sample_id: row for row in rows}, decisions[name], created_at)
            for name, rows in groups.items()
        }
    all_ready = tuple(
        item for name in ("ru_source", "kk_source", "ru_spoof", "kk_spoof") for item in ready[name]
    )
    for rows in ready.values():
        validate_manifest(rows)
        validate_manifest_licenses(rows, ledger)
        require_valid_assets(rows, data_root)
    stage = _stage_outputs(
        root,
        {
            name: value
            for name, value in plan.outputs.items()
            if name not in {"pair_lock_manifest", "pair_lock_receipt"}
        },
    )
    try:
        staged = {name: stage / Path(value).name for name, value in plan.outputs.items()}
        for name, rows in groups.items():
            write_manifest(staged[f"{name}_raw_manifest"], rows)
        for name, rows in ready.items():
            write_manifest(staged[f"{name}_ready_manifest"], rows)
        all_decisions = tuple(
            item
            for name in ("ru_source", "kk_source", "ru_spoof", "kk_spoof")
            for item in decisions[name]
        )
        base._write_csv(
            staged["audio_inventory"],
            base.INVENTORY_FIELDS,
            base._inventory(all_decisions, pair_by_sample, historical_exact),
        )
        receipt = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "status": "materialized_review_required_pair_lock_pending",
            "created_at": created_at,
            "plan": {"path": plan.path, "sha256": plan.sha256},
            "inputs": {
                name: {"path": item.path, "sha256": item.sha256, "rows": item.rows}
                for name, item in sorted(plan.inputs.items())
            },
            "counts": {
                "selected_source_rows": {"ru": len(ru_source), "kk": len(kk_source)},
                "one_shot_synthetic_rows": {"ru": len(ru_spoof), "kk": len(kk_spoof)},
                "raw_assets": sum(len(value) for value in groups.values()),
                "eligible_assets": len(all_ready),
                "eligible_by_cell": {
                    f"{language}/{label}": sum(
                        row.language == language and row.label == label for row in all_ready
                    )
                    for language in ("ru", "kk")
                    for label in ("bonafide", "spoof")
                },
                "irrecoverable_rejected_original_rows": {"ru": 1, "kk": 0},
            },
            "history": history,
            "outputs": {
                name: {
                    "path": plan.outputs[name],
                    "sha256": sha256_file(staged[name]),
                    "rows": base._rows(staged[name]) if staged[name].suffix == ".csv" else None,
                }
                for name in (
                    "ru_source_raw_manifest",
                    "kk_source_raw_manifest",
                    "ru_spoof_raw_manifest",
                    "kk_spoof_raw_manifest",
                    "ru_source_ready_manifest",
                    "kk_source_ready_manifest",
                    "ru_spoof_ready_manifest",
                    "kk_spoof_ready_manifest",
                    "audio_inventory",
                )
            },
            "claims": {
                "raw_audio_extraction_performed": True,
                "synthetic_audio_generated": True,
                "technical_decode_qa_vad_performed": True,
                "full_history_audio_isolation_performed": True,
                "acoustic_review_performed": False,
                "pair_lock_performed": False,
                "detector_checkpoint_loaded": False,
                "calibration_performed": False,
                "detector_inference_performed": False,
                "detector_inference_authorized": False,
                "final_inference_performed": False,
                "replacement_or_backfill": False,
                "resynthesis": False,
                "original_ru_rank_one_irrecoverable_reject": True,
            },
        }
        staged["materialization_receipt"].write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        receipt_sha = sha256_file(staged["materialization_receipt"])
        with _base_contract_context():
            packet = base._review_packet(
                all_ready, receipt_sha, source_text, synthesis_text, pair_by_sample, data_root
            )
        base._write_csv(
            staged["review_packet"], base.PACKET_FIELDS, [asdict(row) for row in packet]
        )
        packet_sha = sha256_file(staged["review_packet"])
        for name, reviewer in (
            ("reviewer_a_template", "reviewer_A_REPLACE_ME"),
            ("reviewer_b_template", "reviewer_B_REPLACE_ME"),
        ):
            base._write_csv(
                staged[name],
                base.REVIEW_FIELDS,
                [
                    {
                        "protocol_id": PROTOCOL_ID,
                        "packet_sha256": packet_sha,
                        "sample_id": row.sample_id,
                        "audio_sha256": row.audio_sha256,
                        "reviewer_pseudo_id": reviewer,
                        "review_status": "inconclusive",
                        "speech_intelligible": "unknown",
                        "lexical_content_preserved": "unknown",
                        "language_preserved": "unknown",
                        "severe_artifacts": "unknown",
                        "notes": "",
                    }
                    for row in packet
                ],
            )
        publish = tuple(
            name for name in plan.outputs if name not in {"pair_lock_manifest", "pair_lock_receipt"}
        )
        if any(
            _project_path(root, plan.outputs[name], f"output {name}").exists() for name in publish
        ):
            raise V4FinalRecoveryError("A recovery output appeared during staging.")
        for name in publish:
            staged[name].replace(_project_path(root, plan.outputs[name], f"output {name}"))
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return plan


def finalize_pair_lock(
    *,
    plan_path: Path,
    project_root: Path,
    data_root: Path,
    reviewer_a: Path,
    reviewer_b: Path,
    created_at: str,
) -> base.Plan:
    """Delegate the exact-byte two-review lock after adapting only the recovery boundary."""

    with _base_contract_context():
        original_load_plan = base.load_plan
        original_load_selection = base._load_selection
        try:
            base.load_plan = load_plan
            base._load_selection = _load_selection
            return base.finalize_pair_lock(
                plan_path=plan_path,
                project_root=project_root,
                data_root=data_root,
                reviewer_a=reviewer_a,
                reviewer_b=reviewer_b,
                created_at=created_at,
            )
        finally:
            base.load_plan = original_load_plan
            base._load_selection = original_load_selection
