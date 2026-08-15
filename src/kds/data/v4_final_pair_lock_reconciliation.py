"""Lock reviewed reconciliation pairs without altering exhausted materialization receipts."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from kds.data import v4_final_materialization as base
from kds.data import v4_final_reconciliation as reconciliation
from kds.data.assets import sha256_file

PROTOCOL_ID = "xlsr-sls-model-v4-final-reconciliation-pair-lock-v1"
AUTHORIZATION_PATH = "configs/research/v4/xlsr_sls_model_v4_final_reconciliation_pair_lock_v1.json"
_INPUTS = {
    "reconciliation_plan",
    "materialization_receipt",
    "review_packet",
    "reviewer_a",
    "reviewer_b",
    "lock_module",
    "runner_script",
}
_PROHIBITIONS = {
    "output_overwrite",
    "source_extraction",
    "synthetic_audio_generation",
    "decoder_execution",
    "detector_checkpoint_loading",
    "calibration",
    "detector_inference",
    "final_inference",
    "replacement_or_backfill",
    "resynthesis",
}


class V4FinalPairLockError(ValueError):
    """Raised when the independent review lock cannot be proven."""


def _verify(binding: base.Binding, root: Path, label: str) -> Path:
    try:
        return base._verify(binding, root, label)
    except base.V4FinalMaterializationError as error:
        raise V4FinalPairLockError(str(error)) from error


def _load_authorization(path: Path, root: Path) -> Mapping[str, base.Binding]:
    resolved = path.resolve(strict=True)
    raw = base._object(resolved, "pair lock authorization")
    if (
        set(raw) != {"schema_version", "protocol_id", "created_at", "inputs", "prohibitions"}
        or raw["schema_version"] != 1
        or raw["protocol_id"] != PROTOCOL_ID
    ):
        raise V4FinalPairLockError("Pair lock authorization schema/protocol is invalid.")
    inputs = base._mapping(raw["inputs"], "pair lock inputs")
    if set(inputs) != _INPUTS:
        raise V4FinalPairLockError("Pair lock authorization input set is invalid.")
    result = {name: base._binding(value, f"inputs.{name}") for name, value in inputs.items()}
    prohibitions = base._mapping(raw["prohibitions"], "pair lock prohibitions")
    if set(prohibitions) != _PROHIBITIONS or any(
        value is not True for value in prohibitions.values()
    ):
        raise V4FinalPairLockError("Pair lock prohibitions are not fail-closed.")
    for name, binding in result.items():
        _verify(binding, root, f"inputs.{name}")
    return result


def _validate_reused_qa_receipt(receipt: Mapping[str, object], plan: base.Plan, root: Path) -> None:
    claims = base._mapping(receipt.get("claims"), "materialization receipt claims")
    receipt_plan = base._mapping(receipt.get("plan"), "materialization receipt plan")
    if (
        receipt.get("protocol_id") != reconciliation.PROTOCOL_ID
        or receipt.get("status") != "materialized_review_required_pair_lock_pending"
        or receipt_plan.get("path") != plan.path
        or receipt_plan.get("sha256") != plan.sha256
        or claims.get("technical_decode_qa_vad_reused_exactly") is not True
        or claims.get("full_history_audio_isolation_performed") is not True
        or claims.get("acoustic_review_performed") is not False
        or claims.get("pair_lock_performed") is not False
        or claims.get("detector_checkpoint_loaded") is not False
        or claims.get("detector_inference_performed") is not False
        or claims.get("final_inference_performed") is not False
    ):
        raise V4FinalPairLockError("Reconciliation receipt does not permit pair locking.")
    outputs = base._mapping(receipt.get("outputs"), "materialization outputs")
    names = {
        "ru_source_raw_manifest",
        "kk_source_raw_manifest",
        "ru_spoof_raw_manifest",
        "kk_spoof_raw_manifest",
        "ru_source_ready_manifest",
        "kk_source_ready_manifest",
        "ru_spoof_ready_manifest",
        "kk_spoof_ready_manifest",
        "audio_inventory",
    }
    if set(outputs) != names:
        raise V4FinalPairLockError("Reconciliation receipt outputs are incomplete.")
    for name, raw_item in outputs.items():
        item = base._mapping(raw_item, f"output {name}")
        target = base._project_path(root, plan.outputs[name], f"output {name}")
        if (
            item.get("path") != plan.outputs[name]
            or item.get("sha256") != sha256_file(target)
            or item.get("rows") != base._rows(target)
        ):
            raise V4FinalPairLockError(f"Reconciliation receipt no longer binds {name}.")


def finalize_pair_lock(
    *, authorization_path: Path, project_root: Path, data_root: Path, created_at: str
) -> base.Plan:
    root = project_root.resolve(strict=True)
    bindings = _load_authorization(authorization_path, root)
    recon_plan_path = _verify(bindings["reconciliation_plan"], root, "reconciliation plan")
    plan = reconciliation.load_plan(recon_plan_path, root)
    for name, expected in (
        ("materialization_receipt", plan.outputs["materialization_receipt"]),
        ("review_packet", plan.outputs["review_packet"]),
        ("reviewer_a", plan.outputs["reviewer_a_template"]),
        ("reviewer_b", plan.outputs["reviewer_b_template"]),
    ):
        if bindings[name].path != expected:
            raise V4FinalPairLockError(
                f"Authorization {name} does not bind the reconciliation output."
            )
    with reconciliation._base_context():
        old_plan, old_selection, old_receipt = (
            base.load_plan,
            base._load_selection,
            base._validate_materialization_receipt,
        )
        try:
            base.load_plan = reconciliation.load_plan
            base._load_selection = reconciliation._load_selection
            base._validate_materialization_receipt = _validate_reused_qa_receipt
            return base.finalize_pair_lock(
                plan_path=recon_plan_path,
                project_root=root,
                data_root=data_root,
                reviewer_a=_verify(bindings["reviewer_a"], root, "reviewer A"),
                reviewer_b=_verify(bindings["reviewer_b"], root, "reviewer B"),
                created_at=created_at,
            )
        finally:
            base.load_plan, base._load_selection, base._validate_materialization_receipt = (
                old_plan,
                old_selection,
                old_receipt,
            )
