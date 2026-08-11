"""Execute one locked exploratory mixed stress test with a frozen XLS-R Stage-B checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import torch
import transformers
from torch import Tensor
from torch.torch_version import TorchVersion

from kds.data.assets import require_valid_assets
from kds.data.dataset import DatasetConfig, ManifestAudioDataset
from kds.data.licenses import load_license_ledger
from kds.data.manifest import ManifestRow
from kds.eval.xlsr_exploratory import (
    ExploratoryMixedInputs,
    XlsrExploratoryMixedPlan,
    XlsrExploratoryMixedPlanError,
    load_xlsr_exploratory_mixed_plan,
    metric_record,
    pair_lock_records,
    validate_exploratory_mixed_inputs,
    xlsr_exploratory_mixed_plan_record,
)
from kds.models import XlsrSlsClassifier
from kds.training import make_audio_loader
from kds.training.frozen_b0 import state_dict_sha256


def _cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen XLS-R exploratory inference requires an available CUDA device.")
    device = torch.device("cuda")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Frozen XLS-R exploratory inference requires CUDA BF16 support.")
    return device


def _environment_record(device: torch.device) -> dict[str, object]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_name": properties.name,
        "device_total_memory_bytes": properties.total_memory,
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }


def _require_unused_outputs(plan: XlsrExploratoryMixedPlan) -> None:
    existing = [
        str(path)
        for path in (plan.outputs.execution_lock, plan.outputs.report)
        if path.exists()
    ]
    if existing:
        raise ValueError(
            "Refusing to run this one-time exploratory plan because output already exists: "
            + ", ".join(existing)
            + "."
        )


def _write_exclusive_json(path: Path, payload: dict[str, object]) -> None:
    """Atomically publish one JSON receipt without allowing replacement."""

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError(f"Refusing to overwrite existing output: {path}") from error
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_stage_b_state(plan: XlsrExploratoryMixedPlan) -> dict[str, Tensor]:
    with torch.serialization.safe_globals([TorchVersion]):
        value: object = torch.load(
            plan.checkpoint.checkpoint.path,
            map_location="cpu",
            weights_only=True,
        )
    if not isinstance(value, dict):
        raise ValueError("Frozen Stage-B checkpoint root must be a dictionary.")
    checkpoint = cast(dict[str, object], value)
    if (
        checkpoint.get("model_name") != "xlsr_sls"
        or checkpoint.get("stage") != "B"
        or checkpoint.get("training_purpose") != "research"
        or checkpoint.get("selected_trainable_state_sha256")
        != plan.checkpoint.selected_trainable_state_sha256
    ):
        raise ValueError(
            "Frozen checkpoint does not match the planned research XLS-R Stage-B model."
        )
    _validate_embedded_encoder_and_head(checkpoint, plan)
    state_value = checkpoint.get("trainable_state_dict")
    if not isinstance(state_value, dict) or not state_value:
        raise ValueError("Frozen Stage-B checkpoint has no trainable_state_dict.")
    state = cast(dict[str, Tensor], state_value)
    if any(
        not isinstance(name, str) or not isinstance(tensor, Tensor)
        for name, tensor in state.items()
    ):
        raise ValueError("Frozen Stage-B trainable_state_dict has invalid entries.")
    if state_dict_sha256(state) != plan.checkpoint.selected_trainable_state_sha256:
        raise ValueError("Frozen Stage-B trainable state hash does not match the run plan.")
    allowed_prefixes = (
        "head.",
        *(f"encoder.encoder.layers.{index}." for index in range(16, 24)),
    )
    if any(not name.startswith(allowed_prefixes) for name in state):
        raise ValueError(
            "Frozen Stage-B state contains parameters outside the Stage-B head/tail scope."
        )
    return state


def _validate_embedded_encoder_and_head(
    checkpoint: Mapping[str, object], plan: XlsrExploratoryMixedPlan
) -> None:
    encoder_value = checkpoint.get("encoder")
    if not isinstance(encoder_value, dict):
        raise ValueError("Frozen Stage-B checkpoint has no embedded Stage-A encoder receipt.")
    stage_a_value = encoder_value.get("record")
    if not isinstance(stage_a_value, dict):
        raise ValueError("Frozen Stage-B checkpoint has an invalid embedded Stage-A receipt.")
    embedded_encoder = stage_a_value.get("encoder")
    embedded_head = stage_a_value.get("head")
    if not isinstance(embedded_encoder, dict) or not isinstance(embedded_head, dict):
        raise ValueError("Frozen Stage-B checkpoint has no embedded encoder/head configuration.")
    embedded_config = embedded_encoder.get("config")
    embedded_weights = embedded_encoder.get("weights")
    if not isinstance(embedded_config, dict) or not isinstance(embedded_weights, dict):
        raise ValueError("Frozen Stage-B checkpoint has invalid embedded encoder file receipts.")
    if (
        embedded_encoder.get("revision") != plan.encoder.revision
        or embedded_config.get("sha256") != plan.encoder.config.sha256
        or embedded_weights.get("sha256") != plan.encoder.weights.sha256
        or embedded_head
        != {
            "attention_size": plan.head.attention_size,
            "classifier_size": plan.head.classifier_size,
            "dropout": plan.head.dropout,
        }
    ):
        raise ValueError(
            "Frozen Stage-B checkpoint encoder/head receipts do not match the run plan."
        )


def _build_frozen_model(
    plan: XlsrExploratoryMixedPlan, state: Mapping[str, Tensor], device: torch.device
) -> XlsrSlsClassifier:
    model = XlsrSlsClassifier.from_pretrained(
        str(plan.encoder.checkpoint_dir),
        attention_size=plan.head.attention_size,
        classifier_size=plan.head.classifier_size,
        dropout=plan.head.dropout,
        local_files_only=True,
    )
    incompatible = model.load_state_dict(dict(state), strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(
            f"Frozen Stage-B state has unexpected keys: {incompatible.unexpected_keys}"
        )
    model.eval()
    return model.to(device)


def _prediction(logit: float, boundary: float) -> str:
    return "spoof" if logit >= boundary else "bonafide"


def _evaluate_once(
    plan: XlsrExploratoryMixedPlan,
    inputs: ExploratoryMixedInputs,
    model: XlsrSlsClassifier,
    device: torch.device,
    audio_root: Path,
) -> dict[str, object]:
    rows = list(inputs.rows)
    dataset = ManifestAudioDataset(
        rows,
        DatasetConfig(
            audio_root=audio_root,
            sample_rate=plan.inference.sample_rate,
            window_samples=plan.inference.window_samples,
            mode="eval",
            seed=plan.run_id,
        ),
    )
    loader = make_audio_loader(
        dataset,
        batch_size=plan.inference.batch_size,
        shuffle=False,
        num_workers=plan.inference.num_workers,
        pin_memory=True,
    )
    row_by_id = {row.sample_id: row for row in rows}
    if len(row_by_id) != len(rows):
        raise ValueError("Exploratory mixed candidate has duplicate sample IDs.")
    outcomes: dict[str, dict[str, object]] = {}
    total_loss = 0.0
    total_correct = 0
    bonafide_correct = 0
    spoof_correct = 0
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            waveforms = batch.waveforms.to(device, non_blocking=True)
            labels = batch.labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = model(waveforms)
                loss_sum = criterion(logits, labels)
            total_loss += float(loss_sum.detach())
            for sample_id, label, logit_tensor in zip(
                batch.sample_ids,
                labels.detach().cpu().tolist(),
                logits.detach().float().cpu().tolist(),
                strict=True,
            ):
                row = row_by_id[sample_id]
                logit = float(logit_tensor)
                prediction = _prediction(logit, plan.inference.raw_logit_decision_boundary)
                correct = prediction == row.label
                if correct:
                    total_correct += 1
                    if row.label == "bonafide":
                        bonafide_correct += 1
                    else:
                        spoof_correct += 1
                if (label >= 0.5) != (row.label == "spoof"):
                    raise RuntimeError(f"Dataset label mismatch for {sample_id}.")
                outcomes[sample_id] = {
                    "sample_id": sample_id,
                    "audio_sha256": row.sha256,
                    "raw_logit": logit,
                    "raw_prediction": prediction,
                    "correct_at_fixed_raw_boundary": correct,
                }
    if len(outcomes) != len(rows):
        raise RuntimeError(
            "Frozen exploratory inference did not produce one outcome per candidate row."
        )
    pair_results = _pair_results(pair_lock_records(inputs.pair_lock), rows, outcomes)
    pairs_both_correct = sum(
        _pair_label_correct(pair, "bonafide") and _pair_label_correct(pair, "spoof")
        for pair in pair_results
    )
    pairs_none_correct = sum(
        not _pair_label_correct(pair, "bonafide") and not _pair_label_correct(pair, "spoof")
        for pair in pair_results
    )
    examples = len(rows)
    pairs = plan.candidate.expected_pairs
    return {
        "classification_rule": {
            "raw_logit_decision_boundary": plan.inference.raw_logit_decision_boundary,
            "boundary_origin": "fixed_model_default_not_selected_or_tuned",
            "calibrated": False,
            "threshold_selection_performed": False,
        },
        "aggregate": {
            "records": examples,
            "pairs": pairs,
            "raw_window_bce_loss": total_loss / examples,
            "raw_decision_accuracy": metric_record(total_correct, examples),
            "bonafide_recall": metric_record(bonafide_correct, pairs),
            "spoof_recall": metric_record(spoof_correct, pairs),
            "balanced_recall": (bonafide_correct / pairs + spoof_correct / pairs) / 2,
            "pair_outcomes": {
                "both_records_correct": pairs_both_correct,
                "exactly_one_record_correct": pairs - pairs_both_correct - pairs_none_correct,
                "neither_record_correct": pairs_none_correct,
            },
        },
        "pair_results": pair_results,
    }


def _pair_results(
    lock_pairs: tuple[dict[str, str], ...],
    rows: list[ManifestRow],
    outcomes: Mapping[str, dict[str, object]],
) -> list[dict[str, object]]:
    rows_by_text: dict[str, dict[str, ManifestRow]] = {}
    for row in rows:
        rows_by_text.setdefault(row.text_hash, {})[row.label] = row
    results: list[dict[str, object]] = []
    for index, lock in enumerate(lock_pairs, start=1):
        matched = rows_by_text[lock["text_hash"]]
        bona = matched["bonafide"]
        spoof = matched["spoof"]
        bona_outcome = outcomes[bona.sample_id]
        spoof_outcome = outcomes[spoof.sample_id]
        results.append(
            {
                "pair_index": index,
                "annotation_id": lock["annotation_id"],
                "component": lock["component"],
                "text_hash": lock["text_hash"],
                "input_transcript_evidence": {
                    "ru_token_indices": lock["ru_evidence_token_indices"],
                    "ru_tokens": lock["ru_evidence_tokens"],
                    "kk_token_indices": lock["kk_evidence_token_indices"],
                    "kk_tokens": lock["kk_evidence_tokens"],
                },
                "bonafide": bona_outcome,
                "spoof": spoof_outcome,
                "spoof_minus_bonafide_raw_logit": _raw_logit(spoof_outcome)
                - _raw_logit(bona_outcome),
                "both_records_correct_at_fixed_raw_boundary": _outcome_correct(bona_outcome)
                and _outcome_correct(spoof_outcome),
            }
        )
    return results


def _pair_label_correct(pair: Mapping[str, object], label: str) -> bool:
    value = pair.get(label)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"Pair result has no {label} outcome.")
    return _outcome_correct(value)


def _outcome_correct(outcome: Mapping[str, object]) -> bool:
    value = outcome.get("correct_at_fixed_raw_boundary")
    if not isinstance(value, bool):
        raise RuntimeError("Pair result has an invalid correctness value.")
    return value


def _raw_logit(outcome: Mapping[str, object]) -> float:
    value = outcome.get("raw_logit")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError("Pair result has an invalid raw logit.")
    return float(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run exactly one research-only exploratory mixed stress test with the frozen XLS-R "
            "Stage-B checkpoint. It never trains, calibrates, or selects a threshold."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args(argv)

    plan = load_xlsr_exploratory_mixed_plan(arguments.plan)
    device = _cuda_device()
    inputs = validate_exploratory_mixed_inputs(plan, load_license_ledger(plan.license_ledger.path))
    require_valid_assets(list(inputs.rows), arguments.audio_root)
    preflight: dict[str, object] = {
        "status": "validated" if arguments.validate_only else "ready_to_execute",
        "mode": "validate_only" if arguments.validate_only else "inference",
        "run_plan": xlsr_exploratory_mixed_plan_record(plan),
        "candidate_records_validated": len(inputs.rows),
        "candidate_pairs_validated": plan.candidate.expected_pairs,
        "environment": _environment_record(device),
        "training_performed": False,
        "calibrated": False,
        "threshold_selection_performed": False,
        "acoustic_language_preservation_gate": "not_performed",
    }
    if arguments.validate_only:
        print(json.dumps(preflight, ensure_ascii=False, allow_nan=False))
        return 0

    _require_unused_outputs(plan)
    state = _load_stage_b_state(plan)
    model = _build_frozen_model(plan, state, device)
    execution_lock = {
        **preflight,
        "status": "inference_started",
        "started_at": datetime.now(UTC).isoformat(),
        "one_time_execution": True,
        "report_path": str(plan.outputs.report),
    }
    _write_exclusive_json(plan.outputs.execution_lock, execution_lock)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)
    result = _evaluate_once(plan, inputs, model, device, arguments.audio_root)
    torch.cuda.synchronize(device)
    report: dict[str, object] = {
        **preflight,
        "status": "ok",
        "execution_lock": {
            "path": str(plan.outputs.execution_lock),
            "sha256": _sha256_file(plan.outputs.execution_lock),
        },
        "frozen_checkpoint": {
            "path": str(plan.checkpoint.checkpoint.path),
            "sha256": plan.checkpoint.checkpoint.sha256,
            "selected_trainable_state_sha256": plan.checkpoint.selected_trainable_state_sha256,
        },
        "elapsed_seconds": time.monotonic() - started,
        "cuda_memory": {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "result": result,
        "final_quality_claim": "prohibited",
        "input_transcript_provenance_limitation": (
            "Synthetic mixed WAV language provenance is intended input text only; no acoustic "
            "language-preservation gate was performed."
        ),
    }
    _write_exclusive_json(plan.outputs.report, report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (XlsrExploratoryMixedPlanError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "error", "detail": str(error)}, ensure_ascii=False))
        raise SystemExit(2) from error
