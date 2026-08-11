#!/usr/bin/env python3
"""Execute one spoof-only ToneSpeak OOD evaluation with the frozen research XLS-R checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import time
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import torch
import transformers
from torch import Tensor
from torch.torch_version import TorchVersion

from kds.data.assets import require_valid_assets, sha256_file
from kds.data.dataset import DatasetConfig, ManifestAudioDataset
from kds.data.licenses import load_license_ledger
from kds.data.manifest import ManifestRow
from kds.eval.xlsr_tone_speak_ood import (
    ToneSpeakOodPlan,
    ToneSpeakOodPlanError,
    load_tone_speak_ood_plan,
    metric_record,
    tone_speak_ood_plan_record,
    validate_tone_speak_ood_inputs,
)
from kds.models import XlsrSlsClassifier
from kds.training import make_audio_loader
from kds.training.frozen_b0 import state_dict_sha256


def _cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("ToneSpeak OOD XLS-R inference requires an available CUDA device.")
    device = torch.device("cuda")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("ToneSpeak OOD XLS-R inference requires CUDA BF16 support.")
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


def _require_unused_outputs(plan: ToneSpeakOodPlan) -> None:
    existing = [
        str(path) for path in (plan.outputs.execution_lock, plan.outputs.report) if path.exists()
    ]
    if existing:
        raise ValueError(
            "Refusing to run this one-time ToneSpeak OOD plan because output already exists: "
            + ", ".join(existing)
            + "."
        )


def _write_exclusive_json(path: Path, payload: dict[str, object]) -> None:
    """Atomically publish a receipt without allowing replacement after execution begins."""

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
                payload, handle, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True
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


def _load_stage_b_state(plan: ToneSpeakOodPlan) -> dict[str, Tensor]:
    with torch.serialization.safe_globals([TorchVersion]):
        value: object = torch.load(
            plan.checkpoint.checkpoint.path, map_location="cpu", weights_only=True
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
    entries_are_valid = all(
        isinstance(name, str) and isinstance(tensor, Tensor) for name, tensor in state.items()
    )
    if not entries_are_valid:
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
    checkpoint: Mapping[str, object], plan: ToneSpeakOodPlan
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
    plan: ToneSpeakOodPlan, state: Mapping[str, Tensor], device: torch.device
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


def _evaluate_once(
    plan: ToneSpeakOodPlan,
    rows: tuple[ManifestRow, ...],
    model: XlsrSlsClassifier,
    device: torch.device,
    audio_root: Path,
) -> dict[str, object]:
    dataset = ManifestAudioDataset(
        list(rows),
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
        raise ValueError("ToneSpeak OOD candidate has duplicate sample IDs.")
    outcomes: dict[str, dict[str, object]] = {}
    total_loss = 0.0
    spoof_correct = 0
    prediction_counts: Counter[str] = Counter()
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
                if label < 0.5:
                    raise RuntimeError(f"Spoof-only dataset label mismatch for {sample_id}.")
                row = row_by_id[sample_id]
                logit = float(logit_tensor)
                prediction = (
                    "spoof"
                    if logit >= plan.inference.raw_logit_decision_boundary
                    else "bonafide"
                )
                correct = prediction == "spoof"
                prediction_counts[prediction] += 1
                if correct:
                    spoof_correct += 1
                outcomes[sample_id] = {
                    "sample_id": sample_id,
                    "text_hash": row.text_hash,
                    "voice_id": row.voice_id,
                    "audio_sha256": row.sha256,
                    "raw_logit": logit,
                    "raw_prediction": prediction,
                    "correct_at_fixed_raw_boundary": correct,
                }
    if len(outcomes) != len(rows):
        raise RuntimeError("ToneSpeak OOD inference did not produce one outcome per candidate row.")
    examples = len(rows)
    return {
        "classification_rule": {
            "raw_logit_decision_boundary": plan.inference.raw_logit_decision_boundary,
            "boundary_origin": "fixed_model_default_not_selected_or_tuned",
            "calibrated": False,
            "threshold_selection_performed": False,
        },
        "aggregate": {
            "records": examples,
            "class_composition": {"spoof": examples, "bonafide": 0},
            "raw_window_bce_loss": total_loss / examples,
            "spoof_recall_at_fixed_raw_boundary": metric_record(spoof_correct, examples),
            "raw_prediction_counts": dict(sorted(prediction_counts.items())),
            "binary_metrics": "unavailable_spoof_only",
        },
        "sample_results": [outcomes[sample_id] for sample_id in sorted(outcomes)],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run exactly one research-only spoof-only ToneSpeak OOD evaluation with the frozen "
            "XLS-R Stage-B checkpoint. It never trains, calibrates, selects a threshold, or "
            "reports binary/final quality."
        )
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args(argv)

    plan = load_tone_speak_ood_plan(arguments.plan)
    rows = validate_tone_speak_ood_inputs(plan, load_license_ledger(plan.license_ledger.path))
    require_valid_assets(list(rows), arguments.audio_root)
    device = _cuda_device()
    preflight: dict[str, object] = {
        "status": "validated" if arguments.validate_only else "ready_to_execute",
        "mode": "validate_only" if arguments.validate_only else "inference",
        "run_plan": tone_speak_ood_plan_record(plan),
        "candidate_records_validated": len(rows),
        "candidate_class_composition": {"spoof": len(rows), "bonafide": 0},
        "environment": _environment_record(device),
        "training_performed": False,
        "calibrated": False,
        "threshold_selection_performed": False,
        "binary_metrics": "unavailable_spoof_only",
        "acoustic_language_preservation_gate": "verified_for_pinned_assets_only",
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
    result = _evaluate_once(plan, rows, model, device, arguments.audio_root)
    torch.cuda.synchronize(device)
    report: dict[str, object] = {
        **preflight,
        "status": "ok",
        "execution_lock": {
            "path": str(plan.outputs.execution_lock),
            "sha256": sha256_file(plan.outputs.execution_lock),
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
        "source_provenance_limitation": (
            "ToneSpeak is a source-card-provided Russian spoof-only source without a bona-fide "
            "class or independent per-row generation logs."
        ),
        "acoustic_gate_scope_limitation": (
            "The completed gate establishes Russian audibility and lexical preservation only for "
            "the pinned 100 WAV bytes."
        ),
    }
    _write_exclusive_json(plan.outputs.report, report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ToneSpeakOodPlanError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "error", "detail": str(error)}, ensure_ascii=False))
        raise SystemExit(2) from error
