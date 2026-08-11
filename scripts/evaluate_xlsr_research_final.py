#!/usr/bin/env python3
"""Fit pinned temperature scaling and execute one confirmatory RU/KK/mixed XLS-R run."""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
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
from kds.eval.calibration import TemperatureScaler, brier_score, expected_calibration_error
from kds.eval.metrics import wilson_interval
from kds.eval.xlsr_research_final import (
    ValidatedFinalInputs,
    XlsrResearchFinalPlan,
    XlsrResearchFinalPlanError,
    final_plan_record,
    load_xlsr_research_final_plan,
    validate_xlsr_research_final_inputs,
)
from kds.models import XlsrSlsClassifier
from kds.training import make_audio_loader
from kds.training.frozen_b0 import state_dict_sha256


def _cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("Research final XLS-R inference requires an available CUDA device.")
    device = torch.device("cuda")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Research final XLS-R inference requires CUDA BF16 support.")
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


def _require_unused_outputs(plan: XlsrResearchFinalPlan) -> None:
    existing = [
        str(path) for path in (plan.outputs.execution_lock, plan.outputs.report) if path.exists()
    ]
    if existing:
        raise ValueError(
            "Refusing to run this one-time research final plan because output already exists: "
            + ", ".join(existing)
            + "."
        )


def _write_exclusive_json(path: Path, payload: dict[str, object]) -> None:
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


def _load_stage_b_state(plan: XlsrResearchFinalPlan) -> dict[str, Tensor]:
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
        raise ValueError("Frozen checkpoint does not match the planned Stage-B model.")
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
        raise ValueError("Frozen Stage-B trainable state hash does not match the plan.")
    allowed_prefixes = (
        "head.",
        *(f"encoder.encoder.layers.{index}." for index in range(16, 24)),
    )
    if any(not name.startswith(allowed_prefixes) for name in state):
        raise ValueError("Stage-B state contains parameters outside the frozen head/tail scope.")
    return state


def _validate_embedded_encoder_and_head(
    checkpoint: Mapping[str, object], plan: XlsrResearchFinalPlan
) -> None:
    encoder_value = checkpoint.get("encoder")
    if not isinstance(encoder_value, dict):
        raise ValueError("Frozen Stage-B checkpoint has no embedded encoder receipt.")
    stage_a_value = encoder_value.get("record")
    if not isinstance(stage_a_value, dict):
        raise ValueError("Frozen Stage-B checkpoint has an invalid Stage-A receipt.")
    embedded_encoder = stage_a_value.get("encoder")
    embedded_head = stage_a_value.get("head")
    if not isinstance(embedded_encoder, dict) or not isinstance(embedded_head, dict):
        raise ValueError("Frozen Stage-B checkpoint lacks encoder/head configuration.")
    embedded_config = embedded_encoder.get("config")
    embedded_weights = embedded_encoder.get("weights")
    if not isinstance(embedded_config, dict) or not isinstance(embedded_weights, dict):
        raise ValueError("Frozen Stage-B checkpoint has invalid encoder file receipts.")
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
        raise ValueError("Frozen Stage-B encoder/head receipts do not match the plan.")


def _build_model(
    plan: XlsrResearchFinalPlan, state: Mapping[str, Tensor], device: torch.device
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
        raise ValueError(f"Stage-B state has unexpected keys: {incompatible.unexpected_keys}")
    model.eval()
    return model.to(device)


def _infer_logits(
    plan: XlsrResearchFinalPlan,
    rows: Sequence[ManifestRow],
    model: XlsrSlsClassifier,
    device: torch.device,
    audio_root: Path,
) -> tuple[Tensor, Tensor]:
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
    logits_by_id: dict[str, float] = {}
    labels_by_id: dict[str, float] = {}
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            waveforms = batch.waveforms.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = model(waveforms)
            for sample_id, label, logit in zip(
                batch.sample_ids,
                batch.labels.tolist(),
                logits.detach().float().cpu().tolist(),
                strict=True,
            ):
                if sample_id in logits_by_id:
                    raise RuntimeError(f"Inference duplicated sample_id={sample_id!r}.")
                logits_by_id[sample_id] = float(logit)
                labels_by_id[sample_id] = float(label)
    if set(logits_by_id) != {row.sample_id for row in rows}:
        raise RuntimeError("Inference did not produce exactly one logit per manifest row.")
    ordered_logits = torch.tensor([logits_by_id[row.sample_id] for row in rows])
    ordered_labels = torch.tensor([labels_by_id[row.sample_id] for row in rows])
    return ordered_logits, ordered_labels


def _metric(correct: int, examples: int) -> dict[str, object]:
    return {
        "correct": correct,
        "examples": examples,
        "value": correct / examples,
        "confidence_interval": asdict(wilson_interval(correct, examples)),
    }


def _binary_metrics(probabilities: Tensor, labels: Tensor) -> dict[str, object]:
    predictions = probabilities >= 0.5
    expected = labels >= 0.5
    bonafide = ~expected
    spoof = expected
    correct = int((predictions == expected).sum())
    bonafide_correct = int((predictions[bonafide] == expected[bonafide]).sum())
    spoof_correct = int((predictions[spoof] == expected[spoof]).sum())
    bonafide_count = int(bonafide.sum())
    spoof_count = int(spoof.sum())
    return {
        "accuracy": _metric(correct, labels.numel()),
        "bonafide_recall": _metric(bonafide_correct, bonafide_count),
        "spoof_recall": _metric(spoof_correct, spoof_count),
        "balanced_accuracy": (bonafide_correct / bonafide_count + spoof_correct / spoof_count) / 2,
        "brier_score": brier_score(probabilities, labels),
        "expected_calibration_error_15_bins": expected_calibration_error(probabilities, labels),
    }


def _layer_report(
    rows: Sequence[ManifestRow], logits: Tensor, temperature: float
) -> dict[str, object]:
    labels = torch.tensor([1.0 if row.label == "spoof" else 0.0 for row in rows])
    probabilities = torch.sigmoid(logits / temperature)
    predictions = probabilities >= 0.5
    by_family: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_family[row.generator_family].append(index)
    strata: dict[str, object] = {}
    for family, indices in sorted(by_family.items()):
        family_probabilities = probabilities[indices]
        family_labels = labels[indices]
        family_expected = family_labels >= 0.5
        family_predictions = family_probabilities >= 0.5
        family_correct = int((family_predictions == family_expected).sum())
        label_counts = Counter(rows[index].label for index in indices)
        family_record: dict[str, object] = {
            "label_counts": dict(sorted(label_counts.items())),
            "class_accuracy": _metric(family_correct, len(indices)),
            "mean_calibrated_spoof_probability": float(family_probabilities.mean()),
        }
        if set(label_counts) == {"bonafide", "spoof"}:
            family_record["binary_metrics"] = _binary_metrics(family_probabilities, family_labels)
        strata[family] = family_record
    pair_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        pair_indices[row.text_hash].append(index)
    both_correct = sum(
        len(indices) == 2 and bool((predictions[indices] == (labels[indices] >= 0.5)).all())
        for indices in pair_indices.values()
    )
    sample_results = []
    for index, row in enumerate(rows):
        sample_results.append(
            {
                "sample_id": row.sample_id,
                "label": row.label,
                "source_name": row.source_name,
                "generator_family": row.generator_family,
                "text_hash": row.text_hash,
                "audio_sha256": row.sha256,
                "raw_logit": float(logits[index]),
                "calibrated_spoof_probability": float(probabilities[index]),
                "prediction": "spoof" if bool(predictions[index]) else "bonafide",
            }
        )
    return {
        "records": len(rows),
        "pairs": len(pair_indices),
        "metrics": _binary_metrics(probabilities, labels),
        "pairs_both_correct": _metric(both_correct, len(pair_indices)),
        "generator_family_metrics": strata,
        "generator_family_counts": dict(
            sorted(Counter(row.generator_family for row in rows).items())
        ),
        "sample_results": sample_results,
    }


def _validate_assets(inputs: ValidatedFinalInputs, audio_root: Path) -> int:
    rows = [
        *inputs.train,
        *inputs.stage_a_dev,
        *inputs.stage_b_dev,
        *inputs.calibration,
        *(row for layer in inputs.final_layers.values() for row in layer),
    ]
    require_valid_assets(rows, audio_root)
    return len(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args(argv)

    plan = load_xlsr_research_final_plan(arguments.plan)
    ledger = load_license_ledger(plan.license_ledger.path)
    inputs = validate_xlsr_research_final_inputs(plan, ledger)
    asset_count = _validate_assets(inputs, arguments.audio_root)
    device = _cuda_device()
    preflight: dict[str, object] = {
        "status": "validated" if arguments.validate_only else "ready_to_execute",
        "mode": "validate_only" if arguments.validate_only else "calibration_and_inference",
        "run_plan": final_plan_record(plan),
        "assets_validated": asset_count,
        "role_rows": {
            "train": len(inputs.train),
            "stage_a_dev": len(inputs.stage_a_dev),
            "stage_b_dev": len(inputs.stage_b_dev),
            "calibration": len(inputs.calibration),
            **{name: len(rows) for name, rows in inputs.final_layers.items()},
        },
        "environment": _environment_record(device),
        "training_performed": False,
        "threshold_selection_performed": False,
        "pooled_language_metric": "prohibited",
    }
    if arguments.validate_only:
        print(json.dumps(preflight, ensure_ascii=False, allow_nan=False))
        return 0

    _require_unused_outputs(plan)
    state = _load_stage_b_state(plan)
    model = _build_model(plan, state, device)
    execution_lock = {
        **preflight,
        "status": "calibration_and_final_inference_started",
        "started_at": datetime.now(UTC).isoformat(),
        "one_time_execution": True,
        "report_path": str(plan.outputs.report),
    }
    _write_exclusive_json(plan.outputs.execution_lock, execution_lock)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats(device)

    calibration_logits, calibration_labels = _infer_logits(
        plan, inputs.calibration, model, device, arguments.audio_root
    )
    scaler = TemperatureScaler()
    calibration_report = scaler.fit(
        calibration_logits,
        calibration_labels,
        max_iter=plan.inference.temperature_max_iter,
    )
    temperature = calibration_report.temperature
    layer_reports: dict[str, object] = {}
    for layer in plan.final_layers:
        rows = inputs.final_layers[layer.name]
        logits, labels = _infer_logits(plan, rows, model, device, arguments.audio_root)
        expected_labels = torch.tensor([1.0 if row.label == "spoof" else 0.0 for row in rows])
        if not torch.equal(labels, expected_labels):
            raise RuntimeError(f"Dataset labels differ from manifest for final layer {layer.name}.")
        layer_reports[layer.name] = {
            "language": layer.language,
            "evidence_kind": layer.evidence_kind,
            "project_exposure": layer.project_exposure,
            "result": _layer_report(rows, logits, temperature),
        }
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
        "calibration": {
            **asdict(calibration_report),
            "records": len(inputs.calibration),
            "manifest": str(plan.calibration.manifest.path),
            "manifest_sha256": plan.calibration.manifest.sha256,
            "threshold_selection_performed": False,
        },
        "final_layers": layer_reports,
        "elapsed_seconds": time.monotonic() - started,
        "cuda_memory": {
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "limitations": [
            "This is a personal-research confirmatory evaluation, not product quality.",
            "The mixed layer was previously inferred with an older checkpoint and is not a "
            "blind project-level final.",
            "The Kazakh Silero layer has source-transcript provenance but no completed "
            "two-review acoustic language gate.",
            "No source provides verified speaker-disjoint provenance.",
            "Metrics are intentionally not pooled across RU, KK and mixed layers.",
        ],
    }
    _write_exclusive_json(plan.outputs.report, report)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (XlsrResearchFinalPlanError, OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "error", "detail": str(error)}, ensure_ascii=False))
        raise SystemExit(2) from error
