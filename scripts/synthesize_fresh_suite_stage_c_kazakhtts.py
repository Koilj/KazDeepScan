"""Synthesize the frozen Stage-C base through the exact fixed-voice KazakhTTS route."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]
import torch

from kds.data.assets import require_valid_assets, sha256_file
from kds.data.kazakhtts import (
    KAZAKHTTS_SOURCE_ID,
    extract_verified_kazakhtts_runtime,
    load_kazakhtts_runtime,
    validate_kazakhtts_text,
)
from kds.data.kazakhtts_candidate import kazakhtts_spoof_row
from kds.data.kazakhtts_inference import (
    load_kazakhtts_models,
    resolve_kazakhtts_device,
    synthesize_kazakhtts_waveform,
)
from kds.data.kazakhtts_text import KAZAKHTTS_TEXT_NORMALIZER_ID
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import (
    ManifestError,
    ManifestRow,
    load_manifest,
    validate_manifest,
    write_manifest,
)
from kds.data.research_tts import (
    ResearchTtsError,
    load_research_tts_model_lock,
    verify_research_tts_model_lock,
)
from kds.eval.fresh_suite_selection import FreshSuiteSelectionError, load_fresh_suite_selection


def _selection_items(plan: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    roles = cast(Mapping[str, object], plan["roles"])
    result: dict[str, Mapping[str, object]] = {}
    for language in ("ru", "kk", "mixed"):
        role = cast(Mapping[str, object], roles[language])
        for item in cast(list[object], role["items"]):
            value = cast(Mapping[str, object], item)
            sample_id = cast(str, value["sample_id"])
            if sample_id in result:
                raise FreshSuiteSelectionError(
                    f"Stage-C selection repeats sample ID {sample_id!r}."
                )
            result[sample_id] = value
    return result


def _require_materialization(
    path: Path, selection: Path, base_manifest: Path
) -> Mapping[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FreshSuiteSelectionError(f"Cannot read Stage-C materialization: {error}") from error
    if not isinstance(value, dict):
        raise FreshSuiteSelectionError("Stage-C materialization must be a JSON object.")
    selected = value.get("selection")
    outputs = value.get("outputs")
    combined = outputs.get("combined_ready") if isinstance(outputs, dict) else None
    counts = value.get("counts")
    if (
        value.get("schema_version") != 1
        or value.get("protocol_id") != "fresh-suite-stage-c-base-materialization-v1"
        or value.get("detector_inference_authorized") is not False
        or value.get("post_selection_backfill") is not False
        or not isinstance(selected, dict)
        or selected.get("path") != selection.as_posix()
        or selected.get("sha256") != sha256_file(selection)
        or not isinstance(combined, dict)
        or combined.get("path") != base_manifest.as_posix()
        or combined.get("sha256") != sha256_file(base_manifest)
        or not isinstance(counts, dict)
        or counts.get("ready") != {"kk": 60, "mixed": 58, "ru": 50}
    ):
        raise FreshSuiteSelectionError("Stage-C base materialization binding is invalid.")
    return cast(Mapping[str, object], value)


def _load_normalization(
    path: Path,
    *,
    selection: Path,
    base_manifest: Path,
    model_lock: Path,
    project_root: Path,
) -> dict[str, Mapping[str, object]]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FreshSuiteSelectionError(
            f"Cannot read Stage-C normalization plan: {error}"
        ) from error
    if not isinstance(value, dict):
        raise FreshSuiteSelectionError("Stage-C normalization plan must be a JSON object.")
    source = value.get("normalizer_source")
    selected = value.get("selection")
    base = value.get("base_manifest")
    model = value.get("model_lock")
    decision = value.get("decision_rule")
    rows = value.get("rows")
    source_path = project_root / "src/kds/data/kazakhtts_text.py"
    if (
        value.get("schema_version") != 1
        or value.get("protocol_id") != "fresh-suite-stage-c-kazakhtts-normalization-v1"
        or value.get("normalizer_id") != KAZAKHTTS_TEXT_NORMALIZER_ID
        or value.get("row_count") != 168
        or not isinstance(source, dict)
        or source.get("path") != "src/kds/data/kazakhtts_text.py"
        or source.get("sha256") != sha256_file(source_path)
        or not isinstance(selected, dict)
        or selected.get("path") != selection.as_posix()
        or selected.get("sha256") != sha256_file(selection)
        or not isinstance(base, dict)
        or base.get("path") != base_manifest.as_posix()
        or base.get("sha256") != sha256_file(base_manifest)
        or not isinstance(model, dict)
        or model.get("path") != model_lock.as_posix()
        or model.get("sha256") != sha256_file(model_lock)
        or not isinstance(decision, dict)
        or decision.get("metric_or_detector_based") is not False
        or decision.get("post_selection_backfill") is not False
        or decision.get("source_text_id_and_hash_preserved") is not True
        or decision.get("detector_inference") != "forbidden"
        or not isinstance(rows, list)
        or len(rows) != 168
    ):
        raise FreshSuiteSelectionError("Stage-C normalization plan binding is invalid.")
    result: dict[str, Mapping[str, object]] = {}
    required_row = {
        "sample_id",
        "language",
        "text_id",
        "source_text_hash",
        "source_text",
        "normalized_text",
        "normalized_text_sha256",
        "changed",
        "operations",
    }
    for raw in rows:
        if not isinstance(raw, dict) or set(raw) != required_row:
            raise FreshSuiteSelectionError("Stage-C normalization row schema is invalid.")
        sample_id = raw.get("sample_id")
        normalized = raw.get("normalized_text")
        digest = raw.get("normalized_text_sha256")
        if (
            not isinstance(sample_id, str)
            or sample_id in result
            or not isinstance(normalized, str)
            or not isinstance(digest, str)
            or hashlib.sha256(normalized.encode()).hexdigest() != digest
        ):
            raise FreshSuiteSelectionError("Stage-C normalization row is malformed or repeated.")
        result[sample_id] = cast(Mapping[str, object], raw)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--materialization", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--normalization-plan", type=Path)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--license-ledger", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--output-text-rejections", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    metadata_outputs = (
        arguments.output_manifest,
        arguments.output_text_rejections,
        arguments.output_report,
    )
    try:
        if (
            arguments.output_directory.exists()
            or any(path.exists() for path in metadata_outputs)
            or any(not path.parent.is_dir() for path in metadata_outputs)
        ):
            raise FreshSuiteSelectionError(
                "KazakhTTS Stage-C outputs must be new with existing parents."
            )
        if len(set(metadata_outputs)) != len(metadata_outputs) or arguments.seed < 0:
            raise FreshSuiteSelectionError("KazakhTTS Stage-C outputs/seed are invalid.")
        project_root = arguments.project_root.resolve(strict=True)
        data_root = arguments.data_root.resolve(strict=True)
        output_directory = arguments.output_directory.resolve(strict=False)
        try:
            output_directory.relative_to(data_root)
        except ValueError as error:
            raise FreshSuiteSelectionError(
                "KazakhTTS Stage-C asset directory must stay below data-root."
            ) from error
        output_directory.parent.mkdir(parents=True, exist_ok=True)
        plan = load_fresh_suite_selection(arguments.selection, project_root)
        _require_materialization(
            arguments.materialization, arguments.selection, arguments.base_manifest
        )
        selection_items = _selection_items(plan)
        base_rows = load_manifest(arguments.base_manifest)
        validate_manifest(base_rows)
        ledger = load_license_ledger(arguments.license_ledger)
        validate_manifest_licenses(base_rows, ledger)
        require_valid_assets(base_rows, data_root)
        language_counts = Counter(row.language for row in base_rows)
        if language_counts != {"ru": 50, "kk": 60, "mixed": 58}:
            raise FreshSuiteSelectionError(
                f"Stage-C base language counts changed: {dict(language_counts)}."
            )
        for row in base_rows:
            item = selection_items.get(row.sample_id)
            if (
                item is None
                or item.get("text_id") != row.text_id
                or item.get("text_hash") != row.text_hash
                or item.get("language") != row.language
            ):
                raise FreshSuiteSelectionError(
                    f"Stage-C base row is outside frozen selection: {row.sample_id!r}."
                )
        normalization = (
            None
            if arguments.normalization_plan is None
            else _load_normalization(
                arguments.normalization_plan,
                selection=arguments.selection,
                base_manifest=arguments.base_manifest,
                model_lock=arguments.model_lock,
                project_root=project_root,
            )
        )
        if normalization is not None and set(normalization) != {row.sample_id for row in base_rows}:
            raise FreshSuiteSelectionError(
                "Stage-C normalization plan does not exactly cover the ready base manifest."
            )
        lock = load_research_tts_model_lock(arguments.model_lock)
        if len(lock.models) != 1:
            raise ResearchTtsError("Stage-C synthesis model lock must contain one model.")
        model = lock.models[0]
        runtime = load_kazakhtts_runtime(model)
        if KAZAKHTTS_SOURCE_ID not in ledger:
            raise LicenseLedgerError([f"Missing synthetic source {KAZAKHTTS_SOURCE_ID!r}."])
        verified = verify_research_tts_model_lock(arguments.model_root, lock)
        device = resolve_kazakhtts_device(arguments.device)
        stage_directory = Path(
            tempfile.mkdtemp(
                prefix=".kds-stage-c-kazakhtts-",
                dir=output_directory.parent,
            )
        )
        stage_metadata = Path(
            tempfile.mkdtemp(
                prefix=".kds-stage-c-kazakhtts-metadata-",
                dir=arguments.output_report.parent,
            )
        )
        try:
            with tempfile.TemporaryDirectory(prefix="kds-stage-c-kazakhtts-runtime-") as temp:
                extracted = extract_verified_kazakhtts_runtime(
                    verified_paths=verified[model.model_id],
                    runtime=runtime,
                    destination=Path(temp) / "runtime",
                )
                accepted: list[tuple[ManifestRow, str]] = []
                text_rejections: list[dict[str, str]] = []
                for base in sorted(base_rows, key=lambda row: row.sample_id):
                    try:
                        source_text = cast(str, selection_items[base.sample_id]["text"])
                        synthesis_text = (
                            source_text
                            if normalization is None
                            else cast(str, normalization[base.sample_id]["normalized_text"])
                        )
                        normalized = validate_kazakhtts_text(
                            synthesis_text, extracted
                        )
                    except ResearchTtsError as error:
                        text_rejections.append(
                            {
                                "sample_id": base.sample_id,
                                "text_id": base.text_id,
                                "text_hash": base.text_hash,
                                "reason": str(error),
                            }
                        )
                        continue
                    accepted.append((base, normalized))
                text_to_speech, vocoder = load_kazakhtts_models(runtime, extracted, device)
                torch.manual_seed(arguments.seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(arguments.seed)
                rows = []
                generated: list[dict[str, object]] = []
                for index, (base, text) in enumerate(accepted, start=1):
                    waveform = synthesize_kazakhtts_waveform(text_to_speech, vocoder, text)
                    sample_key = hashlib.sha256(base.sample_id.encode()).hexdigest()[:20]
                    name = f"{sample_key}-{base.text_hash[:12]}.wav"
                    staged_asset = stage_directory / name
                    sf.write(staged_asset, waveform, runtime.sample_rate, subtype="PCM_16")
                    info = sf.info(str(staged_asset))
                    if (
                        info.samplerate != runtime.sample_rate
                        or info.channels != 1
                        or str(info.format).lower() != "wav"
                        or not math.isfinite(info.duration)
                        or info.duration <= 0
                    ):
                        raise ResearchTtsError(
                            f"KazakhTTS produced invalid WAV for {base.sample_id!r}."
                        )
                    relative_path = (
                        output_directory.relative_to(data_root) / name
                    ).as_posix()
                    row = kazakhtts_spoof_row(
                        base_row=base,
                        model=model,
                        runtime=runtime,
                        relative_path=relative_path,
                        sha256=sha256_file(staged_asset),
                        duration_s=float(info.duration),
                        created_at=arguments.created_at,
                        device=str(device),
                        normalizer_id=(
                            KAZAKHTTS_TEXT_NORMALIZER_ID if normalization is not None else ""
                        ),
                        synthesis_text_sha256=(
                            hashlib.sha256(text.encode()).hexdigest()
                            if normalization is not None
                            else ""
                        ),
                    )
                    rows.append(row)
                    generated.append(
                        {
                            "base_sample_id": base.sample_id,
                            "spoof_sample_id": row.sample_id,
                            "language": base.language,
                            "text_id": base.text_id,
                            "text_hash": base.text_hash,
                            "relative_path": relative_path,
                            "sha256": row.sha256,
                            "size_bytes": staged_asset.stat().st_size,
                            "duration_s": float(info.duration),
                            "sample_rate": int(info.samplerate),
                            "peak_abs_before_pcm16": float(np.max(np.abs(waveform))),
                            "rms_before_pcm16": float(
                                np.sqrt(np.mean(np.square(waveform, dtype=np.float64)))
                            ),
                        }
                    )
                    if index % 10 == 0 or index == len(accepted):
                        print(
                            json.dumps(
                                {"progress": index, "total": len(accepted)},
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
            validate_manifest(rows)
            validate_manifest_licenses(rows, ledger)
            staged_manifest = stage_metadata / arguments.output_manifest.name
            staged_text_rejections = stage_metadata / arguments.output_text_rejections.name
            staged_report = stage_metadata / arguments.output_report.name
            write_manifest(staged_manifest, rows)
            staged_text_rejections.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "selection_sha256": sha256_file(arguments.selection),
                        "selected_base_rows": len(base_rows),
                        "accepted_text_rows": len(rows),
                        "rejected_text_rows": text_rejections,
                        "post_selection_backfill": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            staged_report.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "protocol_id": (
                            "fresh-suite-stage-c-kazakhtts-normalized-synthesis-v2"
                            if normalization is not None
                            else "fresh-suite-stage-c-kazakhtts-synthesis-v1"
                        ),
                        "created_at": arguments.created_at,
                        "seed": arguments.seed,
                        "selection": {
                            "path": arguments.selection.as_posix(),
                            "sha256": sha256_file(arguments.selection),
                        },
                        "materialization": {
                            "path": arguments.materialization.as_posix(),
                            "sha256": sha256_file(arguments.materialization),
                        },
                        "base_manifest": {
                            "path": arguments.base_manifest.as_posix(),
                            "sha256": sha256_file(arguments.base_manifest),
                        },
                        "model_lock": {
                            "path": arguments.model_lock.as_posix(),
                            "sha256": sha256_file(arguments.model_lock),
                        },
                        "normalization_plan": (
                            None
                            if arguments.normalization_plan is None
                            else {
                                "path": arguments.normalization_plan.as_posix(),
                                "sha256": sha256_file(arguments.normalization_plan),
                                "normalizer_id": KAZAKHTTS_TEXT_NORMALIZER_ID,
                            }
                        ),
                        "model_id": model.model_id,
                        "generator_family": model.generator_family,
                        "generator_name": model.generator_name,
                        "generator_version": model.generator_version,
                        "fixed_voice_id": runtime.fixed_voice_id,
                        "device": str(device),
                        "torch_version": torch.__version__,
                        "cuda_version": torch.version.cuda,
                        "selected_base_rows": len(base_rows),
                        "text_rejected_rows": len(text_rejections),
                        "generated_rows": len(rows),
                        "generated_by_language": dict(
                            sorted(Counter(row.language for row in rows).items())
                        ),
                        "output_manifest": {
                            "path": arguments.output_manifest.as_posix(),
                            "sha256": sha256_file(staged_manifest),
                        },
                        "text_rejections": {
                            "path": arguments.output_text_rejections.as_posix(),
                            "sha256": sha256_file(staged_text_rejections),
                        },
                        "detector_inference_performed": False,
                        "full_asset_acoustic_gate_passed": False,
                        "outputs": generated,
                    },
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            if output_directory.exists() or any(
                path.exists() for path in metadata_outputs
            ):
                raise FreshSuiteSelectionError("A Stage-C synthesis output appeared while staging.")
            stage_directory.replace(output_directory)
            staged_manifest.replace(arguments.output_manifest)
            staged_text_rejections.replace(arguments.output_text_rejections)
            staged_report.replace(arguments.output_report)
        finally:
            shutil.rmtree(stage_directory, ignore_errors=True)
            shutil.rmtree(stage_metadata, ignore_errors=True)
    except (
        FreshSuiteSelectionError,
        ImportError,
        LicenseLedgerError,
        ManifestError,
        OSError,
        ResearchTtsError,
        RuntimeError,
        ValueError,
    ) as error:
        issues = (
            list(error.issues)
            if isinstance(error, (LicenseLedgerError, ManifestError))
            else [str(error)]
        )
        print(json.dumps({"status": "error", "issues": issues}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "generated": len(rows),
                "text_rejected": len(text_rejections),
                "output_manifest": str(arguments.output_manifest),
                "output_report": str(arguments.output_report),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
