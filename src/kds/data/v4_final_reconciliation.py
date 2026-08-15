"""Publication-only v4 final reconciliation after a failed salvage packet build.

The preceding salvage one-shot is exhausted.  This module has no extraction,
TTS, decoder, detector or inference path: it verifies existing assets and
decode journals, re-applies isolation, and exposes only complete ready pairs to
the independent review gate.
"""

# ruff: noqa: E501

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import soundfile as sf  # type: ignore[import-untyped]

from kds.data import v4_final_materialization as base
from kds.data import v4_final_salvage_materialization as salvage
from kds.data.assets import require_valid_assets, sha256_file
from kds.data.kazakhtts import load_kazakhtts_runtime
from kds.data.kazakhtts_text import KAZAKHTTS_TEXT_NORMALIZER_ID
from kds.data.licenses import load_license_ledger, validate_manifest_licenses
from kds.data.manifest import ManifestRow, validate_manifest, write_manifest
from kds.data.research_tts import load_research_tts_model_lock
from kds.data.v4_audio_gate import V4DecodeResult

PROTOCOL_ID = "xlsr-sls-model-v4-final-reconciliation-v1"
OUTPUTS = {
    "ru_source_raw_manifest": "data/manifests/v4/xlsr_sls_model_v4_final_reconciliation_ru_source_raw_v1.csv",
    "kk_source_raw_manifest": "data/manifests/v4/xlsr_sls_model_v4_final_reconciliation_kk_source_raw_v1.csv",
    "ru_spoof_raw_manifest": "data/manifests/v4/xlsr_sls_model_v4_final_reconciliation_ru_qwen_raw_v1.csv",
    "kk_spoof_raw_manifest": "data/manifests/v4/xlsr_sls_model_v4_final_reconciliation_kk_kazakhtts_raw_v1.csv",
    "ru_source_ready_manifest": "data/manifests/v4/xlsr_sls_model_v4_final_reconciliation_ru_source_ready_v1.csv",
    "kk_source_ready_manifest": "data/manifests/v4/xlsr_sls_model_v4_final_reconciliation_kk_source_ready_v1.csv",
    "ru_spoof_ready_manifest": "data/manifests/v4/xlsr_sls_model_v4_final_reconciliation_ru_qwen_ready_v1.csv",
    "kk_spoof_ready_manifest": "data/manifests/v4/xlsr_sls_model_v4_final_reconciliation_kk_kazakhtts_ready_v1.csv",
    "audio_inventory": "data/manifests/v4/xlsr_sls_model_v4_final_reconciliation_audio_gate_inventory_v1.csv",
    "review_packet": "data/manifests/v4/xlsr_sls_model_v4_final_reconciliation_acoustic_language_packet_v1.csv",
    "reviewer_a_template": "data/manifests/v4/xlsr_sls_model_v4_final_reconciliation_acoustic_language_reviewer_a_v1.csv",
    "reviewer_b_template": "data/manifests/v4/xlsr_sls_model_v4_final_reconciliation_acoustic_language_reviewer_b_v1.csv",
    "materialization_receipt": "docs/artifacts/v4/xlsr_sls_model_v4_final_reconciliation_v1.json",
    "pair_lock_manifest": "data/manifests/v4/xlsr_sls_model_v4_final_reconciliation_pairs_frozen_v1.csv",
    "pair_lock_receipt": "docs/artifacts/v4/xlsr_sls_model_v4_final_reconciliation_pair_lock_v1.json",
}
_REQUIRED = {
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
_PROHIBITIONS = {
    "network_downloads",
    "source_extraction",
    "synthetic_audio_generation",
    "resynthesis",
    "replacement_or_backfill",
    "decoder_execution",
    "detector_checkpoint_loading",
    "calibration",
    "detector_inference",
    "final_inference",
    "output_overwrite",
    "pair_lock_before_two_reviews",
}
DECODE_ROOT = "processed/v4/xlsr_sls_model_v4_final_salvage_materialization_v1"


class V4FinalReconciliationError(ValueError):
    """Raised when existing salvaged evidence cannot be published safely."""


def _verify(binding: base.Binding, root: Path, label: str) -> Path:
    try:
        return base._verify(binding, root, label)
    except base.V4FinalMaterializationError as error:
        raise V4FinalReconciliationError(str(error)) from error


def _path(root: Path, relative: str, label: str) -> Path:
    try:
        return base._project_path(root, relative, label)
    except base.V4FinalMaterializationError as error:
        raise V4FinalReconciliationError(str(error)) from error


@contextmanager
def _base_context() -> Iterator[None]:
    replacements = {
        "PROTOCOL_ID": PROTOCOL_ID,
        "RU_SOURCE_ID": salvage.RU_SOURCE_ID,
        "KK_SOURCE_ID": salvage.KK_SOURCE_ID,
        "RU_SPOOF_ID": salvage.RU_SPOOF_ID,
        "KK_SPOOF_ID": salvage.KK_SPOOF_ID,
    }
    old = {name: getattr(base, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in old.items():
            setattr(base, name, value)


def load_plan(path: Path, project_root: Path) -> base.Plan:
    root = project_root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    raw = base._object(resolved, "reconciliation plan")
    expected = {"schema_version", "protocol_id", "created_at", "inputs", "outputs", "prohibitions"}
    if set(raw) != expected or raw["schema_version"] != 1 or raw["protocol_id"] != PROTOCOL_ID:
        raise V4FinalReconciliationError("Reconciliation plan schema/protocol is invalid.")
    raw_inputs = base._mapping(raw["inputs"], "reconciliation inputs")
    if set(raw_inputs) != _REQUIRED:
        raise V4FinalReconciliationError("Reconciliation plan input set is invalid.")
    inputs = {name: base._binding(item, f"inputs.{name}") for name, item in raw_inputs.items()}
    outputs = {
        name: base._safe_path(item, f"output {name}")
        for name, item in base._mapping(raw["outputs"], "outputs").items()
    }
    prohibitions = base._mapping(raw["prohibitions"], "prohibitions")
    if (
        outputs != OUTPUTS
        or set(prohibitions) != _PROHIBITIONS
        or any(item is not True for item in prohibitions.values())
    ):
        raise V4FinalReconciliationError("Reconciliation outputs/prohibitions changed.")
    for name, binding in inputs.items():
        _verify(binding, root, f"inputs.{name}")
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise V4FinalReconciliationError("Plan must be below project root.") from error
    plan = base.Plan(
        relative,
        sha256_file(resolved),
        base._timestamp(raw["created_at"], "created_at"),
        inputs,
        {},
        DECODE_ROOT,
        "",
        {},
        outputs,
    )
    _load_selection(plan, root)
    return plan


def _salvage_plan(plan: base.Plan, root: Path) -> base.Plan:
    return salvage.load_plan(_verify(plan.inputs["salvage_plan"], root, "salvage plan"), root)


def _load_selection(plan: base.Plan, root: Path) -> tuple[base.SelectedRow, ...]:
    source = salvage._load_selection(_salvage_plan(plan, root), root)
    selection = salvage._read_selection(
        _verify(plan.inputs["salvage_selection"], root, "salvage selection")
    )
    if source != selection or len(selection) != 997:
        raise V4FinalReconciliationError(
            "Reconciliation selection does not match exhausted salvage scope."
        )
    return selection


def _journal(path: Path, label: str) -> list[dict[str, object]]:
    try:
        result = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V4FinalReconciliationError(f"Cannot read {label}.") from error
    if not all(isinstance(item, dict) for item in result):
        raise V4FinalReconciliationError(f"{label} contains a non-object event.")
    return result


def _completed_spoofs(
    rows: Sequence[ManifestRow],
    selected: Mapping[str, base.SelectedRow],
    journal: Path,
    raw_root: str,
    prefix: str,
    lock_path: Path,
    source_id: str,
    created_at: str,
    qwen: bool,
) -> tuple[ManifestRow, ...]:
    events = _journal(journal, "synthetic journal")
    planned = [item for item in events if item.get("event") == "planned"]
    generated = [item for item in events if item.get("event") == "generated"]
    expected = {row.sample_id for row in rows}
    if (
        len(events) != 2 * len(rows)
        or {item.get("sample_id") for item in planned} != expected
        or {item.get("base_sample_id") for item in generated} != expected
    ):
        raise V4FinalReconciliationError(
            "Synthetic journal is not a complete exact one-shot trace."
        )
    generated_by_base = {str(item["base_sample_id"]): item for item in generated}
    lock = load_research_tts_model_lock(lock_path)
    if len(lock.models) != 1:
        raise V4FinalReconciliationError("Synthetic model lock must contain exactly one route.")
    model = lock.models[0]
    runtime = None if qwen else load_kazakhtts_runtime(model)
    output: list[ManifestRow] = []
    directory = Path("data") / raw_root
    files: set[str] = set()
    for source in sorted(rows, key=lambda item: selected[item.sample_id].selection_rank):
        item = selected[source.sample_id]
        filename = f"{prefix}_{item.selection_rank:03d}_{source.text_hash[:12]}.wav"
        planned_item = next(
            event for event in planned if event.get("sample_id") == source.sample_id
        )
        generated_item = generated_by_base[source.sample_id]
        path = directory / filename
        if (
            planned_item.get("output") != filename
            or not path.is_file()
            or generated_item.get("audio_sha256") != sha256_file(path)
        ):
            raise V4FinalReconciliationError("Synthetic WAV does not match its one-shot journal.")
        info = sf.info(path)
        if info.channels != 1 or info.duration <= 0:
            raise V4FinalReconciliationError("Synthetic WAV is invalid.")
        with _base_context():
            row = base._spoof_row(
                source,
                model,
                source_id,
                path.relative_to("data").as_posix(),
                sha256_file(path),
                float(info.duration),
                int(info.samplerate),
                created_at,
                item.synthesis_seed,
                item.synthesis_text_sha256,
                "cuda:0",
            )
        if qwen:
            row = replace(row, voice_id="qwen3_tts_customvoice:aiden")
        else:
            assert runtime is not None
            row = replace(
                row,
                voice_id=f"{model.model_id}:{runtime.fixed_voice_id}",
                augmentation_chain=f"text_normalizer={KAZAKHTTS_TEXT_NORMALIZER_ID};synthesis_text_sha256={item.synthesis_text_sha256};reference_audio=forbidden;voice_cloning=false",
            )
        output.append(row)
        files.add(filename)
    actual = {
        path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()
    }
    if actual != files:
        raise V4FinalReconciliationError(
            "Synthetic raw namespace has unexpected/missing WAV files."
        )
    validate_manifest(output)
    return tuple(output)


def _source_and_raw(
    plan: base.Plan,
    root: Path,
    data_root: Path,
    common_voice_archive: Path,
    fleurs_root: Path,
    created_at: str,
) -> tuple[dict[str, tuple[ManifestRow, ...]], dict[str, str], dict[str, str]]:
    selection = _load_selection(plan, root)
    ru_source, kk_source, source_text, synthesis_text = salvage._source_rows_from_existing(
        selection, common_voice_archive, fleurs_root, data_root, created_at
    )
    selected = {row.sample_id: row for row in selection}
    ru_spoof = _completed_spoofs(
        ru_source,
        selected,
        _verify(plan.inputs["recovery_qwen_journal"], root, "recovery Qwen journal"),
        salvage.PARTIAL_RAW_ROOTS["ru_spoof"],
        "ru_qwen",
        _verify(plan.inputs["qwen_model_lock"], root, "Qwen lock"),
        salvage.RU_SPOOF_ID,
        created_at,
        True,
    )
    partial_kk_source = tuple(
        row for row in kk_source if selected[row.sample_id].selection_rank <= 271
    )
    partial_kk = _completed_spoofs(
        partial_kk_source,
        selected,
        _verify(plan.inputs["recovery_kk_journal"], root, "recovery KK journal"),
        salvage.PARTIAL_RAW_ROOTS["kk_spoof"],
        "kk_kazakhtts",
        _verify(plan.inputs["kazakhtts_model_lock"], root, "KazakhTTS lock"),
        salvage.KK_SPOOF_ID,
        created_at,
        False,
    )
    remaining_kk_source = tuple(
        row for row in kk_source if selected[row.sample_id].selection_rank > 271
    )
    remaining_kk = _completed_spoofs(
        remaining_kk_source,
        selected,
        _verify(plan.inputs["salvage_kk_journal"], root, "salvage KK journal"),
        "raw/v4/xlsr_sls_model_v4_final_salvage_materialization_v1/kk_kazakhtts_remaining",
        "kk_kazakhtts",
        _verify(plan.inputs["kazakhtts_model_lock"], root, "KazakhTTS lock"),
        salvage.KK_SPOOF_ID,
        created_at,
        False,
    )
    return (
        {
            "ru_source": ru_source,
            "kk_source": kk_source,
            "ru_spoof": ru_spoof,
            "kk_spoof": tuple((*partial_kk, *remaining_kk)),
        },
        source_text,
        synthesis_text,
    )


def _decode_results(
    path: Path, rows: Sequence[ManifestRow], data_root: Path
) -> dict[str, V4DecodeResult]:
    events = _journal(path, "decode journal")
    try:
        decoded = [V4DecodeResult(**cast(Any, item)) for item in events]
    except TypeError as error:
        raise V4FinalReconciliationError("Decode journal schema changed.") from error
    by_raw = {(row.relative_path, row.sha256): row for row in rows}
    if len(decoded) != len(rows) or len(
        {(item.raw_relative_path, item.raw_sha256) for item in decoded}
    ) != len(rows):
        raise V4FinalReconciliationError("Decode journal coverage is incomplete.")
    result: dict[str, V4DecodeResult] = {}
    for item in decoded:
        row = by_raw.get((item.raw_relative_path, item.raw_sha256))
        target = data_root / item.decoded_relative_path
        if row is None or not target.is_file() or sha256_file(target) != item.decoded_audio_sha256:
            raise V4FinalReconciliationError(
                "Decode journal no longer binds an exact canonical asset."
            )
        result[row.sample_id] = replace(item, sample_id=row.sample_id)
    return result


def _require_ledger(plan: base.Plan, root: Path) -> None:
    with _base_context():
        try:
            base._require_ledger(plan, root)
        except base.V4FinalMaterializationError as error:
            raise V4FinalReconciliationError(str(error)) from error


def _stage_outputs(root: Path, outputs: Mapping[str, str]) -> Path:
    targets = [_path(root, value, f"output {name}") for name, value in outputs.items()]
    if any(path.exists() or not path.parent.is_dir() for path in targets):
        raise V4FinalReconciliationError(
            "Reconciliation outputs must be new with existing parents."
        )
    return Path(tempfile.mkdtemp(prefix=".kds-v4-reconciliation-", dir=targets[0].parent))


def preflight_materialization(
    *,
    plan_path: Path,
    project_root: Path,
    data_root: Path,
    common_voice_archive: Path,
    fleurs_release_root: Path,
    created_at: str,
) -> base.Plan:
    root = project_root.resolve(strict=True)
    base._timestamp(created_at, "created_at")
    plan = load_plan(plan_path, root)
    if data_root.resolve(strict=True) != _path(root, "data", "data root"):
        raise V4FinalReconciliationError("data_root must be the project data directory.")
    _require_ledger(plan, root)
    try:
        base._verify_common_voice_archive(common_voice_archive)
    except base.V4FinalMaterializationError as error:
        raise V4FinalReconciliationError(str(error)) from error
    groups, _source, _synthesis = _source_and_raw(
        plan, root, data_root, common_voice_archive, fleurs_release_root, created_at
    )
    for name, rows in groups.items():
        _decode_results(
            _verify(plan.inputs[f"{name}_decode_journal"], root, f"{name} decode journal"),
            rows,
            data_root,
        )
    with _base_context():
        try:
            base._history(plan, root)
        except base.V4FinalMaterializationError as error:
            raise V4FinalReconciliationError(str(error)) from error
    if any(
        _path(root, value, "output").exists() or not _path(root, value, "output").parent.is_dir()
        for value in plan.outputs.values()
    ):
        raise V4FinalReconciliationError(
            "Reconciliation outputs must be new with existing parents."
        )
    return plan


def run_materialization(
    *,
    plan_path: Path,
    project_root: Path,
    data_root: Path,
    common_voice_archive: Path,
    fleurs_release_root: Path,
    created_at: str,
) -> base.Plan:
    root = project_root.resolve(strict=True)
    plan = preflight_materialization(
        plan_path=plan_path,
        project_root=root,
        data_root=data_root,
        common_voice_archive=common_voice_archive,
        fleurs_release_root=fleurs_release_root,
        created_at=created_at,
    )
    selection = _load_selection(plan, root)
    ranks = {row.sample_id: row.selection_rank for row in selection}
    pair = {row.sample_id: row.pair_key for row in selection}
    groups, source_text, synthesis_text = _source_and_raw(
        plan, root, data_root, common_voice_archive, fleurs_release_root, created_at
    )
    decoded = {
        name: _decode_results(
            _verify(plan.inputs[f"{name}_decode_journal"], root, f"{name} decode journal"),
            rows,
            data_root,
        )
        for name, rows in groups.items()
    }
    ledger = load_license_ledger(_verify(plan.inputs["materialization_ledger"], root, "ledger"))
    with _base_context():
        historical_exact, historical_signatures, history = base._history(plan, root)
        for source_name, spoof_name in (("ru_source", "ru_spoof"), ("kk_source", "kk_spoof")):
            sources = {row.text_id: row for row in groups[source_name]}
            for spoof in groups[spoof_name]:
                source = sources.get(spoof.text_id)
                if source is None or source.text_hash != spoof.text_hash:
                    raise V4FinalReconciliationError("Spoof lost its frozen source pairing.")
                ranks[spoof.sample_id] = ranks[source.sample_id]
                pair[spoof.sample_id] = pair[source.sample_id]
        decisions = {
            name: base._decisions(
                rows, decoded[name], ranks, historical_exact, historical_signatures
            )
            for name, rows in groups.items()
        }
        individual = {
            name: base._ready({row.sample_id: row for row in rows}, decisions[name], created_at)
            for name, rows in groups.items()
        }
    ready: dict[str, tuple[ManifestRow, ...]] = {}
    for language in ("ru", "kk"):
        source_name, spoof_name = f"{language}_source", f"{language}_spoof"
        sources = {row.text_id: row for row in individual[source_name]}
        spoofs = {row.text_id: row for row in individual[spoof_name]}
        keys = set(sources).intersection(spoofs)
        ready[source_name] = tuple(row for row in individual[source_name] if row.text_id in keys)
        ready[spoof_name] = tuple(row for row in individual[spoof_name] if row.text_id in keys)
    all_ready = tuple(
        row for name in ("ru_source", "kk_source", "ru_spoof", "kk_spoof") for row in ready[name]
    )
    for rows in (*groups.values(), *ready.values()):
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
            base._inventory(all_decisions, pair, historical_exact),
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
                "raw_assets": sum(len(rows) for rows in groups.values()),
                "individually_eligible_assets": sum(len(rows) for rows in individual.values()),
                "complete_eligible_pairs": len(all_ready) // 2,
                "review_assets": len(all_ready),
                "complete_pairs_by_language": {
                    language: len(ready[f"{language}_source"]) for language in ("ru", "kk")
                },
            },
            "history": history,
            "outputs": {
                name: {
                    "path": plan.outputs[name],
                    "sha256": sha256_file(staged[name]),
                    "rows": base._rows(staged[name]),
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
                "source_extraction_performed": False,
                "synthetic_audio_generated": False,
                "decoder_execution_performed": False,
                "technical_decode_qa_vad_reused_exactly": True,
                "full_history_audio_isolation_performed": True,
                "acoustic_review_performed": False,
                "pair_lock_performed": False,
                "detector_checkpoint_loaded": False,
                "calibration_performed": False,
                "detector_inference_performed": False,
                "final_inference_performed": False,
                "replacement_or_backfill": False,
                "resynthesis": False,
            },
        }
        staged["materialization_receipt"].write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with _base_context():
            packet = base._review_packet(
                all_ready,
                sha256_file(staged["materialization_receipt"]),
                source_text,
                synthesis_text,
                pair,
                data_root,
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
        if any(_path(root, plan.outputs[name], "output").exists() for name in publish):
            raise V4FinalReconciliationError("Output appeared during staging.")
        for name in publish:
            staged[name].replace(_path(root, plan.outputs[name], "output"))
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
    with _base_context():
        old_plan, old_selection = base.load_plan, base._load_selection
        try:
            base.load_plan, base._load_selection = load_plan, _load_selection
            return base.finalize_pair_lock(
                plan_path=plan_path,
                project_root=project_root,
                data_root=data_root,
                reviewer_a=reviewer_a,
                reviewer_b=reviewer_b,
                created_at=created_at,
            )
        finally:
            base.load_plan, base._load_selection = old_plan, old_selection
