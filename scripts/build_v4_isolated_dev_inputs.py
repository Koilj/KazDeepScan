"""Build the isolated XLS-R+SLS model-v4 bilingual dev inputs without training a detector."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import cast

import soundfile as sf  # type: ignore[import-untyped]
import torch

from kds.data.assets import (
    AssetValidationError,
    require_valid_assets,
    resolve_asset_path,
    sha256_file,
)
from kds.data.ksc_slr102 import (
    KSC_ARCHIVE_NAME,
    ExtractedKscAsset,
    KscIngestionError,
    attach_ksc_transcripts,
    extract_ksc_audio_slice,
    inspect_extracted_ksc_audio,
    ksc_manifest_rows,
    load_ksc_metadata_from_archive,
    select_ksc_records_from_archive_excluding_texts,
)
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
from kds.data.silero_v4 import (
    SileroV4Error,
    load_silero_v4_model,
    load_silero_v4_runtime,
    normalize_silero_v4_text,
    synthesize_silero_v4,
)
from kds.data.v4_audio_gate import (
    V4AudioGateError,
    V4AudioSignature,
    V4DecodedCandidate,
    V4DecodedDecision,
    V4DecodeResult,
    V4DecodeTask,
    append_v4_decode_journal,
    decide_v4_decoded_audio_eligibility,
    load_v4_decode_journal,
    run_v4_decode_task,
)
from kds.data.v4_dev_inputs import (
    V4_KK_DEV_SILERO_SOURCE_ID,
    V4DevInputsError,
    build_v4_combined_dev_manifest,
    freeze_v4_kk_dev_pairs,
    replace_with_decoded_v4_dev_row,
    v4_kk_dev_silero_spoof_row,
)
from kds.data.v4_selection import V4SelectionError, load_v4_exposure_inventory

PROTOCOL_ID = "xlsr-sls-model-v4-isolated-dev-inputs-v1"
PLAN_SCHEMA_VERSION = 1
SOURCE_DECODE_NAMESPACE = "processed/v4/xlsr_sls_model_v4_dev_inputs_v1/source"
SPOOF_DECODE_NAMESPACE = "processed/v4/xlsr_sls_model_v4_dev_inputs_v1/spoof"
RUNTIME_ROOT = "artifacts/v4/xlsr_sls_model_v4_dev_inputs_v1"

INVENTORY_FIELDS = (
    "kind",
    "selection_rank",
    "sample_id",
    "text_id",
    "text_hash",
    "raw_relative_path",
    "raw_audio_sha256",
    "decoded_relative_path",
    "decoded_audio_sha256",
    "preparation_status",
    "eligibility_status",
    "rejection_reason",
    "historical_exact_match_count",
    "historical_near_match_count",
    "within_pool_near_match_count",
)


class V4DevPlanError(ValueError):
    """Raised if the immutable v4 dev-input plan cannot be applied safely."""


@dataclass(frozen=True, slots=True)
class Binding:
    path: str
    sha256: str
    rows: int | None


@dataclass(frozen=True, slots=True)
class Plan:
    path: str
    sha256: str
    created_at: str
    inputs: Mapping[str, Binding]
    seed: str
    candidate_pairs: int
    frozen_pairs: int
    source_split: str
    archive_size_bytes: int
    archive_sha256: str
    slice_name: str
    outputs: Mapping[str, str]


def _now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _json_object(path: Path, label: str) -> dict[str, object]:
    try:
        result: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V4DevPlanError(f"Cannot read {label}: {path}") from error
    if not isinstance(result, dict):
        raise V4DevPlanError(f"{label} must be a JSON object.")
    return cast(dict[str, object], result)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise V4DevPlanError(f"{label} must be a JSON object.")
    return cast(dict[str, object], value)


def _safe_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise V4DevPlanError(f"{label} must be a non-empty project-relative path.")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or ".." in parsed.parts or "\\" in value or value == ".":
        raise V4DevPlanError(f"{label} is not a safe project-relative path.")
    return parsed.as_posix()


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise V4DevPlanError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise V4DevPlanError(f"{label} must be a positive integer.")
    return value


def _binding(value: object, label: str) -> Binding:
    raw = _mapping(value, label)
    if set(raw) != {"path", "sha256", "rows"}:
        raise V4DevPlanError(f"{label} must contain exactly path, sha256 and rows.")
    rows = raw["rows"]
    if rows is not None:
        _positive_int(rows, f"{label}.rows")
    return Binding(
        path=_safe_path(raw["path"], f"{label}.path"),
        sha256=_sha256(raw["sha256"], f"{label}.sha256"),
        rows=cast(int | None, rows),
    )


def _project_path(project_root: Path, relative: str, label: str) -> Path:
    candidate = (project_root / relative).resolve()
    try:
        candidate.relative_to(project_root)
    except ValueError as error:
        raise V4DevPlanError(f"{label} resolves outside the project root.") from error
    return candidate


def _csv_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise V4DevPlanError(f"Cannot count CSV rows in {path}.") from error


def _verify_binding(binding: Binding, project_root: Path, label: str) -> Path:
    path = _project_path(project_root, binding.path, label)
    if not path.is_file() or sha256_file(path) != binding.sha256:
        raise V4DevPlanError(f"{label} binding does not match: {binding.path}")
    if binding.rows is not None and _csv_rows(path) != binding.rows:
        raise V4DevPlanError(f"{label} row count changed: {binding.path}")
    return path


def load_plan(path: Path, project_root: Path) -> Plan:
    """Load the strict pre-execution contract for one isolated v4 dev materialization."""

    root = project_root.resolve(strict=True)
    plan_path = path.resolve(strict=True)
    try:
        relative = plan_path.relative_to(root).as_posix()
    except ValueError as error:
        raise V4DevPlanError("v4 dev-input plan must be inside the project root.") from error
    raw = _json_object(plan_path, "v4 dev-input plan")
    if set(raw) != {
        "schema_version",
        "protocol_id",
        "created_at",
        "inputs",
        "selection",
        "outputs",
        "prohibitions",
    }:
        raise V4DevPlanError("v4 dev-input plan keys are invalid.")
    if raw["schema_version"] != PLAN_SCHEMA_VERSION or raw["protocol_id"] != PROTOCOL_ID:
        raise V4DevPlanError("v4 dev-input plan protocol/schema is invalid.")
    created_at = raw["created_at"]
    if not isinstance(created_at, str):
        raise V4DevPlanError("v4 dev-input plan timestamp is invalid.")
    try:
        datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise V4DevPlanError("v4 dev-input plan timestamp is invalid.") from error
    raw_inputs = _mapping(raw["inputs"], "v4 dev-input plan inputs")
    required_inputs = {
        "roles_and_selection",
        "pyara_dev_manifest",
        "combined_train_manifest",
        "license_ledger",
        "silero_model_lock",
        "historical_fingerprint_inventory",
        "ksc_module",
        "silero_module",
        "audio_gate_module",
        "dev_input_module",
        "runner_script",
    }
    if set(raw_inputs) != required_inputs:
        raise V4DevPlanError("v4 dev-input plan inputs are incomplete.")
    inputs = {
        name: _binding(raw_inputs[name], f"v4 dev input {name}") for name in sorted(required_inputs)
    }
    selection = _mapping(raw["selection"], "v4 dev selection")
    if set(selection) != {
        "seed",
        "source_split",
        "candidate_pairs",
        "frozen_pairs",
        "archive",
        "slice_name",
        "detector_feedback",
        "backfill_outside_reserve",
    }:
        raise V4DevPlanError("v4 dev selection keys are invalid.")
    seed = selection["seed"]
    if not isinstance(seed, str) or not seed:
        raise V4DevPlanError("v4 dev selection seed is invalid.")
    if selection["source_split"] != "dev":
        raise V4DevPlanError("v4 KK dev must use only the original KSC dev split.")
    candidate_pairs = _positive_int(selection["candidate_pairs"], "candidate_pairs")
    frozen_pairs = _positive_int(selection["frozen_pairs"], "frozen_pairs")
    if candidate_pairs <= frozen_pairs:
        raise V4DevPlanError("v4 dev candidates must include a predeclared reserve.")
    archive = _mapping(selection["archive"], "v4 dev archive")
    if set(archive) != {"filename", "size_bytes", "sha256"}:
        raise V4DevPlanError("v4 dev archive binding keys are invalid.")
    if archive["filename"] != KSC_ARCHIVE_NAME:
        raise V4DevPlanError("v4 dev archive filename is invalid.")
    archive_size = _positive_int(archive["size_bytes"], "archive.size_bytes")
    archive_sha = _sha256(archive["sha256"], "archive.sha256")
    slice_name = selection["slice_name"]
    if (
        not isinstance(slice_name, str)
        or not slice_name
        or not slice_name.replace("-", "").replace("_", "").isalnum()
    ):
        raise V4DevPlanError("v4 dev slice-name is invalid.")
    if (
        selection["detector_feedback"] is not False
        or selection["backfill_outside_reserve"] is not False
    ):
        raise V4DevPlanError("v4 dev selection must prohibit detector feedback and backfill.")
    outputs = _mapping(raw["outputs"], "v4 dev outputs")
    required_outputs = {
        "source_raw_manifest",
        "source_ready_manifest",
        "spoof_raw_manifest",
        "spoof_ready_manifest",
        "audio_gate_inventory",
        "kk_frozen_pairs_manifest",
        "combined_dev_manifest",
        "receipt",
    }
    if set(outputs) != required_outputs:
        raise V4DevPlanError("v4 dev outputs are incomplete.")
    safe_outputs = {name: _safe_path(value, f"output {name}") for name, value in outputs.items()}
    if len(set(safe_outputs.values())) != len(safe_outputs):
        raise V4DevPlanError("v4 dev output paths must be distinct.")
    prohibitions = _mapping(raw["prohibitions"], "v4 dev prohibitions")
    if prohibitions != {
        "network": True,
        "detector_or_logit_feedback": True,
        "actual_training_execution": True,
        "checkpoint_selection": True,
        "calibration": True,
        "final_inference": True,
        "output_overwrite": True,
    }:
        raise V4DevPlanError("v4 dev prohibitions are not fail-closed.")
    for name, binding in inputs.items():
        _verify_binding(binding, root, f"v4 dev input {name}")
    return Plan(
        path=relative,
        sha256=sha256_file(plan_path),
        created_at=created_at,
        inputs=inputs,
        seed=seed,
        candidate_pairs=candidate_pairs,
        frozen_pairs=frozen_pairs,
        source_split="dev",
        archive_size_bytes=archive_size,
        archive_sha256=archive_sha,
        slice_name=slice_name,
        outputs=safe_outputs,
    )


def _validate_role_binding(plan: Plan, project_root: Path) -> None:
    roles = _json_object(
        _project_path(project_root, plan.inputs["roles_and_selection"].path, "roles"),
        "v4 roles-and-selection",
    )
    raw_roles = _mapping(roles.get("roles"), "v4 roles")
    dev = _mapping(raw_roles.get("dev"), "v4 dev role")
    if dev != {
        "ru": "pyara_ru_v7:historical_dev_role_reuse_only",
        "kk_bonafide": "ksc_slr102:dev_only",
        "kk_spoof": "silero_v4_kk:dev_only",
        "selection_metric": "dev_loss_only",
    }:
        raise V4DevPlanError("v4 roles no longer reserve the required isolated dev layer.")


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise V4DevPlanError("This write-once v4 Silero dev run requires an available CUDA device.")
    return device


def _decoded_relative_path(raw_sha256: str, namespace: str) -> str:
    if (
        len(raw_sha256) != 64
        or any(character not in "0123456789abcdef" for character in raw_sha256)
    ):
        raise V4DevPlanError("v4 dev raw audio SHA-256 is invalid.")
    return f"{namespace}/{raw_sha256[:2]}/{raw_sha256}.wav"


def _history(
    path: Path,
) -> tuple[dict[str, tuple[str, ...]], tuple[V4AudioSignature, ...]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except OSError as error:
        raise V4DevPlanError("Cannot read v4 historical fingerprint inventory.") from error
    expected = {
        "manifest_audio_sha256",
        "reference_count",
        "references",
        "local_asset_state",
        "selected_relative_path",
        "canonical_audio_sha256",
        "duration_s",
        "speech_seconds",
        "audio_fingerprint_v1",
        "fingerprint_status",
        "detail",
    }
    if not rows or set(rows[0]) != expected:
        raise V4DevPlanError("v4 historical fingerprint inventory schema is invalid.")
    exact: dict[str, list[str]] = {}
    by_canonical: dict[str, V4AudioSignature] = {}
    for row in rows:
        try:
            references = json.loads(row["references"])
            if not isinstance(references, list) or not all(
                isinstance(item, str) for item in references
            ):
                raise TypeError
        except (TypeError, json.JSONDecodeError) as error:
            raise V4DevPlanError("v4 history references are invalid.") from error
        exact.setdefault(row["manifest_audio_sha256"], []).extend(cast(list[str], references))
        if row["fingerprint_status"] != "fingerprinted":
            continue
        canonical = row["canonical_audio_sha256"]
        reference = f"history:{canonical}"
        exact.setdefault(canonical, []).append(reference)
        signature = V4AudioSignature(
            identity=reference,
            audio_sha256=canonical,
            fingerprint=row["audio_fingerprint_v1"],
            speech_seconds=float(row["speech_seconds"]),
        )
        prior = by_canonical.get(canonical)
        if prior is not None and prior != signature:
            raise V4DevPlanError("v4 history canonical audio has inconsistent signatures.")
        by_canonical[canonical] = signature
    return (
        {key: tuple(sorted(set(value))) for key, value in exact.items()},
        tuple(by_canonical.values()),
    )


def _run_decode_tasks(
    tasks: Mapping[str, V4DecodeTask], journal: Path, workers: int, stage: str
) -> dict[str, V4DecodeResult]:
    results = load_v4_decode_journal(journal, tasks)
    remaining = [task for sample_id, task in tasks.items() if sample_id not in results]
    if not remaining:
        return results
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures: dict[Future[V4DecodeResult], V4DecodeTask] = {
            executor.submit(run_v4_decode_task, task): task for task in remaining
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            append_v4_decode_journal(journal, result)
            results[result.sample_id] = result
            if completed % 50 == 0 or completed == len(remaining):
                print(
                    json.dumps(
                        {
                            "status": "progress",
                            "stage": stage,
                            "completed": len(results),
                            "total": len(tasks),
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
    return results


def _decision_inventory_rows(
    kind: str,
    ranks: Mapping[str, int],
    decisions: Sequence[V4DecodedDecision],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for decision in decisions:
        decoded = decision.candidate.result
        result.append(
            {
                "kind": kind,
                "selection_rank": ranks[decoded.sample_id],
                "sample_id": decoded.sample_id,
                "text_id": "",
                "text_hash": "",
                "raw_relative_path": decoded.raw_relative_path,
                "raw_audio_sha256": decoded.raw_sha256,
                "decoded_relative_path": decoded.decoded_relative_path,
                "decoded_audio_sha256": decoded.decoded_audio_sha256,
                "preparation_status": decoded.preparation_status,
                "eligibility_status": decision.eligibility_status,
                "rejection_reason": decision.rejection_reason,
                "historical_exact_match_count": len(decision.historical_exact_matches),
                "historical_near_match_count": len(decision.historical_near_matches),
                "within_pool_near_match_count": len(decision.within_pool_near_matches),
            }
        )
    return result


def _require_disjoint(left: Sequence[ManifestRow], right: Sequence[ManifestRow]) -> None:
    for field in ("sample_id", "sha256", "text_hash", "parent_group_id"):
        overlap = {getattr(row, field) for row in left}.intersection(
            getattr(row, field) for row in right
        )
        if overlap:
            raise V4DevInputsError(
                f"v4 train/dev overlap on {field}: {len(overlap)}; example={min(overlap)!r}."
            )


def _publish(
    *,
    plan: Plan,
    project_root: Path,
    data_root: Path,
    source_raw: Sequence[ManifestRow],
    source_ready: Sequence[ManifestRow],
    spoof_raw: Sequence[ManifestRow],
    spoof_ready: Sequence[ManifestRow],
    inventory_rows: Sequence[Mapping[str, object]],
    kk_frozen: Sequence[ManifestRow],
    combined_dev: Sequence[ManifestRow],
    source_decisions: Sequence[V4DecodedDecision],
    spoof_decisions: Sequence[V4DecodedDecision],
    text_rejected: int,
    exposure_rows: int,
    exposure_bindings: Sequence[Mapping[str, object]],
    source_journal: Path,
    spoof_journal: Path,
    created_at: str,
) -> None:
    outputs = {
        name: _project_path(project_root, value, f"output {name}")
        for name, value in plan.outputs.items()
    }
    if any(path.exists() or not path.parent.is_dir() for path in outputs.values()):
        raise V4DevPlanError("v4 dev output exists already or has no parent directory.")
    ledger = load_license_ledger(project_root / plan.inputs["license_ledger"].path)
    for rows in (source_raw, source_ready, spoof_raw, spoof_ready, kk_frozen, combined_dev):
        validate_manifest(rows)
        validate_manifest_licenses(rows, ledger)
    for rows in (source_ready, spoof_ready, kk_frozen, combined_dev):
        require_valid_assets(rows, data_root)
    published: list[tuple[Path, Path]] = []
    try:
        with tempfile.TemporaryDirectory(
            prefix="kds-v4-dev-inputs-", dir=project_root
        ) as stage_name:
            stage = Path(stage_name)
            staged = {name: stage / path.name for name, path in outputs.items()}
            write_manifest(staged["source_raw_manifest"], source_raw)
            write_manifest(staged["source_ready_manifest"], source_ready)
            write_manifest(staged["spoof_raw_manifest"], spoof_raw)
            write_manifest(staged["spoof_ready_manifest"], spoof_ready)
            with staged["audio_gate_inventory"].open("x", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=INVENTORY_FIELDS, lineterminator="\n")
                writer.writeheader()
                writer.writerows(
                    sorted(
                        inventory_rows,
                        key=lambda row: (str(row["kind"]), int(row["selection_rank"])),
                    )
                )
            write_manifest(staged["kk_frozen_pairs_manifest"], kk_frozen)
            write_manifest(staged["combined_dev_manifest"], combined_dev)
            receipt = {
                "schema_version": 1,
                "protocol_id": PROTOCOL_ID,
                "created_at": created_at,
                "state": "isolated_dev_inputs_frozen_training_contract_pending",
                "bindings": {
                    "plan": {"path": plan.path, "sha256": plan.sha256},
                    **{name: asdict(binding) for name, binding in sorted(plan.inputs.items())},
                    "runtime": {
                        "source_decode_journal": {
                            "path": source_journal.relative_to(project_root).as_posix(),
                            "sha256": sha256_file(source_journal),
                            "rows": len(source_decisions),
                        },
                        "spoof_decode_journal": {
                            "path": spoof_journal.relative_to(project_root).as_posix(),
                            "sha256": sha256_file(spoof_journal),
                            "rows": len(spoof_decisions),
                        },
                    },
                    "project_history": {
                        "rows_with_version_duplicates": exposure_rows,
                        "manifest_bindings": list(exposure_bindings),
                    },
                },
                "selection": {
                    "source_split": plan.source_split,
                    "seed": plan.seed,
                    "candidate_pairs": plan.candidate_pairs,
                    "frozen_pairs": plan.frozen_pairs,
                    "reserve_pairs": plan.candidate_pairs - plan.frozen_pairs,
                    "order": "selection_rank_then_text_id",
                    "text_rejected_before_audio_extraction": text_rejected,
                    "detector_feedback": False,
                    "backfill_outside_reserve": False,
                },
                "outputs": {
                    name: {
                        "path": plan.outputs[name],
                        "sha256": sha256_file(path),
                        "rows": _csv_rows(path),
                    }
                    for name, path in staged.items()
                    if name != "receipt"
                },
                "audio_gate": {
                    "canonical_decode": "ffmpeg mono pcm_s16le 16000 Hz",
                    "technical_decode_qa_vad_performed": True,
                    "historical_exact_audio_screen_performed": True,
                    "historical_near_audio_screen_performed": True,
                    "within_pool_exact_and_near_audio_screen_performed": True,
                    "source_decision_counts": dict(
                        sorted(
                            Counter(item.eligibility_status for item in source_decisions).items()
                        )
                    ),
                    "spoof_decision_counts": dict(
                        sorted(Counter(item.eligibility_status for item in spoof_decisions).items())
                    ),
                },
                "claims": {
                    "synthesis_performed": True,
                    "detector_or_logit_feedback_used": False,
                    "actual_training_execution": False,
                    "checkpoint_selection": False,
                    "calibration": False,
                    "final_inference": False,
                    "speaker_independence": "not_verified_speaker_independent",
                    "kk_probability_claim": False,
                },
                "next_gate": (
                    "create a separate full v4 training contract that hash-pins this combined dev "
                    "manifest, the combined train manifest, runtime, hyperparameters and "
                    "write-once outputs"
                ),
            }
            staged["receipt"].write_text(
                json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            for name, output in outputs.items():
                os.link(staged[name], output)
                published.append((output, staged[name]))
    except (OSError, ValueError) as error:
        for output, staged_path in reversed(published):
            try:
                if output.samefile(staged_path):
                    output.unlink()
            except OSError:
                pass
        raise V4DevPlanError(f"Cannot publish v4 dev packet atomically: {error}") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/research/v4/xlsr_sls_model_v4_dev_inputs_v1.json"),
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--model-root", type=Path, default=Path("models/research/silero_v4_cyrillic_v1")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--created-at", default=_now())
    arguments = parser.parse_args(argv)
    try:
        if arguments.workers not in range(1, 65):
            raise V4DevPlanError("v4 dev decode workers must be between 1 and 64.")
        datetime.fromisoformat(arguments.created_at.replace("Z", "+00:00"))
        project_root = arguments.project_root.resolve(strict=True)
        data_root = arguments.data_root.resolve(strict=True)
        plan = load_plan(arguments.plan, project_root)
        _validate_role_binding(plan, project_root)
        device = _device(arguments.device)
        if (
            arguments.archive.name != KSC_ARCHIVE_NAME
            or not arguments.archive.is_file()
            or arguments.archive.stat().st_size != plan.archive_size_bytes
        ):
            raise V4DevPlanError("KSC archive does not match the dev-input size/name binding.")
        if (
            plan.archive_sha256
            != "a200aa3ab6b0284a7241ac357951fa5422f6fea855a30c1ab2fa1559c3f0d149"
        ):
            raise V4DevPlanError(
                "KSC archive hash no longer matches the verified local ledger value."
            )
        outputs = {
            name: _project_path(project_root, value, name) for name, value in plan.outputs.items()
        }
        if any(path.exists() for path in outputs.values()):
            raise V4DevPlanError("v4 dev outputs are write-once and one already exists.")
        source_directory = data_root / "raw" / "ksc_slr102" / "slices" / plan.slice_name
        spoof_directory = (
            data_root / "raw" / V4_KK_DEV_SILERO_SOURCE_ID / "slices" / plan.slice_name
        )
        if source_directory.exists() or spoof_directory.exists():
            raise V4DevPlanError(
                "v4 dev raw slice already exists; refusing replacement or resynthesis."
            )
        exposure = load_v4_exposure_inventory(data_root / "manifests", project_root)
        pyara = load_manifest(project_root / plan.inputs["pyara_dev_manifest"].path)
        validate_manifest(pyara)
        ledger = load_license_ledger(project_root / plan.inputs["license_ledger"].path)
        validate_manifest_licenses(pyara, ledger)
        require_valid_assets(pyara, data_root)
        text_rejected = 0

        def text_eligible(text: str) -> bool:
            nonlocal text_rejected
            try:
                normalize_silero_v4_text(text)
            except SileroV4Error:
                text_rejected += 1
                return False
            return True

        selected_index, _archive_report = select_ksc_records_from_archive_excluding_texts(
            arguments.archive,
            load_ksc_metadata_from_archive(arguments.archive, [plan.source_split]),
            plan.candidate_pairs,
            plan.seed,
            excluded_utterance_ids={
                sample.removeprefix("ksc_slr102:")
                for sample in exposure.sample_ids
                if sample.startswith("ksc_slr102:")
            },
            excluded_text_hashes=exposure.text_hashes,
            transcript_filter=text_eligible,
        )
        source_directory.parent.mkdir(parents=True, exist_ok=True)
        extracted = extract_ksc_audio_slice(
            arguments.archive,
            (item.utterance_id for item in selected_index),
            source_directory,
            excluded_text_hashes=exposure.text_hashes,
        )
        selected = attach_ksc_transcripts(selected_index, source_directory)
        transcripts = {item.utterance_id: item.transcript for item in selected}
        assets: dict[str, ExtractedKscAsset] = {}
        for utterance_id, raw_path in extracted.items():
            duration_s, original_sr, codec = inspect_extracted_ksc_audio(raw_path)
            assets[utterance_id] = ExtractedKscAsset(
                utterance_id=utterance_id,
                relative_path=raw_path.relative_to(data_root).as_posix(),
                sha256=sha256_file(raw_path),
                duration_s=duration_s,
                original_sr=original_sr,
                codec=codec,
            )
        source_raw = tuple(ksc_manifest_rows(selected, assets, arguments.created_at))
        validate_manifest(source_raw)
        validate_manifest_licenses(source_raw, ledger)
        source_ranks = {row.sample_id: index for index, row in enumerate(source_raw, start=1)}
        source_tasks = {
            row.sample_id: V4DecodeTask(
                sample_id=row.sample_id,
                raw_relative_path=row.relative_path,
                raw_sha256=row.sha256,
                source_path=str(resolve_asset_path(data_root, row.relative_path)),
                decoded_relative_path=_decoded_relative_path(
                    row.sha256, namespace=SOURCE_DECODE_NAMESPACE
                ),
                destination_path=str(
                    resolve_asset_path(
                        data_root,
                        _decoded_relative_path(row.sha256, SOURCE_DECODE_NAMESPACE),
                    )
                ),
            )
            for row in source_raw
        }
        runtime = project_root / RUNTIME_ROOT
        runtime.mkdir(parents=True, exist_ok=True)
        source_results = _run_decode_tasks(
            source_tasks,
            runtime / "source_decode_qa.jsonl",
            arguments.workers,
            "v4_dev_source_decode_qa",
        )
        exact, historical_signatures = _history(
            project_root / plan.inputs["historical_fingerprint_inventory"].path
        )
        source_decisions = decide_v4_decoded_audio_eligibility(
            tuple(
                V4DecodedCandidate(
                    selection_rank=source_ranks[sample_id],
                    language="kk",
                    label="bonafide",
                    result=result,
                )
                for sample_id, result in source_results.items()
            ),
            exact,
            historical_signatures,
        )
        source_by_id = {row.sample_id: row for row in source_raw}
        source_ready = tuple(
            replace_with_decoded_v4_dev_row(
                source_by_id[item.candidate.result.sample_id],
                relative_path=item.candidate.result.decoded_relative_path,
                sha256=item.candidate.result.decoded_audio_sha256,
                duration_s=item.candidate.result.duration_s,
                created_at=arguments.created_at,
            )
            for item in source_decisions
            if item.eligibility_status == "eligible"
        )
        if not source_ready:
            raise V4DevInputsError("No KSC dev rows passed technical and audio-leakage QA.")
        lock = load_research_tts_model_lock(project_root / plan.inputs["silero_model_lock"].path)
        if len(lock.models) != 1:
            raise V4DevPlanError("The v4 dev Silero lock must contain exactly one model.")
        model_spec = lock.models[0]
        runtime_spec = load_silero_v4_runtime(model_spec)
        verified = verify_research_tts_model_lock(arguments.model_root, lock)
        model = load_silero_v4_model(
            verified[model_spec.model_id][runtime_spec.package_path], runtime_spec, device
        )
        assignments = [
            (
                row,
                runtime_spec.profiles_by_language["kk"][
                    index % len(runtime_spec.profiles_by_language["kk"])
                ],
            )
            for index, row in enumerate(sorted(source_ready, key=lambda item: item.sample_id))
        ]
        spoof_directory.parent.mkdir(parents=True, exist_ok=True)
        stage_directory = Path(
            tempfile.mkdtemp(prefix="kds-v4-dev-silero-", dir=spoof_directory.parent)
        )
        try:
            spoof_raw_list: list[ManifestRow] = []
            for base_row, profile in assignments:
                utterance_id = base_row.sample_id.removeprefix("ksc_slr102:")
                text = transcripts.get(utterance_id)
                if text is None:
                    raise V4DevInputsError(
                        f"Missing selected KSC transcript for {base_row.sample_id!r}."
                    )
                key = hashlib.sha256(
                    f"{base_row.sample_id}:{profile.voice_id}".encode()
                ).hexdigest()[:16]
                relative = (
                    Path("raw")
                    / V4_KK_DEV_SILERO_SOURCE_ID
                    / "slices"
                    / plan.slice_name
                    / model_spec.model_id
                    / "kk"
                    / f"{key}.wav"
                )
                output = stage_directory / model_spec.model_id / "kk" / f"{key}.wav"
                output.parent.mkdir(parents=True, exist_ok=True)
                synthesize_silero_v4(
                    model=model, profile=profile, text=text, runtime=runtime_spec, output=output
                )
                info = sf.info(str(output))
                if (
                    info.duration <= 0
                    or info.samplerate != runtime_spec.sample_rate
                    or str(info.format).lower() != "wav"
                ):
                    raise V4DevInputsError(
                        f"Silero produced an invalid v4 dev WAV for {base_row.sample_id!r}."
                    )
                spoof_raw_list.append(
                    v4_kk_dev_silero_spoof_row(
                        base_row=base_row,
                        model=model_spec,
                        profile=profile,
                        relative_path=relative.as_posix(),
                        sha256=sha256_file(output),
                        duration_s=float(info.duration),
                        original_sr=int(info.samplerate),
                        created_at=arguments.created_at,
                        device=f"local_{device.type}_silero_v4_fastpitch_hifigan",
                    )
                )
            if spoof_directory.exists():
                raise V4DevPlanError("v4 dev Silero slice appeared while staging.")
            spoof_directory.parent.mkdir(parents=True, exist_ok=True)
            stage_directory.replace(spoof_directory)
        finally:
            shutil.rmtree(stage_directory, ignore_errors=True)
        spoof_raw = tuple(spoof_raw_list)
        validate_manifest(spoof_raw)
        validate_manifest_licenses(spoof_raw, ledger)
        require_valid_assets(spoof_raw, data_root)
        spoof_ranks = {
            row.sample_id: source_ranks["ksc_slr102:" + row.text_id.removeprefix("ksc_slr102:")]
            for row in spoof_raw
        }
        spoof_tasks = {
            row.sample_id: V4DecodeTask(
                sample_id=row.sample_id,
                raw_relative_path=row.relative_path,
                raw_sha256=row.sha256,
                source_path=str(resolve_asset_path(data_root, row.relative_path)),
                decoded_relative_path=_decoded_relative_path(
                    row.sha256, namespace=SPOOF_DECODE_NAMESPACE
                ),
                destination_path=str(
                    resolve_asset_path(
                        data_root,
                        _decoded_relative_path(row.sha256, SPOOF_DECODE_NAMESPACE),
                    )
                ),
            )
            for row in spoof_raw
        }
        spoof_results = _run_decode_tasks(
            spoof_tasks,
            runtime / "spoof_decode_qa.jsonl",
            arguments.workers,
            "v4_dev_spoof_decode_qa",
        )
        source_signatures = tuple(
            V4AudioSignature(
                identity=f"v4-dev-source:{item.candidate.result.sample_id}",
                audio_sha256=item.candidate.result.decoded_audio_sha256,
                fingerprint=item.candidate.result.audio_fingerprint_v1,
                speech_seconds=item.candidate.result.speech_seconds,
            )
            for item in source_decisions
            if item.eligibility_status == "eligible"
        )
        for signature in source_signatures:
            exact[signature.audio_sha256] = tuple(
                sorted({*exact.get(signature.audio_sha256, ()), signature.identity})
            )
        spoof_decisions = decide_v4_decoded_audio_eligibility(
            tuple(
                V4DecodedCandidate(
                    selection_rank=spoof_ranks[sample_id],
                    language="kk",
                    label="spoof",
                    result=result,
                )
                for sample_id, result in spoof_results.items()
            ),
            exact,
            (*historical_signatures, *source_signatures),
        )
        spoof_by_id = {row.sample_id: row for row in spoof_raw}
        spoof_ready = tuple(
            replace_with_decoded_v4_dev_row(
                spoof_by_id[item.candidate.result.sample_id],
                relative_path=item.candidate.result.decoded_relative_path,
                sha256=item.candidate.result.decoded_audio_sha256,
                duration_s=item.candidate.result.duration_s,
                created_at=arguments.created_at,
            )
            for item in spoof_decisions
            if item.eligibility_status == "eligible"
        )
        rank_by_text = {row.text_id: source_ranks[row.sample_id] for row in source_raw}
        kk_frozen = freeze_v4_kk_dev_pairs(
            source_ready, spoof_ready, source_ranks=rank_by_text, target_pairs=plan.frozen_pairs
        )
        combined_dev = build_v4_combined_dev_manifest(pyara, kk_frozen)
        train = load_manifest(project_root / plan.inputs["combined_train_manifest"].path)
        _require_disjoint(combined_dev, train)
        inventory_rows = _decision_inventory_rows("source", source_ranks, source_decisions)
        inventory_rows.extend(_decision_inventory_rows("spoof", spoof_ranks, spoof_decisions))
        text_by_sample = {
            row.sample_id: (row.text_id, row.text_hash) for row in (*source_raw, *spoof_raw)
        }
        for row in inventory_rows:
            row["text_id"], row["text_hash"] = text_by_sample[str(row["sample_id"])]
        _publish(
            plan=plan,
            project_root=project_root,
            data_root=data_root,
            source_raw=source_raw,
            source_ready=source_ready,
            spoof_raw=spoof_raw,
            spoof_ready=spoof_ready,
            inventory_rows=inventory_rows,
            kk_frozen=kk_frozen,
            combined_dev=combined_dev,
            source_decisions=source_decisions,
            spoof_decisions=spoof_decisions,
            text_rejected=text_rejected,
            exposure_rows=exposure.rows,
            exposure_bindings=exposure.manifest_bindings,
            source_journal=runtime / "source_decode_qa.jsonl",
            spoof_journal=runtime / "spoof_decode_qa.jsonl",
            created_at=arguments.created_at,
        )
    except (
        AssetValidationError,
        KscIngestionError,
        LicenseLedgerError,
        ManifestError,
        ResearchTtsError,
        SileroV4Error,
        V4AudioGateError,
        V4DevInputsError,
        V4DevPlanError,
        V4SelectionError,
        OSError,
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
                "status": "complete",
                "receipt": plan.outputs["receipt"],
                "frozen_kk_pairs": plan.frozen_pairs,
                "training_authorized": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
