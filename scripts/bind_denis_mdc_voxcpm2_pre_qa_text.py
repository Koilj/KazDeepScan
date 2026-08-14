"""Freeze the exact 64 Denis ready texts and the one-attempt VoxCPM2 contract."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import tarfile
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Protocol, cast

from kds.data.assets import require_valid_assets, sha256_file
from kds.data.denis import (
    DENIS_ARCHIVE_EXPECTED_SHA256,
    DENIS_ARCHIVE_EXPECTED_SIZE_BYTES,
    DENIS_SOURCE_ID,
    DenisRecord,
    inspect_denis_archive,
)
from kds.data.denis_voxcpm2_candidate import (
    DENIS_VOXCPM2_SOURCE_ID,
    DENIS_VOXCPM2_TEXT_BINDING_PROTOCOL_ID,
    DENIS_VOXCPM2_VOICE_ID,
)
from kds.data.licenses import LicenseLedgerError, load_license_ledger, validate_manifest_licenses
from kds.data.manifest import ManifestError, ManifestRow, load_manifest, validate_manifest
from kds.data.research_tts import ResearchTtsError, load_research_tts_model_lock
from kds.data.voxcpm2 import VOXCPM2_MODEL_REVISION, VOXCPM2_SOURCE_REVISION
from kds.data.voxcpm2_text_only import (
    VOXCPM2_FIXED_SEED,
    bind_text,
    collapse_whitespace,
)

_MATERIALIZATION_PROTOCOL_ID = "denis-1-0-mdc-pre-qa-materialization-v1"
_MODEL_ID = "openbmb_voxcpm2_official_text_only"
_READY_ROWS = 64
_TARGET_ROWS = 79
_HEX = frozenset("0123456789abcdef")


class DenisVoxCPM2TextBindingError(ValueError):
    """Raised when the ready Denis text cannot be frozen before candidate synthesis."""


class FrozenDenisSelectionRow(Protocol):
    selection_rank: int
    sample_id: str
    member_stem: str
    category: str
    text_id: str
    literal_text_sha256: str
    whitespace_canonical_text_sha256: str
    nfkc_whitespace_canonical_text_sha256: str


SelectionLoader = Callable[[Path, Path, Path], tuple[FrozenDenisSelectionRow, ...]]
SelectionBinder = Callable[[Sequence[FrozenDenisSelectionRow], Sequence[DenisRecord]], object]


def _selection_helpers() -> tuple[SelectionLoader, SelectionBinder]:
    """Load the existing audited selection parser in script and module launch modes."""

    try:
        module = importlib.import_module("scripts.materialize_denis_mdc_pre_qa")
    except ModuleNotFoundError:
        module = importlib.import_module("materialize_denis_mdc_pre_qa")
    return (
        cast(SelectionLoader, module.load_frozen_denis_selection),
        cast(SelectionBinder, module.bind_denis_selection),
    )


def _json_object(path: Path, label: str) -> Mapping[str, object]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DenisVoxCPM2TextBindingError(f"Cannot read {label}: {path}.") from error
    if not isinstance(payload, dict):
        raise DenisVoxCPM2TextBindingError(f"{label} must be a JSON object.")
    return cast(Mapping[str, object], payload)


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value).difference(_HEX):
        raise DenisVoxCPM2TextBindingError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _safe_member(value: str) -> None:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts or "\\" in value:
        raise DenisVoxCPM2TextBindingError("Denis TAR contains an unsafe member path.")


def _project_file(project_root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise DenisVoxCPM2TextBindingError(f"{label} path must be non-empty.")
    try:
        path = (project_root / value).resolve(strict=True)
        path.relative_to(project_root)
    except (OSError, ValueError) as error:
        raise DenisVoxCPM2TextBindingError(
            f"{label} must be an existing file beneath project root."
        ) from error
    return path


def _require_materialization(
    path: Path,
    *,
    selection_csv: Path,
    selection_receipt: Path,
    ready_manifest: Path,
) -> None:
    receipt = _json_object(path, "Denis materialization receipt")
    selection = receipt.get("selection")
    outputs = receipt.get("outputs")
    qa = receipt.get("technical_qa")
    target = receipt.get("target_outcome")
    claims = receipt.get("claims")
    if not all(isinstance(value, Mapping) for value in (selection, outputs, qa, target, claims)):
        raise DenisVoxCPM2TextBindingError("Denis materialization receipt structure is invalid.")
    selection = cast(Mapping[str, object], selection)
    outputs = cast(Mapping[str, object], outputs)
    qa = cast(Mapping[str, object], qa)
    target = cast(Mapping[str, object], target)
    claims = cast(Mapping[str, object], claims)
    csv_binding = selection.get("csv")
    receipt_binding = selection.get("receipt")
    ready_binding = outputs.get("ready_manifest")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("protocol_id") != _MATERIALIZATION_PROTOCOL_ID
        or not isinstance(csv_binding, Mapping)
        or csv_binding.get("path") != selection_csv.as_posix()
        or csv_binding.get("rows") != _TARGET_ROWS
        or _sha256(csv_binding.get("sha256"), "Selection CSV SHA-256") != sha256_file(selection_csv)
        or not isinstance(receipt_binding, Mapping)
        or receipt_binding.get("path") != selection_receipt.as_posix()
        or _sha256(receipt_binding.get("sha256"), "Selection receipt SHA-256")
        != sha256_file(selection_receipt)
        or selection.get("target_pairs") != _TARGET_ROWS
        or selection.get("post_selection_replacement_or_backfill") is not False
        or not isinstance(ready_binding, Mapping)
        or ready_binding.get("path") != ready_manifest.as_posix()
        or ready_binding.get("rows") != _READY_ROWS
        or _sha256(ready_binding.get("sha256"), "Ready manifest SHA-256")
        != sha256_file(ready_manifest)
        or qa.get("raw_rows") != _TARGET_ROWS
        or qa.get("ready_rows") != _READY_ROWS
        or qa.get("rejected_rows") != 15
        or qa.get("reused_rows") != 0
        or qa.get("replacement_or_backfill") is not False
        or target.get("target_ready_pairs") != _TARGET_ROWS
        or target.get("minimum_ready_pairs") != 60
        or target.get("actual_ready_pairs") != _READY_ROWS
        or target.get("status") != "minimum_60_met_but_target_not_met"
        or claims.get("future_synthesis_must_use_only_ready_frozen_texts") is not True
        or claims.get("synthetic_audio_generated") is not False
        or claims.get("training_data_overlap_unverified") is not True
        or claims.get("single_speaker") is not True
        or claims.get("speaker_independent") is not False
        or claims.get("detector_inference_performed") is not False
        or claims.get("detector_inference_authorized") is not False
    ):
        raise DenisVoxCPM2TextBindingError(
            "Denis materialization receipt does not authorize exact ready-text binding."
        )


def _validate_ready_rows(rows: Sequence[ManifestRow]) -> None:
    if (
        len(rows) != _READY_ROWS
        or len({row.sample_id for row in rows}) != _READY_ROWS
        or len({row.text_id for row in rows}) != _READY_ROWS
        or len({row.text_hash for row in rows}) != _READY_ROWS
        or any(
            row.split != "ood"
            or row.label != "bonafide"
            or row.language != "ru"
            or row.source_name != DENIS_SOURCE_ID
            for row in rows
        )
    ):
        raise DenisVoxCPM2TextBindingError(
            "Binding base must be exactly 64 unique ready Denis bona-fide rows."
        )


def _require_model_gate(project_root: Path, model_lock: Path) -> Mapping[str, object]:
    lock = load_research_tts_model_lock(model_lock)
    if (
        lock.protocol_id != "voxcpm2-official-text-only-v1"
        or len(lock.models) != 1
        or lock.models[0].model_id != _MODEL_ID
    ):
        raise DenisVoxCPM2TextBindingError("Model lock is not the one official VoxCPM2 route.")
    runtime = lock.models[0].runtime
    exact_values = {
        "source_revision": VOXCPM2_SOURCE_REVISION,
        "device": "cuda",
        "sample_rate": 48_000,
        "local_files_only": True,
        "text_input_only": True,
        "default_voice_identity": "unknown_not_claimed",
        "reference_audio_policy": "forbidden_null_only",
        "prompt_audio_policy": "forbidden_null_only",
        "prompt_text_policy": "forbidden_null_only",
        "voice_cloning": False,
        "lora_weights_policy": "forbidden_null_only",
        "load_denoiser": False,
        "normalize": False,
        "denoise": False,
        "retry_badcase": False,
        "retry_badcase_max_times": 1,
        "streaming": False,
        "seed": VOXCPM2_FIXED_SEED,
        "cfg_value": 2.0,
        "inference_timesteps": 10,
        "min_len": 2,
        "max_len": 4096,
        "runtime_environment_materialized": True,
        "cuda_load_verified": True,
        "text_only_smoke_verified": True,
        "cuda_smoke_rerun_policy": "forbidden",
    }
    if any(runtime.get(name) != value for name, value in exact_values.items()):
        raise DenisVoxCPM2TextBindingError("VoxCPM2 runtime contract changed from its gate.")
    for path_field, hash_field in (
        ("artifact_source_receipt", "artifact_source_receipt_sha256"),
        ("project_history_receipt", "project_history_receipt_sha256"),
        ("pre_inference_failure_receipt", "pre_inference_failure_receipt_sha256"),
        ("cuda_smoke_receipt", "cuda_smoke_receipt_sha256"),
    ):
        path = _project_file(project_root, runtime.get(path_field), path_field)
        if sha256_file(path) != _sha256(runtime.get(hash_field), hash_field):
            raise DenisVoxCPM2TextBindingError(f"Pinned gate receipt changed: {path_field}.")
    wrapper = _project_file(project_root, runtime.get("wrapper_module"), "wrapper module")
    if sha256_file(wrapper) != _sha256(runtime.get("wrapper_sha256"), "wrapper SHA-256"):
        raise DenisVoxCPM2TextBindingError("Frozen VoxCPM2 wrapper changed.")
    smoke = _json_object(
        _project_file(project_root, runtime.get("cuda_smoke_receipt"), "CUDA smoke receipt"),
        "CUDA smoke receipt",
    )
    smoke_claims = smoke.get("claims")
    network = smoke.get("network_policy")
    if (
        smoke.get("protocol_id") != "voxcpm2-official-text-only-cuda-smoke-v1"
        or not isinstance(smoke_claims, Mapping)
        or smoke_claims.get("text_only_default_voice_smoke_verified") is not True
        or smoke_claims.get("reference_or_prompt_audio_used") is not False
        or smoke_claims.get("semantic_normalizer_used") is not False
        or smoke_claims.get("denoiser_used") is not False
        or smoke_claims.get("retry_or_resynthesis_used") is not False
        or smoke_claims.get("candidate_text_used") is not False
        or smoke_claims.get("detector_inference_performed") is not False
        or not isinstance(network, Mapping)
        or network.get("observed_upstream_network_attempts") != 0
    ):
        raise DenisVoxCPM2TextBindingError("VoxCPM2 CUDA smoke receipt is not admissible.")
    return runtime


def _read_texts(archive_path: Path, selected: Sequence[FrozenDenisSelectionRow]) -> dict[str, str]:
    wanted = {f"{row.member_stem}.txt": row for row in selected}
    texts: dict[str, str] = {}
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            for member in archive:
                _safe_member(member.name)
                row = wanted.get(member.name)
                if row is None:
                    continue
                handle = archive.extractfile(member)
                if not member.isfile() or handle is None:
                    raise DenisVoxCPM2TextBindingError(
                        f"Selected Denis transcript is not a regular file: {member.name}."
                    )
                try:
                    text = handle.read().decode("utf-8")
                except UnicodeDecodeError as error:
                    raise DenisVoxCPM2TextBindingError(
                        f"Selected Denis transcript is not UTF-8: {member.name}."
                    ) from error
                texts[row.sample_id] = text
    except (OSError, tarfile.TarError) as error:
        raise DenisVoxCPM2TextBindingError("Cannot read the pinned Denis archive.") from error
    if set(texts) != {row.sample_id for row in selected}:
        raise DenisVoxCPM2TextBindingError("Pinned Denis archive lacks a selected transcript.")
    return texts


def _binding_digest(rows: Sequence[Mapping[str, object]]) -> str:
    fields = (
        "selection_rank",
        "sample_id",
        "member_stem",
        "text_id",
        "literal_text_sha256",
        "collapse_whitespace_text_sha256",
        "nfkc_collapse_whitespace_text_sha256",
        "literal_text_utf8_bytes",
        "collapse_whitespace_text_utf8_bytes",
        "ready_audio_sha256",
    )
    material = "\n".join("\t".join(str(row[field]) for field in fields) for row in rows).encode(
        "utf-8"
    )
    return hashlib.sha256(material).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--license-ledger", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--selection-csv", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--materialization-receipt", type=Path, required=True)
    parser.add_argument("--ready-manifest", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--synthesis-runner", type=Path, required=True)
    parser.add_argument("--preprocess-runner", type=Path, required=True)
    parser.add_argument("--technical-qa-publisher", type=Path, required=True)
    parser.add_argument("--raw-directory", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path, required=True)
    parser.add_argument("--synthesis-receipt", type=Path, required=True)
    parser.add_argument("--ready-spoof-manifest", type=Path, required=True)
    parser.add_argument("--rejection-report", type=Path, required=True)
    parser.add_argument("--technical-qa-receipt", type=Path, required=True)
    parser.add_argument("--bound-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        datetime.fromisoformat(arguments.bound_at.replace("Z", "+00:00"))
        if arguments.output.exists() or not arguments.output.parent.is_dir():
            raise DenisVoxCPM2TextBindingError(
                "Text-binding receipt must be new with an existing parent."
            )
        project_root = arguments.project_root.resolve(strict=True)
        data_root = arguments.data_root.resolve(strict=True)
        for path in (
            arguments.selection_csv,
            arguments.selection_receipt,
            arguments.materialization_receipt,
            arguments.ready_manifest,
            arguments.model_lock,
            arguments.synthesis_runner,
            arguments.preprocess_runner,
            arguments.technical_qa_publisher,
        ):
            resolved = path.resolve(strict=True)
            resolved.relative_to(project_root)
        output_paths = (
            arguments.raw_manifest,
            arguments.synthesis_receipt,
            arguments.ready_spoof_manifest,
            arguments.rejection_report,
            arguments.technical_qa_receipt,
        )
        if (
            len(set(output_paths)) != len(output_paths)
            or arguments.raw_directory.exists()
            or any(path.exists() or not path.parent.is_dir() for path in output_paths)
        ):
            raise DenisVoxCPM2TextBindingError(
                "All future synthesis/QA outputs must be distinct, absent and parented."
            )
        arguments.raw_directory.resolve().relative_to(data_root)
        _require_materialization(
            arguments.materialization_receipt,
            selection_csv=arguments.selection_csv,
            selection_receipt=arguments.selection_receipt,
            ready_manifest=arguments.ready_manifest,
        )
        load_frozen_denis_selection, bind_denis_selection = _selection_helpers()
        selection = load_frozen_denis_selection(
            arguments.selection_csv, arguments.selection_receipt, project_root
        )
        inspection = inspect_denis_archive(arguments.archive.resolve(strict=True))
        bind_denis_selection(selection, inspection.records)
        ready_rows = tuple(load_manifest(arguments.ready_manifest))
        validate_manifest(ready_rows)
        _validate_ready_rows(ready_rows)
        ledger = load_license_ledger(arguments.license_ledger)
        validate_manifest_licenses(ready_rows, ledger)
        require_valid_assets(ready_rows, data_root)
        if DENIS_VOXCPM2_SOURCE_ID not in ledger:
            raise LicenseLedgerError(["Official VoxCPM2 source is absent from the ledger."])
        runtime = _require_model_gate(project_root, arguments.model_lock)
        ready_by_id = {row.sample_id: row for row in ready_rows}
        selected_ready = tuple(row for row in selection if row.sample_id in ready_by_id)
        if len(selected_ready) != _READY_ROWS:
            raise DenisVoxCPM2TextBindingError(
                "Ready manifest is not an exact subset of the frozen selection."
            )
        texts = _read_texts(arguments.archive, selected_ready)
        bound_rows: list[dict[str, object]] = []
        for row in selected_ready:
            ready = ready_by_id[row.sample_id]
            text = texts[row.sample_id]
            binding = bind_text(text)
            canonical = collapse_whitespace(text)
            nfkc = unicodedata.normalize("NFKC", canonical)
            if (
                ready.text_id != row.text_id
                or ready.text_hash != row.whitespace_canonical_text_sha256
                or binding.literal_sha256 != row.literal_text_sha256
                or binding.collapse_whitespace_sha256 != row.whitespace_canonical_text_sha256
                or hashlib.sha256(nfkc.encode("utf-8")).hexdigest()
                != row.nfkc_whitespace_canonical_text_sha256
                or not 1 <= len(canonical.encode("utf-8")) <= 4096
            ):
                raise DenisVoxCPM2TextBindingError(
                    f"Denis literal/canonical text binding changed for {row.sample_id!r}."
                )
            bound_rows.append(
                {
                    "selection_rank": row.selection_rank,
                    "sample_id": row.sample_id,
                    "member_stem": row.member_stem,
                    "category": row.category,
                    "text_id": row.text_id,
                    "literal_text_sha256": binding.literal_sha256,
                    "collapse_whitespace_text_sha256": binding.collapse_whitespace_sha256,
                    "nfkc_collapse_whitespace_text_sha256": (
                        row.nfkc_whitespace_canonical_text_sha256
                    ),
                    "literal_text_utf8_bytes": len(text.encode("utf-8")),
                    "collapse_whitespace_text_utf8_bytes": len(canonical.encode("utf-8")),
                    "ready_audio_sha256": ready.sha256,
                }
            )
        receipt = {
            "schema_version": 1,
            "protocol_id": DENIS_VOXCPM2_TEXT_BINDING_PROTOCOL_ID,
            "bound_at": arguments.bound_at,
            "candidate_state": (
                "immutable 64-row Denis ready-text binding and one-attempt VoxCPM2 contract; "
                "no candidate synthesis, acoustic review, pairing or detector inference"
            ),
            "inputs": {
                "selection_csv": {
                    "path": arguments.selection_csv.as_posix(),
                    "sha256": sha256_file(arguments.selection_csv),
                    "rows": _TARGET_ROWS,
                },
                "selection_receipt": {
                    "path": arguments.selection_receipt.as_posix(),
                    "sha256": sha256_file(arguments.selection_receipt),
                },
                "materialization_receipt": {
                    "path": arguments.materialization_receipt.as_posix(),
                    "sha256": sha256_file(arguments.materialization_receipt),
                },
                "ready_manifest": {
                    "path": arguments.ready_manifest.as_posix(),
                    "sha256": sha256_file(arguments.ready_manifest),
                    "rows": _READY_ROWS,
                },
                "denis_archive": {
                    "path": str(arguments.archive.resolve(strict=True)),
                    "expected_size_bytes": DENIS_ARCHIVE_EXPECTED_SIZE_BYTES,
                    "expected_sha256": DENIS_ARCHIVE_EXPECTED_SHA256,
                    "identity_verified_before_text_read": True,
                },
                "model_lock": {
                    "path": arguments.model_lock.as_posix(),
                    "sha256": sha256_file(arguments.model_lock),
                    "model_id": _MODEL_ID,
                    "model_revision": VOXCPM2_MODEL_REVISION,
                    "source_revision": VOXCPM2_SOURCE_REVISION,
                },
                "model_gate_receipts": {
                    name: {
                        "path": runtime[name],
                        "sha256": runtime[f"{name}_sha256"],
                    }
                    for name in ("artifact_source_receipt", "project_history_receipt")
                },
                "cuda_smoke_receipt": {
                    "path": runtime["cuda_smoke_receipt"],
                    "sha256": runtime["cuda_smoke_receipt_sha256"],
                    "rerun_policy": "forbidden",
                },
            },
            "frozen_programs": {
                "binding_runner": {
                    "path": Path(__file__).as_posix(),
                    "sha256": sha256_file(Path(__file__)),
                },
                "text_only_wrapper": {
                    "path": runtime["wrapper_module"],
                    "sha256": runtime["wrapper_sha256"],
                },
                "synthesis_runner": {
                    "path": arguments.synthesis_runner.as_posix(),
                    "sha256": sha256_file(arguments.synthesis_runner),
                },
                "preprocess_runner": {
                    "path": arguments.preprocess_runner.as_posix(),
                    "sha256": sha256_file(arguments.preprocess_runner),
                },
                "technical_qa_publisher": {
                    "path": arguments.technical_qa_publisher.as_posix(),
                    "sha256": sha256_file(arguments.technical_qa_publisher),
                },
            },
            "synthesis_contract": {
                "bound_rows": _READY_ROWS,
                "model_loads": 1,
                "attempts_per_bound_text": 1,
                "total_generation_calls_required": _READY_ROWS,
                "fixed_seed_per_call": VOXCPM2_FIXED_SEED,
                "pass_collapse_whitespace_text_only": True,
                "external_text_normalizer_or_stress_model": "forbidden",
                "reference_audio": None,
                "prompt_audio": None,
                "prompt_text": None,
                "voice_cloning": False,
                "lora": None,
                "load_denoiser": False,
                "normalize": False,
                "denoise": False,
                "retry_badcase": False,
                "resynthesis_after_failure": "forbidden",
                "replacement_reselection_or_backfill": "forbidden",
                "network": "bwrap --unshare-net plus offline environment and socket guard",
                "detector_or_metric_access_during_generation": "forbidden",
                "output_profile": {
                    "sample_rate_hz": 48_000,
                    "channels": 1,
                    "container": "WAV",
                    "subtype": "PCM_16",
                    "voice_id": DENIS_VOXCPM2_VOICE_ID,
                    "voice_identity_claim": "unknown_not_claimed",
                },
            },
            "output_contract": {
                "raw_directory": arguments.raw_directory.as_posix(),
                "raw_manifest": arguments.raw_manifest.as_posix(),
                "synthesis_receipt": arguments.synthesis_receipt.as_posix(),
                "ready_spoof_manifest": arguments.ready_spoof_manifest.as_posix(),
                "technical_qa_rejection_report": arguments.rejection_report.as_posix(),
                "technical_qa_receipt": arguments.technical_qa_receipt.as_posix(),
            },
            "category_counts": dict(
                sorted(Counter(str(row["category"]) for row in bound_rows).items())
            ),
            "text_binding_sha256": _binding_digest(bound_rows),
            "rows": bound_rows,
            "claims": {
                "text_binding_frozen": True,
                "ready_rows_only": True,
                "plaintext_transcripts_persisted_in_receipt": False,
                "synthetic_audio_generated": False,
                "acoustic_review_performed": False,
                "pairing_performed": False,
                "detector_inference_performed": False,
                "detector_inference_authorized": False,
                "external_source_and_generator_family_holdout": True,
                "training_data_overlap_unverified": True,
                "single_bonafide_speaker": True,
                "speaker_independent": False,
                "speaker_robust": False,
            },
        }
        with arguments.output.open("x", encoding="utf-8") as output:
            json.dump(receipt, output, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
    except (
        LicenseLedgerError,
        ManifestError,
        OSError,
        ResearchTtsError,
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
                "bound_rows": len(bound_rows),
                "text_binding_sha256": receipt["text_binding_sha256"],
                "receipt": str(arguments.output),
                "receipt_sha256": sha256_file(arguments.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
