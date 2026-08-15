"""Fail-closed metadata selection for the future XLS-R+SLS v4 final inputs.

This gate is deliberately before extraction and synthesis.  It re-audits the
current project history, reserves fresh Common Voice RU and FLEURS KK source
identities, and verifies only the *metadata* constraints of the fixed Qwen and
KazakhTTS routes.  It never reads an audio payload, starts a TTS runtime,
loads the detector checkpoint, or produces logits.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import cast

from kds.data.assets import sha256_file
from kds.data.common_voice import (
    COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SHA256,
    COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SIZE_BYTES,
    CommonVoiceRecord,
    load_common_voice_metadata_from_archive,
)
from kds.data.fleurs import FLEURS_REVISION, FleursRecord, inspect_fleurs_release
from kds.data.kazakhtts_text import KazakhTtsTextError, normalize_kazakhtts_stage_c_text
from kds.data.licenses import APPROVED_LICENSE_STATUSES, LicenseLedgerEntry, load_license_ledger
from kds.data.manifest import REQUIRED_FIELDS, ManifestRow, load_manifest, validate_manifest
from kds.data.qwen3_tts_customvoice import QWEN3_TTS_CUSTOMVOICE_MODEL_ID
from kds.data.research_tts import ResearchTtsModel, load_research_tts_model_lock
from kds.eval.candidate_exposure import CandidateExposureError, configured_role_scope
from kds.eval.common_voice_metadata_screen import (
    CommonVoiceMetadataIdentity,
    common_voice_metadata_identity,
    screen_common_voice_ru_test_metadata,
)

V4_FINAL_INPUT_SCHEMA_VERSION = 1
V4_FINAL_INPUT_PROTOCOL_ID = "xlsr-sls-model-v4-final-inputs-v1"
V4_FINAL_SELECTION_FIELDS = (
    "language",
    "selection_rank",
    "sample_id",
    "source_member",
    "source_split",
    "parent_group_id",
    "speaker_pseudo_id",
    "text_id",
    "text_hash",
    "synthesis_text_sha256",
    "synthesis_seed",
    "normalization_operations",
)

_HEX = frozenset("0123456789abcdef")
_FINAL_SOURCE_IDS = (
    "common_voice_ru_v24_v4_final",
    "google_fleurs_kk_v1_v4_final",
    "qwen3_tts_customvoice_aiden_v4_final",
    "issai_kazakhtts2_male2_tacotron2_pwg_v4_final",
)


class V4FinalInputError(ValueError):
    """Raised when a final-input metadata contract cannot prove isolation."""


@dataclass(frozen=True, slots=True)
class FileBinding:
    """One project-relative immutable input and optional exact CSV row count."""

    path: str
    sha256: str
    rows: int | None


@dataclass(frozen=True, slots=True)
class SourceSelection:
    """A deterministic metadata-only source selection policy."""

    seed: str
    target_pairs: int


@dataclass(frozen=True, slots=True)
class V4FinalInputPlan:
    """The strict pre-materialization contract for the four-cell final."""

    path: str
    sha256: str
    created_at: str
    inputs: Mapping[str, FileBinding]
    ru: SourceSelection
    kk: SourceSelection
    config_root: str
    manifest_root: str
    output_selection: str
    output_receipt: str


@dataclass(frozen=True, slots=True)
class CandidateIdentity:
    """Source metadata sufficient for deterministic selection without audio bytes."""

    language: str
    sample_id: str
    source_member: str
    source_split: str
    parent_group_id: str
    speaker_pseudo_id: str
    text_id: str
    text_hash: str
    synthesis_text_sha256: str
    synthesis_seed: str
    normalization_operations: str


@dataclass(frozen=True, slots=True)
class FinalSelectionResult:
    """The selected metadata rows and the evidence needed for the immutable receipt."""

    ru: tuple[CandidateIdentity, ...]
    kk: tuple[CandidateIdentity, ...]
    common_voice_screen: Mapping[str, object]
    fleurs_audit: Mapping[str, object]
    configured_scope: Mapping[str, object]


def _json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V4FinalInputError(f"Cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise V4FinalInputError(f"{label} must be a JSON object.")
    return cast(dict[str, object], value)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise V4FinalInputError(f"{label} must be a JSON object.")
    return cast(dict[str, object], value)


def _safe_project_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise V4FinalInputError(f"{label} must be a non-empty project-relative path.")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or ".." in parsed.parts or "\\" in value or value == ".":
        raise V4FinalInputError(f"{label} is not a safe project-relative path.")
    return parsed.as_posix()


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise V4FinalInputError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise V4FinalInputError(f"{label} must be a positive integer.")
    return value


def _timestamp(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise V4FinalInputError(f"{label} must be a non-empty ISO-8601 timestamp.")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise V4FinalInputError(f"{label} must be an ISO-8601 timestamp.") from error
    return value


def _binding(value: object, label: str) -> FileBinding:
    raw = _mapping(value, label)
    if set(raw) != {"path", "sha256", "rows"}:
        raise V4FinalInputError(f"{label} must contain exactly path, sha256 and rows.")
    rows = raw["rows"]
    if rows is not None:
        _positive_int(rows, f"{label}.rows")
    return FileBinding(
        path=_safe_project_path(raw["path"], f"{label}.path"),
        sha256=_sha256(raw["sha256"], f"{label}.sha256"),
        rows=cast(int | None, rows),
    )


def _source_selection(value: object, label: str) -> SourceSelection:
    raw = _mapping(value, label)
    if set(raw) != {"seed", "target_pairs"}:
        raise V4FinalInputError(f"{label} must contain exactly seed and target_pairs.")
    seed = raw["seed"]
    if not isinstance(seed, str) or not seed or "\x00" in seed:
        raise V4FinalInputError(f"{label}.seed must be non-empty and contain no NUL.")
    return SourceSelection(
        seed=seed,
        target_pairs=_positive_int(raw["target_pairs"], f"{label}.target_pairs"),
    )


def _project_path(project_root: Path, relative_path: str, label: str) -> Path:
    candidate = (project_root / relative_path).resolve()
    try:
        candidate.relative_to(project_root)
    except ValueError as error:
        raise V4FinalInputError(f"{label} resolves outside the project root.") from error
    return candidate


def _relative_to_root(path: Path, project_root: Path, label: str) -> str:
    try:
        return path.resolve(strict=True).relative_to(project_root).as_posix()
    except (OSError, ValueError) as error:
        raise V4FinalInputError(f"{label} escapes the project root: {path}") from error


def _csv_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise V4FinalInputError(f"Cannot count CSV rows in {path}.") from error


def _verify_binding(binding: FileBinding, project_root: Path, label: str) -> Path:
    path = _project_path(project_root, binding.path, label)
    if not path.is_file():
        raise V4FinalInputError(f"{label} is missing: {binding.path}")
    if sha256_file(path) != binding.sha256:
        raise V4FinalInputError(f"{label} SHA-256 mismatch: {binding.path}")
    if binding.rows is not None and _csv_rows(path) != binding.rows:
        raise V4FinalInputError(f"{label} row count mismatch: {binding.path}")
    return path


def load_v4_final_input_plan(path: Path, project_root: Path) -> V4FinalInputPlan:
    """Load a strictly shaped, hash-pinned final-input metadata contract."""

    root = project_root.resolve(strict=True)
    plan_path = path.resolve(strict=True)
    relative_plan = _relative_to_root(plan_path, root, "Final-input plan")
    raw = _json_object(plan_path, "v4 final-input plan")
    expected_top = {
        "schema_version",
        "protocol_id",
        "created_at",
        "inputs",
        "sources",
        "selection",
        "scope",
        "outputs",
        "prohibitions",
    }
    if set(raw) != expected_top:
        raise V4FinalInputError("v4 final-input plan has unexpected keys.")
    if raw["schema_version"] != V4_FINAL_INPUT_SCHEMA_VERSION:
        raise V4FinalInputError("Unsupported v4 final-input plan schema.")
    if raw["protocol_id"] != V4_FINAL_INPUT_PROTOCOL_ID:
        raise V4FinalInputError("v4 final-input protocol_id mismatch.")

    input_values = _mapping(raw["inputs"], "inputs")
    required_inputs = {
        "roles_and_selection",
        "training_receipt",
        "train_manifest",
        "dev_manifest",
        "calibration_report",
        "calibration_pair_lock",
        "license_ledger",
        "final_license_ledger",
        "final_readiness_audit",
        "fleurs_artifact_lock",
        "qwen_model_lock",
        "kazakhtts_model_lock",
        "final_inputs_module",
        "common_voice_module",
        "common_voice_screen_module",
        "fleurs_module",
        "kazakhtts_text_module",
        "research_tts_module",
        "runner_script",
    }
    if set(input_values) != required_inputs:
        raise V4FinalInputError("v4 final-input plan inputs are incomplete.")
    inputs = {name: _binding(input_values[name], f"inputs.{name}") for name in sorted(input_values)}

    sources = _mapping(raw["sources"], "sources")
    expected_sources = {"common_voice_ru_v24", "google_fleurs_kk_v1", "qwen", "kazakhtts"}
    if set(sources) != expected_sources:
        raise V4FinalInputError("v4 final-input plan sources are incomplete.")
    common_voice = _mapping(sources["common_voice_ru_v24"], "sources.common_voice_ru_v24")
    if set(common_voice) != {"archive_name", "size_bytes", "sha256", "source_split"} or (
        common_voice["archive_name"] != "cv-corpus-24.0-2025-12-05-ru.tar.gz"
        or common_voice["size_bytes"] != COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SIZE_BYTES
        or common_voice["sha256"] != COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SHA256
        or common_voice["source_split"] != "test"
    ):
        raise V4FinalInputError("Common Voice source binding is invalid.")
    fleurs = _mapping(sources["google_fleurs_kk_v1"], "sources.google_fleurs_kk_v1")
    if set(fleurs) != {"locale", "revision", "source_split"} or (
        fleurs["locale"] != "kk_kz"
        or fleurs["revision"] != FLEURS_REVISION
        or fleurs["source_split"] != "train"
    ):
        raise V4FinalInputError("FLEURS source binding is invalid.")
    for source_name, expected_model_id in (
        ("qwen", QWEN3_TTS_CUSTOMVOICE_MODEL_ID),
        ("kazakhtts", "issai_kazakhtts2_male2_tacotron2_pwg"),
    ):
        route = _mapping(sources[source_name], f"sources.{source_name}")
        if set(route) != {"model_id", "text_input_only", "reference_audio", "voice_cloning"} or (
            route["model_id"] != expected_model_id
            or route["text_input_only"] is not True
            or route["reference_audio"] != "forbidden"
            or route["voice_cloning"] is not False
        ):
            raise V4FinalInputError(f"{source_name} route binding is invalid.")

    selection = _mapping(raw["selection"], "selection")
    if set(selection) != {"ru", "kk", "one_record_per_text_group", "post_selection_backfill"}:
        raise V4FinalInputError("v4 final-input selection policy is invalid.")
    if (
        selection["one_record_per_text_group"] is not True
        or selection["post_selection_backfill"] is not False
    ):
        raise V4FinalInputError("Final selection must be one-record-per-text with no backfill.")
    ru = _source_selection(selection["ru"], "selection.ru")
    kk = _source_selection(selection["kk"], "selection.kk")
    if ru.target_pairs != 500 or kk.target_pairs != 500:
        raise V4FinalInputError("v4 final-input contract must retain 500 pairs per language.")

    scope = _mapping(raw["scope"], "scope")
    if set(scope) != {"config_root", "manifest_root"}:
        raise V4FinalInputError("Final-input scope is invalid.")
    config_root = _safe_project_path(scope["config_root"], "scope.config_root")
    manifest_root = _safe_project_path(scope["manifest_root"], "scope.manifest_root")
    if not _project_path(root, config_root, "scope.config_root").is_dir():
        raise V4FinalInputError("scope.config_root is not an existing directory.")
    if not _project_path(root, manifest_root, "scope.manifest_root").is_dir():
        raise V4FinalInputError("scope.manifest_root is not an existing directory.")

    outputs = _mapping(raw["outputs"], "outputs")
    if set(outputs) != {"metadata_selection", "receipt"}:
        raise V4FinalInputError("v4 final-input outputs are invalid.")
    output_selection = _safe_project_path(
        outputs["metadata_selection"], "outputs.metadata_selection"
    )
    output_receipt = _safe_project_path(outputs["receipt"], "outputs.receipt")
    if output_selection == output_receipt:
        raise V4FinalInputError("v4 final-input outputs must be distinct.")

    prohibitions = _mapping(raw["prohibitions"], "prohibitions")
    required_prohibitions = {
        "raw_audio_extraction",
        "synthetic_audio_generation",
        "audio_qa",
        "acoustic_review",
        "pairing",
        "checkpoint_loading",
        "calibration",
        "temperature_fitting",
        "final_inference",
        "detector_inference",
        "output_overwrite",
        "network_downloads",
    }
    if set(prohibitions) != required_prohibitions or any(
        value is not True for value in prohibitions.values()
    ):
        raise V4FinalInputError("v4 final-input prohibitions must all be true.")

    return V4FinalInputPlan(
        path=relative_plan,
        sha256=sha256_file(plan_path),
        created_at=_timestamp(raw["created_at"], "created_at"),
        inputs=inputs,
        ru=ru,
        kk=kk,
        config_root=config_root,
        manifest_root=manifest_root,
        output_selection=output_selection,
        output_receipt=output_receipt,
    )


def _manifest_like_csv(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            fields = next(csv.reader(handle), [])
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise V4FinalInputError(
            f"Cannot inspect manifest inventory file {path}: {error}"
        ) from error
    return REQUIRED_FIELDS.issubset(fields)


def _manifest_inventory(
    *, project_root: Path, manifest_root: Path
) -> tuple[list[ManifestRow], list[dict[str, object]]]:
    rows: list[ManifestRow] = []
    bindings: list[dict[str, object]] = []
    for path in sorted(manifest_root.rglob("*.csv")):
        if not _manifest_like_csv(path):
            continue
        loaded = load_manifest(path)
        validate_manifest(loaded)
        rows.extend(loaded)
        bindings.append(
            {
                "path": _relative_to_root(path, project_root, "Manifest inventory"),
                "sha256": sha256_file(path),
                "rows": len(loaded),
            }
        )
    if not bindings:
        raise V4FinalInputError("Final-input audit found no valid manifest inventory.")
    return rows, bindings


def _selection_rank(seed: str, language: str, sample_id: str) -> bytes:
    return hashlib.sha256(f"{seed}\x00{language}\x00{sample_id}".encode()).digest()


def select_distinct_metadata_candidates(
    candidates: Iterable[CandidateIdentity], *, seed: str, target: int
) -> tuple[CandidateIdentity, ...]:
    """Select fresh records by public metadata with unique source groups and texts."""

    materialized = tuple(candidates)
    if not materialized:
        raise V4FinalInputError("Final metadata selection has no candidates.")
    sample_ids = [candidate.sample_id for candidate in materialized]
    if len(sample_ids) != len(set(sample_ids)):
        raise V4FinalInputError("Final metadata selection has duplicate sample IDs.")
    selected: list[CandidateIdentity] = []
    groups: set[str] = set()
    texts: set[str] = set()
    for candidate in sorted(
        materialized,
        key=lambda item: (_selection_rank(seed, item.language, item.sample_id), item.sample_id),
    ):
        if candidate.parent_group_id in groups or candidate.text_hash in texts:
            continue
        selected.append(candidate)
        groups.add(candidate.parent_group_id)
        texts.add(candidate.text_hash)
        if len(selected) == target:
            break
    if len(selected) != target:
        raise V4FinalInputError(
            "Final metadata selection has insufficient disjoint source/text capacity: "
            f"required {target}, selected {len(selected)}."
        )
    return tuple(selected)


def _qwen_synthesis_seed(source_text: str) -> str:
    return str(int.from_bytes(hashlib.sha256(source_text.encode("utf-8")).digest()[:4], "big"))


def _ru_candidates(
    records: Sequence[CommonVoiceRecord],
    surviving: Sequence[CommonVoiceMetadataIdentity],
) -> tuple[CandidateIdentity, ...]:
    allowed = {item.sample_id for item in surviving}
    candidates: list[CandidateIdentity] = []
    for record in records:
        identity = common_voice_metadata_identity(record)
        if identity.sample_id not in allowed:
            continue
        if (
            not record.sentence
            or "\x00" in record.sentence
            or len(record.sentence.encode("utf-8")) > 4096
        ):
            continue
        candidates.append(
            CandidateIdentity(
                language="ru",
                sample_id=identity.sample_id,
                source_member=record.clip_name,
                source_split=record.split,
                parent_group_id=identity.parent_group_id,
                speaker_pseudo_id=identity.speaker_pseudo_id,
                text_id=record.sentence_id,
                text_hash=identity.text_hash,
                synthesis_text_sha256=identity.text_hash,
                synthesis_seed=_qwen_synthesis_seed(record.sentence),
                normalization_operations="literal_utf8_source_text",
            )
        )
    return tuple(candidates)


def _fleurs_history_values(rows: Sequence[ManifestRow]) -> dict[str, set[str]]:
    return {
        "sample_id": {row.sample_id for row in rows},
        "text_hash": {row.text_hash for row in rows},
        "parent_group_id": {row.parent_group_id for row in rows},
    }


def _kk_candidates(
    records: Sequence[FleursRecord], history: Sequence[ManifestRow]
) -> tuple[tuple[CandidateIdentity, ...], Mapping[str, object]]:
    values = _fleurs_history_values(history)
    candidates: list[CandidateIdentity] = []
    overlap_counts: Counter[str] = Counter()
    normalizer_rejections: Counter[str] = Counter()
    for record in records:
        sample_id = f"google_fleurs_kk_v1:{record.filename.removesuffix('.wav')}"
        parent_group_id = f"google_fleurs_kk_v1:prompt:{record.prompt_id}"
        overlap_fields = [
            field
            for field, value in (
                ("sample_id", sample_id),
                ("text_hash", record.text_hash),
                ("parent_group_id", parent_group_id),
            )
            if value in values[field]
        ]
        if overlap_fields:
            overlap_counts.update(overlap_fields)
            continue
        try:
            normalized = normalize_kazakhtts_stage_c_text(record.transcript, "kk")
        except KazakhTtsTextError as error:
            normalizer_rejections[str(error)] += 1
            continue
        candidates.append(
            CandidateIdentity(
                language="kk",
                sample_id=sample_id,
                source_member=record.filename,
                source_split=record.source_split,
                parent_group_id=parent_group_id,
                speaker_pseudo_id="google_fleurs_kk_v1:unknown",
                text_id=parent_group_id,
                text_hash=record.text_hash,
                synthesis_text_sha256=normalized.normalized_sha256,
                synthesis_seed="",
                normalization_operations=";".join(normalized.operations),
            )
        )
    return tuple(candidates), {
        "speaker_group_policy": (
            "FLEURS has no published speaker IDs; its shared 'unknown' placeholder is not an "
            "identity and is never used as an exclusion or uniqueness key. prompt_id/text group "
            "is the sole FLEURS grouping key; speaker independence is not claimed."
        ),
        "historical_overlap_counts": {field: overlap_counts[field] for field in sorted(values)},
        "normalizer_rejections": {
            "records": sum(normalizer_rejections.values()),
            "reason_counts": dict(sorted(normalizer_rejections.items())),
        },
        "eligible_records": len(candidates),
        "eligible_text_groups": len({candidate.text_hash for candidate in candidates}),
    }


def _require_final_roles(path: Path) -> None:
    raw = _json_object(path, "v4 roles-and-selection contract")
    roles = _mapping(raw.get("roles"), "roles")
    final = _mapping(roles.get("final"), "roles.final")
    expected = {
        "ru_bonafide": "common_voice_ru_v24:fresh_client_groups:final_only",
        "ru_spoof": "qwen3_tts_customvoice_aiden:fresh_text_only:final_only",
        "kk_bonafide": "google_fleurs_kk_v1:fresh_groups:final_only",
        "kk_spoof": "kazakhtts2_tacotron2_pwg:fresh_text_only:final_only",
        "target_rows_per_cell": 500,
        "project_history_claim": "exact_assets_fresh_but_generator_families_previously_studied",
    }
    if final != expected:
        raise V4FinalInputError("v4 roles-and-selection final policy has changed.")


def _require_completed_calibration(path: Path) -> None:
    raw = _json_object(path, "v4 RU calibration report")
    if (
        raw.get("status") != "ok"
        or raw.get("calibration_scoring_performed") is not True
        or raw.get("temperature_fitted") is not True
        or raw.get("final_inference_performed") is not False
        or raw.get("threshold_selection_performed") is not False
    ):
        raise V4FinalInputError("v4 RU calibration report has an invalid final-input boundary.")


def _require_final_ledger(path: Path) -> dict[str, LicenseLedgerEntry]:
    ledger = load_license_ledger(path)
    if tuple(sorted(ledger)) != _FINAL_SOURCE_IDS:
        raise V4FinalInputError(
            "Final metadata ledger must contain exactly four final route entries."
        )
    for source_id in _FINAL_SOURCE_IDS:
        entry = ledger[source_id]
        if entry.status not in APPROVED_LICENSE_STATUSES:
            raise V4FinalInputError(f"Final metadata ledger entry is not approved: {source_id}")
        if entry.train_dev_test_use != "prohibited" or entry.ood_evaluation_use != "prohibited":
            raise V4FinalInputError(
                f"Final metadata ledger entry must prohibit materialization/evaluation: {source_id}"
            )
    return ledger


def _require_fixed_text_routes(qwen_lock_path: Path, kazakhtts_lock_path: Path) -> None:
    qwen_lock = load_research_tts_model_lock(qwen_lock_path)
    kazakhtts_lock = load_research_tts_model_lock(kazakhtts_lock_path)
    if len(qwen_lock.models) != 1 or len(kazakhtts_lock.models) != 1:
        raise V4FinalInputError("Each final TTS route must have exactly one locked model.")
    qwen = qwen_lock.models[0]
    kazakhtts = kazakhtts_lock.models[0]
    _require_qwen_route(qwen)
    _require_kazakhtts_route(kazakhtts)


def _require_qwen_route(model: ResearchTtsModel) -> None:
    runtime = model.runtime
    if (
        model.model_id != QWEN3_TTS_CUSTOMVOICE_MODEL_ID
        or model.generator_family != "qwen3_tts_customvoice_gguf"
        or runtime.get("target_language") != "ru"
        or runtime.get("text_input_only") is not True
        or runtime.get("reference_audio_policy") != "forbidden"
        or runtime.get("voice_cloning") is not False
        or runtime.get("voice_design") != "forbidden"
        or runtime.get("fixed_speaker_name") != "aiden"
    ):
        raise V4FinalInputError("Qwen route is not the pinned fixed text-only Aiden route.")


def _require_kazakhtts_route(model: ResearchTtsModel) -> None:
    runtime = model.runtime
    if (
        model.model_id != "issai_kazakhtts2_male2_tacotron2_pwg"
        or model.generator_family != "tacotron2_parallelwavegan_fixed_voice_tts"
        or runtime.get("supported_languages") != ["kk"]
        or runtime.get("reference_audio_policy") != "forbidden"
        or runtime.get("voice_cloning") is not False
        or runtime.get("fixed_voice_id") != "ISSAI_KazakhTTS2_M2"
    ):
        raise V4FinalInputError("KazakhTTS route is not the pinned fixed text-only KK route.")


def _selection_payload(candidate: CandidateIdentity, selection_rank: int) -> dict[str, object]:
    return {
        "language": candidate.language,
        "selection_rank": selection_rank,
        "sample_id": candidate.sample_id,
        "source_member": candidate.source_member,
        "source_split": candidate.source_split,
        "parent_group_id": candidate.parent_group_id,
        "speaker_pseudo_id": candidate.speaker_pseudo_id,
        "text_id": candidate.text_id,
        "text_hash": candidate.text_hash,
        "synthesis_text_sha256": candidate.synthesis_text_sha256,
        "synthesis_seed": candidate.synthesis_seed,
        "normalization_operations": candidate.normalization_operations,
    }


def _rows_fingerprint(rows: Sequence[CandidateIdentity]) -> str:
    payload = [
        _selection_payload(candidate, index)
        for index, candidate in enumerate(rows, start=1)
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def _write_selection(
    path: Path, ru: Sequence[CandidateIdentity], kk: Sequence[CandidateIdentity]
) -> None:
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=V4_FINAL_SELECTION_FIELDS, lineterminator="\n")
        writer.writeheader()
        for rows in (ru, kk):
            writer.writerows(
                _selection_payload(candidate, index)
                for index, candidate in enumerate(rows, start=1)
            )


def run_v4_final_input_selection(
    *,
    plan_path: Path,
    project_root: Path,
    common_voice_archive: Path,
    fleurs_release_root: Path,
) -> tuple[V4FinalInputPlan, FinalSelectionResult, str]:
    """Perform the one-time metadata selection and atomically publish its immutable receipt."""

    root = project_root.resolve(strict=True)
    plan = load_v4_final_input_plan(plan_path, root)
    verified = {
        name: _verify_binding(binding, root, f"inputs.{name}")
        for name, binding in plan.inputs.items()
    }
    _require_final_roles(verified["roles_and_selection"])
    _require_completed_calibration(verified["calibration_report"])
    _require_final_ledger(verified["final_license_ledger"])
    _require_fixed_text_routes(verified["qwen_model_lock"], verified["kazakhtts_model_lock"])

    selection_path = _project_path(root, plan.output_selection, "outputs.metadata_selection")
    receipt_path = _project_path(root, plan.output_receipt, "outputs.receipt")
    if (
        selection_path.exists()
        or receipt_path.exists()
        or not selection_path.parent.is_dir()
        or not receipt_path.parent.is_dir()
    ):
        raise V4FinalInputError(
            "Final-input outputs must be new and have existing parent directories."
        )

    records = tuple(load_common_voice_metadata_from_archive(common_voice_archive, ("test",)))
    try:
        common_voice_screen = screen_common_voice_ru_test_metadata(
            records=records,
            project_root=root,
            config_root=_project_path(root, plan.config_root, "scope.config_root"),
            manifest_root=_project_path(root, plan.manifest_root, "scope.manifest_root"),
            created_at=plan.created_at,
        )
    except CandidateExposureError as error:
        raise V4FinalInputError(str(error)) from error
    ru_pool = _ru_candidates(records, common_voice_screen.surviving)
    ru_selected = select_distinct_metadata_candidates(
        ru_pool, seed=plan.ru.seed, target=plan.ru.target_pairs
    )

    _fleurs_report, fleurs_rows = inspect_fleurs_release(fleurs_release_root, "kk_kz")
    configured_rows, config_bindings, configured_manifest_bindings = configured_role_scope(
        root, _project_path(root, plan.config_root, "scope.config_root")
    )
    inventory_rows, inventory_bindings = _manifest_inventory(
        project_root=root,
        manifest_root=_project_path(root, plan.manifest_root, "scope.manifest_root"),
    )
    kk_pool, fleurs_audit = _kk_candidates(
        fleurs_rows["train"], [*configured_rows, *inventory_rows]
    )
    kk_selected = select_distinct_metadata_candidates(
        kk_pool, seed=plan.kk.seed, target=plan.kk.target_pairs
    )

    result = FinalSelectionResult(
        ru=ru_selected,
        kk=kk_selected,
        common_voice_screen={
            **common_voice_screen.receipt,
            "archive": {
                "name": common_voice_archive.name,
                "expected_size_bytes": COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SIZE_BYTES,
                "expected_sha256": COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SHA256,
                "identity_verified_before_metadata_read": True,
            },
            "qwen_literal_text_eligible_records": len(ru_pool),
            "qwen_literal_text_eligible_text_groups": len({item.text_hash for item in ru_pool}),
        },
        fleurs_audit={
            **fleurs_audit,
            "locale": "kk_kz",
            "source_split": "train",
            "release_revision": FLEURS_REVISION,
            "source_records": len(fleurs_rows["train"]),
            "source_text_groups": len({item.text_hash for item in fleurs_rows["train"]}),
        },
        configured_scope={
            "configuration_files": config_bindings,
            "configured_manifests": configured_manifest_bindings,
            "configured_rows": len(configured_rows),
            "manifest_inventory": inventory_bindings,
            "inventory_rows": len(inventory_rows),
        },
    )

    stage = Path(tempfile.mkdtemp(prefix=".kds-v4-final-inputs-", dir=receipt_path.parent))
    try:
        staged_selection = stage / selection_path.name
        _write_selection(staged_selection, result.ru, result.kk)
        receipt = {
            "schema_version": 1,
            "protocol_id": V4_FINAL_INPUT_PROTOCOL_ID,
            "created_at": plan.created_at,
            "status": "ok",
            "plan": {"path": plan.path, "sha256": plan.sha256},
            "inputs": {
                name: {"path": binding.path, "sha256": binding.sha256, "rows": binding.rows}
                for name, binding in sorted(plan.inputs.items())
            },
            "selection": {
                "ru": {
                    "seed": plan.ru.seed,
                    "target_pairs": plan.ru.target_pairs,
                    "selected_pairs": len(result.ru),
                    "selected_client_groups": len({item.parent_group_id for item in result.ru}),
                    "selected_text_groups": len({item.text_hash for item in result.ru}),
                    "selected_rows_sha256": _rows_fingerprint(result.ru),
                },
                "kk": {
                    "seed": plan.kk.seed,
                    "target_pairs": plan.kk.target_pairs,
                    "selected_pairs": len(result.kk),
                    "selected_prompt_groups": len({item.parent_group_id for item in result.kk}),
                    "selected_text_groups": len({item.text_hash for item in result.kk}),
                    "selected_rows_sha256": _rows_fingerprint(result.kk),
                },
                "one_record_per_text_group": True,
                "post_selection_backfill": False,
                "selection_uses_audio_or_duration": False,
                "selection_uses_detector_or_model_output": False,
                "selection_uses_model_metrics_or_final_errors": False,
            },
            "current_history_scope": result.configured_scope,
            "common_voice_ru": result.common_voice_screen,
            "fleurs_kk": result.fleurs_audit,
            "outputs": {
                "metadata_selection": {
                    "path": plan.output_selection,
                    "sha256": sha256_file(staged_selection),
                    "rows": len(result.ru) + len(result.kk),
                }
            },
            "claims": {
                "raw_audio_extraction_performed": False,
                "synthetic_audio_generated": False,
                "audio_qa_performed": False,
                "acoustic_review_performed": False,
                "pairing_performed": False,
                "checkpoint_loaded": False,
                "calibration_performed": False,
                "temperature_fitted": False,
                "final_inference_performed": False,
                "detector_inference_performed": False,
                "detector_inference_authorized": False,
                "speaker_independence": "not_verified_speaker_independent",
                "kk_calibrated_probability_claim": False,
                "future_materialization_requires_separate_contract": True,
            },
        }
        staged_receipt = stage / receipt_path.name
        staged_receipt.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if selection_path.exists() or receipt_path.exists():
            raise V4FinalInputError("A final-input output appeared while staging.")
        staged_selection.replace(selection_path)
        staged_receipt.replace(receipt_path)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return plan, result, sha256_file(receipt_path)
