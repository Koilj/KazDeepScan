"""Fail-closed metadata gate for the XLS-R+SLS v4 RU calibration inputs.

The gate is deliberately earlier than materialization: it verifies the selected v4
checkpoint and current project history, chooses fresh *metadata identities* from
the pinned VoxForge archive, and validates the pinned eSpeak route.  It never
extracts WAV bytes, invokes eSpeak, loads the detector checkpoint, fits a
temperature, or runs detector inference.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import cast

from kds.data.assets import sha256_file
from kds.data.espeakng import EspeakNgRuntime, load_espeakng_runtime
from kds.data.licenses import APPROVED_LICENSE_STATUSES, load_license_ledger
from kds.data.manifest import REQUIRED_FIELDS, ManifestRow, load_manifest, validate_manifest
from kds.data.research_tts import (
    ResearchTtsModel,
    load_research_tts_model_lock,
    verify_research_tts_model_lock,
)
from kds.data.voxforge import (
    VOXFORGE_RU_ARCHIVE_EXPECTED_SHA256,
    VOXFORGE_RU_ARCHIVE_EXPECTED_SIZE_BYTES,
    VOXFORGE_RU_SOURCE_ID,
    VoxForgeRuRecord,
    load_voxforge_ru_metadata,
)
from kds.eval.voxforge_metadata_screen import (
    VoxForgeMetadataIdentity,
    voxforge_metadata_identity,
)

V4_CALIBRATION_INPUT_SCHEMA_VERSION = 1
V4_CALIBRATION_INPUT_PROTOCOL_ID = "xlsr-sls-model-v4-calibration-inputs-v1"
V4_CALIBRATION_METADATA_FIELDS = (
    "selection_rank",
    "sample_id",
    "submission_pseudo_id",
    "prompt_id",
    "parent_group_id",
    "speaker_pseudo_id",
    "prompt_text_hash",
    "original_prompt_text_hash",
)


class V4CalibrationInputError(ValueError):
    """Raised when the v4 calibration-input gate cannot prove isolation."""


@dataclass(frozen=True, slots=True)
class FileBinding:
    """A hash-pinned project input, optionally with an exact CSV row count."""

    path: str
    sha256: str
    rows: int | None


@dataclass(frozen=True, slots=True)
class V4CalibrationSelection:
    """The metadata-only policy for fresh VoxForge calibration candidates."""

    seed: str
    target_text_groups: int
    historical_text_overlap: str


@dataclass(frozen=True, slots=True)
class V4CalibrationCheckpoint:
    """The ignored local checkpoint whose file identity is checked but never loaded."""

    path: str
    file_sha256: str
    selected_state_sha256: str


@dataclass(frozen=True, slots=True)
class V4CalibrationInputPlan:
    """Strict pre-execution contract for the v4 calibration-input metadata gate."""

    path: str
    sha256: str
    created_at: str
    inputs: Mapping[str, FileBinding]
    checkpoint: V4CalibrationCheckpoint
    voxforge_archive_name: str
    voxforge_archive_size_bytes: int
    voxforge_archive_sha256: str
    espeak_model_id: str
    espeak_model_root: str
    selection: V4CalibrationSelection
    config_root: str
    manifest_root: str
    output_selection: str
    output_receipt: str


@dataclass(frozen=True, slots=True)
class V4CalibrationMetadataRow:
    """One selected source identity before any WAV byte is extracted."""

    selection_rank: int
    sample_id: str
    submission_pseudo_id: str
    prompt_id: str
    parent_group_id: str
    speaker_pseudo_id: str
    prompt_text_hash: str
    original_prompt_text_hash: str


def _json_object(path: Path, label: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise V4CalibrationInputError(f"Cannot read {label}: {path}") from error
    if not isinstance(value, dict):
        raise V4CalibrationInputError(f"{label} must be a JSON object.")
    return cast(dict[str, object], value)


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise V4CalibrationInputError(f"{label} must be a JSON object.")
    return cast(dict[str, object], value)


def _safe_project_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise V4CalibrationInputError(f"{label} must be a non-empty project-relative path.")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or ".." in parsed.parts or "\\" in value or value == ".":
        raise V4CalibrationInputError(f"{label} is not a safe project-relative path.")
    return parsed.as_posix()


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise V4CalibrationInputError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise V4CalibrationInputError(f"{label} must be a positive integer.")
    return value


def _binding(value: object, label: str) -> FileBinding:
    raw = _mapping(value, label)
    if set(raw) != {"path", "sha256", "rows"}:
        raise V4CalibrationInputError(f"{label} must contain exactly path, sha256 and rows.")
    rows = raw["rows"]
    if rows is not None:
        _positive_int(rows, f"{label}.rows")
    return FileBinding(
        path=_safe_project_path(raw["path"], f"{label}.path"),
        sha256=_sha256(raw["sha256"], f"{label}.sha256"),
        rows=cast(int | None, rows),
    )


def _project_path(project_root: Path, relative_path: str, label: str) -> Path:
    candidate = (project_root / relative_path).resolve()
    try:
        candidate.relative_to(project_root)
    except ValueError as error:
        raise V4CalibrationInputError(f"{label} resolves outside the project root.") from error
    return candidate


def _relative_to_root(path: Path, project_root: Path, label: str) -> str:
    try:
        return path.resolve(strict=True).relative_to(project_root).as_posix()
    except ValueError as error:
        raise V4CalibrationInputError(f"{label} escapes the project root: {path}") from error


def _csv_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise V4CalibrationInputError(f"Cannot count CSV rows in {path}.") from error


def _verify_binding(binding: FileBinding, project_root: Path, label: str) -> Path:
    path = _project_path(project_root, binding.path, label)
    if not path.is_file():
        raise V4CalibrationInputError(f"{label} is missing: {binding.path}")
    if sha256_file(path) != binding.sha256:
        raise V4CalibrationInputError(f"{label} SHA-256 mismatch: {binding.path}")
    if binding.rows is not None and _csv_rows(path) != binding.rows:
        raise V4CalibrationInputError(f"{label} row count mismatch: {binding.path}")
    return path


def _parse_timestamp(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise V4CalibrationInputError(f"{label} must be a non-empty ISO-8601 timestamp.")
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise V4CalibrationInputError(f"{label} must be an ISO-8601 timestamp.") from error
    return value


def load_v4_calibration_input_plan(path: Path, project_root: Path) -> V4CalibrationInputPlan:
    """Load and validate the immutable metadata-only calibration-input contract."""

    root = project_root.resolve(strict=True)
    plan_path = path.resolve(strict=True)
    relative_plan = _relative_to_root(plan_path, root, "Calibration-input plan")
    raw = _json_object(plan_path, "v4 calibration-input plan")
    expected_top = {
        "schema_version",
        "protocol_id",
        "created_at",
        "inputs",
        "checkpoint",
        "sources",
        "selection",
        "scope",
        "outputs",
        "prohibitions",
    }
    if set(raw) != expected_top:
        raise V4CalibrationInputError("v4 calibration-input plan has unexpected keys.")
    if raw["schema_version"] != V4_CALIBRATION_INPUT_SCHEMA_VERSION:
        raise V4CalibrationInputError("Unsupported v4 calibration-input plan schema.")
    if raw["protocol_id"] != V4_CALIBRATION_INPUT_PROTOCOL_ID:
        raise V4CalibrationInputError("v4 calibration-input protocol_id mismatch.")

    input_values = _mapping(raw["inputs"], "inputs")
    required_inputs = {
        "roles_and_selection",
        "training_plan",
        "training_receipt",
        "training_contract",
        "train_manifest",
        "dev_manifest",
        "license_ledger",
        "voxforge_source_audit",
        "espeak_model_lock",
        "calibration_module",
        "voxforge_module",
        "voxforge_identity_module",
        "research_tts_module",
        "espeak_module",
        "runner_script",
    }
    if set(input_values) != required_inputs:
        raise V4CalibrationInputError("v4 calibration-input plan inputs are incomplete.")
    inputs = {name: _binding(input_values[name], f"inputs.{name}") for name in sorted(input_values)}

    checkpoint_raw = _mapping(raw["checkpoint"], "checkpoint")
    if set(checkpoint_raw) != {"path", "file_sha256", "selected_state_sha256"}:
        raise V4CalibrationInputError("checkpoint must contain path and both SHA-256 digests.")
    checkpoint = V4CalibrationCheckpoint(
        path=_safe_project_path(checkpoint_raw["path"], "checkpoint.path"),
        file_sha256=_sha256(checkpoint_raw["file_sha256"], "checkpoint.file_sha256"),
        selected_state_sha256=_sha256(
            checkpoint_raw["selected_state_sha256"], "checkpoint.selected_state_sha256"
        ),
    )

    sources = _mapping(raw["sources"], "sources")
    if set(sources) != {"voxforge", "espeak"}:
        raise V4CalibrationInputError("sources must contain exactly voxforge and espeak.")
    voxforge = _mapping(sources["voxforge"], "sources.voxforge")
    if set(voxforge) != {"archive_name", "size_bytes", "sha256"}:
        raise V4CalibrationInputError("sources.voxforge has invalid keys.")
    archive_name = voxforge["archive_name"]
    if not isinstance(archive_name, str) or not archive_name:
        raise V4CalibrationInputError("sources.voxforge.archive_name must be non-empty.")
    archive_size = _positive_int(voxforge["size_bytes"], "sources.voxforge.size_bytes")
    archive_sha256 = _sha256(voxforge["sha256"], "sources.voxforge.sha256")
    if (
        archive_size != VOXFORGE_RU_ARCHIVE_EXPECTED_SIZE_BYTES
        or archive_sha256 != VOXFORGE_RU_ARCHIVE_EXPECTED_SHA256
    ):
        raise V4CalibrationInputError("sources.voxforge must use the pinned VoxForge RU archive.")

    espeak = _mapping(sources["espeak"], "sources.espeak")
    if set(espeak) != {"model_id", "model_root"}:
        raise V4CalibrationInputError("sources.espeak has invalid keys.")
    model_id = espeak["model_id"]
    if not isinstance(model_id, str) or not model_id:
        raise V4CalibrationInputError("sources.espeak.model_id must be non-empty.")

    selection_raw = _mapping(raw["selection"], "selection")
    if set(selection_raw) != {
        "seed",
        "target_text_groups",
        "historical_text_overlap",
        "new_exact_source_assets_required",
        "new_contributor_groups_required",
        "historical_espeak_text_exclusion_required",
    }:
        raise V4CalibrationInputError("selection has invalid keys.")
    seed = selection_raw["seed"]
    if not isinstance(seed, str) or not seed or "\x00" in seed:
        raise V4CalibrationInputError("selection.seed must be a non-empty NUL-free string.")
    historical_text_overlap = selection_raw["historical_text_overlap"]
    if historical_text_overlap != "disclose_only_non_v4_roles":
        raise V4CalibrationInputError("Historical text policy must remain disclose-only.")
    if (
        selection_raw["new_exact_source_assets_required"] is not True
        or selection_raw["new_contributor_groups_required"] is not True
        or selection_raw["historical_espeak_text_exclusion_required"] is not True
    ):
        raise V4CalibrationInputError(
            "Calibration selection must require fresh source/eSpeak inputs."
        )
    selection = V4CalibrationSelection(
        seed=seed,
        target_text_groups=_positive_int(selection_raw["target_text_groups"], "target_text_groups"),
        historical_text_overlap=historical_text_overlap,
    )

    scope = _mapping(raw["scope"], "scope")
    if set(scope) != {"config_root", "manifest_root"}:
        raise V4CalibrationInputError("scope has invalid keys.")
    config_root = _safe_project_path(scope["config_root"], "scope.config_root")
    manifest_root = _safe_project_path(scope["manifest_root"], "scope.manifest_root")

    outputs = _mapping(raw["outputs"], "outputs")
    if set(outputs) != {"metadata_selection", "receipt"}:
        raise V4CalibrationInputError("outputs has invalid keys.")
    output_selection = _safe_project_path(
        outputs["metadata_selection"], "outputs.metadata_selection"
    )
    output_receipt = _safe_project_path(outputs["receipt"], "outputs.receipt")
    if output_selection == output_receipt:
        raise V4CalibrationInputError("Metadata selection and receipt outputs must differ.")

    prohibitions = _mapping(raw["prohibitions"], "prohibitions")
    expected_prohibitions = {
        "raw_audio_extraction": True,
        "synthetic_audio_generation": True,
        "audio_qa": True,
        "acoustic_review": True,
        "pairing": True,
        "checkpoint_loading": True,
        "calibration": True,
        "temperature_fitting": True,
        "final_inference": True,
        "detector_inference": True,
        "output_overwrite": True,
        "network_downloads": True,
    }
    if prohibitions != expected_prohibitions:
        raise V4CalibrationInputError("v4 calibration-input prohibitions are not fail-closed.")

    for name, binding in inputs.items():
        _verify_binding(binding, root, f"inputs.{name}")
    if not _project_path(root, config_root, "scope.config_root").is_dir():
        raise V4CalibrationInputError("scope.config_root is not an existing directory.")
    if not _project_path(root, manifest_root, "scope.manifest_root").is_dir():
        raise V4CalibrationInputError("scope.manifest_root is not an existing directory.")
    return V4CalibrationInputPlan(
        path=relative_plan,
        sha256=sha256_file(plan_path),
        created_at=_parse_timestamp(raw["created_at"], "created_at"),
        inputs=inputs,
        checkpoint=checkpoint,
        voxforge_archive_name=archive_name,
        voxforge_archive_size_bytes=archive_size,
        voxforge_archive_sha256=archive_sha256,
        espeak_model_id=model_id,
        espeak_model_root=_safe_project_path(espeak["model_root"], "sources.espeak.model_root"),
        selection=selection,
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
        raise V4CalibrationInputError(
            f"Cannot inspect manifest inventory file {path}: {error}"
        ) from error
    return REQUIRED_FIELDS.issubset(fields)


def _manifest_inventory(
    project_root: Path, manifest_root: Path
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
        raise V4CalibrationInputError("Calibration audit found no valid manifest inventory.")
    return rows, bindings


def _manifest_references(value: object) -> list[str]:
    references: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "manifest":
                if isinstance(item, str):
                    references.append(item)
                elif isinstance(item, dict) and isinstance(item.get("path"), str):
                    references.append(cast(str, item["path"]))
            references.extend(_manifest_references(item))
    elif isinstance(value, list):
        for item in value:
            references.extend(_manifest_references(item))
    return references


def _configured_role_scope(
    project_root: Path, config_root: Path
) -> tuple[list[ManifestRow], list[dict[str, object]], list[dict[str, object]]]:
    configs = sorted(config_root.rglob("*.json"))
    if not configs:
        raise V4CalibrationInputError("Calibration audit found no research configuration files.")
    manifest_paths: set[Path] = set()
    config_bindings: list[dict[str, object]] = []
    for config in configs:
        raw = _json_object(config, "Research configuration")
        references = _manifest_references(raw)
        config_bindings.append(
            {
                "path": _relative_to_root(config, project_root, "Research configuration"),
                "sha256": sha256_file(config),
                "manifest_references": len(references),
            }
        )
        for reference in references:
            resolved = (config.parent / reference).resolve()
            _relative_to_root(resolved, project_root, "Configured manifest")
            if resolved.suffix != ".csv" or not resolved.is_file():
                raise V4CalibrationInputError(
                    f"Configured manifest is missing or not a CSV: {reference!r} from {config}."
                )
            manifest_paths.add(resolved)
    rows: list[ManifestRow] = []
    manifest_bindings: list[dict[str, object]] = []
    for path in sorted(manifest_paths):
        loaded = load_manifest(path)
        validate_manifest(loaded)
        rows.extend(loaded)
        manifest_bindings.append(
            {
                "path": _relative_to_root(path, project_root, "Configured manifest"),
                "sha256": sha256_file(path),
                "rows": len(loaded),
            }
        )
    return rows, config_bindings, manifest_bindings


def _metadata_value(
    identity: V4CalibrationMetadataRow | VoxForgeMetadataIdentity, field: str
) -> str:
    values = {
        "sample_id": identity.sample_id,
        "text_hash": identity.prompt_text_hash,
        "parent_group_id": identity.parent_group_id,
        "speaker_pseudo_id": identity.speaker_pseudo_id,
    }
    try:
        return values[field]
    except KeyError as error:
        raise V4CalibrationInputError(f"Invalid VoxForge metadata field: {field}") from error


def _row_values(rows: Sequence[ManifestRow]) -> dict[str, set[str]]:
    return {
        "sample_id": {row.sample_id for row in rows},
        "text_hash": {row.text_hash for row in rows},
        "parent_group_id": {row.parent_group_id for row in rows},
        "speaker_pseudo_id": {row.speaker_pseudo_id for row in rows},
    }


def _overlap_counts(
    identities: Sequence[V4CalibrationMetadataRow | VoxForgeMetadataIdentity],
    rows: Sequence[ManifestRow],
) -> dict[str, int]:
    values = _row_values(rows)
    return {
        field: sum(_metadata_value(identity, field) in values[field] for identity in identities)
        for field in values
    }


def _selection_rank(seed: str, domain: str, value: str) -> bytes:
    return hashlib.sha256(f"{seed}\x00{domain}\x00{value}".encode()).digest()


def select_fresh_voxforge_metadata_candidates(
    *,
    records: Sequence[VoxForgeRuRecord],
    historical_rows: Sequence[ManifestRow],
    target_text_groups: int,
    selection_seed: str,
) -> tuple[V4CalibrationMetadataRow, ...]:
    """Select fresh source/group identities without reading WAV payloads or detector outputs."""

    if target_text_groups <= 0 or not selection_seed or "\x00" in selection_seed:
        raise V4CalibrationInputError("Calibration metadata selection has invalid target or seed.")
    prior_source_rows = [row for row in historical_rows if row.source_name == VOXFORGE_RU_SOURCE_ID]
    prior_source_samples = {row.sample_id for row in prior_source_rows}
    prior_source_groups = {row.parent_group_id for row in prior_source_rows}
    historical_espeak_texts = {
        row.text_hash for row in historical_rows if row.generator_family == "formant_rule_based_tts"
    }
    eligible: list[tuple[VoxForgeRuRecord, VoxForgeMetadataIdentity]] = []
    for record in records:
        identity = voxforge_metadata_identity(record)
        if (
            identity.sample_id in prior_source_samples
            or identity.parent_group_id in prior_source_groups
            or identity.prompt_text_hash in historical_espeak_texts
        ):
            continue
        eligible.append((record, identity))
    if not eligible:
        raise V4CalibrationInputError("No fresh VoxForge metadata records remain for calibration.")

    by_text_group: dict[
        str, dict[str, list[tuple[VoxForgeRuRecord, VoxForgeMetadataIdentity]]]
    ] = {}
    original_hash_by_text: dict[str, str] = {}
    for record, identity in eligible:
        text_hash = _metadata_value(identity, "text_hash")
        original_hash = identity.original_prompt_text_hash
        previous_original_hash = original_hash_by_text.setdefault(text_hash, original_hash)
        if previous_original_hash != original_hash:
            raise V4CalibrationInputError(
                "One canonical VoxForge text maps to multiple original transcript hashes."
            )
        group = _metadata_value(identity, "parent_group_id")
        by_text_group.setdefault(text_hash, {}).setdefault(group, []).append((record, identity))
    if target_text_groups > len(by_text_group):
        raise V4CalibrationInputError("Calibration target exceeds fresh VoxForge text capacity.")

    selected_texts = sorted(
        by_text_group,
        key=lambda text_hash: (
            _selection_rank(selection_seed, "prompt-text-group", text_hash),
            text_hash,
        ),
    )[:target_text_groups]
    group_to_text: dict[str, str] = {}

    def augment(text_hash: str, seen_groups: set[str]) -> bool:
        groups = sorted(
            by_text_group[text_hash],
            key=lambda group: (
                _selection_rank(selection_seed, f"contributor-for-prompt-text:{text_hash}", group),
                group,
            ),
        )
        for group in groups:
            if group in seen_groups:
                continue
            seen_groups.add(group)
            prior_text = group_to_text.get(group)
            if prior_text is None or augment(prior_text, seen_groups):
                group_to_text[group] = text_hash
                return True
        return False

    for text_hash in selected_texts:
        if not augment(text_hash, set()):
            raise V4CalibrationInputError(
                "Fresh VoxForge texts cannot be matched to distinct unused contributor groups."
            )
    text_to_group = {text_hash: group for group, text_hash in group_to_text.items()}
    if set(text_to_group) != set(selected_texts):
        raise V4CalibrationInputError("Calibration matching did not retain every selected text.")

    selected: list[V4CalibrationMetadataRow] = []
    for rank, text_hash in enumerate(selected_texts, start=1):
        group = text_to_group[text_hash]
        _record, identity = min(
            by_text_group[text_hash][group],
            key=lambda pair: (
                _selection_rank(
                    selection_seed,
                    f"record-in-text-contributor:{text_hash}:{group}",
                    _metadata_value(pair[1], "sample_id"),
                ),
                _metadata_value(pair[1], "sample_id"),
            ),
        )
        sample_id = _metadata_value(identity, "sample_id")
        submission = sample_id.split(":submission:", 1)[1].split(":prompt:", 1)[0]
        prompt_id = sample_id.rsplit(":prompt:", 1)[1]
        selected.append(
            V4CalibrationMetadataRow(
                selection_rank=rank,
                sample_id=sample_id,
                submission_pseudo_id=submission,
                prompt_id=prompt_id,
                parent_group_id=_metadata_value(identity, "parent_group_id"),
                speaker_pseudo_id=_metadata_value(identity, "speaker_pseudo_id"),
                prompt_text_hash=text_hash,
                original_prompt_text_hash=identity.original_prompt_text_hash,
            )
        )
    _validate_metadata_selection(selected, target_text_groups)
    return tuple(selected)


def _validate_metadata_selection(
    rows: Sequence[V4CalibrationMetadataRow], target_text_groups: int
) -> None:
    if len(rows) != target_text_groups:
        raise V4CalibrationInputError("Calibration metadata selection does not meet its target.")
    fields = (
        "selection_rank",
        "sample_id",
        "parent_group_id",
        "speaker_pseudo_id",
        "prompt_text_hash",
        "original_prompt_text_hash",
    )
    for field in fields:
        values = [getattr(row, field) for row in rows]
        if len(values) != len(set(values)):
            raise V4CalibrationInputError(f"Calibration metadata selection repeats {field}.")
    if [row.selection_rank for row in rows] != list(range(1, target_text_groups + 1)):
        raise V4CalibrationInputError("Calibration metadata selection ranks are not contiguous.")


def _require_training_state(
    *,
    plan: V4CalibrationInputPlan,
    training_receipt_path: Path,
    project_root: Path,
) -> dict[str, object]:
    receipt = _json_object(training_receipt_path, "v4 training receipt")
    required = {
        "status": "ok",
        "mode": "train",
        "calibration_performed": False,
        "final_inference_performed": False,
        "frozen_final_evaluation_performed": False,
        "selected_model_state_sha256": plan.checkpoint.selected_state_sha256,
    }
    for field, expected in required.items():
        if receipt.get(field) != expected:
            raise V4CalibrationInputError(f"v4 training receipt has unexpected {field!r}.")
    run_plan = _mapping(receipt.get("run_plan"), "v4 training receipt run_plan")
    training_plan = plan.inputs["training_plan"]
    if run_plan.get("plan_sha256") != training_plan.sha256:
        raise V4CalibrationInputError("Training receipt is not bound to the pinned training plan.")
    outputs = _mapping(run_plan.get("outputs"), "v4 training receipt outputs")
    checkpoint_path = _project_path(project_root, plan.checkpoint.path, "checkpoint")
    receipt_checkpoint = outputs.get("checkpoint")
    if not isinstance(receipt_checkpoint, str) or not receipt_checkpoint:
        raise V4CalibrationInputError("v4 training receipt lacks a checkpoint path.")
    if Path(receipt_checkpoint).resolve() != checkpoint_path:
        raise V4CalibrationInputError(
            "Training receipt checkpoint path does not match the contract."
        )
    if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != plan.checkpoint.file_sha256:
        raise V4CalibrationInputError(
            "Ignored local checkpoint file is missing or hash-mismatched."
        )
    return {
        "training_receipt": _relative_to_root(
            training_receipt_path, project_root, "Training receipt"
        ),
        "training_receipt_sha256": sha256_file(training_receipt_path),
        "checkpoint": {
            "path": plan.checkpoint.path,
            "file_sha256": plan.checkpoint.file_sha256,
            "selected_state_sha256": plan.checkpoint.selected_state_sha256,
            "loaded": False,
        },
    }


def _require_v4_role_isolation(
    plan: V4CalibrationInputPlan, project_root: Path
) -> tuple[list[ManifestRow], list[ManifestRow]]:
    roles_path = _verify_binding(plan.inputs["roles_and_selection"], project_root, "roles")
    roles = _json_object(roles_path, "v4 roles-and-selection contract")
    calibration = _mapping(_mapping(roles.get("roles"), "roles").get("calibration"), "calibration")
    required_calibration = {
        "ru_bonafide": (
            "voxforge_ru_mdc_2026_05:fresh_exact_assets:calibration_candidate_pending_gate"
        ),
        "ru_spoof": "espeakng_ru:calibration_only",
        "kk": "not_available_from_disjoint_local_lineage",
        "purpose": "ru_temperature_only",
    }
    if calibration != required_calibration:
        raise V4CalibrationInputError(
            "v4 role contract does not retain the pending RU-only calibration gate."
        )
    role_isolation = _mapping(roles.get("role_isolation"), "role_isolation")
    source_roots = _mapping(role_isolation.get("source_lineage_roots"), "source_lineage_roots")
    tts_roots = _mapping(role_isolation.get("tts_family_roots"), "tts_family_roots")
    if source_roots.get("calibration") != [VOXFORGE_RU_SOURCE_ID]:
        raise V4CalibrationInputError("v4 calibration source lineage root changed.")
    if tts_roots.get("calibration") != ["espeakng_formant_tts"]:
        raise V4CalibrationInputError("v4 calibration TTS-family root changed.")

    train_path = _verify_binding(plan.inputs["train_manifest"], project_root, "train manifest")
    dev_path = _verify_binding(plan.inputs["dev_manifest"], project_root, "dev manifest")
    train = load_manifest(train_path)
    dev = load_manifest(dev_path)
    validate_manifest(train)
    validate_manifest(dev)
    if len(train) != 20_000 or len(dev) != 1_917:
        raise V4CalibrationInputError("v4 train/dev manifests do not have their frozen row counts.")
    return train, dev


def _require_source_and_route(
    *, plan: V4CalibrationInputPlan, project_root: Path, archive_path: Path
) -> dict[str, object]:
    if not archive_path.is_file() or archive_path.name != plan.voxforge_archive_name:
        raise V4CalibrationInputError(
            "VoxForge archive path/name does not match the calibration contract."
        )
    if archive_path.stat().st_size != plan.voxforge_archive_size_bytes:
        raise V4CalibrationInputError(
            "VoxForge archive size does not match the calibration contract."
        )
    if sha256_file(archive_path) != plan.voxforge_archive_sha256:
        raise V4CalibrationInputError(
            "VoxForge archive SHA-256 does not match the calibration contract."
        )
    source_audit_path = _verify_binding(
        plan.inputs["voxforge_source_audit"], project_root, "VoxForge source audit"
    )
    source_audit = _json_object(source_audit_path, "VoxForge source audit")
    required_source_audit = {
        "source_id": VOXFORGE_RU_SOURCE_ID,
        "archive_size_bytes": plan.voxforge_archive_size_bytes,
        "archive_sha256": plan.voxforge_archive_sha256,
        "submissions": 644,
        "wav_files": 6_412,
        "canonical_prompt_texts": 81,
    }
    for field, expected in required_source_audit.items():
        if source_audit.get(field) != expected:
            raise V4CalibrationInputError(f"VoxForge source audit has unexpected {field!r}.")

    ledger_path = _verify_binding(plan.inputs["license_ledger"], project_root, "license ledger")
    ledger = load_license_ledger(ledger_path)
    for source_id in (VOXFORGE_RU_SOURCE_ID, "fleurs_ru_v1_espeakng"):
        entry = ledger.get(source_id)
        if (
            entry is None
            or entry.status not in APPROVED_LICENSE_STATUSES
            or entry.train_dev_test_use not in {"research_only", "product_allowed"}
        ):
            raise V4CalibrationInputError(
                f"Calibration source/route {source_id!r} is not research-approved in the ledger."
            )

    lock_path = _verify_binding(plan.inputs["espeak_model_lock"], project_root, "eSpeak model lock")
    lock = load_research_tts_model_lock(lock_path)
    if len(lock.models) != 1 or lock.models[0].model_id != plan.espeak_model_id:
        raise V4CalibrationInputError(
            "Calibration contract does not bind exactly one pinned eSpeak model."
        )
    model: ResearchTtsModel = lock.models[0]
    runtime: EspeakNgRuntime = load_espeakng_runtime(model)
    if runtime.voice != "ru" or model.generator_family != "formant_rule_based_tts":
        raise V4CalibrationInputError(
            "Calibration eSpeak route is not the pinned Russian formant route."
        )
    model_root = _project_path(project_root, plan.espeak_model_root, "eSpeak model root")
    verified = verify_research_tts_model_lock(model_root, lock)
    if model.model_id not in verified:
        raise V4CalibrationInputError(
            "eSpeak model lock verification did not return the selected model."
        )
    return {
        "voxforge": {
            "archive_name": plan.voxforge_archive_name,
            "size_bytes": plan.voxforge_archive_size_bytes,
            "sha256": plan.voxforge_archive_sha256,
            "metadata_only": True,
        },
        "espeak": {
            "model_id": model.model_id,
            "generator_family": model.generator_family,
            "voice": runtime.voice,
            "profile_count": len(runtime.profiles),
            "model_lock": _relative_to_root(lock_path, project_root, "eSpeak model lock"),
            "model_lock_sha256": sha256_file(lock_path),
            "model_artifacts_verified": len(verified[model.model_id]),
            "model_loaded": False,
            "synthetic_audio_generated": False,
        },
    }


def audit_v4_calibration_inputs(
    *,
    plan_path: Path,
    project_root: Path,
    voxforge_archive: Path,
    audited_at: str,
) -> tuple[tuple[V4CalibrationMetadataRow, ...], dict[str, object]]:
    """Audit and freeze metadata identities while keeping calibration execution prohibited."""

    _parse_timestamp(audited_at, "audited_at")
    root = project_root.resolve(strict=True)
    plan = load_v4_calibration_input_plan(plan_path, root)
    output_selection = _project_path(root, plan.output_selection, "metadata selection output")
    output_receipt = _project_path(root, plan.output_receipt, "receipt output")
    if (
        output_selection.exists()
        or output_receipt.exists()
        or not output_selection.parent.is_dir()
        or not output_receipt.parent.is_dir()
    ):
        raise V4CalibrationInputError(
            "Calibration metadata outputs must be new with existing parents."
        )

    train, dev = _require_v4_role_isolation(plan, root)
    training_state = _require_training_state(
        plan=plan,
        training_receipt_path=_verify_binding(
            plan.inputs["training_receipt"], root, "training receipt"
        ),
        project_root=root,
    )
    source_and_route = _require_source_and_route(
        plan=plan, project_root=root, archive_path=voxforge_archive.resolve(strict=True)
    )
    records = load_voxforge_ru_metadata(voxforge_archive)
    inventory_root = _project_path(root, plan.manifest_root, "scope.manifest_root")
    config_root = _project_path(root, plan.config_root, "scope.config_root")
    inventory_rows, inventory_bindings = _manifest_inventory(root, inventory_root)
    configured_rows, config_bindings, configured_manifest_bindings = _configured_role_scope(
        root, config_root
    )
    selected = select_fresh_voxforge_metadata_candidates(
        records=records,
        historical_rows=inventory_rows,
        target_text_groups=plan.selection.target_text_groups,
        selection_seed=plan.selection.seed,
    )
    train_overlap = _overlap_counts(selected, train)
    dev_overlap = _overlap_counts(selected, dev)
    configured_overlap = _overlap_counts(selected, configured_rows)
    inventory_overlap = _overlap_counts(selected, inventory_rows)
    if any(train_overlap.values()) or any(dev_overlap.values()):
        raise V4CalibrationInputError("Selected calibration metadata overlaps frozen v4 train/dev.")
    if any(
        configured_overlap[field] for field in ("sample_id", "parent_group_id", "speaker_pseudo_id")
    ):
        raise V4CalibrationInputError(
            "Selected calibration metadata overlaps a configured role identity."
        )
    if any(
        inventory_overlap[field] for field in ("sample_id", "parent_group_id", "speaker_pseudo_id")
    ):
        raise V4CalibrationInputError(
            "Selected calibration metadata reuses historical source identity."
        )

    source_rows = [row for row in inventory_rows if row.source_name == VOXFORGE_RU_SOURCE_ID]
    historical_espeak_rows = [
        row for row in inventory_rows if row.generator_family == "formant_rule_based_tts"
    ]
    selected_texts = {row.prompt_text_hash for row in selected}
    if selected_texts.intersection({row.text_hash for row in historical_espeak_rows}):
        raise V4CalibrationInputError("Selected texts overlap historic eSpeak outputs.")
    receipt: dict[str, object] = {
        "schema_version": V4_CALIBRATION_INPUT_SCHEMA_VERSION,
        "protocol_id": V4_CALIBRATION_INPUT_PROTOCOL_ID,
        "status": "metadata_inputs_frozen_materialization_contract_required",
        "audited_at": audited_at,
        "plan": {"path": plan.path, "sha256": plan.sha256},
        "training_state": training_state,
        "source_and_route": source_and_route,
        "inputs": {
            name: {"path": binding.path, "sha256": binding.sha256, "rows": binding.rows}
            for name, binding in plan.inputs.items()
        },
        "current_project_scope": {
            "configuration_directory": plan.config_root,
            "configuration_files": config_bindings,
            "configured_manifests": configured_manifest_bindings,
            "configured_rows": len(configured_rows),
            "manifest_inventory_directory": plan.manifest_root,
            "inventory_manifests": inventory_bindings,
            "inventory_rows": len(inventory_rows),
        },
        "historical_exclusions": {
            "voxforge_source_manifest_rows": len(source_rows),
            "voxforge_source_sample_ids": len({row.sample_id for row in source_rows}),
            "voxforge_source_contributor_groups": len({row.parent_group_id for row in source_rows}),
            "historic_formant_tts_rows": len(historical_espeak_rows),
            "historic_formant_tts_text_groups": len(
                {row.text_hash for row in historical_espeak_rows}
            ),
            "policy": (
                "exclude every previous VoxForge exact sample and contributor group; exclude every "
                "text that has already produced a formant-rule-based TTS asset"
            ),
        },
        "selection": {
            "kind": "seeded_maximum_text_to_fresh_contributor_matching",
            "selection_seed": plan.selection.seed,
            "target_text_groups": plan.selection.target_text_groups,
            "selected_records": len(selected),
            "selected_contributor_groups": len({row.parent_group_id for row in selected}),
            "selected_prompt_text_groups": len(selected_texts),
            "rows_sha256": _metadata_rows_sha256(selected),
            "metadata_only": True,
            "raw_audio_sha256_available": False,
        },
        "isolation": {
            "v4_train_overlap_counts": train_overlap,
            "v4_dev_overlap_counts": dev_overlap,
            "configured_role_overlap_counts": configured_overlap,
            "manifest_inventory_overlap_counts": inventory_overlap,
            "historical_text_overlap_policy": plan.selection.historical_text_overlap,
            "historical_text_overlap_count": inventory_overlap["text_hash"],
            "historical_text_overlap_nonblocking_reason": (
                "the v4 role contract requires fresh exact VoxForge assets and contributor "
                "groups, not novel prompt texts versus historical non-v4 research layers; "
                "v4 train/dev text "
                "overlap remains a blocking zero-tolerance condition"
            ),
        },
        "claims": {
            "new_exact_voxforge_source_assets_selected": True,
            "new_voxforge_contributor_groups_selected": True,
            "selected_texts_absent_from_historic_formant_tts_outputs": True,
            "v4_train_dev_isolation_passed": True,
            "speaker_independence": "not_verified_speaker_independent",
            "calibration_language_scope": "ru_only",
            "kk_probability_claim": False,
            "raw_audio_extraction_performed": False,
            "synthetic_audio_generated": False,
            "checkpoint_loaded": False,
            "calibration_performed": False,
            "temperature_fitted": False,
            "final_inference_performed": False,
            "detector_inference_performed": False,
        },
        "next_gate": {
            "name": "v4_calibration_materialization_and_audio_isolation_contract",
            "required_before_calibration": True,
            "required_inputs": [
                "fresh_frozen_license_ledger_for_voxforge_and_new_espeak_derivative",
                "archive_rebinding_and_exact_raw_audio_hashes",
                "one-shot_text_only_espeak_synthesis_contract",
                "decode_qa_vad_and_exact_near_audio_leakage_gate",
                "complete_pair_lock_before_checkpoint_scoring",
            ],
            "still_prohibited": ["temperature_fitting", "final_inference", "detector_feedback"],
        },
    }
    return selected, receipt


def _metadata_rows_sha256(rows: Sequence[V4CalibrationMetadataRow]) -> str:
    payload = [asdict(row) for row in rows]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def write_v4_calibration_input_outputs(
    *,
    plan: V4CalibrationInputPlan,
    project_root: Path,
    selection: Sequence[V4CalibrationMetadataRow],
    receipt: Mapping[str, object],
) -> None:
    """Write the metadata selection and receipt atomically without overwriting either output."""

    root = project_root.resolve(strict=True)
    selection_path = _project_path(root, plan.output_selection, "metadata selection output")
    receipt_path = _project_path(root, plan.output_receipt, "receipt output")
    if (
        selection_path.exists()
        or receipt_path.exists()
        or not selection_path.parent.is_dir()
        or not receipt_path.parent.is_dir()
    ):
        raise V4CalibrationInputError(
            "Calibration metadata outputs must be new with existing parents."
        )
    stage = Path(tempfile.mkdtemp(prefix=".kds-v4-calibration-inputs-", dir=root))
    try:
        staged_selection = stage / selection_path.name
        with staged_selection.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=V4_CALIBRATION_METADATA_FIELDS, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(asdict(row) for row in selection)
        staged_receipt = stage / receipt_path.name
        receipt_payload = dict(receipt)
        receipt_payload["outputs"] = {
            "metadata_selection": {
                "path": plan.output_selection,
                "sha256": sha256_file(staged_selection),
                "rows": len(selection),
            },
            "receipt": {"path": plan.output_receipt},
        }
        with staged_receipt.open("x", encoding="utf-8") as handle:
            json.dump(receipt_payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        if selection_path.exists() or receipt_path.exists():
            raise V4CalibrationInputError("A calibration metadata output appeared during staging.")
        staged_selection.replace(selection_path)
        staged_receipt.replace(receipt_path)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
