"""Fail-closed helpers for freezing the Stage-C bona-fide selection."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import cast

from kds.data.assets import sha256_file
from kds.data.manifest import ManifestRow, validate_manifest


class FreshSuiteSelectionError(ValueError):
    """Raised when Stage-C selection inputs cannot support a frozen candidate."""


def require_stage_c_language_gate(path: Path) -> str:
    """Require the completed three-language gate while preserving the inference boundary."""

    value = _json_object(path, "Stage-C language gate")
    if (
        value.get("schema_version") != 1
        or value.get("protocol_id") != "fresh-suite-stage-c-kazakhtts-acoustic-gate-v1"
        or value.get("all_languages_passed") is not True
        or value.get("approved_input_languages") != ["kk", "mixed", "ru"]
        or value.get("detector_inference_authorized") is not False
    ):
        raise FreshSuiteSelectionError(
            "Stage-C language gate must approve exactly kk/mixed/ru without authorizing inference."
        )
    results = value.get("results")
    if not isinstance(results, list) or len(results) != 3:
        raise FreshSuiteSelectionError("Stage-C language gate must contain three results.")
    decisions: dict[str, str] = {}
    for raw in results:
        if not isinstance(raw, dict):
            raise FreshSuiteSelectionError("Stage-C language result must be an object.")
        language = raw.get("language")
        decision = raw.get("decision")
        if not isinstance(language, str) or not isinstance(decision, str) or language in decisions:
            raise FreshSuiteSelectionError("Stage-C language results are malformed or duplicated.")
        decisions[language] = decision
    if decisions != {"kk": "pass", "mixed": "pass", "ru": "pass"}:
        raise FreshSuiteSelectionError("Every Stage-C language decision must be pass.")
    return sha256_file(path)


def require_fresh_inventory_v2(path: Path) -> str:
    """Require the audited source capacities used by the all-eligible selection rule."""

    value = _json_object(path, "fresh-suite inventory")
    inventory = value.get("inventory")
    if (
        value.get("schema_version") != 1
        or value.get("protocol_id") != "fresh-research-suite-source-inventory-v2"
        or not isinstance(inventory, dict)
    ):
        raise FreshSuiteSelectionError("Fresh-suite inventory v2 has an invalid contract.")
    expected = {
        "ru": ("fresh_release_unique_text_groups_before_extraction_and_qa", 55),
        "kk": ("fresh_qa_ready_unique_text_groups", 60),
        "mixed": ("fresh_qa_ready_mixed_rows", 58),
    }
    for language, (field, count) in expected.items():
        role = inventory.get(language)
        if not isinstance(role, dict) or role.get(field) != count:
            raise FreshSuiteSelectionError(
                f"Fresh-suite inventory {language!r} must bind {field}={count}."
            )
    return sha256_file(path)


def select_all_fresh_ready_rows(
    ready_rows: Sequence[ManifestRow],
    exposed_rows: Sequence[ManifestRow],
    *,
    source_name: str,
    language: str,
    code_switch: str,
    expected_count: int,
) -> tuple[ManifestRow, ...]:
    """Select every ready text group absent from a fixed exposed role."""

    validate_manifest(ready_rows)
    validate_manifest(exposed_rows)
    eligible_ready = [
        row
        for row in ready_rows
        if row.source_name == source_name
        and row.language == language
        and row.code_switch == code_switch
        and row.label == "bonafide"
        and row.split == "test"
        and row.codec == "wav"
        and row.relative_path.startswith("processed/")
    ]
    if len(eligible_ready) != len(ready_rows):
        raise FreshSuiteSelectionError(
            f"Ready pool for {language!r} contains a row outside its strict source role."
        )
    exposed_texts = {
        row.text_hash
        for row in exposed_rows
        if row.source_name == source_name and row.label == "bonafide"
    }
    fresh = [row for row in eligible_ready if row.text_hash not in exposed_texts]
    by_text: dict[str, ManifestRow] = {}
    for row in fresh:
        if row.text_hash in by_text:
            raise FreshSuiteSelectionError(
                f"Fresh {language!r} ready pool repeats text group {row.text_hash}."
            )
        by_text[row.text_hash] = row
    selected = tuple(sorted(by_text.values(), key=lambda row: row.sample_id))
    if len(selected) != expected_count:
        raise FreshSuiteSelectionError(
            f"Fresh {language!r} ready pool has {len(selected)} groups; expected {expected_count}."
        )
    return selected


def selection_item(
    *,
    sample_id: str,
    source_name: str,
    language: str,
    code_switch: str,
    parent_group_id: str,
    text_id: str,
    text_hash: str,
    text: str,
    source_member: str,
    base_row: ManifestRow | None,
) -> dict[str, object]:
    """Render one exact text selection, optionally bound to an already-ready base WAV."""

    if not text or sha256_text(text) != text_hash:
        raise FreshSuiteSelectionError(f"Selection text hash mismatch for {sample_id!r}.")
    if base_row is not None and (
        base_row.sample_id != sample_id
        or base_row.source_name != source_name
        or base_row.language != language
        or base_row.code_switch != code_switch
        or base_row.parent_group_id != parent_group_id
        or base_row.text_id != text_id
        or base_row.text_hash != text_hash
    ):
        raise FreshSuiteSelectionError(f"Selection/base provenance mismatch for {sample_id!r}.")
    return {
        "sample_id": sample_id,
        "source_name": source_name,
        "language": language,
        "code_switch": code_switch,
        "parent_group_id": parent_group_id,
        "text_id": text_id,
        "text_hash": text_hash,
        "text": text,
        "source_member": source_member,
        "base_asset": (
            None
            if base_row is None
            else {
                "relative_path": base_row.relative_path,
                "sha256": base_row.sha256,
                "duration_s": base_row.duration_s,
                "original_sr": base_row.original_sr,
                "codec": base_row.codec,
            }
        ),
    }


def require_unique_selection(items: Iterable[Mapping[str, object]], expected_count: int) -> None:
    """Reject duplicate IDs/texts and count drift across the frozen three-role selection."""

    rows = list(items)
    sample_ids = [item.get("sample_id") for item in rows]
    text_hashes = [item.get("text_hash") for item in rows]
    if len(rows) != expected_count:
        raise FreshSuiteSelectionError(
            f"Stage-C selection has {len(rows)} rows; expected {expected_count}."
        )
    if any(not isinstance(value, str) or not value for value in sample_ids + text_hashes):
        raise FreshSuiteSelectionError("Stage-C selection contains an empty ID or text hash.")
    if len(set(sample_ids)) != len(rows) or len(set(text_hashes)) != len(rows):
        raise FreshSuiteSelectionError("Stage-C selection repeats a sample or text group.")


def input_binding(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FreshSuiteSelectionError(f"Selection input does not exist: {path}")
    return {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def load_fresh_suite_selection(path: Path, project_root: Path) -> Mapping[str, object]:
    """Load a frozen selection, verify every input binding and preserve no-inference rules."""

    value = _json_object(path, "Stage-C selection")
    required = {
        "schema_version",
        "protocol_id",
        "created_at",
        "seed",
        "bindings",
        "inventory_sha256",
        "language_gate_sha256",
        "fleurs_revision",
        "fleurs_release_artifacts",
        "generator",
        "selection_contract",
        "roles",
        "selected_count",
        "detector_inference_authorized",
    }
    if set(value) != required or (
        value.get("schema_version") != 1
        or value.get("protocol_id") != "fresh-suite-stage-c-selection-v1"
        or value.get("selected_count") != 173
        or value.get("detector_inference_authorized") is not False
    ):
        raise FreshSuiteSelectionError("Stage-C selection has an invalid root contract.")
    resolved_root = project_root.resolve(strict=True)
    bindings = value.get("bindings")
    if not isinstance(bindings, list) or not bindings:
        raise FreshSuiteSelectionError("Stage-C selection bindings must be a non-empty array.")
    seen_paths: set[str] = set()
    for raw in bindings:
        if not isinstance(raw, dict) or set(raw) != {"path", "sha256", "size_bytes"}:
            raise FreshSuiteSelectionError("Stage-C selection binding has an invalid schema.")
        relative = raw.get("path")
        digest = raw.get("sha256")
        size = raw.get("size_bytes")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or relative in seen_paths
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
        ):
            raise FreshSuiteSelectionError("Stage-C selection binding is malformed or repeated.")
        seen_paths.add(relative)
        candidate = (resolved_root / relative).resolve(strict=False)
        try:
            candidate.relative_to(resolved_root)
        except ValueError as error:
            raise FreshSuiteSelectionError(
                f"Stage-C selection binding escapes project root: {relative!r}."
            ) from error
        if (
            not candidate.is_file()
            or candidate.stat().st_size != size
            or sha256_file(candidate) != digest
        ):
            raise FreshSuiteSelectionError(
                f"Stage-C selection binding is missing or changed: {relative!r}."
            )
    contract = value.get("selection_contract")
    generator = value.get("generator")
    if (
        not isinstance(contract, dict)
        or contract.get("post_selection_backfill") is not False
        or contract.get("detector_inference") != "forbidden"
        or contract.get("metrics_reported_separately") != ["kk", "mixed", "ru"]
        or not isinstance(generator, dict)
        or generator.get("reference_audio") != "forbidden"
        or generator.get("voice_cloning") != "forbidden"
    ):
        raise FreshSuiteSelectionError("Stage-C selection weakens a safety or reporting rule.")
    roles = value.get("roles")
    if not isinstance(roles, dict) or set(roles) != {"ru", "kk", "mixed"}:
        raise FreshSuiteSelectionError("Stage-C selection must contain RU, KK and mixed roles.")
    expected = {
        "ru": (55, "selected_pre_qa"),
        "kk": (60, "qa_ready"),
        "mixed": (58, "qa_ready"),
    }
    items: list[Mapping[str, object]] = []
    item_fields = {
        "sample_id",
        "source_name",
        "language",
        "code_switch",
        "parent_group_id",
        "text_id",
        "text_hash",
        "text",
        "source_member",
        "base_asset",
    }
    for language, (count, stage) in expected.items():
        role = roles.get(language)
        if (
            not isinstance(role, dict)
            or set(role) != {"stage", "selected_count", "items"}
            or role.get("stage") != stage
            or role.get("selected_count") != count
            or not isinstance(role.get("items"), list)
            or len(role["items"]) != count
        ):
            raise FreshSuiteSelectionError(f"Stage-C selection role {language!r} is invalid.")
        for item in role["items"]:
            if not isinstance(item, dict) or set(item) != item_fields:
                raise FreshSuiteSelectionError(
                    f"Stage-C selection role {language!r} has an invalid item schema."
                )
            text = item.get("text")
            text_hash = item.get("text_hash")
            base_asset = item.get("base_asset")
            if (
                item.get("language") != language
                or not isinstance(text, str)
                or not isinstance(text_hash, str)
                or sha256_text(text) != text_hash
                or (language == "ru" and base_asset is not None)
                or (language != "ru" and not isinstance(base_asset, dict))
            ):
                raise FreshSuiteSelectionError(
                    f"Stage-C selection role {language!r} has an invalid item binding."
                )
            items.append(cast(Mapping[str, object], item))
    require_unique_selection(items, 173)
    return value


def sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_object(path: Path, label: str) -> Mapping[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FreshSuiteSelectionError(f"Cannot read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise FreshSuiteSelectionError(f"{label} must be a JSON object.")
    return cast(Mapping[str, object], value)
