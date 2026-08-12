from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from kds.data.manifest import ManifestRow
from kds.eval.fresh_suite_selection import (
    FreshSuiteSelectionError,
    load_fresh_suite_selection,
    require_stage_c_language_gate,
    require_unique_selection,
    select_all_fresh_ready_rows,
    selection_item,
    sha256_text,
)


def _row(sample: str, text: str) -> ManifestRow:
    digest = sha256_text(text)
    return ManifestRow(
        sample_id=sample,
        relative_path=f"processed/{digest[:2]}/{digest}.wav",
        sha256=digest,
        split="test",
        label="bonafide",
        language="kk",
        code_switch="false",
        parent_group_id=f"group:{digest}",
        source_name="google_fleurs_kk_v1",
        source_license="CC-BY-4.0",
        rights_basis="test",
        speaker_pseudo_id="unknown",
        text_id=f"text:{digest}",
        text_hash=digest,
        duration_s=3.0,
        generator_family="",
        generator_name="",
        generator_version="",
        voice_id="",
        clone_consent_id="",
        device="source",
        capture_route="source",
        original_sr=16_000,
        codec="wav",
        augmentation_chain="",
        augmentation_seed="",
        created_at="2026-08-12T00:00:00Z",
    )


def test_selection_takes_only_unexposed_ready_text_groups() -> None:
    old = _row("old", "ескі мәтін")
    fresh = _row("fresh", "жаңа мәтін")

    selected = select_all_fresh_ready_rows(
        [old, fresh],
        [old],
        source_name="google_fleurs_kk_v1",
        language="kk",
        code_switch="false",
        expected_count=1,
    )

    assert selected == (fresh,)


def test_selection_item_rejects_changed_text() -> None:
    row = _row("fresh", "жаңа мәтін")

    with pytest.raises(FreshSuiteSelectionError, match="text hash mismatch"):
        selection_item(
            sample_id=row.sample_id,
            source_name=row.source_name,
            language=row.language,
            code_switch=row.code_switch,
            parent_group_id=row.parent_group_id,
            text_id=row.text_id,
            text_hash=row.text_hash,
            text="өзгерген мәтін",
            source_member="test/1.wav",
            base_row=row,
        )


def test_selection_rejects_cross_role_text_reuse() -> None:
    items = [
        {"sample_id": "ru:1", "text_hash": "a" * 64},
        {"sample_id": "kk:1", "text_hash": "a" * 64},
    ]

    with pytest.raises(FreshSuiteSelectionError, match="repeats"):
        require_unique_selection(items, 2)


def test_language_gate_preserves_detector_boundary(tmp_path: Path) -> None:
    path = tmp_path / "gate.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_id": "fresh-suite-stage-c-kazakhtts-acoustic-gate-v1",
                "all_languages_passed": True,
                "approved_input_languages": ["kk", "mixed", "ru"],
                "detector_inference_authorized": False,
                "results": [
                    {"language": language, "decision": "pass"}
                    for language in ("kk", "mixed", "ru")
                ],
            }
        ),
        encoding="utf-8",
    )

    assert len(require_stage_c_language_gate(path)) == 64
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["detector_inference_authorized"] = True
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(FreshSuiteSelectionError, match="without authorizing inference"):
        require_stage_c_language_gate(path)


def test_ready_pool_rejects_wrong_source() -> None:
    row = _row("fresh", "жаңа мәтін")
    wrong = replace(row, source_name="wrong")
    with pytest.raises(FreshSuiteSelectionError, match="outside its strict source role"):
        select_all_fresh_ready_rows(
            [wrong],
            [row],
            source_name="google_fleurs_kk_v1",
            language="kk",
            code_switch="false",
            expected_count=1,
        )


def test_published_stage_c_selection_and_all_bindings_are_valid() -> None:
    plan = load_fresh_suite_selection(
        Path("data/manifests/fresh_suite_stage_c_selection_v1.json"), Path(".")
    )

    assert plan["selected_count"] == 173
    assert plan["detector_inference_authorized"] is False
