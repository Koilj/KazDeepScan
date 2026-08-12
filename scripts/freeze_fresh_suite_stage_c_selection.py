"""Freeze all currently eligible RU/KK/mixed Stage-C source groups before synthesis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kds.data.fleurs import (
    FLEURS_REVISION,
    FleursIngestionError,
    fleurs_locale_spec,
    inspect_fleurs_release,
    select_fleurs_records,
)
from kds.data.ksc2_mixed_candidate import (
    Ksc2MixedCandidateError,
    load_published_mixed_review,
)
from kds.data.manifest import ManifestError, ManifestRow, load_manifest, validate_manifest
from kds.data.research_tts import ResearchTtsError, load_research_tts_model_lock
from kds.eval.fresh_suite_selection import (
    FreshSuiteSelectionError,
    input_binding,
    require_fresh_inventory_v2,
    require_stage_c_language_gate,
    require_unique_selection,
    select_all_fresh_ready_rows,
    selection_item,
)


def _load_manifests(paths: list[Path]) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for path in paths:
        current = load_manifest(path)
        validate_manifest(current)
        rows.extend(current)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleurs-release-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--language-gate", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--generator-route-gate", type=Path, required=True)
    parser.add_argument("--ru-exposed-manifest", type=Path, action="append", required=True)
    parser.add_argument("--kk-ready-manifest", type=Path, required=True)
    parser.add_argument("--kk-exposed-manifest", type=Path, action="append", required=True)
    parser.add_argument("--mixed-ready-manifest", type=Path, action="append", required=True)
    parser.add_argument("--mixed-exposed-manifest", type=Path, action="append", required=True)
    parser.add_argument(
        "--mixed-review",
        type=Path,
        nargs=2,
        action="append",
        metavar=("CSV", "RECEIPT"),
        required=True,
    )
    parser.add_argument("--seed", required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        if arguments.output.exists() or not arguments.output.parent.is_dir():
            raise FreshSuiteSelectionError(
                "Stage-C selection output must be new with an existing parent."
            )
        if not arguments.seed:
            raise FreshSuiteSelectionError("Stage-C selection seed must not be empty.")
        inventory_hash = require_fresh_inventory_v2(arguments.inventory)
        language_gate_hash = require_stage_c_language_gate(arguments.language_gate)
        ru_report, ru_records = inspect_fleurs_release(arguments.fleurs_release_root, "ru_ru")
        kk_report, kk_records = inspect_fleurs_release(arguments.fleurs_release_root, "kk_kz")

        ru_exposed = _load_manifests(arguments.ru_exposed_manifest)
        exposed_ru_texts = {
            row.text_hash
            for row in ru_exposed
            if row.source_name == "google_fleurs_ru_v1" and row.label == "bonafide"
        }
        ru_selected = select_fleurs_records(
            ru_records["test"],
            55,
            arguments.seed,
            excluded_text_hashes=exposed_ru_texts,
        )
        if len({record.text_hash for record in ru_selected}) != 55:
            raise FreshSuiteSelectionError("RU selection did not produce 55 unique text groups.")

        kk_ready = load_manifest(arguments.kk_ready_manifest)
        kk_exposed = _load_manifests(arguments.kk_exposed_manifest)
        kk_selected = select_all_fresh_ready_rows(
            kk_ready,
            kk_exposed,
            source_name="google_fleurs_kk_v1",
            language="kk",
            code_switch="false",
            expected_count=60,
        )
        kk_records_by_id = {
            f"google_fleurs_kk_v1:{record.filename.removesuffix('.wav')}": record
            for record in kk_records["test"]
        }

        mixed_ready = _load_manifests(arguments.mixed_ready_manifest)
        mixed_exposed = _load_manifests(arguments.mixed_exposed_manifest)
        mixed_selected = select_all_fresh_ready_rows(
            mixed_ready,
            mixed_exposed,
            source_name="ksc2_v1",
            language="mixed",
            code_switch="true",
            expected_count=58,
        )
        mixed_evidence = {}
        for review_csv, review_receipt in arguments.mixed_review:
            for item in load_published_mixed_review(review_csv, review_receipt):
                if item.annotation_id in mixed_evidence:
                    raise FreshSuiteSelectionError(
                        f"Mixed reviews repeat annotation {item.annotation_id!r}."
                    )
                mixed_evidence[item.annotation_id] = item

        ru_spec = fleurs_locale_spec("ru_ru")
        items_ru = [
            selection_item(
                sample_id=f"{ru_spec.source_id}:{record.filename.removesuffix('.wav')}",
                source_name=ru_spec.source_id,
                language="ru",
                code_switch="false",
                parent_group_id=f"{ru_spec.source_id}:prompt:{record.prompt_id}",
                text_id=f"{ru_spec.source_id}:prompt:{record.prompt_id}",
                text_hash=record.text_hash,
                text=record.transcript,
                source_member=f"test/{record.filename}",
                base_row=None,
            )
            for record in ru_selected
        ]
        items_kk = []
        for row in kk_selected:
            record = kk_records_by_id.get(row.sample_id)
            if record is None:
                raise FreshSuiteSelectionError(
                    f"Selected KK row is absent from the pinned release: {row.sample_id!r}."
                )
            items_kk.append(
                selection_item(
                    sample_id=row.sample_id,
                    source_name=row.source_name,
                    language=row.language,
                    code_switch=row.code_switch,
                    parent_group_id=row.parent_group_id,
                    text_id=row.text_id,
                    text_hash=row.text_hash,
                    text=record.transcript,
                    source_member=f"test/{record.filename}",
                    base_row=row,
                )
            )
        items_mixed = []
        for row in mixed_selected:
            evidence = mixed_evidence.get(row.sample_id)
            if evidence is None:
                raise FreshSuiteSelectionError(
                    f"Selected mixed row lacks semantic evidence: {row.sample_id!r}."
                )
            items_mixed.append(
                selection_item(
                    sample_id=row.sample_id,
                    source_name=row.source_name,
                    language=row.language,
                    code_switch=row.code_switch,
                    parent_group_id=row.parent_group_id,
                    text_id=row.text_id,
                    text_hash=row.text_hash,
                    text=evidence.transcript,
                    source_member=evidence.audio_relative_path,
                    base_row=row,
                )
            )

        all_items = [*items_ru, *items_kk, *items_mixed]
        require_unique_selection(all_items, 173)
        lock = load_research_tts_model_lock(arguments.model_lock)
        if len(lock.models) != 1:
            raise FreshSuiteSelectionError("Stage-C model lock must contain one exact route.")
        model = lock.models[0]
        route = json.loads(arguments.generator_route_gate.read_text(encoding="utf-8"))
        audit = route.get("audit") if isinstance(route, dict) else None
        if (
            not isinstance(audit, dict)
            or audit.get("novelty_claim") != "unseen_exact_generator_route"
            or audit.get("exact_route_overlap_rows") != 0
        ):
            raise FreshSuiteSelectionError("Stage-C generator route gate did not pass.")

        input_paths = [
            arguments.inventory,
            arguments.language_gate,
            arguments.model_lock,
            arguments.generator_route_gate,
            *arguments.ru_exposed_manifest,
            arguments.kk_ready_manifest,
            *arguments.kk_exposed_manifest,
            *arguments.mixed_ready_manifest,
            *arguments.mixed_exposed_manifest,
            *(path for pair in arguments.mixed_review for path in pair),
        ]
        payload = {
            "schema_version": 1,
            "protocol_id": "fresh-suite-stage-c-selection-v1",
            "created_at": arguments.created_at,
            "seed": arguments.seed,
            "bindings": [input_binding(path) for path in input_paths],
            "inventory_sha256": inventory_hash,
            "language_gate_sha256": language_gate_hash,
            "fleurs_revision": FLEURS_REVISION,
            "fleurs_release_artifacts": {
                "ru_ru": dict(sorted(ru_report.artifacts.items())),
                "kk_kz": dict(sorted(kk_report.artifacts.items())),
            },
            "generator": {
                "model_id": model.model_id,
                "generator_family": model.generator_family,
                "generator_name": model.generator_name,
                "generator_version": model.generator_version,
                "model_lock_sha256": input_binding(arguments.model_lock)["sha256"],
                "route_gate_sha256": input_binding(arguments.generator_route_gate)["sha256"],
                "reference_audio": "forbidden",
                "voice_cloning": "forbidden",
            },
            "selection_contract": {
                "rule": (
                    "all currently eligible fresh groups; no metric- or detector-based selection"
                ),
                "one_recording_per_text_group": True,
                "post_selection_backfill": False,
                "qa_rejections_must_be_accounted": True,
                "metrics_reported_separately": ["kk", "mixed", "ru"],
                "source_independent_claim": False,
                "speaker_independent_claim": False,
                "detector_inference": "forbidden",
            },
            "roles": {
                "ru": {"stage": "selected_pre_qa", "selected_count": 55, "items": items_ru},
                "kk": {"stage": "qa_ready", "selected_count": 60, "items": items_kk},
                "mixed": {
                    "stage": "qa_ready",
                    "selected_count": 58,
                    "items": items_mixed,
                },
            },
            "selected_count": len(all_items),
            "detector_inference_authorized": False,
        }
        arguments.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (
        FleursIngestionError,
        FreshSuiteSelectionError,
        Ksc2MixedCandidateError,
        ManifestError,
        OSError,
        ResearchTtsError,
        ValueError,
    ) as error:
        print(json.dumps({"status": "error", "issues": [str(error)]}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(arguments.output),
                "selected": {"ru": len(items_ru), "kk": len(items_kk), "mixed": len(items_mixed)},
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
