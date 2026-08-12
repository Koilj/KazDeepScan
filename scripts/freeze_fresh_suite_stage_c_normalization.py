"""Freeze normalized KazakhTTS input text for every ready Stage-C base row before synthesis."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from kds.data.assets import sha256_file
from kds.data.kazakhtts import (
    extract_verified_kazakhtts_runtime,
    load_kazakhtts_runtime,
    validate_kazakhtts_text,
)
from kds.data.kazakhtts_text import (
    KAZAKHTTS_TEXT_NORMALIZER_ID,
    KazakhTtsTextError,
    normalize_kazakhtts_stage_c_text,
)
from kds.data.manifest import ManifestError, load_manifest, validate_manifest
from kds.data.research_tts import (
    ResearchTtsError,
    load_research_tts_model_lock,
    verify_research_tts_model_lock,
)
from kds.eval.fresh_suite_selection import FreshSuiteSelectionError, load_fresh_suite_selection


def _selection_texts(plan: Mapping[str, object]) -> dict[str, str]:
    roles = cast(Mapping[str, object], plan["roles"])
    result: dict[str, str] = {}
    for language in ("ru", "kk", "mixed"):
        role = cast(Mapping[str, object], roles[language])
        for raw in cast(list[object], role["items"]):
            item = cast(Mapping[str, object], raw)
            result[cast(str, item["sample_id"])] = cast(str, item["text"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--created-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        if arguments.output.exists() or not arguments.output.parent.is_dir():
            raise FreshSuiteSelectionError(
                "Stage-C normalization output must be new with an existing parent."
            )
        plan = load_fresh_suite_selection(
            arguments.selection, arguments.project_root.resolve(strict=True)
        )
        selected_texts = _selection_texts(plan)
        base = load_manifest(arguments.base_manifest)
        validate_manifest(base)
        lock = load_research_tts_model_lock(arguments.model_lock)
        if len(lock.models) != 1:
            raise ResearchTtsError("Stage-C normalization model lock must contain one model.")
        model = lock.models[0]
        runtime = load_kazakhtts_runtime(model)
        verified = verify_research_tts_model_lock(arguments.model_root, lock)
        rows: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory(prefix="kds-stage-c-normalization-") as temp:
            extracted = extract_verified_kazakhtts_runtime(
                verified_paths=verified[model.model_id],
                runtime=runtime,
                destination=Path(temp) / "runtime",
            )
            for base_row in sorted(base, key=lambda item: item.sample_id):
                source_text = selected_texts.get(base_row.sample_id)
                if source_text is None:
                    raise FreshSuiteSelectionError(
                        f"Stage-C normalization base is not selected: {base_row.sample_id!r}."
                    )
                result = normalize_kazakhtts_stage_c_text(source_text, base_row.language)
                validated = validate_kazakhtts_text(result.normalized, extracted)
                if validated != result.normalized:
                    raise FreshSuiteSelectionError(
                        f"Stage-C normalization is not runtime-canonical: {base_row.sample_id!r}."
                    )
                rows.append(
                    {
                        "sample_id": base_row.sample_id,
                        "language": base_row.language,
                        "text_id": base_row.text_id,
                        "source_text_hash": base_row.text_hash,
                        "source_text": source_text,
                        "normalized_text": result.normalized,
                        "normalized_text_sha256": result.normalized_sha256,
                        "changed": result.normalized != source_text,
                        "operations": list(result.operations),
                    }
                )
        if len(rows) != 168 or len({row["sample_id"] for row in rows}) != 168:
            raise FreshSuiteSelectionError("Stage-C normalization must cover 168 unique rows.")
        source_path = Path("src/kds/data/kazakhtts_text.py")
        payload = {
            "schema_version": 1,
            "protocol_id": "fresh-suite-stage-c-kazakhtts-normalization-v1",
            "created_at": arguments.created_at,
            "normalizer_id": KAZAKHTTS_TEXT_NORMALIZER_ID,
            "normalizer_source": {
                "path": source_path.as_posix(),
                "sha256": sha256_file(source_path),
            },
            "selection": {
                "path": arguments.selection.as_posix(),
                "sha256": sha256_file(arguments.selection),
            },
            "base_manifest": {
                "path": arguments.base_manifest.as_posix(),
                "sha256": sha256_file(arguments.base_manifest),
            },
            "model_lock": {
                "path": arguments.model_lock.as_posix(),
                "sha256": sha256_file(arguments.model_lock),
            },
            "row_count": len(rows),
            "changed_by_language": dict(
                sorted(
                    Counter(
                        cast(str, row["language"]) for row in rows if row["changed"]
                    ).items()
                )
            ),
            "decision_rule": {
                "purpose": "character-inventory compatibility only",
                "metric_or_detector_based": False,
                "post_selection_backfill": False,
                "source_text_id_and_hash_preserved": True,
                "full_asset_acoustic_review_required": True,
                "detector_inference": "forbidden",
            },
            "rows": rows,
        }
        arguments.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (
        FreshSuiteSelectionError,
        KazakhTtsTextError,
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
                "rows": len(rows),
                "changed_by_language": payload["changed_by_language"],
                "output": str(arguments.output),
                "sha256": sha256_file(arguments.output),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
