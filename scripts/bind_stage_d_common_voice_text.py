"""Bind literal Common Voice text to the frozen Stage-D 73-WAV candidate."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
from pathlib import Path

from kds.data.assets import sha256_file
from kds.data.common_voice import load_common_voice_metadata_from_archive
from kds.data.manifest import ManifestError, load_manifest, validate_manifest
from kds.data.research_tts import (
    ResearchTtsError,
    load_research_tts_model_lock,
    verify_research_tts_model_lock,
)


class StageDTextBindingError(ValueError):
    """Raised when the frozen candidate cannot be re-bound to literal source text."""


def _symbols_from_verified_source(path: Path) -> str:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as error:
        raise StageDTextBindingError(
            f"Cannot read verified Dialogs-RU symbols source: {error}"
        ) from error
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "symbols"
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, str) and value:
                return value
    raise StageDTextBindingError(
        "Verified Dialogs-RU symbols source has no literal symbols string."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--common-voice-archive", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--bound-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        if arguments.output.exists() or not arguments.output.parent.is_dir():
            raise StageDTextBindingError("Text-binding output must be new with an existing parent.")
        candidate = load_manifest(arguments.candidate_manifest)
        validate_manifest(candidate)
        if (
            len(candidate) != 73
            or any(
                row.split != "test"
                or row.label != "bonafide"
                or row.language != "ru"
                or row.source_name != "common_voice_ru_v24"
                for row in candidate
            )
        ):
            raise StageDTextBindingError(
                "Stage-D candidate must be exactly 73 Common Voice RU test rows."
            )
        if len({row.sample_id for row in candidate}) != len(candidate):
            raise StageDTextBindingError("Stage-D candidate repeats a sample ID.")
        lock = load_research_tts_model_lock(arguments.model_lock)
        if len(lock.models) != 1:
            raise ResearchTtsError("Stage-D text binding requires one locked Dialogs-RU model.")
        model = lock.models[0]
        verified = verify_research_tts_model_lock(arguments.model_root, lock)
        symbols = _symbols_from_verified_source(
            verified[model.model_id]["upstream/text/symbols.py"]
        )
        archive_sha256 = sha256_file(arguments.common_voice_archive)
        records = load_common_voice_metadata_from_archive(
            arguments.common_voice_archive, ["train", "dev", "test"]
        )
        by_id = {
            f"common_voice_ru_v24:{Path(record.clip_name).stem}": record for record in records
        }
        rows: list[dict[str, object]] = []
        for candidate_row in sorted(candidate, key=lambda row: row.sample_id):
            record = by_id.get(candidate_row.sample_id)
            if record is None:
                raise StageDTextBindingError(
                    f"Common Voice archive lacks frozen candidate row {candidate_row.sample_id!r}."
                )
            text_sha256 = hashlib.sha256(record.sentence.encode("utf-8")).hexdigest()
            if text_sha256 != candidate_row.text_hash:
                raise StageDTextBindingError(
                    f"Common Voice literal text hash differs for {candidate_row.sample_id!r}."
                )
            token_text = " ".join(record.sentence.lower().split())
            dropped = tuple(
                sorted(set(character for character in token_text if character not in symbols))
            )
            rows.append(
                {
                    "sample_id": candidate_row.sample_id,
                    "text_id": candidate_row.text_id,
                    "text_hash": candidate_row.text_hash,
                    "source_split": record.split,
                    "literal_text_sha256": text_sha256,
                    "upstream_basic_cleaner_text_sha256": hashlib.sha256(
                        token_text.encode("utf-8")
                    ).hexdigest(),
                    "upstream_tokenizer_dropped_characters": list(dropped),
                }
            )
        binding_digest = hashlib.sha256(
            "\n".join(
                f"{row['sample_id']}\t{row['text_hash']}\t{row['upstream_basic_cleaner_text_sha256']}"
                for row in rows
            ).encode("utf-8")
        ).hexdigest()
        report = {
            "schema_version": 1,
            "protocol_id": "stage-d-common-voice-ru-literal-text-binding-v1",
            "bound_at": arguments.bound_at,
            "candidate_manifest": {
                "path": arguments.candidate_manifest.as_posix(),
                "sha256": sha256_file(arguments.candidate_manifest),
                "rows": len(candidate),
            },
            "common_voice_archive": {
                "path": str(arguments.common_voice_archive),
                "size_bytes": arguments.common_voice_archive.stat().st_size,
                "sha256": archive_sha256,
            },
            "model_lock": {
                "path": arguments.model_lock.as_posix(),
                "sha256": sha256_file(arguments.model_lock),
            },
            "input_contract": {
                "synthesis_uses_literal_source_text": True,
                "text_replacement_or_reselection": "forbidden",
                "external_normalizer_or_stress_model": "forbidden",
                "upstream_basic_cleaner": "lowercase_and_collapse_whitespace_only",
                "tokenizer_drops_are_disclosed_not_rewritten": True,
                "detector_or_metric_used": False,
            },
            "rows": rows,
            "rows_by_original_common_voice_split": dict(
                sorted(Counter(str(row["source_split"]) for row in rows).items())
            ),
            "rows_with_tokenizer_drops": sum(
                bool(row["upstream_tokenizer_dropped_characters"]) for row in rows
            ),
            "text_binding_sha256": binding_digest,
        }
        with arguments.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except (
        ManifestError,
        ResearchTtsError,
        StageDTextBindingError,
        OSError,
        ValueError,
    ) as error:
        issues = list(error.issues) if isinstance(error, ManifestError) else [str(error)]
        print(json.dumps({"status": "error", "issues": issues}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(arguments.output),
                "output_sha256": sha256_file(arguments.output),
                "rows": len(rows),
                "rows_with_tokenizer_drops": report["rows_with_tokenizer_drops"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
