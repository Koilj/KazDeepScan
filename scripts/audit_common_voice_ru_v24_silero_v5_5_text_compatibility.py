"""Publish a write-once V5.5 literal-text screen for Common Voice RU metadata survivors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kds.data.assets import sha256_file
from kds.data.common_voice import (
    COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SHA256,
    COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SIZE_BYTES,
    CommonVoiceIngestionError,
    load_common_voice_metadata_from_archive,
)
from kds.data.research_tts import ResearchTtsError, load_research_tts_model_lock
from kds.data.silero_v5_5 import load_silero_v5_5_runtime
from kds.eval.candidate_exposure import CandidateExposureError
from kds.eval.common_voice_metadata_screen import (
    screen_common_voice_ru_test_metadata,
    screen_silero_v5_5_literal_text_compatibility,
)


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CandidateExposureError(f"Cannot read {label}: {error}") from error
    if not isinstance(payload, dict):
        raise CandidateExposureError(f"{label} must be a JSON object.")
    return payload


def _require_parent_screen(
    *, path: Path, expected_payload: dict[str, object]
) -> None:
    actual_payload = _load_json_object(path, "metadata screen receipt")
    if actual_payload != expected_payload:
        raise CandidateExposureError(
            "Metadata screen receipt does not match current archive/config/manifest inputs. "
            "Publish a new metadata screen before the literal-text screen."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--config-root", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--metadata-screen", type=Path, required=True)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--audited-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        if arguments.output.exists() or not arguments.output.parent.is_dir():
            raise CandidateExposureError(
                "Literal-text-screen output must be new with an existing parent."
            )
        records = load_common_voice_metadata_from_archive(arguments.archive, ("test",))
        metadata_screen = screen_common_voice_ru_test_metadata(
            records=records,
            project_root=arguments.project_root,
            config_root=arguments.config_root,
            manifest_root=arguments.manifest_root,
            created_at=arguments.audited_at,
        )
        expected_parent = {
            **metadata_screen.receipt,
            "archive": {
                "path": str(arguments.archive),
                "expected_size_bytes": COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SIZE_BYTES,
                "expected_sha256": COMMON_VOICE_RU_V24_ARCHIVE_EXPECTED_SHA256,
                "identity_verified_before_metadata_read": True,
            },
        }
        _require_parent_screen(path=arguments.metadata_screen, expected_payload=expected_parent)
        lock = load_research_tts_model_lock(arguments.model_lock)
        if len(lock.models) != 1:
            raise ResearchTtsError(
                "Literal-text screen requires exactly one V5.5 model lock entry."
            )
        model = lock.models[0]
        runtime = load_silero_v5_5_runtime(model)
        compatibility = screen_silero_v5_5_literal_text_compatibility(
            records=records, metadata_screen=metadata_screen
        )
        input_metadata_screen = compatibility.receipt["input_metadata_screen"]
        if not isinstance(input_metadata_screen, dict):
            raise CandidateExposureError("Literal-text screen has an invalid input-screen receipt.")
        payload = {
            **compatibility.receipt,
            "audited_at": arguments.audited_at,
            "archive": expected_parent["archive"],
            "input_metadata_screen": {
                **input_metadata_screen,
                "path": str(arguments.metadata_screen),
                "sha256": sha256_file(arguments.metadata_screen),
            },
            "model_lock": {
                "path": str(arguments.model_lock),
                "sha256": sha256_file(arguments.model_lock),
                "model_id": model.model_id,
                "runtime_kind": model.runtime["kind"],
                "fixed_speaker": runtime.fixed_speaker,
                "sample_rate": runtime.sample_rate,
            },
        }
        with arguments.output.open("x", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, ensure_ascii=False, indent=2, sort_keys=True)
            file_handle.write("\n")
        strict_group_exclusion = compatibility.receipt["strict_group_exclusion"]
        if not isinstance(strict_group_exclusion, dict):
            raise CandidateExposureError(
                "Literal-text screen has an invalid group-exclusion receipt."
            )
        surviving_client_groups = strict_group_exclusion.get("surviving_client_groups")
        if not isinstance(surviving_client_groups, int):
            raise CandidateExposureError(
                "Literal-text screen has no surviving-client-group count."
            )
    except (
        CandidateExposureError,
        CommonVoiceIngestionError,
        ResearchTtsError,
        OSError,
        ValueError,
    ) as error:
        print(json.dumps({"status": "error", "issues": [str(error)]}, ensure_ascii=False))
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(arguments.output),
                "output_sha256": sha256_file(arguments.output),
                "surviving_records": len(compatibility.surviving),
                "surviving_client_groups": surviving_client_groups,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
