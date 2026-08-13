"""Publish the write-once exact-route audit for VoxForge RU Qwen3-TTS CustomVoice."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from kds.data.assets import sha256_file
from kds.data.manifest import (
    REQUIRED_FIELDS,
    ManifestError,
    ManifestRow,
    load_manifest,
    validate_manifest,
)
from kds.data.qwen3_tts_customvoice import (
    Qwen3TtsCustomVoiceError,
    load_qwen3_tts_customvoice,
)
from kds.data.research_tts import ResearchTtsError, load_research_tts_model_lock
from kds.eval.generator_route_gate import GeneratorRouteGateError, audit_generator_route_exposure

_HISTORICAL_MODEL_IDENTIFIERS = frozenset(
    {
        "qwen3-tts-12hz-0.6b-customvoice",
        "qwen3-tts-customvoice",
        "qwen3 tts 0.6b customvoice q8_0 gguf",
    }
)


def _is_manifest(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            fields = next(csv.reader(handle), [])
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise GeneratorRouteGateError(
            f"Cannot inspect manifest candidate {path}: {error}"
        ) from error
    return REQUIRED_FIELDS.issubset(fields)


def _project_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(project_root).as_posix()
    except ValueError as error:
        raise GeneratorRouteGateError(f"Path is outside the project: {path}") from error


def _historical_identifier_evidence(
    exposure_rows: dict[str, list[ManifestRow]],
) -> dict[str, object]:
    matches: list[str] = []
    versions: Counter[str] = Counter()
    for path, rows in sorted(exposure_rows.items()):
        for row in rows:
            if (
                row.label != "spoof"
                or row.generator_name.casefold() not in _HISTORICAL_MODEL_IDENTIFIERS
            ):
                continue
            matches.append(f"{path}:{row.sample_id}")
            versions[row.generator_version] += 1
    return {
        "identifiers": sorted(_HISTORICAL_MODEL_IDENTIFIERS),
        "rows": len(matches),
        "examples": matches[:20],
        "versions": dict(sorted(versions.items())),
        "interpretation": (
            "Any historical Qwen3 CustomVoice identifier without an immutable revision and weights "
            "hash would make exact-route novelty unprovable. Zero matching rows are required."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--manifest-directory", type=Path, required=True)
    parser.add_argument("--audited-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        if arguments.output.exists() or not arguments.output.parent.is_dir():
            raise GeneratorRouteGateError("Route-audit output must be new with an existing parent.")
        project_root = Path.cwd().resolve(strict=True)
        lock = load_research_tts_model_lock(arguments.model_lock)
        if len(lock.models) != 1:
            raise ResearchTtsError("Qwen3-TTS route audit requires exactly one model lock entry.")
        model = lock.models[0]
        runtime = load_qwen3_tts_customvoice(arguments.model_root, model)
        exposure_rows: dict[str, list[ManifestRow]] = {}
        inputs: list[dict[str, object]] = []
        for path in sorted(arguments.manifest_directory.glob("*.csv")):
            if not _is_manifest(path):
                continue
            rows = load_manifest(path)
            validate_manifest(rows)
            spoof_rows = [row for row in rows if row.label == "spoof"]
            if not spoof_rows:
                continue
            project_path = _project_path(path, project_root)
            exposure_rows[project_path] = rows
            inputs.append(
                {
                    "path": project_path,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "rows": len(rows),
                    "spoof_rows": len(spoof_rows),
                }
            )
        gate = audit_generator_route_exposure(
            model=model,
            exposure_manifests=exposure_rows,
            fixed_voice_aliases=("aiden", "qwen3_tts_customvoice:aiden"),
        )
        if gate["exact_route_overlap_rows"] != 0:
            raise GeneratorRouteGateError("Unexpected non-zero Qwen3-TTS exact-route overlap.")
        historical_identifier = _historical_identifier_evidence(exposure_rows)
        if historical_identifier["rows"] != 0:
            raise GeneratorRouteGateError(
                "Historical manifests contain a Qwen3 CustomVoice identifier; exact-route novelty "
                "cannot be proven."
            )
        report = {
            "schema_version": 1,
            "protocol_id": "voxforge-ru-mdc-qwen3-tts-customvoice-aiden-exact-route-audit-v1",
            "audited_at": arguments.audited_at,
            "model_lock": {
                "path": _project_path(arguments.model_lock, project_root),
                "sha256": sha256_file(arguments.model_lock),
                "size_bytes": arguments.model_lock.stat().st_size,
                "model_id": model.model_id,
                "artifact_sha256": [artifact.sha256 for artifact in model.artifacts],
            },
            "runtime_policy": {
                "fixed_voice_id": "qwen3_tts_customvoice:aiden",
                "fixed_speaker_name": "aiden",
                "target_language": runtime.target_language,
                "sample_rate": runtime.sample_rate,
                "reference_audio": "forbidden",
                "voice_cloning": False,
                "voice_design": "forbidden",
                "text_input_only": True,
                "external_text_normalizer": "forbidden",
                "external_stress_model": "forbidden",
                "runtime_auto_download": "forbidden",
                "runtime_health_checked": True,
                "synthesis_performed": False,
            },
            "exposure_inputs": inputs,
            "route_gate": gate,
            "historical_qwen3_identifier_evidence": historical_identifier,
            "claims": {
                "exact_generator_route_absent_from_historical_manifests": True,
                "architecture_family_novelty": False,
                "speaker_independence": False,
                "reference_audio_or_voice_cloning_used": False,
                "detector_inference_performed": False,
                "detector_inference_authorized": False,
            },
        }
        with arguments.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except (
        GeneratorRouteGateError,
        ManifestError,
        Qwen3TtsCustomVoiceError,
        ResearchTtsError,
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
                "exposure_manifest_count": len(exposure_rows),
                "exact_route_overlap_rows": gate["exact_route_overlap_rows"],
                "synthesis_performed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
