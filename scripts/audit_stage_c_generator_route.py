"""Publish a write-once exact-route exposure audit for the Stage-C TTS candidate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from kds.data.assets import sha256_file
from kds.data.kazakhtts import load_kazakhtts_runtime
from kds.data.manifest import (
    REQUIRED_FIELDS,
    ManifestError,
    ManifestRow,
    load_manifest,
    validate_manifest,
)
from kds.data.research_tts import ResearchTtsError, load_research_tts_model_lock
from kds.eval.generator_route_gate import (
    GeneratorRouteGateError,
    audit_generator_route_exposure,
)


def _is_manifest(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            fieldnames = csv.reader(handle).__next__()
    except (OSError, UnicodeDecodeError, csv.Error, StopIteration) as error:
        raise GeneratorRouteGateError(f"Cannot inspect exposure CSV {path}: {error}") from error
    return REQUIRED_FIELDS.issubset(fieldnames)


def _project_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(project_root).as_posix()
    except ValueError as error:
        raise GeneratorRouteGateError(f"Path is outside the project: {path}") from error


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prove exact Stage-C generator-route novelty against all stored manifests."
    )
    parser.add_argument("--model-lock", type=Path, required=True)
    parser.add_argument("--manifest-directory", type=Path, required=True)
    parser.add_argument("--fixed-voice-alias", action="append", required=True)
    parser.add_argument("--audited-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    try:
        if arguments.output.exists():
            raise GeneratorRouteGateError(
                f"Refusing to overwrite generator-route audit: {arguments.output}"
            )
        if not arguments.output.parent.is_dir():
            raise GeneratorRouteGateError(
                f"Generator-route audit parent does not exist: {arguments.output.parent}"
            )
        project_root = Path.cwd().resolve(strict=True)
        lock = load_research_tts_model_lock(arguments.model_lock)
        if len(lock.models) != 1:
            raise ResearchTtsError("Stage-C candidate lock must contain exactly one model.")
        model = lock.models[0]
        runtime = load_kazakhtts_runtime(model)
        exposure_rows: dict[str, list[ManifestRow]] = {}
        input_bindings: list[dict[str, object]] = []
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
            input_bindings.append(
                {
                    "path": project_path,
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                    "rows": len(rows),
                    "spoof_rows": len(spoof_rows),
                }
            )
        audit = audit_generator_route_exposure(
            model=model,
            exposure_manifests=exposure_rows,
            fixed_voice_aliases=arguments.fixed_voice_alias,
        )
        payload = {
            "schema_version": 1,
            "protocol_id": "fresh-suite-stage-c-generator-route-gate-v1",
            "audited_at": arguments.audited_at,
            "model_lock": {
                "path": _project_path(arguments.model_lock, project_root),
                "sha256": sha256_file(arguments.model_lock),
                "size_bytes": arguments.model_lock.stat().st_size,
                "model_id": model.model_id,
                "artifact_sha256": [artifact.sha256 for artifact in model.artifacts],
            },
            "runtime_policy": {
                "fixed_voice_id": runtime.fixed_voice_id,
                "reference_audio": "forbidden",
                "voice_cloning": False,
                "officially_supported_languages": list(runtime.supported_languages),
                "conditional_acoustic_smoke_languages": list(
                    runtime.conditional_smoke_languages
                ),
            },
            "exposure_inputs": input_bindings,
            "audit": audit,
        }
        with arguments.output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    except (
        GeneratorRouteGateError,
        ManifestError,
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
                "audit": audit,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
