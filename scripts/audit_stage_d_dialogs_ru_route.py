"""Publish the write-once Stage-D Dialogs-RU exact-route exposure audit."""

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
from kds.data.research_tts import ResearchTtsError, load_research_tts_model_lock
from kds.eval.generator_route_gate import GeneratorRouteGateError, audit_generator_route_exposure


def _is_manifest(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            fields = csv.reader(handle).__next__()
    except (OSError, UnicodeDecodeError, csv.Error, StopIteration) as error:
        raise GeneratorRouteGateError(
            f"Cannot inspect manifest candidate {path}: {error}"
        ) from error
    return REQUIRED_FIELDS.issubset(fields)


def _project_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve(strict=True).relative_to(project_root).as_posix()
    except ValueError as error:
        raise GeneratorRouteGateError(f"Path is outside the project: {path}") from error


def _legacy_vits2_evidence(exposure_rows: dict[str, list[ManifestRow]]) -> dict[str, object]:
    matches: list[str] = []
    routes: Counter[tuple[str, str, str]] = Counter()
    for path, rows in sorted(exposure_rows.items()):
        for row in rows:
            if row.label != "spoof" or "vits2" not in row.generator_name.casefold():
                continue
            matches.append(f"{path}:{row.sample_id}")
            routes[(row.generator_family, row.generator_name, row.generator_version)] += 1
    return {
        "rows": len(matches),
        "examples": matches[:20],
        "routes": [
            {
                "generator_family": family,
                "generator_name": name,
                "generator_version": version,
                "rows": count,
            }
            for (family, name, version), count in sorted(routes.items())
        ],
        "interpretation": (
            "These historical rows demonstrate that generic VITS2 identifiers occurred, but their "
            "stored metadata lacks the Dialogs-RU checkpoint SHA-256 and locked runtime revision. "
            "They therefore prevent an architecture-family novelty claim without proving "
            "exact-route reuse."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-lock", type=Path, required=True)
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
            raise ResearchTtsError("Stage-D route audit requires exactly one model lock entry.")
        model = lock.models[0]
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
            fixed_voice_aliases=("Masha", "Маша", "dialogs_ru_vits2:Masha:neutral"),
        )
        if gate["exact_route_overlap_rows"] != 0:
            raise GeneratorRouteGateError("Unexpected non-zero exact Dialogs-RU route overlap.")
        report = {
            "schema_version": 1,
            "protocol_id": "stage-d-dialogs-ru-vits2-exact-route-audit-v1",
            "audited_at": arguments.audited_at,
            "model_lock": {
                "path": _project_path(arguments.model_lock, project_root),
                "sha256": sha256_file(arguments.model_lock),
                "size_bytes": arguments.model_lock.stat().st_size,
            },
            "exposure_inputs": inputs,
            "route_gate": gate,
            "legacy_vits2_evidence": _legacy_vits2_evidence(exposure_rows),
            "claims": {
                "exact_generator_route_absent_from_historical_manifests": True,
                "architecture_family_novelty": False,
                "speaker_independence": False,
                "reference_audio_or_voice_cloning_used": False,
            },
        }
        with arguments.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except (GeneratorRouteGateError, ResearchTtsError, ManifestError, OSError, ValueError) as error:
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
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
