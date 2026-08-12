"""Evidence-based novelty gate for a future research generator route.

Historical manifests do not contain reliable architecture identifiers.  The gate
therefore proves only what the stored evidence can support: the exact
family/name/version route is absent.  Family and fixed-speaker alias overlap are
reported separately and never relabelled as architecture or speaker independence.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

from kds.data.manifest import ManifestRow
from kds.data.research_tts import ResearchTtsModel


class GeneratorRouteGateError(ValueError):
    """Raised when an exact generator route was already exposed to the project."""


def generator_route(model: ResearchTtsModel) -> tuple[str, str, str]:
    """Return the exact manifest-level route bound to the checkpoint lock."""

    return (model.generator_family, model.generator_name, model.generator_version)


def audit_generator_route_exposure(
    *,
    model: ResearchTtsModel,
    exposure_manifests: Mapping[str, Sequence[ManifestRow]],
    fixed_voice_aliases: Iterable[str],
) -> dict[str, object]:
    """Reject an exact route reuse and disclose weaker family/voice overlaps."""

    aliases = tuple(sorted(set(fixed_voice_aliases)))
    if not aliases or any(not alias.strip() for alias in aliases):
        raise GeneratorRouteGateError("At least one non-empty fixed voice alias is required.")
    candidate = generator_route(model)
    route_counts: Counter[tuple[str, str, str]] = Counter()
    family_rows: list[str] = []
    voice_alias_rows: dict[str, list[str]] = {alias: [] for alias in aliases}
    spoof_rows = 0
    for path, rows in sorted(exposure_manifests.items()):
        for row in rows:
            if row.label != "spoof":
                continue
            spoof_rows += 1
            route_counts[(row.generator_family, row.generator_name, row.generator_version)] += 1
            if row.generator_family == model.generator_family:
                family_rows.append(f"{path}:{row.sample_id}")
            voice_parts = set(row.voice_id.split(":"))
            for alias in aliases:
                if row.voice_id == alias or alias in voice_parts:
                    voice_alias_rows[alias].append(f"{path}:{row.sample_id}")
    if spoof_rows == 0:
        raise GeneratorRouteGateError("Exposure inventory contains no spoof rows.")
    exact_count = route_counts.get(candidate, 0)
    if exact_count:
        raise GeneratorRouteGateError(
            f"Candidate exact generator route already appears in {exact_count} exposure rows."
        )
    return {
        "novelty_claim": "unseen_exact_generator_route",
        "architecture_independence_claim": False,
        "speaker_independence_claim": False,
        "candidate_route": {
            "generator_family": candidate[0],
            "generator_name": candidate[1],
            "generator_version": candidate[2],
        },
        "exposure_manifest_count": len(exposure_manifests),
        "exposure_spoof_rows": spoof_rows,
        "observed_exact_route_count": len(route_counts),
        "exact_route_overlap_rows": exact_count,
        "generator_family_overlap_rows": len(family_rows),
        "generator_family_overlap_examples": family_rows[:20],
        "fixed_voice_alias_overlap": {
            alias: {
                "rows": len(matches),
                "examples": matches[:20],
            }
            for alias, matches in voice_alias_rows.items()
        },
        "interpretation": (
            "Only the exact checkpoint-bound manifest route is proven unseen. Historical "
            "architecture metadata is incomplete; voice alias overlap is disclosed separately."
        ),
    }
