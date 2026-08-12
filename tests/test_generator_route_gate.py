from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from kds.data.manifest import ManifestRow
from kds.data.research_tts import load_research_tts_model_lock
from kds.eval.generator_route_gate import (
    GeneratorRouteGateError,
    audit_generator_route_exposure,
)
from tests.factories import manifest_mapping


def _model():
    return load_research_tts_model_lock(
        Path("configs/research/kazakhtts_tacotron2_pwg_v1_models.json")
    ).models[0]


def _spoof_row(*, family: str, name: str, version: str, voice_id: str) -> ManifestRow:
    return ManifestRow.from_mapping(
        manifest_mapping(
            sample_id="exposed-spoof",
            label="spoof",
            generator_family=family,
            generator_name=name,
            generator_version=version,
            voice_id=voice_id,
        ),
        row_number=2,
    )


def test_generator_route_gate_accepts_new_route_but_discloses_voice_alias_overlap() -> None:
    model = _model()
    exposed = _spoof_row(
        family="piper_neural_tts",
        name="Piper kk_KZ-issai-high",
        version="piper-tts-1.6.0",
        voice_id="piper_kk_issai_high:ISSAI_KazakhTTS2_M2",
    )

    report = audit_generator_route_exposure(
        model=model,
        exposure_manifests={"prior.csv": [exposed]},
        fixed_voice_aliases=["ISSAI_KazakhTTS2_M2"],
    )

    assert report["novelty_claim"] == "unseen_exact_generator_route"
    assert report["architecture_independence_claim"] is False
    assert report["speaker_independence_claim"] is False
    overlaps = report["fixed_voice_alias_overlap"]
    assert isinstance(overlaps, dict)
    assert overlaps["ISSAI_KazakhTTS2_M2"]["rows"] == 1


def test_generator_route_gate_rejects_exact_route_reuse() -> None:
    model = _model()
    exposed = _spoof_row(
        family=model.generator_family,
        name=model.generator_name,
        version=model.generator_version,
        voice_id="ISSAI_KazakhTTS2_M2",
    )

    with pytest.raises(GeneratorRouteGateError, match="already appears"):
        audit_generator_route_exposure(
            model=model,
            exposure_manifests={"prior.csv": [exposed]},
            fixed_voice_aliases=["ISSAI_KazakhTTS2_M2"],
        )


def test_generator_route_gate_does_not_confuse_family_and_exact_route() -> None:
    model = _model()
    related = replace(model, generator_version="different-checkpoint")
    exposed = _spoof_row(
        family=related.generator_family,
        name=related.generator_name,
        version=related.generator_version,
        voice_id="different-voice",
    )

    report = audit_generator_route_exposure(
        model=model,
        exposure_manifests={"prior.csv": [exposed]},
        fixed_voice_aliases=["ISSAI_KazakhTTS2_M2"],
    )

    assert report["exact_route_overlap_rows"] == 0
    assert report["generator_family_overlap_rows"] == 1
