from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

from kds.data.espeakng import (
    ESPEAKNG_RUNTIME_KIND,
    _safe_link_target,
    _safe_member_path,
    load_espeakng_runtime,
)
from kds.data.research_tts import ResearchTtsError, load_research_tts_model_lock


def test_espeakng_lock_pins_compact_kazakh_formant_runtime() -> None:
    lock = load_research_tts_model_lock(Path("configs/research/espeakng_kk_v1_models.json"))

    assert len(lock.models) == 1
    model = lock.models[0]
    runtime = load_espeakng_runtime(model)

    assert model.generator_family == "formant_rule_based_tts"
    assert runtime.voice == "kk"
    assert runtime.sample_rate == 22_050
    assert len(runtime.profiles) == 12
    assert {profile.speed_wpm for profile in runtime.profiles} == {150, 175, 200, 225}
    assert sum(artifact.expected_size_bytes for artifact in model.artifacts) < 2 * 1024**3
    assert model.runtime["kind"] == ESPEAKNG_RUNTIME_KIND
    assert not any("reference" in artifact.relative_path for artifact in model.artifacts)


def test_espeakng_russian_lock_reuses_only_the_pinned_formant_runtime() -> None:
    lock = load_research_tts_model_lock(Path("configs/research/espeakng_ru_v1_models.json"))

    assert len(lock.models) == 1
    model = lock.models[0]
    runtime = load_espeakng_runtime(model)

    assert model.model_id == "espeakng_russian_formant"
    assert model.destination == "espeakng_kazakh_formant"
    assert runtime.voice == "ru"
    assert len(runtime.profiles) == 12
    assert all(profile.voice_id.startswith("ru:") for profile in runtime.profiles)
    assert not any("reference" in artifact.relative_path for artifact in model.artifacts)


def test_espeakng_package_member_and_link_paths_cannot_escape_runtime_root() -> None:
    assert _safe_member_path("./usr/bin/espeak-ng") == PurePosixPath("usr/bin/espeak-ng")
    assert _safe_link_target(
        PurePosixPath("usr/share/doc/espeak-ng/changelog.gz"), "../libespeak-ng1/changelog.gz"
    ) == PurePosixPath("usr/share/doc/libespeak-ng1/changelog.gz")

    with pytest.raises(ResearchTtsError, match="escapes runtime root"):
        _safe_member_path("../../outside")
    with pytest.raises(ResearchTtsError, match="escapes runtime root"):
        _safe_link_target(PurePosixPath("file"), "../outside")
