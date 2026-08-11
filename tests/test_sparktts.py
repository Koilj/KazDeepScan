from __future__ import annotations

import io
import tarfile
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

import torch

from kds.data.research_tts import load_research_tts_model_lock
from kds.data.sparktts import (
    _SOURCE_FILES,
    SparkTtsRuntime,
    extract_verified_sparktts_source,
    load_sparktts_runtime,
)


def _runtime() -> SparkTtsRuntime:
    lock = load_research_tts_model_lock(Path("configs/research/sparktts_kk_v1_models.json"))
    assert len(lock.models) == 1
    return load_sparktts_runtime(lock.models[0])


def _synthesis_script() -> Any:
    spec = spec_from_file_location("kds_sparktts_synthesis", "scripts/synthesize_ksc_sparktts.py")
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_sparktts_lock_pins_non_cloning_controlled_components_under_limit() -> None:
    lock = load_research_tts_model_lock(Path("configs/research/sparktts_kk_v1_models.json"))
    runtime = _runtime()

    assert sum(artifact.expected_size_bytes for artifact in lock.models[0].artifacts) < 2 * 1024**3
    assert runtime.sample_rate == 16_000
    assert len(runtime.profiles) == 12
    assert {profile.gender for profile in runtime.profiles} == {"female", "male"}
    assert "wav2vec2-large-xlsr-53" not in {
        artifact.relative_path for artifact in lock.models[0].artifacts
    }


def test_sparktts_source_extraction_excludes_reference_audio_code(tmp_path: Path) -> None:
    runtime = _runtime()
    archive = tmp_path / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for relative_path in sorted(_SOURCE_FILES):
            content = f"source:{relative_path}".encode()
            member = tarfile.TarInfo(f"{runtime.source_archive_root}/sparktts/{relative_path}")
            member.size = len(content)
            handle.addfile(member, io.BytesIO(content))
        excluded = tarfile.TarInfo(
            f"{runtime.source_archive_root}/sparktts/models/audio_tokenizer.py"
        )
        excluded.size = len(b"requires wav2vec2")
        handle.addfile(excluded, io.BytesIO(b"requires wav2vec2"))

    source_root = extract_verified_sparktts_source(archive, runtime, tmp_path / "source")

    assert (source_root / "sparktts" / "models" / "bicodec.py").read_text(
        encoding="utf-8"
    ) == "source:models/bicodec.py"
    assert not (source_root / "sparktts" / "models" / "audio_tokenizer.py").exists()


def test_sparktts_bicodec_adapter_preserves_upstream_control_axis() -> None:
    script = _synthesis_script()
    tensor = script._bicodec_global_tensor([1, 2, 3], torch.device("cpu"))

    assert tensor.dtype is torch.long
    assert tensor.shape == (1, 1, 3)
    assert script._attempt_seed(17, 0) == 17
    assert script._attempt_seed(17, 1) == script._attempt_seed(17, 1)
