from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.synthesize_voxforge_ru_mdc_qwen3_tts_customvoice_pre_qa import (
    VoxForgeQwenSynthesisError,
    require_text_binding,
)

_BASE = Path("data/manifests/voxforge_ru_mdc_2026_05_pre_qa_ready_v1.csv")
_BINDING = Path(
    "data/manifests/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_pre_qa_text_binding_v1.json"
)
_LOCK = Path("configs/research/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_v1_models.json")
_ARTIFACT_LOCK = Path(
    "data/licenses/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_v1_artifact_lock.json"
)
_AUDIT = Path(
    "data/manifests/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_exact_route_audit_v1.json"
)
_ARCHIVE = Path("/home/ruslan/Downloads/1778250273870-voxforge-ru.tar.gz")


def test_require_text_binding_accepts_completed_79_row_receipt() -> None:
    binding = require_text_binding(
        _BINDING,
        ready_manifest=_BASE,
        archive=_ARCHIVE,
        model_lock=_LOCK,
        artifact_lock=_ARTIFACT_LOCK,
        route_audit=_AUDIT,
    )

    assert len(binding) == 79
    assert len({row["text_hash"] for row in binding.values()}) == 79


def test_require_text_binding_rejects_generated_claim_before_one_shot_run(tmp_path: Path) -> None:
    payload = json.loads(_BINDING.read_text(encoding="utf-8"))
    payload["claims"]["synthetic_audio_generated"] = True
    changed = tmp_path / "binding.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(VoxForgeQwenSynthesisError, match="synthesis governance"):
        require_text_binding(
            changed,
            ready_manifest=_BASE,
            archive=_ARCHIVE,
            model_lock=_LOCK,
            artifact_lock=_ARTIFACT_LOCK,
            route_audit=_AUDIT,
        )
