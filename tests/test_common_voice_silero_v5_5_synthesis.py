from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.synthesize_common_voice_ru_v24_silero_v5_5_pre_qa import (
    CommonVoiceSileroV55SynthesisError,
    require_text_binding,
)

_BASE = Path("data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_ready_v1.csv")
_BINDING = Path("data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_text_binding_v1.json")
_LOCK = Path("configs/research/silero_v5_5_ru_eugene_v1_models.json")
_AUDIT = Path("data/manifests/silero_v5_5_ru_eugene_exact_route_audit_v1.json")
_ARCHIVE = Path("/home/ruslan/Downloads/cv-corpus-24.0-2025-12-05-ru.tar.gz")


def test_require_text_binding_accepts_completed_75_row_receipt() -> None:
    binding = require_text_binding(
        _BINDING,
        base_manifest=_BASE,
        archive=_ARCHIVE,
        model_lock=_LOCK,
        route_audit=_AUDIT,
    )

    assert len(binding) == 75
    assert len({row["text_hash"] for row in binding.values()}) == 75


def test_require_text_binding_rejects_synthetic_claim_before_synthesis(tmp_path: Path) -> None:
    payload = json.loads(_BINDING.read_text(encoding="utf-8"))
    payload["claims"]["synthetic_audio_generated"] = True
    changed = tmp_path / "binding.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CommonVoiceSileroV55SynthesisError, match="governance boundary"):
        require_text_binding(
            changed,
            base_manifest=_BASE,
            archive=_ARCHIVE,
            model_lock=_LOCK,
            route_audit=_AUDIT,
        )
