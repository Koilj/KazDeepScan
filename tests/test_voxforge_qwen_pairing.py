from __future__ import annotations

from pathlib import Path

from scripts.publish_voxforge_ru_mdc_qwen3_tts_customvoice_pre_qa_pairs import (
    require_synthesis_receipt,
    require_technical_qa_receipt,
)

_BASE = Path("data/manifests/voxforge_ru_mdc_2026_05_pre_qa_ready_v1.csv")
_RAW = Path("data/manifests/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_pre_qa_raw_v1.csv")
_READY = Path("data/manifests/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_pre_qa_ready_v1.csv")
_SYNTHESIS = Path(
    "data/manifests/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_pre_qa_synthesis_v1.json"
)
_TECHNICAL_QA = Path(
    "data/manifests/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_pre_qa_technical_qa_v1.json"
)


def test_receipts_authorize_exact_79_pair_boundary() -> None:
    require_synthesis_receipt(_SYNTHESIS, base_manifest=_BASE, raw_manifest=_RAW)
    require_technical_qa_receipt(
        _TECHNICAL_QA,
        raw_manifest=_RAW,
        ready_manifest=_READY,
        synthesis_receipt=_SYNTHESIS,
    )
