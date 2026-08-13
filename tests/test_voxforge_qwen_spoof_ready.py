from __future__ import annotations

from pathlib import Path

import pytest

from kds.data.manifest import load_manifest
from scripts.publish_voxforge_ru_mdc_qwen3_tts_customvoice_pre_qa_spoof_ready import (
    VoxForgeQwenSpoofReadyError,
    rejection_accounting,
    require_synthesis_receipt,
)

_RAW = Path("data/manifests/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_pre_qa_raw_v1.csv")
_SYNTHESIS = Path(
    "data/manifests/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_pre_qa_synthesis_v1.json"
)


def test_require_synthesis_receipt_accepts_completed_79_row_raw_layer() -> None:
    require_synthesis_receipt(_SYNTHESIS, _RAW)


def test_rejection_accounting_requires_exact_raw_partition() -> None:
    raw_rows = load_manifest(_RAW)[:2]
    report = {
        "input_manifest": str(_RAW),
        "reused_rows": 0,
        "published_rows": 1,
        "rejected_rows": [
            {
                "sample_id": raw_rows[1].sample_id,
                "relative_path": raw_rows[1].relative_path,
                "detail": "Audio is not trainable: insufficient_speech (insufficient_speech).",
            }
        ],
    }

    rejected = rejection_accounting(
        raw_rows=raw_rows,
        ready_rows=raw_rows[:1],
        report=report,
        raw_manifest=_RAW,
    )

    assert [item["sample_id"] for item in rejected] == [raw_rows[1].sample_id]


def test_rejection_accounting_rejects_missing_raw_row() -> None:
    raw_rows = load_manifest(_RAW)[:2]

    with pytest.raises(VoxForgeQwenSpoofReadyError, match="every raw Qwen asset"):
        rejection_accounting(
            raw_rows=raw_rows,
            ready_rows=(),
            report={
                "input_manifest": str(_RAW),
                "reused_rows": 0,
                "published_rows": 0,
                "rejected_rows": [],
            },
            raw_manifest=_RAW,
        )
