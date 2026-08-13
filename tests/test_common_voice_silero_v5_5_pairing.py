from __future__ import annotations

from pathlib import Path

from scripts.publish_common_voice_ru_v24_silero_v5_5_pre_qa_pairs import (
    require_technical_qa_receipt,
)


def test_technical_qa_receipt_returns_all_33_rejected_spoof_ids() -> None:
    rejected = require_technical_qa_receipt(
        Path("data/manifests/silero_v5_5_ru_eugene_pre_qa_technical_qa_v1.json"),
        raw_manifest=Path("data/manifests/silero_v5_5_ru_eugene_pre_qa_raw_v1.csv"),
        ready_manifest=Path("data/manifests/silero_v5_5_ru_eugene_pre_qa_ready_v1.csv"),
    )

    assert len(rejected) == 33
