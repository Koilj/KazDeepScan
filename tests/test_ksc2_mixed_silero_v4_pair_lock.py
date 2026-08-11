from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.publish_ksc2_mixed_silero_v4_pair_lock import main


def test_pair_lock_binds_candidate_to_explicit_review_evidence(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "pair-lock.json"
    exit_code = main(
        [
            "--review-csv",
            "data/manifests/ksc2_test_mixed_ai_review_v1.csv",
            "--review-receipt",
            "data/licenses/ksc2_test_mixed_ai_review_v1_receipt.json",
            "--base-ready-manifest",
            "data/manifests/ksc2_test_mixed_ai_review_v1_ready.csv",
            "--raw-spoof-manifest",
            "data/manifests/ksc2_mixed_v1_silero_v4_raw.csv",
            "--ready-spoof-manifest",
            "data/manifests/ksc2_mixed_v1_silero_v4_ready.csv",
            "--text-rejection-report",
            "data/manifests/ksc2_mixed_v1_silero_v4_text_rejections.json",
            "--audio-rejection-report",
            "data/manifests/ksc2_mixed_v1_silero_v4_audio_rejections.json",
            "--candidate-manifest",
            "data/manifests/ksc2_mixed_v1_silero_v4_candidate_30.csv",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["pair_count"] == 30
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["pairs"]) == 30
    assert all(
        pair["ru_evidence_tokens"] and pair["kk_evidence_tokens"] for pair in payload["pairs"]
    )
