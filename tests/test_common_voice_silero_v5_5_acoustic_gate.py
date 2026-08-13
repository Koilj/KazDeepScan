from __future__ import annotations

from pathlib import Path

from kds.data.manifest import load_manifest
from kds.eval.common_voice_silero_v5_5_acoustic_gate import build_packet


def test_build_packet_requires_exact_84_asset_candidate() -> None:
    candidate = load_manifest(
        Path("data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_pairs_v1.csv")
    )
    transcripts = {
        row.sample_id: "Точный русский текст"
        for row in candidate
        if row.label == "bonafide"
    }

    packet = build_packet(candidate, transcripts)

    assert len(packet) == 84
    assert {row.label for row in packet} == {"bonafide", "spoof"}
