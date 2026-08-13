from __future__ import annotations

from pathlib import Path

from kds.data.manifest import load_manifest
from kds.eval.voxforge_qwen_acoustic_gate import build_packet


def test_build_packet_requires_exact_158_asset_voxforge_qwen_candidate() -> None:
    candidate = load_manifest(
        Path("data/manifests/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_pre_qa_pairs_v1.csv")
    )
    transcripts = {
        row.sample_id: "Точный русский текст" for row in candidate if row.label == "bonafide"
    }

    packet = build_packet(candidate, transcripts)

    assert len(packet) == 158
    assert {row.label for row in packet} == {"bonafide", "spoof"}
