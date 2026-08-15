from __future__ import annotations

from pathlib import Path

from kds.data.qwen3_tts_customvoice import Qwen3TtsCustomVoice, Qwen3TtsCustomVoiceText
from kds.data.v4_final_recovery_materialization import (
    KK_SOURCE_ID,
    KK_SPOOF_ID,
    OUTPUTS,
    RU_SOURCE_ID,
    RU_SPOOF_ID,
    _load_selection,
    load_plan,
)


def test_recovery_qwen_command_uses_an_absolute_output_path() -> None:
    runtime = Qwen3TtsCustomVoice(
        executable=Path("/bin/true"),
        talker_path=Path("/models/talker.gguf"),
        codec_path=Path("/models/codec.gguf"),
        cuda_library_dirs=(),
        fixed_speaker_name="aiden",
        sample_rate=24_000,
        target_language="ru",
        temperature=0.9,
        max_new_tokens=512,
    )
    command = runtime.command_for(
        Qwen3TtsCustomVoiceText(source_text="Тест", seed=1),
        Path("/tmp/kds-recovery-output.wav").resolve(),
    )

    assert command[command.index("--tts-output") + 1] == "/tmp/kds-recovery-output.wav"


def test_recovery_outputs_and_sources_are_separate_from_the_failed_contract() -> None:
    assert all("final_recovery" in path for path in OUTPUTS.values())
    assert len({RU_SOURCE_ID, KK_SOURCE_ID, RU_SPOOF_ID, KK_SPOOF_ID}) == 4


def test_revalidated_contract_excludes_only_the_attempted_ru_rank_one() -> None:
    root = Path(__file__).resolve().parents[1]
    plan = load_plan(
        root / "configs/research/v4/"
        "xlsr_sls_model_v4_final_recovery_materialization_v1_revalidated.json",
        root,
    )
    selected = _load_selection(plan, root)

    assert len(selected) == 999
    assert sum(row.language == "ru" for row in selected) == 499
    assert sum(row.language == "kk" for row in selected) == 500
    assert not any(row.language == "ru" and row.selection_rank == 1 for row in selected)
