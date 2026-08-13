from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from kds.data.voxcpm2_text_only import (
    VOXCPM2_FIXED_SEED,
    VoxCPM2TextOnlyError,
    bind_text,
    generation_kwargs,
    local_model_load_kwargs,
    synthesize_text_only,
)


class _FakeModel:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def generate(self, **kwargs: object) -> Any:
        self.kwargs = kwargs
        return [0.0]


class _UpstreamLikeModel:
    """Mirror VoxCPM.generate, which owns the non-streaming keyword itself."""

    def generate(self, *args: object, **kwargs: object) -> list[float]:
        return self._generate(*args, streaming=False, **kwargs)

    def _generate(self, *args: object, streaming: bool, **kwargs: object) -> list[float]:
        assert not args
        assert streaming is False
        assert kwargs["reference_wav_path"] is None
        return [0.0]


def test_generation_contract_is_text_only_and_single_attempt() -> None:
    literal = "  Тест\u00a0 строки  "
    kwargs = generation_kwargs(literal, bind_text(literal))

    assert kwargs["text"] == "Тест строки"
    assert kwargs["prompt_wav_path"] is None
    assert kwargs["prompt_text"] is None
    assert kwargs["reference_wav_path"] is None
    assert kwargs["normalize"] is False
    assert kwargs["denoise"] is False
    assert kwargs["retry_badcase"] is False
    assert kwargs["retry_badcase_max_times"] == 1
    assert kwargs["seed"] == VOXCPM2_FIXED_SEED


def test_generation_contract_rejects_changed_literal() -> None:
    with pytest.raises(VoxCPM2TextOnlyError, match="binding mismatch"):
        generation_kwargs("другой текст", bind_text("исходный текст"))


def test_narrow_wrapper_passes_no_uncontrolled_arguments() -> None:
    model = _FakeModel()
    text = "Проверка"

    assert synthesize_text_only(model, text, bind_text(text)) == [0.0]
    assert model.kwargs == generation_kwargs(text, bind_text(text))


def test_wrapper_does_not_duplicate_upstream_owned_streaming_keyword() -> None:
    text = "Проверка интерфейса"

    assert synthesize_text_only(_UpstreamLikeModel(), text, bind_text(text)) == [0.0]
    assert "streaming" not in generation_kwargs(text, bind_text(text))


def test_local_loader_disables_remote_denoiser_and_lora(tmp_path: Path) -> None:
    kwargs = local_model_load_kwargs(tmp_path)

    assert kwargs["hf_model_id"] == str(tmp_path.resolve())
    assert kwargs["local_files_only"] is True
    assert kwargs["load_denoiser"] is False
    assert kwargs["zipenhancer_model_id"] is None
    assert kwargs["lora_config"] is None
    assert kwargs["lora_weights_path"] is None
