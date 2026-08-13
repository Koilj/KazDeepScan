from __future__ import annotations

import pytest
import torch

from kds.data.utrobinmv_vits import UtrobinmvVits, UtrobinmvVitsError


class _FakeTokenizer:
    unk_token_id = 41

    def __call__(self, text: str, *, return_tensors: str) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        if "-" in text:
            ids = [3, self.unk_token_id, 4]
        else:
            ids = [3, 4, 5]
        return {
            "input_ids": torch.tensor([ids]),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        }


def _runtime() -> UtrobinmvVits:
    return UtrobinmvVits(
        model=object(),
        tokenizer=_FakeTokenizer(),
        torch=torch,
        sample_rate=16_000,
        fixed_speaker_id=0,
    )


def test_prepare_text_uses_only_lowercase_and_collapsed_whitespace() -> None:
    prepared = _runtime().prepare_text("  ПрИвЕт\nмир! ")

    assert prepared.source_text == "  ПрИвЕт\nмир! "
    assert prepared.token_text == "привет мир!"
    assert prepared.input_ids == (3, 4, 5)
    assert prepared.attention_mask == (1, 1, 1)


def test_prepare_text_rejects_unknown_token_instead_of_rewriting() -> None:
    with pytest.raises(UtrobinmvVitsError, match="outside the pinned tokenizer vocabulary"):
        _runtime().prepare_text("из-под")
