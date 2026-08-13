"""Fail-closed runtime for pinned UtrobinTTS Russian VITS safetensors.

The route exposes exactly one built-in female speaker and accepts only literal
source text after the model card's required lowercase/whitespace preparation.
It does not accept reference audio, cloning, external accentuation, SSML, or
any arbitrary generation controls.
"""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from kds.data.research_tts import (
    ResearchTtsModel,
    verify_research_tts_model_bundle,
)

UTROBINMV_VITS_RUNTIME_KIND = "transformers_vits_safetensors_fixed_female_cpu"
UTROBINMV_VITS_MODEL_ID = "utrobinmv_tts_ru_vits_female"


class UtrobinmvVitsError(ValueError):
    """Raised when the locked UtrobinTTS route cannot safely synthesize text."""


@dataclass(frozen=True, slots=True)
class UtrobinmvVitsText:
    """One literal transcript prepared for the locked upstream tokenizer."""

    source_text: str
    token_text: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]


@dataclass(slots=True)
class UtrobinmvVits:
    """Loaded fixed-speaker CPU model without a reference-audio input surface."""

    model: Any
    tokenizer: Any
    torch: Any
    sample_rate: int
    fixed_speaker_id: int

    def prepare_text(self, source_text: str) -> UtrobinmvVitsText:
        """Prepare one source transcript with no lexical or stress rewrite.

        The pinned model card says input must be lowercase. Lowercasing and whitespace collapse
        are the only preparation; the source string itself remains the provenance and seed input.
        Unknown tokens fail closed instead of being silently replaced by the tokenizer's ``<unk>``.
        """

        if not isinstance(source_text, str) or not source_text.strip():
            raise UtrobinmvVitsError("UtrobinTTS synthesis requires non-empty source text.")
        if any(
            character.isspace() and character not in {" ", "\t", "\n", "\r"}
            for character in source_text
        ):
            raise UtrobinmvVitsError(
                "UtrobinTTS source text contains unsupported control whitespace."
            )
        token_text = " ".join(source_text.lower().split())
        encoded = self.tokenizer(token_text, return_tensors="pt")
        raw_input_ids = encoded.get("input_ids")
        raw_attention_mask = encoded.get("attention_mask")
        if raw_input_ids is None or raw_attention_mask is None:
            raise UtrobinmvVitsError("Pinned UtrobinTTS tokenizer returned incomplete tensors.")
        input_ids = tuple(int(value) for value in raw_input_ids[0].tolist())
        attention_mask = tuple(int(value) for value in raw_attention_mask[0].tolist())
        if not input_ids or len(input_ids) != len(attention_mask):
            raise UtrobinmvVitsError("Pinned UtrobinTTS tokenizer returned invalid tensor lengths.")
        unknown_id = self.tokenizer.unk_token_id
        if not isinstance(unknown_id, int):
            raise UtrobinmvVitsError("Pinned UtrobinTTS tokenizer has no integer unknown token ID.")
        if unknown_id in input_ids:
            raise UtrobinmvVitsError(
                "UtrobinTTS source text has characters outside the pinned tokenizer vocabulary."
            )
        return UtrobinmvVitsText(
            source_text=source_text,
            token_text=token_text,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def synthesize(self, prepared: UtrobinmvVitsText) -> np.ndarray:
        """Generate one deterministic fixed-voice CPU waveform from prepared literal text."""

        seed = int.from_bytes(
            hashlib.sha256(prepared.source_text.encode("utf-8")).digest()[:8], "big"
        )
        self.torch.manual_seed(seed)
        input_ids = self.torch.tensor([prepared.input_ids], dtype=self.torch.long)
        attention_mask = self.torch.tensor([prepared.attention_mask], dtype=self.torch.long)
        speaker_id = self.torch.tensor([self.fixed_speaker_id], dtype=self.torch.long)
        with self.torch.inference_mode():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                speaker_id=speaker_id,
            )
        waveform = getattr(output, "waveform", None)
        if waveform is None or getattr(waveform, "ndim", None) != 2:
            raise UtrobinmvVitsError("UtrobinTTS model returned an invalid waveform shape.")
        audio = waveform[0].detach().cpu().to(self.torch.float32).numpy()
        if audio.size == 0 or not np.isfinite(audio).all():
            raise UtrobinmvVitsError("UtrobinTTS model produced empty or non-finite audio.")
        peak = float(np.abs(audio).max())
        if peak <= 0:
            raise UtrobinmvVitsError("UtrobinTTS model produced a silent waveform.")
        normalized = (audio / peak * 0.95 * 32767.0).astype(np.int16)
        return cast(np.ndarray[Any, np.dtype[np.int16]], normalized)


def load_utrobinmv_vits(model_root: Path, model: ResearchTtsModel) -> UtrobinmvVits:
    """Verify and load one exact safe-tensors-only UtrobinTTS bundle on CPU."""

    if model.model_id != UTROBINMV_VITS_MODEL_ID:
        raise UtrobinmvVitsError(f"Unexpected UtrobinTTS model id: {model.model_id!r}.")
    runtime = _validated_runtime(model.runtime)
    verified = verify_research_tts_model_bundle(model_root, model)
    required_artifacts = {
        _required_string(runtime, "model_config_path"),
        _required_string(runtime, "weights_path"),
        _required_string(runtime, "vocab_path"),
        _required_string(runtime, "tokenizer_config_path"),
        _required_string(runtime, "special_tokens_map_path"),
    }
    if not required_artifacts.issubset(verified):
        raise UtrobinmvVitsError("UtrobinTTS lock lacks required verified runtime artifacts.")
    try:
        transformers = importlib.import_module("transformers")
        torch = importlib.import_module("torch")
        auto_tokenizer = transformers.AutoTokenizer
        vits_model = transformers.VitsModel
    except (ImportError, AttributeError) as error:
        raise UtrobinmvVitsError(
            "UtrobinTTS runtime requires the project's locked ml dependencies."
        ) from error
    expected_transformers = _required_string(runtime, "transformers_version")
    if getattr(transformers, "__version__", None) != expected_transformers:
        raise UtrobinmvVitsError(
            "UtrobinTTS runtime requires transformers "
            f"{expected_transformers}, got {getattr(transformers, '__version__', None)!r}."
        )
    bundle_root = (model_root / model.destination).resolve(strict=True)
    try:
        loaded_model = vits_model.from_pretrained(
            bundle_root,
            local_files_only=True,
            use_safetensors=True,
        ).to("cpu")
        loaded_tokenizer = auto_tokenizer.from_pretrained(bundle_root, local_files_only=True)
    except (OSError, ValueError, RuntimeError) as error:
        raise UtrobinmvVitsError(
            "Cannot load the verified UtrobinTTS safetensors bundle."
        ) from error
    loaded_model.eval()
    config = getattr(loaded_model, "config", None)
    if (
        getattr(config, "model_type", None) != "vits"
        or getattr(config, "sampling_rate", None) != _required_int(runtime, "sample_rate")
        or getattr(config, "num_speakers", None) != 2
        or getattr(loaded_tokenizer, "unk_token_id", None) != 41
    ):
        raise UtrobinmvVitsError("UtrobinTTS loaded configuration differs from the locked route.")
    return UtrobinmvVits(
        model=loaded_model,
        tokenizer=loaded_tokenizer,
        torch=torch,
        sample_rate=_required_int(runtime, "sample_rate"),
        fixed_speaker_id=_required_nonnegative_int(runtime, "fixed_speaker_id"),
    )


def _validated_runtime(runtime: Mapping[str, object]) -> Mapping[str, object]:
    if _required_string(runtime, "kind") != UTROBINMV_VITS_RUNTIME_KIND:
        raise UtrobinmvVitsError("UtrobinTTS lock has an unsupported runtime kind.")
    if _required_string(runtime, "reference_audio_policy") != "forbidden":
        raise UtrobinmvVitsError("UtrobinTTS lock must forbid reference audio.")
    if _required_bool(runtime, "voice_cloning", "UtrobinTTS runtime"):
        raise UtrobinmvVitsError("UtrobinTTS lock must forbid voice cloning.")
    if _required_bool(runtime, "text_input_only", "UtrobinTTS runtime") is not True:
        raise UtrobinmvVitsError("UtrobinTTS route must accept text input only.")
    if _required_string(runtime, "external_text_normalizer") != "forbidden":
        raise UtrobinmvVitsError("UtrobinTTS lock must forbid external text normalization.")
    if _required_string(runtime, "external_stress_model") != "forbidden":
        raise UtrobinmvVitsError("UtrobinTTS lock must forbid external stress models.")
    if _required_nonnegative_int(runtime, "fixed_speaker_id") != 0:
        raise UtrobinmvVitsError("UtrobinTTS route must lock built-in female speaker_id=0.")
    if _required_string(runtime, "fixed_voice_id") != "utrobinmv_vits:woman:0":
        raise UtrobinmvVitsError("UtrobinTTS lock has an unexpected fixed voice ID.")
    if _required_string(runtime, "device") != "cpu":
        raise UtrobinmvVitsError("UtrobinTTS route is intentionally locked to CPU.")
    if _required_int(runtime, "sample_rate") != 16_000:
        raise UtrobinmvVitsError("UtrobinTTS route must use a 16 kHz sample rate.")
    if _required_string(runtime, "text_preparation") != "lowercase_and_collapse_whitespace_only":
        raise UtrobinmvVitsError("UtrobinTTS text preparation must remain fixed and minimal.")
    if _required_string(runtime, "unknown_token_policy") != "reject":
        raise UtrobinmvVitsError("UtrobinTTS lock must reject unknown tokenizer characters.")
    if _required_string(runtime, "weights_format") != "safetensors_only":
        raise UtrobinmvVitsError("UtrobinTTS lock must use safetensors only.")
    for key in (
        "model_config_path",
        "weights_path",
        "vocab_path",
        "tokenizer_config_path",
        "special_tokens_map_path",
        "transformers_version",
    ):
        _required_string(runtime, key)
    return runtime


def _required_string(runtime: Mapping[str, object], key: str) -> str:
    value = runtime.get(key)
    if not isinstance(value, str) or not value:
        raise UtrobinmvVitsError(f"UtrobinTTS runtime needs non-empty {key!r}.")
    return value


def _required_bool(runtime: Mapping[str, object], key: str, label: str) -> bool:
    value = runtime.get(key)
    if not isinstance(value, bool):
        raise UtrobinmvVitsError(f"{label} needs boolean {key!r}.")
    return value


def _required_int(runtime: Mapping[str, object], key: str) -> int:
    value = runtime.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise UtrobinmvVitsError(f"UtrobinTTS runtime needs positive integer {key!r}.")
    return value


def _required_nonnegative_int(runtime: Mapping[str, object], key: str) -> int:
    value = runtime.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise UtrobinmvVitsError(f"UtrobinTTS runtime needs non-negative integer {key!r}.")
    return value
