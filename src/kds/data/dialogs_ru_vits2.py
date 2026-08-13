"""Fail-closed local runtime for the pinned Dialogs-RU VITS2 checkpoint.

The upstream repository's ``tts.py`` calls ``torch.load`` with pickle enabled.
This module deliberately does not import that file.  It verifies every locked
upstream byte first, loads only tensor data with ``weights_only=True``, rejects
partial state dictionaries, and exposes only the fixed Masha/neutral text-only
profile required by the Stage-D research protocol.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import torch

from kds.data.research_tts import (
    ResearchTtsModel,
    verify_research_tts_model_bundle,
)

DIALOGS_RU_VITS2_RUNTIME_KIND = "dialogs_ru_vits2_weights_only_cpu"
DIALOGS_RU_VITS2_MODEL_ID = "dialogs_ru_vits2_masha_neutral"


class DialogsRuVits2Error(ValueError):
    """Raised when the locked Dialogs-RU route cannot be used safely."""


@dataclass(frozen=True, slots=True)
class DialogsRuVits2Text:
    """A fixed, auditable model input prepared from one source transcript."""

    source_text: str
    token_text: str
    token_ids: tuple[int, ...]
    dropped_characters: tuple[str, ...]


@dataclass(slots=True)
class DialogsRuVits2:
    """Loaded fixed-profile model; no reference audio or profile arguments exist."""

    model: torch.nn.Module
    text_to_sequence: Any
    symbols: tuple[str, ...]
    add_blank: bool
    sample_rate: int
    noise_scale: float
    noise_scale_w: float
    length_scale: float

    def prepare_text(self, source_text: str) -> DialogsRuVits2Text:
        """Tokenize one literal source transcript without external normalization.

        The upstream training cleaner lowercases and collapses whitespace.  It is
        part of the locked model source.  No text normalizer, stress model, ASR,
        LID, or model output is invoked here.  Unsupported punctuation is
        disclosed instead of being silently rewritten; tokenization otherwise
        follows the locked upstream vocabulary exactly.
        """

        if not isinstance(source_text, str) or not source_text.strip():
            raise DialogsRuVits2Error("Dialogs-RU synthesis requires non-empty source text.")
        token_ids = tuple(
            int(value) for value in self.text_to_sequence(source_text, ["basic_cleaners"])
        )
        if not token_ids:
            raise DialogsRuVits2Error("Dialogs-RU source text has no recognized model characters.")
        token_text = " ".join(source_text.lower().split())
        vocabulary = set(self.symbols)
        dropped = tuple(
            sorted(set(character for character in token_text if character not in vocabulary))
        )
        return DialogsRuVits2Text(
            source_text=source_text,
            token_text=token_text,
            token_ids=token_ids,
            dropped_characters=dropped,
        )

    @torch.inference_mode()
    def synthesize(self, prepared: DialogsRuVits2Text) -> np.ndarray:
        """Synthesize a CPU waveform using the locked Masha/neutral profile only."""

        # VITS samples a latent even with fixed profile and fixed hyperparameters.
        # The seed derives solely from the pre-committed literal source text, not
        # from detector output or quality feedback.
        seed = int.from_bytes(
            hashlib.sha256(prepared.source_text.encode("utf-8")).digest()[:8], "big"
        )
        torch.manual_seed(seed)
        token_ids = list(prepared.token_ids)
        if self.add_blank:
            interspersed: list[int] = [0] * (len(token_ids) * 2 + 1)
            interspersed[1::2] = token_ids
            token_ids = interspersed
        x = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
        x_lengths = torch.tensor([len(token_ids)], dtype=torch.long)
        speaker_id = torch.tensor([0], dtype=torch.long)  # Masha, frozen by the lock.
        emotion_id = torch.tensor([0], dtype=torch.long)  # neutral, frozen by the lock.
        output = cast(Any, self.model).infer(
            x,
            x_lengths,
            sid=speaker_id,
            emotion=emotion_id,
            noise_scale=self.noise_scale,
            noise_scale_w=self.noise_scale_w,
            length_scale=self.length_scale,
        )
        if not isinstance(output, (tuple, list)) or not output:
            raise DialogsRuVits2Error("Dialogs-RU model returned no waveform tensor.")
        waveform = output[0]
        if not isinstance(waveform, torch.Tensor) or waveform.ndim != 3:
            raise DialogsRuVits2Error("Dialogs-RU model returned an invalid waveform shape.")
        audio = waveform[0, 0].detach().cpu().to(torch.float32).numpy()
        if audio.size == 0 or not np.isfinite(audio).all():
            raise DialogsRuVits2Error("Dialogs-RU model produced empty or non-finite audio.")
        peak = float(np.abs(audio).max())
        if peak <= 0:
            raise DialogsRuVits2Error("Dialogs-RU model produced a silent waveform.")
        return (audio / peak * 0.95 * 32767.0).astype(np.int16)


def load_dialogs_ru_vits2(model_root: Path, model: ResearchTtsModel) -> DialogsRuVits2:
    """Verify and load exactly one pinned Dialogs-RU bundle on CPU.

    This loader is intentionally incompatible with unpinned bundles and with
    checkpoints requiring pickle deserialization.
    """

    if model.model_id != DIALOGS_RU_VITS2_MODEL_ID:
        raise DialogsRuVits2Error(f"Unexpected Dialogs-RU model id: {model.model_id!r}.")
    runtime = _validated_runtime(model.runtime)
    verified = verify_research_tts_model_bundle(model_root, model)
    source_directory = _required_string(runtime, "source_directory")
    config_path = verified[_required_string(runtime, "config_path")]
    checkpoint_path = verified[_required_string(runtime, "checkpoint_path")]
    bundle_root = (model_root / model.destination).resolve(strict=True)
    source_root = (bundle_root / source_directory).resolve(strict=True)
    try:
        source_root.relative_to(bundle_root)
    except ValueError as error:
        raise DialogsRuVits2Error(
            "Dialogs-RU source directory escapes its verified bundle."
        ) from error
    if not source_root.is_dir():
        raise DialogsRuVits2Error("Dialogs-RU verified source directory is missing.")
    if not config_path.is_relative_to(source_root):
        raise DialogsRuVits2Error("Dialogs-RU config must reside in the verified source directory.")
    if not checkpoint_path.is_file():  # Defensive: bundle verification should already ensure this.
        raise DialogsRuVits2Error("Dialogs-RU verified checkpoint is missing.")

    config = _load_config(config_path)
    with _verified_source_imports(source_root) as imported:
        models_module = imported["models"]
        text_module = imported["text"]
        symbols_module = imported["text.symbols"]
        synthesizer = getattr(models_module, "SynthesizerTrn", None)
        text_to_sequence = getattr(text_module, "text_to_sequence", None)
        raw_symbols = getattr(symbols_module, "symbols", None)
        if (
            not callable(synthesizer)
            or not callable(text_to_sequence)
            or not isinstance(raw_symbols, str)
        ):
            raise DialogsRuVits2Error(
                "Pinned Dialogs-RU source has an unexpected public interface."
            )
        cfg_data = _mapping(config, "data", "Dialogs-RU config")
        cfg_train = _mapping(config, "train", "Dialogs-RU config")
        cfg_model = _mapping(config, "model", "Dialogs-RU config")
        try:
            net_g = synthesizer(
                len(raw_symbols),
                80,
                _config_int(cfg_train, "segment_size", "Dialogs-RU config train")
                // _config_int(cfg_data, "hop_length", "Dialogs-RU config data"),
                n_speakers=_config_int(cfg_data, "n_speakers", "Dialogs-RU config data"),
                n_emotions=_config_int(cfg_data, "n_emotions", "Dialogs-RU config data"),
                **cfg_model,
            ).to("cpu")
        except (KeyError, TypeError, ValueError) as error:
            raise DialogsRuVits2Error(
                f"Cannot construct pinned Dialogs-RU model: {error}"
            ) from error
        net_g.eval()
        _load_weights_only_strict(checkpoint_path, net_g)

    return DialogsRuVits2(
        model=net_g,
        text_to_sequence=text_to_sequence,
        symbols=tuple(raw_symbols),
        add_blank=_required_bool(cfg_data, "add_blank", "Dialogs-RU config data"),
        sample_rate=_required_int(runtime, "sample_rate"),
        noise_scale=_required_float(runtime, "noise_scale"),
        noise_scale_w=_required_float(runtime, "noise_scale_w"),
        length_scale=_required_float(runtime, "length_scale"),
    )


def _validated_runtime(runtime: Mapping[str, object]) -> Mapping[str, object]:
    if _required_string(runtime, "kind") != DIALOGS_RU_VITS2_RUNTIME_KIND:
        raise DialogsRuVits2Error("Dialogs-RU lock has an unsupported runtime kind.")
    if _required_string(runtime, "reference_audio_policy") != "forbidden":
        raise DialogsRuVits2Error("Dialogs-RU lock must forbid reference audio.")
    if _required_bool(runtime, "voice_cloning", "Dialogs-RU runtime"):
        raise DialogsRuVits2Error("Dialogs-RU lock must forbid voice cloning.")
    if _required_int(runtime, "fixed_speaker_id") != 0:
        raise DialogsRuVits2Error("Dialogs-RU runtime must lock Masha speaker_id=0.")
    if _required_int(runtime, "fixed_emotion_id") != 0:
        raise DialogsRuVits2Error("Dialogs-RU runtime must lock neutral emotion_id=0.")
    if _required_string(runtime, "fixed_voice_id") != "dialogs_ru_vits2:Masha:neutral":
        raise DialogsRuVits2Error("Dialogs-RU runtime has an unexpected fixed voice id.")
    if _required_string(runtime, "torch_load_policy") != "weights_only_true_strict_state_dict":
        raise DialogsRuVits2Error("Dialogs-RU runtime must require a weights_only strict loader.")
    if _required_string(runtime, "device") != "cpu":
        raise DialogsRuVits2Error("Dialogs-RU runtime is intentionally locked to CPU.")
    if _required_int(runtime, "sample_rate") != 22050:
        raise DialogsRuVits2Error("Dialogs-RU runtime sample rate must be 22050 Hz.")
    for key in ("source_directory", "config_path", "checkpoint_path"):
        _required_string(runtime, key)
    for key in ("noise_scale", "noise_scale_w", "length_scale"):
        value = _required_float(runtime, key)
        if value < 0:
            raise DialogsRuVits2Error(f"Dialogs-RU runtime {key} must not be negative.")
    return runtime


def _load_config(path: Path) -> Mapping[str, object]:
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise DialogsRuVits2Error(f"Cannot read verified Dialogs-RU config: {error}") from error
    if not isinstance(raw, dict):
        raise DialogsRuVits2Error("Dialogs-RU config root must be a JSON object.")
    return cast(Mapping[str, object], raw)


@contextmanager
def _verified_source_imports(source_root: Path) -> Iterator[dict[str, ModuleType]]:
    """Import only source modules that resolve under the verified bundle root."""

    module_names = (
        "S_monotonic_align",
        "attentions",
        "commons",
        "models",
        "modules",
        "pqmf",
        "stft",
        "text",
        "text.cleaners",
        "text.symbols",
        "transforms",
    )
    previous: dict[str, ModuleType | None] = {name: sys.modules.get(name) for name in module_names}
    for name in module_names:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(source_root))
    try:
        imported = {name: importlib.import_module(name) for name in module_names}
        for name, module in imported.items():
            module_path = Path(getattr(module, "__file__", "")).resolve(strict=True)
            if not module_path.is_relative_to(source_root):
                raise DialogsRuVits2Error(
                    f"Dialogs-RU module {name!r} did not load from its verified source bundle."
                )
        yield imported
    finally:
        sys.path.pop(0)
        for name in module_names:
            sys.modules.pop(name, None)
        for name, previous_module in previous.items():
            if previous_module is not None:
                sys.modules[name] = previous_module


def _load_weights_only_strict(path: Path, model: torch.nn.Module) -> None:
    try:
        payload: object = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:  # torch's exact exception classes differ across supported releases.
        raise DialogsRuVits2Error(
            "Dialogs-RU checkpoint was rejected by the required "
            "torch.load(weights_only=True) path: "
            f"{error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise DialogsRuVits2Error("Dialogs-RU checkpoint payload must be a mapping.")
    raw_state = payload.get("model", payload)
    if not isinstance(raw_state, Mapping) or not raw_state:
        raise DialogsRuVits2Error("Dialogs-RU checkpoint has no non-empty model state mapping.")
    state: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise DialogsRuVits2Error(
                "Dialogs-RU checkpoint state must contain only string tensor entries."
            )
        state[key] = value
    expected = model.state_dict()
    missing = sorted(set(expected).difference(state))
    unexpected = sorted(set(state).difference(expected))
    wrong_shapes = sorted(
        key for key in expected.keys() & state.keys() if expected[key].shape != state[key].shape
    )
    if missing or unexpected or wrong_shapes:
        details = []
        if missing:
            details.append(f"missing={missing[:5]}")
        if unexpected:
            details.append(f"unexpected={unexpected[:5]}")
        if wrong_shapes:
            details.append(f"wrong_shapes={wrong_shapes[:5]}")
        raise DialogsRuVits2Error("Dialogs-RU checkpoint state is not exact: " + "; ".join(details))
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise DialogsRuVits2Error(f"Dialogs-RU strict state load failed: {error}") from error


def _mapping(raw: Mapping[str, object], key: str, label: str) -> Mapping[str, object]:
    value = raw.get(key)
    if not isinstance(value, dict):
        raise DialogsRuVits2Error(f"{label} field {key!r} must be an object.")
    return cast(Mapping[str, object], value)


def _config_int(raw: Mapping[str, object], key: str, label: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise DialogsRuVits2Error(f"{label} field {key!r} must be an integer.")
    return value


def _required_string(raw: Mapping[str, object], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DialogsRuVits2Error(f"Dialogs-RU runtime field {key!r} must be a non-empty string.")
    return value.strip()


def _required_int(raw: Mapping[str, object], key: str) -> int:
    value = raw.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise DialogsRuVits2Error(f"Dialogs-RU runtime field {key!r} must be an integer.")
    return value


def _required_float(raw: Mapping[str, object], key: str) -> float:
    value = raw.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DialogsRuVits2Error(f"Dialogs-RU runtime field {key!r} must be a number.")
    return float(value)


def _required_bool(raw: Mapping[str, object], key: str, label: str) -> bool:
    value = raw.get(key)
    if not isinstance(value, bool):
        raise DialogsRuVits2Error(f"{label} field {key!r} must be a boolean.")
    return value
