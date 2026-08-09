from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import torch
from torch import Tensor, nn


class XlsrEncoderConfig(Protocol):
    hidden_size: int
    num_hidden_layers: int
    conv_kernel: list[int]
    conv_stride: list[int]


@dataclass(frozen=True, slots=True)
class SlsHeadConfig:
    hidden_size: int
    hidden_state_count: int
    attention_size: int = 128
    classifier_size: int = 256
    dropout: float = 0.2

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.hidden_state_count <= 0:
            raise ValueError("hidden_size and hidden_state_count must be positive.")
        if self.attention_size <= 0 or self.classifier_size <= 0:
            raise ValueError("attention_size and classifier_size must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")


class LearnableLayerMix(nn.Module):
    """SLS-style learned softmax mixture over every SSL hidden state."""

    def __init__(self, hidden_state_count: int) -> None:
        super().__init__()
        if hidden_state_count <= 0:
            raise ValueError("hidden_state_count must be positive.")
        self.layer_logits = nn.Parameter(torch.zeros(hidden_state_count))

    def forward(self, hidden_states: tuple[Tensor, ...]) -> Tensor:
        if len(hidden_states) != self.layer_logits.numel():
            raise ValueError(
                f"Expected {self.layer_logits.numel()} hidden states, "
                f"received {len(hidden_states)}."
            )
        reference_shape = hidden_states[0].shape
        matching_shapes = all(state.shape == reference_shape for state in hidden_states)
        if len(reference_shape) != 3 or not matching_shapes:
            raise ValueError(
                "All hidden states must have the same [batch, frames, features] shape."
            )
        weights = torch.softmax(self.layer_logits, dim=0)
        stacked = torch.stack(hidden_states, dim=-1)
        return (stacked * weights).sum(dim=-1)


class AttentiveStatisticsPooling(nn.Module):
    """Masked attentive mean+standard-deviation pooling over SSL frames."""

    def __init__(self, hidden_size: int, attention_size: int) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, attention_size),
            nn.Tanh(),
            nn.Linear(attention_size, 1),
        )

    def forward(self, features: Tensor, feature_mask: Tensor) -> Tensor:
        if features.ndim != 3:
            raise ValueError("features must have shape [batch, frames, hidden_size].")
        if feature_mask.shape != features.shape[:2] or feature_mask.dtype is not torch.bool:
            raise ValueError("feature_mask must be a bool tensor with shape [batch, frames].")
        if not bool(feature_mask.any(dim=1).all()):
            raise ValueError("Each recording must retain at least one valid feature frame.")

        attention_logits = self.attention(features).squeeze(-1)
        attention_logits = attention_logits.masked_fill(~feature_mask, -torch.inf)
        attention_weights = torch.softmax(attention_logits, dim=-1).unsqueeze(-1)
        mean = (attention_weights * features).sum(dim=1)
        variance = (attention_weights * (features - mean.unsqueeze(1)).square()).sum(dim=1)
        standard_deviation = torch.sqrt(variance.clamp_min(1e-6))
        return torch.cat((mean, standard_deviation), dim=-1)


class SlsBinaryHead(nn.Module):
    """Layer mix, attentive statistics pooling, and one raw spoof logit."""

    def __init__(self, config: SlsHeadConfig) -> None:
        super().__init__()
        self.config = config
        self.layer_mix = LearnableLayerMix(config.hidden_state_count)
        self.pooling = AttentiveStatisticsPooling(config.hidden_size, config.attention_size)
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size * 2),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 2, config.classifier_size),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_size, 1),
        )

    def forward(self, hidden_states: tuple[Tensor, ...], feature_mask: Tensor) -> Tensor:
        mixed_features = self.layer_mix(hidden_states)
        pooled = self.pooling(mixed_features, feature_mask)
        return cast(Tensor, self.classifier(pooled).squeeze(-1))


class XlsrSlsClassifier(nn.Module):
    """XLS-R encoder plus SLS head; it exposes raw window-level spoof logits only."""

    def __init__(
        self,
        encoder: nn.Module,
        head_config: SlsHeadConfig,
        conv_kernel: tuple[int, ...],
        conv_stride: tuple[int, ...],
    ) -> None:
        super().__init__()
        if len(conv_kernel) != len(conv_stride) or not conv_kernel:
            raise ValueError(
                "conv_kernel and conv_stride must be non-empty tuples of equal length."
            )
        self.encoder = encoder
        self.head = SlsBinaryHead(head_config)
        self._conv_kernel = conv_kernel
        self._conv_stride = conv_stride

    @classmethod
    def from_pretrained(cls, model_name: str = "facebook/wav2vec2-xls-r-300m") -> XlsrSlsClassifier:
        from transformers import AutoModel

        encoder = cast(nn.Module, AutoModel.from_pretrained(model_name))
        config = cast(XlsrEncoderConfig, encoder.config)
        head_config = SlsHeadConfig(
            hidden_size=int(config.hidden_size),
            hidden_state_count=int(config.num_hidden_layers) + 1,
        )
        return cls(
            encoder=encoder,
            head_config=head_config,
            conv_kernel=tuple(int(value) for value in config.conv_kernel),
            conv_stride=tuple(int(value) for value in config.conv_stride),
        )

    def forward(self, input_values: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if input_values.ndim != 2:
            raise ValueError("input_values must have shape [batch, samples].")
        if attention_mask is not None and attention_mask.shape != input_values.shape:
            raise ValueError("attention_mask must have the same shape as input_values.")
        encoder_output = self.encoder(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = cast(tuple[Tensor, ...] | None, encoder_output.hidden_states)
        if hidden_states is None:
            raise RuntimeError("Encoder did not return hidden states required by the SLS head.")
        feature_mask = self._feature_mask(input_values, attention_mask, hidden_states[-1].shape[1])
        return cast(Tensor, self.head(hidden_states, feature_mask))

    def _feature_mask(
        self, input_values: Tensor, attention_mask: Tensor | None, feature_frames: int
    ) -> Tensor:
        if attention_mask is None:
            lengths = torch.full(
                (input_values.shape[0],), input_values.shape[1], device=input_values.device
            )
        else:
            lengths = attention_mask.to(dtype=torch.long).sum(dim=-1)
        for kernel, stride in zip(self._conv_kernel, self._conv_stride, strict=True):
            lengths = torch.div(lengths - kernel, stride, rounding_mode="floor") + 1
        lengths = lengths.clamp(min=1, max=feature_frames)
        indices = torch.arange(feature_frames, device=input_values.device)
        return indices.unsqueeze(0) < lengths.unsqueeze(1)
