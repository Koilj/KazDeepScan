# VoxForge RU — UtrobinTTS VITS route review, 13 августа 2026

**Статус:** rejected. Ни selection-bound WAV extraction, ни synthesis, ни QA, ни pairing, ни
detector inference этим review не разрешены.

## Проверенный кандидат

Read-only review закрепил public model
[`utrobinmv/tts_ru_free_hf_vits_low_multispeaker`](https://huggingface.co/utrobinmv/tts_ru_free_hf_vits_low_multispeaker)
на commit `0e9a39581aac24f21487004f7e4f6c4c8c441de2`. Pinned card declares `Apache-2.0`, Russian
plain-text input, two built-in speakers and 16 kHz VITS; project wrapper зафиксировал только
speaker `0` (`woman`), CPU, safetensors и lowercase/whitespace preparation. Reference audio,
voice cloning, random speaker, external accentizer и other text models запрещены.

Six model-card/config/tokenizer/safetensors files (`60,366,570` bytes) были locally SHA-256
verified, safetensors header reported `708` tensor entries and a local CPU model load succeeded.
No waveform was generated. Full lock:
[artifact receipt](../data/licenses/voxforge_ru_mdc_utrobinmv_vits_female_v1_artifact_lock.json),
SHA-256 `e7599203b511938e0c3abb0a8ea0c337c0bb6243bea95ef710594c32bd34cb0d`.

The model card does not document the training data or verified speaker identity. Thus even a
successful route would have remained personal-research only and could not support architecture-
or speaker-independence claims.

## Почему кандидат отклонён

Первичный immutable audit по полной тройке `family/name/version` увидел `0` exact-hash matches
в `18,764` historical spoof rows, но это оказалось недостаточным. Его собственный legacy review
выявил `76` historical rows с тем же source model identifier
`tts_ru_free_hf_vits_low_multispeaker` и `generator_version=unspecified_by_source`.

Без historical commit и hash нельзя доказать, что candidate отличается от уже использованной
модели. Поэтому первоначальный claim `unseen_exact_generator_route` отозван, а receipt нельзя
считать pass:

- provisional audit SHA-256 `8b193c389f75396db1b54544fa46b6fb8fc11a77fea82c807f5b455c30d7d935`;
- immutable [reconciliation receipt](../data/manifests/voxforge_ru_mdc_utrobinmv_vits_female_exact_route_reconciliation_v1.json),
  SHA-256 `3a3eb9b6c5e5a7cb48d2e1b4f00c87b076e43a9642b7fb71fc8be87120a978f3`.

The audit code was changed to fail closed on this exact historical-name condition. The locked
model files stay local and ignored; they are retained only to make the rejected review
reproducible and must not be used as an alternative or backfill route.

## Следующий безопасный шаг

Find a new text-only Russian TTS candidate whose model identifier is absent from all historical
spoof manifests, then pin its source revision, license, files, fixed public voice and parameters
before any VoxForge WAV is decoded. Existing VITS-family evidence means a future candidate must
not claim architecture-family novelty unless stronger historical evidence supports it.
