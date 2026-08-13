# VoxForge RU / Qwen3-TTS CustomVoice `aiden` — literal-text binding v1

**Статус:** completed write-once text binding. No synthetic WAV, pairing, acoustic review or detector inference was performed.

## Immutable inputs

Only the `79` ready bona-fide rows from the completed 81-row materialization were admitted. The binding rechecked the ready/raw manifests, materialization receipt, frozen selection CSV/receipt, accepted Qwen route audit, artifact lock, model lock and exact VoxForge archive before source metadata read. It confirms the 79 rows are an unchanged subset of the original 81 selections; the two `signal_too_quiet` rejects remain absent and cannot be replaced.

All `79/79` rows re-bound to the same source prompt, contributor-group and both transcript hash layers. The binding stores no transcript: only hashes, byte length and the deterministic seed derived from the literal UTF-8 source text. Byte lengths range from `73` to `334`, below the fixed 4,096-byte runtime limit. `PROMPTS` whitespace canonicalization is the source parser's sole transformation; no external normalizer, stress model, text replacement, selection change, audio, detector or metric entered this gate.

Receipt: [text binding JSON](../data/manifests/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_pre_qa_text_binding_v1.json), SHA-256 `cedd26e3149bc5212772ef1699a180a489fac5c59e89f65bb299f229ee965d4a`; deterministic binding digest `fb2cef2bb3d2fe8628885ddef16c4acc5eed7e940f9d2aec77c76d2cf9ed5581`.

## Следующий безопасный шаг

Generate exactly one local Qwen3-TTS CustomVoice / fixed `aiden` WAV per bound row, using the locked local talker, codec, CUDA runtime, literal text and recorded deterministic seed. A failed synthesis or technical-QA row permanently reduces the set: no regeneration, replacement or backfill. Pairing, acoustic/language review and detector inference remain separately prohibited.
