# VoxForge RU / Qwen3-TTS CustomVoice `aiden` — literal-text binding v1

**Статус:** completed write-once text binding. The downstream one-shot synthesis, technical QA,
exact pairing, `158/158` acoustic review gate and exactly one governed detector run are complete.

## Immutable inputs

Only the `79` ready bona-fide rows from the completed 81-row materialization were admitted. The binding rechecked the ready/raw manifests, materialization receipt, frozen selection CSV/receipt, accepted Qwen route audit, artifact lock, model lock and exact VoxForge archive before source metadata read. It confirms the 79 rows are an unchanged subset of the original 81 selections; the two `signal_too_quiet` rejects remain absent and cannot be replaced.

All `79/79` rows re-bound to the same source prompt, contributor-group and both transcript hash layers. The binding stores no transcript: only hashes, byte length and the deterministic seed derived from the literal UTF-8 source text. Byte lengths range from `73` to `334`, below the fixed 4,096-byte runtime limit. `PROMPTS` whitespace canonicalization is the source parser's sole transformation; no external normalizer, stress model, text replacement, selection change, audio, detector or metric entered this gate.

Receipt: [text binding JSON](../data/manifests/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_pre_qa_text_binding_v1.json), SHA-256 `cedd26e3149bc5212772ef1699a180a489fac5c59e89f65bb299f229ee965d4a`; deterministic binding digest `fb2cef2bb3d2fe8628885ddef16c4acc5eed7e940f9d2aec77c76d2cf9ed5581`.

## Follow-on state

The [non-candidate CUDA smoke](voxforge_ru_mdc_qwen3_tts_customvoice_cuda_smoke_v1.md), the
one-shot [synthesis receipt](voxforge_ru_mdc_qwen3_tts_customvoice_pre_qa_synthesis_v1.md) and
its [technical QA receipt](voxforge_ru_mdc_qwen3_tts_customvoice_pre_qa_technical_qa_v1.md) now
account for all `79` bound rows, and the [exact pair lock](voxforge_ru_mdc_qwen3_tts_customvoice_pre_qa_pairing_v1.md)
contains `79` pairs. The subsequent
[two-review gate](voxforge_ru_mdc_qwen3_tts_customvoice_pre_qa_acoustic_gate_v1.md) passed all
`158` exact assets. No regeneration, replacement or backfill is permitted. The mandatory
exposure/contract/preflight steps subsequently passed and exactly one inference run completed; see
the [final receipt](research_xlsr_sls_stage_b_v2_voxforge_ru_mdc_qwen3_tts_customvoice_aiden_v1.md).
Repeat inference and final-error tuning are prohibited.
