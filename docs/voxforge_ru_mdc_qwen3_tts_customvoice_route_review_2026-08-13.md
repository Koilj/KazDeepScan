# VoxForge RU — Qwen3-TTS CustomVoice / Aiden route review, 13 августа 2026

**Статус:** accepted exact text-only spoof route. Этот admission не создавал WAV и не
разрешает detector inference, pairing или какую-либо product/commercial use.

## Зафиксированный маршрут

Принят локальный CUDA route: Qwen CustomVoice `0.6B` Q8_0 GGUF, отдельный 12 Hz tokenizer
GGUF и CrispASR `v0.8.28`. Весь operational bundle состоит из шести файлов с суммарным размером
`1,610,363,823` bytes и проходит local size/SHA-256 verification перед каждым обращением к
весам. Runtime safely extracts только allow-listed files из pinned archive, проверяет каждый
inner hash, требует project CUDA-12 `cudart`/`cublas` libraries и проверяет наличие
`qwen3-tts-customvoice` backend до synthesis.

- [model lock](../configs/research/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_v1_models.json),
  SHA-256 `6ec1a828ad72738341a3226e73cde65ca9a87b3ee11fac43f2ca40173dd98d8d`;
- [artifact lock](../data/licenses/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_v1_artifact_lock.json),
  SHA-256 `94a8d1515b0062e5382202dc723fef474ec47bdf09c9d353cfa02bb1d5569855`;
- local wrapper: [qwen3_tts_customvoice.py](../src/kds/data/qwen3_tts_customvoice.py),
  SHA-256 `f02d93fd7ee064539771a868c5033bc163bec1c94c838eb1152c52eb7eb677b7`.

The conversion and tokenizer cards declare Apache-2.0 and point to the Apache-2.0 Qwen
upstreams. The official CustomVoice card declares Russian support; the selected baked `aiden`
token itself is described as English. Therefore this route makes no Russian-native-voice,
verified-person, speaker-group or speaker-independence claim. The required full acoustic/language
review remains a real gate, not a formality.

## Fail-closed controls

Only literal non-empty source UTF-8 text can reach the command. It fixes `--backend
qwen3-tts-customvoice`, local talker/codec paths, `--voice aiden`, `--target-lang ru`, CUDA,
`temperature=0.9` and `max_new_tokens=512`; the deterministic seed is derived from that exact
literal text. Reference WAV, `--ref-text`, VoiceDesign `--instruct`, random profiles, external
normalizer/stress models and runtime auto-download are not exposed.

No VoxForge WAV was opened, no synthetic WAV was written and no detector inference was run during
artifact or route verification.

## Historical route gate

The immutable [exact-route audit](../data/manifests/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_exact_route_audit_v1.json)
SHA-256 `234e1a49ecc06f4fc7025c4e713af858279d4c35b1d366b4ab052d7a847f5513` scanned `59`
manifests with `18,764` historical spoof rows:

- exact family/name/version overlap: `0`;
- legacy Qwen CustomVoice identifier overlap: `0`;
- fixed `aiden` / fully qualified alias overlap: `0 / 0`;
- family overlap: `0`.

This proves only that the exact pinned route was absent from stored historical manifests. It does
not prove architecture-family novelty or speaker independence; those claims remain forbidden.
The previously rejected UtrobinTTS route remains rejected and cannot be used as a backfill.

## Следующий безопасный шаг

The already frozen `81` pseudonymous selections have been materialized from the byte-pinned archive
and technical QA retained `79`; two quiet rejects are accounted with no replacement. The `79`
literal texts are bound. Exactly one synthesis per bound row is next; pairing, acoustic review and
detector inference stay prohibited until their own immutable receipts pass.
