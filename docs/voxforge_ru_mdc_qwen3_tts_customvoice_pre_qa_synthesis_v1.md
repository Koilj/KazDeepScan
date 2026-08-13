# VoxForge RU / Qwen3-TTS CustomVoice `aiden` — pre-QA synthesis v1

**Статус:** completed one-shot raw synthetic layer; normal technical QA has also completed.
Pairing, full acoustic/language review and detector inference remain prohibited.

## Входы и зафиксированный route

Run принял ровно `79` frozen ready VoxForge bona-fide rows из
[base manifest](../data/manifests/voxforge_ru_mdc_2026_05_pre_qa_ready_v1.csv), SHA-256
`001eba24021673327a17b7fc94418b334c902007cdee507ddba6a17a06f660e4`, и `79/79` literal-text
bindings. До первого model call повторно проверены archive identity, binding, exact-route audit,
artifact/model locks, license ledger и every base asset. Source text перечитан только из
byte-pinned local archive; transcript не записывается в synthesis receipt.

Каждая строка получила ровно одну CUDA-попытку через локальный fixed
`qwen3-tts-customvoice:aiden` route: literal UTF-8 text, recorded deterministic seed,
`--target-lang ru`, `temperature=0.9` и `max_new_tokens=512`. Reference audio, cloning,
VoiceDesign, внешний normalizer/stress model, auto-download, reselection, replacement и
backfill не использовались. English-описание baked `aiden` token по-прежнему не даёт
Russian-native, speaker, speaker-independence или architecture-independence claim.

## Результат

Все `79/79` попыток успешно создали отдельный non-empty mono 24 kHz WAV; failed attempts: `0`.
Raw layer содержит `513.44` s audio (`3.28`–`14.80` s per row), с unique spoof sample IDs,
text IDs и text hashes. Сохранён один fixed profile и `79` matching literal-text seeds.

- [raw spoof manifest](../data/manifests/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_pre_qa_raw_v1.csv):
  `79` rows, SHA-256 `d378466b47dc4e05e575457ebdda5677ad685465df7d70cc6a9c2d7207f0f990`;
- [one-shot synthesis receipt](../data/manifests/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_pre_qa_synthesis_v1.json):
  SHA-256 `dd9414b215ea9203454ea651cd781acb8f9b7b8def9df7b53155f4b373e477d5`.

Raw WAVs stay only in ignored
`data/raw/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_v1/slices/pre_qa`; model and audio bytes
are not added to Git.

## Следующий безопасный шаг

Normal technical QA retained all `79` rows and the exact `79`-pair lock is now published in the
[pairing receipt](voxforge_ru_mdc_qwen3_tts_customvoice_pre_qa_pairing_v1.md). No row may be
resynthesized, replaced or backfilled; full two-review acoustic/language review must now pass
before detector inference can be considered.
