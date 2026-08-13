# Common Voice RU v24 / Silero V5.5 `eugene` — immutable pre-QA selection v1

**Статус:** selection завершён; это ещё не extraction, audio QA, synthesis, acoustic review,
pairing или detector inference.

## Решение о размере

Заморожено `80` записей — ровно по одной на отдельную Common Voice `client` group. Это
консервативный pre-QA buffer относительно historical `55` Stage-D/v3 pairs, но не прогноз
yield после decode/QA/VAD или acoustic review. Размер не был выбран по logits, metrics или final
errors.

## Входы и воспроизводимое правило

Pinned Common Voice archive `/home/ruslan/Downloads/cv-corpus-24.0-2025-12-05-ru.tar.gz`
проверен до metadata read: `7,008,716,262` bytes, SHA-256
`9a2ed32a0574f74f505cd7740a599f0b9edc9f52ba1e7d6624b66f258db4c0ea`.

Selection заново проверил точное соответствие обоим parent receipts текущим pinned
archive/config/manifest inputs:

- metadata exposure screen:
  [`data/manifests/common_voice_ru_v24_full_test_metadata_exposure_screen_v1.json`](../data/manifests/common_voice_ru_v24_full_test_metadata_exposure_screen_v1.json),
  SHA-256 `f862ae667195c733c7deb6bf25f304a6287890ca87d4dc0ee7cb5e06aa6f46b3`;
- V5.5 literal-text screen:
  [`data/manifests/common_voice_ru_v24_full_test_silero_v5_5_literal_text_screen_v1.json`](../data/manifests/common_voice_ru_v24_full_test_silero_v5_5_literal_text_screen_v1.json),
  SHA-256 `4356c3ecbf3a9b68dd7a5d5f4e2ed9347d9c6f105d63d558bfc03dd1403b23d0`;
- fixed model lock:
  [`configs/research/silero_v5_5_ru_eugene_v1_models.json`](../configs/research/silero_v5_5_ru_eugene_v1_models.json),
  SHA-256 `39fc9f4748286593ff39fe51688212215c312b4b1d880dc9539e7f43d9ce8edd`.

Из `5,600` literal-text-compatible records в `1,337` surviving client groups selector с seed
`2026-08-13-silero-v5-5-eugene-pre-qa-candidate-v1` сначала ранжирует client groups по
SHA-256, затем независимо ранжирует records внутри каждой выбранной group. Он берёт первые `80`
groups и только один record из каждой. Ties детерминированно разрешаются по идентификатору.
Правило не смотрит на audio/duration, detector/model output, metrics или final errors.

## Неизменяемые outputs

- selection CSV:
  [`data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_selection_v1.csv`](../data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_selection_v1.csv),
  `80` rows, SHA-256 `73eaf22706419b275517500ebb25973510e8dcccaa94a54f45b4fe2a787f6b50`;
- write-once receipt:
  [`data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_selection_v1.json`](../data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_selection_v1.json),
  SHA-256 `a7f0a1b5c3a152c87692e5e9d7d4ac2e02b2b64d9b6933f7f79f29b4e6b6d7ad`.

Read-only verification confirms `80/80` unique sample IDs, `80/80` unique client groups and
`80/80` unique text hashes. The CSV stores only the exact clip name and metadata identities/hashes;
the literal transcript remains resolvable only from the pinned archive during the later binding
step.

## Ограничения и следующий безопасный шаг

At the time of selection, no MP3 was extracted, no WAV was created, no synthetic audio was
generated, and no QA, review or detector inference had happened. The historical `55` Stage-D/v3
pairs, historical `73` selection and its `18` QA rejects are not a reserve and cannot be used for
backfill.

The subsequent [materialization receipt](common_voice_ru_v24_silero_v5_5_eugene_pre_qa_materialization_v1.md)
accounts for all 80 frozen MP3s: 75 reached ready WAV and five were rejected for
`insufficient_speech`, with no replacement or backfill. The following
[literal-text binding receipt](common_voice_ru_v24_silero_v5_5_eugene_pre_qa_text_binding_v1.md)
verified the exact source text of all 75 ready rows without rewrite. The subsequent
[synthesis receipt](common_voice_ru_v24_silero_v5_5_eugene_pre_qa_synthesis_v1.md) stores one
raw fixed-profile WAV per bound text. The subsequent
[technical-QA receipt](common_voice_ru_v24_silero_v5_5_eugene_pre_qa_spoof_technical_qa_v1.md)
retained `42` and accounted `33` rejects. The matching
[42-pair lock](common_voice_ru_v24_silero_v5_5_eugene_pre_qa_pairing_v1.md) is frozen and passed
its subsequent technical acoustic gate, immutable contract and exactly one detector-inference run.
Its original selection cannot be changed, replaced or tuned against final errors.
