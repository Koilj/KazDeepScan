# Common Voice RU v24 / Silero V5.5 `eugene` — pre-QA synthesis v1

**Статус:** completed raw synthetic layer and technical decode/QA/VAD accounting; acoustic review,
binary pairing и detector inference ещё не выполнены.

## Входы и строгий route

Run принял ровно `75` ready bona fide rows from
[base manifest](../data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_ready_v1.csv),
literal-text binding `75/75`, exact-route audit и pinned Common Voice archive. До model load
повторно верифицированы оба local bundle artifacts: `v5_5_ru.pt` SHA-256
`50081637b602126ee06cb3bc8a744d25651d2da149ee8864b9a379bfdd934437` и `source.tar.gz` SHA-256
`cca6d3e6e34e03f9fe30c4e33ee2de8e89aa384f95bc0f3143c51af7a72765aa`.

Единственный permitted call route — local CPU, built-in fixed `eugene`, literal source text,
48 kHz, no reference audio, cloning, random profile, SSML, `voice_path` или external
normalizer. CC-BY-NC-SA-4.0 ограничивает результат personal-research scope; product, training,
calibration и detector inference не разрешены.

## Результат

Сохранено ровно `75` mono PCM-16 48 kHz synthetic WAV — один на каждый bound text. All `75/75`
spoof sample IDs, text IDs и text hashes unique; каждый raw asset прошёл SHA-256 asset check и
license-ledger validation. Нет replacement, backfill либо reuse of old Stage-D/v3 pairs.

- [raw spoof manifest](../data/manifests/silero_v5_5_ru_eugene_pre_qa_raw_v1.csv): `75` rows,
  SHA-256 `82b220dbe4663d472a01d9e85e0d7f2309577fe772a0275a27dffeb7102460ae`;
- [synthesis receipt](../data/manifests/silero_v5_5_ru_eugene_pre_qa_synthesis_v1.json):
  SHA-256 `61f5e81712c602fc8ee857f008527878e77a7ccf1e73af277bf3d803c75e9ab6`.

Raw WAVs live only in ignored
`data/raw/silero_v5_5_ru_eugene_v1/slices/common_voice_ru_v24_pre_qa_v1`; no raw audio or model
byte is added to Git.

## Output-write correction

Before this completed write-once run, an initial writer invocation stopped before creating any
WAV byte, manifest or receipt because its temporary filename did not retain a `.wav` suffix.
The output directory was verified empty and removed; the writer now refuses non-WAV temporary
names. The completed receipt above is the sole persisted synthesis result: one saved WAV per
bound text. This correction did not select, replace, backfill, QA or review any candidate row.

## Следующий безопасный шаг

Normal QA завершён в
[separate receipt](common_voice_ru_v24_silero_v5_5_eugene_pre_qa_spoof_technical_qa_v1.md):
`42` spoof rows retained and `33` insufficient-speech rejects are accounted. A failed synthetic
QA row cannot be resynthesized, replaced или backfilled. Только после immutable 42-pair lock и
two independent full-asset acoustic reviews можно готовить новый governed inference plan.
