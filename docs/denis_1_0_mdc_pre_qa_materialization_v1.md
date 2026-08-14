# Denis 1.0 — frozen metadata selection и bona-fide QA/VAD v1

**Статус:** completed write-once metadata selection и source-bound technical materialization.
Это minimum `64`-row bona-fide layer, не target-`79` success, не paired candidate, не acoustic
review и не detector result.

## Current pre-selection exposure gate

Первый Denis exposure receipt был корректен на момент intake. До selection в project configs
появился завершённый official VoxCPM2 lock, поэтому selector fail closed отклонил устаревший
inventory binding. Новый immutable
[`source_exposure_screen_v2`](../data/manifests/denis_1_0_mdc_source_exposure_screen_v2.json)
добавил только этот config: `35` configs, те же `19` configured manifests / `12,555` rows и
полный inventory `95` manifests / `40,682` rows. Direct sample/audio/three-text-hash overlap
остался `0`; historical `ru_RU-denis-medium` lineage осталась `12` unique samples (`11` train,
`1` dev). Receipt SHA-256:
`d140918a60d437f41d209b57803058179bb1d8cfd7ae8e7db217788d0b9841cb`.

## Frozen metadata selection

Seed `2026-08-14-denis-1-0-mdc-pre-qa-candidate-v1` заморозил ровно `79` source identities.
Это полный target, не скрытый резерв. Selector сначала seeded-ranks три source categories, затем
seeded-ranks rows внутри каждой category и выдаёт их round-robin. Итоговое распределение:

| Category | Frozen rows |
| --- | ---: |
| General | `27` |
| Chat | `26` |
| CustomerService | `26` |

Ranking не использует source duration, waveform, QA/VAD, detector/model outputs, metrics или
historical final errors. Все `79` sample IDs, exact source-audio hashes и literal,
whitespace-collapse, NFKC+whitespace text hashes уникальны и связаны до extraction. Все rows
намеренно сохраняют одну group `denis_1_0_mdc:speaker:single`; category balance не превращает
одного человека в speaker diversity.

- [selection CSV](../data/manifests/denis_1_0_mdc_pre_qa_selection_v1.csv): SHA-256
  `04f927ac2e85d27c098e931ff08bdd9a0c086aa24e599611a687d695e60418f7`;
- [selection receipt](../data/manifests/denis_1_0_mdc_pre_qa_selection_v1.json): SHA-256
  `5e9dd93290eece14f738cab06e665d61a47d0e79cb5e1730198574471b2fc37c`.

## Exact extraction и normal technical QA

Materializer повторно проверил exact archive `109,594,943` bytes / SHA-256
`75e2c63c5082df7623c6a98c529718b22015dfbd2d38a1ea328635f4dd4ccf9b`, selection и оба
parent receipts. Из TAR извлечены только `79/79` frozen `.webm` members. Raw local filenames
сохраняют source suffix, а manifest явно сообщает фактический Ogg/Opus payload; raw audio
остаётся в ignored `data/raw/` и не добавляется в Git.

Обычный `AudioPreparationPipeline` выполнил ffprobe/ffmpeg decode, mono PCM-16 16 kHz,
waveform QA и WebRTC VAD с minimum speech `2.5` s:

| Результат | Rows |
| --- | ---: |
| Raw exact source rows | `79` |
| Ready normalized WAV | `64` |
| `insufficient_speech` rejects | `15` |
| Reuse / replacement / backfill | `0 / false / false` |

Ready by category: General `23`, Chat `17`, CustomerService `24`. Rejections by category:
`4 / 9 / 2`; все `15` перечислены по sample ID в receipt. Target `79` не достигнут, но
predeclared minimum `60` пройден. Оставшиеся `1,071` source rows не являются резервом: новый
selection, replacement или backfill по QA outcome запрещены.

- [raw manifest](../data/manifests/denis_1_0_mdc_pre_qa_raw_v1.csv): `79` rows, SHA-256
  `055be5875ad590b71be850308f73b813da9bd802156a2ae76c2125375dc4ed20`;
- [ready manifest](../data/manifests/denis_1_0_mdc_pre_qa_ready_v1.csv): `64` rows, SHA-256
  `ae71cca7dcc2854cccca67565f81a0696acc665a6648b9889f5d9abd267891d8`;
- [materialization receipt](../data/manifests/denis_1_0_mdc_pre_qa_materialization_v1.json):
  SHA-256 `c36fc8bcc60c16d5d2493c4bf8b77719f32ca3d9da9ba15d51054b9ee16d5386`.

## Ограничения и следующий gate

Сохраняется только следующая маркировка:

> external human-source / generator-family holdout candidate; TTS training-data overlap
> unverified; likely historical Denis speaker-lineage exposure; single-speaker; not
> speaker-independent or speaker-robust; personal research only.

Generator-family часть пока только route eligibility: synthetic half ещё не создана. Следующий
безопасный этап — отдельный immutable 64-row VoxCPM2 literal/canonical text-binding и one-shot
synthesis contract с exact ready manifest, model/runtime hashes, seed/parameters и запретом
reference/prompt audio, normalizer, denoiser, retry, replacement и backfill. До такого contract
candidate synthesis не выполнять; detector inference остаётся запрещён до synthetic QA, exact
pair lock, двух независимых full-asset acoustic reviews, current exposure audit и отдельного
write-once evaluation contract.
