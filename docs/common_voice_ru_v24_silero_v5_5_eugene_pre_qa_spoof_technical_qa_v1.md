# Common Voice RU v24 / Silero V5.5 `eugene` — spoof technical QA v1

**Статус:** raw synthesis layer fully accounted by normal technical QA. Acoustic review, binary
pairing и detector inference не начаты.

## Вход и результат

QA принял только `75` raw fixed-`eugene` WAV from
[synthesis manifest](../data/manifests/silero_v5_5_ru_eugene_pre_qa_raw_v1.csv), SHA-256
`82b220dbe4663d472a01d9e85e0d7f2309577fe772a0275a27dffeb7102460ae`, и проверил его prior
one-WAV-per-bound-text receipt before processing.

`AudioPreparationPipeline` decoded every raw WAV, normalized it to mono PCM WAV at 16 kHz,
applied quality checks and WebRTC VAD.

| Outcome | Rows |
| --- | ---: |
| Raw fixed-eugene spoof layer | 75 |
| Ready spoof WAV layer | 42 |
| Fully accounted rejects | 33 |

All 33 rejects are `insufficient_speech`. There were zero reused rows and no resynthesis,
replacement or backfill. `42/42` retained spoof sample IDs, text IDs and text hashes are unique;
the ready assets and both raw/ready manifests pass license-ledger and SHA-256 asset validation.

## Versioned outputs

- [ready spoof manifest](../data/manifests/silero_v5_5_ru_eugene_pre_qa_ready_v1.csv): `42`
  rows, SHA-256 `60df83502521fb225dbf8af44f80395c6493c9c4fc698a06fdb0c567b74a7036`;
- [generic rejection report](../data/manifests/silero_v5_5_ru_eugene_pre_qa_rejections_v1.json):
  SHA-256 `46a46887c3cf760c8ada215e40a8868b4290584a66e684f8daf447cca1b84f85`;
- [technical-QA receipt](../data/manifests/silero_v5_5_ru_eugene_pre_qa_technical_qa_v1.json):
  SHA-256 `e9c9fccbba13272cf537e5bad0e9ce70bd2dcd47fbd309bdb4469208453e46db`.

## Следующий безопасный шаг

The exact 42-pair candidate is now frozen in
[pairing receipt](common_voice_ru_v24_silero_v5_5_eugene_pre_qa_pairing_v1.md). It does not use
the remaining 33 bona fide-only rows or any old Stage-D/v3 pair as a substitute. Full-asset
acoustic review must independently pass for every retained bona-fide and spoof byte before any
new inference plan can be made.
