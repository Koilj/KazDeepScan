# Common Voice RU v24 / Silero V5.5 `eugene` — pre-QA materialization v1

**Статус:** extraction и technical decode/QA/VAD завершены. Это ещё не synthetic generation,
acoustic review, binary pairing или detector inference.

## Вход и неизменяемая граница

Materialization принял только frozen 80-row selection:

- [selection CSV](../data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_selection_v1.csv),
  SHA-256 `73eaf22706419b275517500ebb25973510e8dcccaa94a54f45b4fe2a787f6b50`;
- [selection receipt](../data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_selection_v1.json),
  SHA-256 `a7f0a1b5c3a152c87692e5e9d7d4ac2e02b2b64d9b6933f7f79f29b4e6b6d7ad`.

Before reading metadata and extraction, the local archive
`/home/ruslan/Downloads/cv-corpus-24.0-2025-12-05-ru.tar.gz` was rechecked at
`7,008,716,262` bytes and SHA-256
`9a2ed32a0574f74f505cd7740a599f0b9edc9f52ba1e7d6624b66f258db4c0ea`.
The materializer revalidated the selection CSV hash, all three parent input hashes, and every
selected `sample_id`, `clip_name`, test split, client group, sentence ID and text hash against
the pinned archive TSV before extracting audio.

## Результат technical QA

Exactly `80` frozen MP3 files were extracted locally (raw audio remains ignored by Git). The
normal `AudioPreparationPipeline` decoded each asset, normalized it to mono PCM WAV at 16 kHz,
checked audio quality and ran WebRTC VAD.

| Outcome | Rows |
| --- | ---: |
| Frozen raw Common Voice records | 80 |
| Ready 16 kHz WAV records | 75 |
| Accounted rejects | 5 |

Every rejection is `insufficient_speech`; no assets were reused, replaced, reselected or
backfilled. Raw selection had `80/80` unique sample IDs, client groups and text hashes; ready
rows preserve `75/75` uniqueness.

## Versioned outputs

- [raw manifest](../data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_raw_v1.csv):
  `80` rows, SHA-256 `5543e1e88b688cb79b4401a7ef68ba525d75321cbe98a4837bf7347867a2a9a5`;
- [ready manifest](../data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_ready_v1.csv):
  `75` rows, SHA-256 `2b183adbfcac9b1a6022dd35c2f8b6ec8f111c01b4b3364596c53aff8906192a`;
- [materialization receipt](../data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_materialization_v1.json):
  SHA-256 `da5f4a79cf0444d0f2e905c4f178760967306d3a393753736e72cc4a27a0da3e`.

The receipt records each rejected sample and binds the raw/ready manifests, selection receipt
and archive. It explicitly records zero synthetic assets, zero acoustic reviews and zero detector
inferences.

## Следующий безопасный шаг

Bind the exact literal archive texts for only the 75 ready rows, then create exactly one
fixed-profile V5.5/eugene synthetic WAV per bound text. The five rejected rows are permanently
out of this candidate: no replacement, reselection, regeneration or backfill is authorized.
The later full-asset acoustic reviews must cover the exact bona-fide and synthetic bytes before
any paired evaluation plan can be prepared.
