# Новые кандидаты для independent Stage-B final

> **Исторический review.** Stage-B/Stage-C routes из этого документа уже выполнены; его source
> exclusions (включая RuASD upstream) остаются в силе. Текущая классификация будущих independent,
> external-holdout и same-family layers находятся в
> [review от 14 августа](external_holdout_policy_and_voxcpm2_candidates_2026-08-14.md). Denis
> source intake и official VoxCPM2 artifact/source/history/runtime/smoke gates завершены. Frozen
> Denis selection зафиксировал 79 rows, а bona-fide QA/VAD оставил minimum layer `64` без
> backfill. Последующий 64-row binding/one-shot synthesis дал `64/64` raw, но synthetic QA/VAD
> оставил `53`, поэтому exact route завершён `stop_below_minimum_60` без pairing или inference.

Проверка проведена 10–11 августа 2026 года. В этот список включён только источник,
который не является частью RuASD/KSC2, публикует исходное аудио и транскрипты с
понятной лицензией и может дать bona-fide половину будущих отдельных `ru` и `kk`
final-слоёв. Это **не** готовый общий RU/KK/mixed binary anti-spoof dataset и не
разрешение запускать final: RU и KK spoof halves теперь созданы разрешённой text-only
TTS-family, но проверяемый mixed layer всё ещё отсутствует.

## Google FLEURS — исходное аудио

- Официальная карточка: <https://huggingface.co/datasets/google/fleurs>.
- Закреплённая ревизия: [`4683b04af03d2d9549064c7d72060a9a94bb6046`](https://huggingface.co/datasets/google/fleurs/tree/4683b04af03d2d9549064c7d72060a9a94bb6046/data).
- Лицензия: `CC-BY-4.0`.
- Это не производная RuASD: FLEURS — отдельный Google benchmark, построенный на
  FLoRes; в нём есть самостоятельные `ru_ru` и `kk_kz` конфигурации. Внутри FLEURS
  train-speakers отделены от dev/test-speakers, но публичного speaker ID нет.
- Исходные данные — bona-fide read speech, 16 kHz. Их допустимо рассматривать для
  `ru` и `kk`, но не для `mixed` final.
- Объём полного `kk_kz`: `3,861,486,625` bytes, `4,425` записей (train 3,200 /
  validation 369 / test 856). Полный `ru_ru`: `2,636,505,060` bytes, `3,693`
  записей (train 2,562 / validation 356 / test 775).

Скачивать следует в разные каталоги, например
`/home/ruslan/Downloads/FLEURS/kk_kz/` и `/home/ruslan/Downloads/FLEURS/ru_ru/`.
Нужны все три audio archive и все три TSV для каждого языка; ничего не
распаковывать и не менять до read-only audit.

### Казахский (`kk_kz`)

- [train audio](https://huggingface.co/datasets/google/fleurs/resolve/4683b04af03d2d9549064c7d72060a9a94bb6046/data/kk_kz/audio/train.tar.gz?download=true)
- [validation audio](https://huggingface.co/datasets/google/fleurs/resolve/4683b04af03d2d9549064c7d72060a9a94bb6046/data/kk_kz/audio/dev.tar.gz?download=true)
- [test audio](https://huggingface.co/datasets/google/fleurs/resolve/4683b04af03d2d9549064c7d72060a9a94bb6046/data/kk_kz/audio/test.tar.gz?download=true)
- [train metadata](https://huggingface.co/datasets/google/fleurs/resolve/4683b04af03d2d9549064c7d72060a9a94bb6046/data/kk_kz/train.tsv?download=true)
- [validation metadata](https://huggingface.co/datasets/google/fleurs/resolve/4683b04af03d2d9549064c7d72060a9a94bb6046/data/kk_kz/dev.tsv?download=true)
- [test metadata](https://huggingface.co/datasets/google/fleurs/resolve/4683b04af03d2d9549064c7d72060a9a94bb6046/data/kk_kz/test.tsv?download=true)

### Русский (`ru_ru`)

- [train audio](https://huggingface.co/datasets/google/fleurs/resolve/4683b04af03d2d9549064c7d72060a9a94bb6046/data/ru_ru/audio/train.tar.gz?download=true)
- [validation audio](https://huggingface.co/datasets/google/fleurs/resolve/4683b04af03d2d9549064c7d72060a9a94bb6046/data/ru_ru/audio/dev.tar.gz?download=true)
- [test audio](https://huggingface.co/datasets/google/fleurs/resolve/4683b04af03d2d9549064c7d72060a9a94bb6046/data/ru_ru/audio/test.tar.gz?download=true)
- [train metadata](https://huggingface.co/datasets/google/fleurs/resolve/4683b04af03d2d9549064c7d72060a9a94bb6046/data/ru_ru/train.tsv?download=true)
- [validation metadata](https://huggingface.co/datasets/google/fleurs/resolve/4683b04af03d2d9549064c7d72060a9a94bb6046/data/ru_ru/dev.tsv?download=true)
- [test metadata](https://huggingface.co/datasets/google/fleurs/resolve/4683b04af03d2d9549064c7d72060a9a94bb6046/data/ru_ru/test.tsv?download=true)

## Почему остальные найденные наборы не добавлены

- `issai/Multilingual_Speech_Dataset` содержит русский OpenSTT; OpenSTT уже входит
  в full RuASD. `GOLOS`, `SOVA`, `RUSLAN`, `RuLS` и `M-AILABS` также уже присутствуют
  в RuASD и не годятся как independent final source.
- `issai/KazakhTTS`, `issai/kazakh-speech-commands` и
  `Flamme-VRM/kazakh-speech-dataset` используют KSC/KazakhTTS/KSC2 либо synthetic
  audio. Это не независимое дополнение к скачиваемому KSC2.
- `TilQazyna/Til-Audio` gated, содержит 250 GB смешанных upstream-источников и
  machine-ASR transcripts; без первичного provenance/license audit не годится.
- Common Voice Spontaneous Russian 4.0 технически чистый и малый (442 validated
  clips), но использует ту же платформу Common Voice, которая уже представлена в
  RuASD. У старого RuASD нет сопоставимых client IDs, поэтому speaker overlap нельзя
  строго исключить.
- LRLspoof по-прежнему исключён: казахская часть spoof-only, а архив примерно 452 GB
  и не поддерживает selective Kazakh download.

## Intake completed

Владелец вручную предоставил full release в `/home/ruslan/Downloads/FLEURS/data/`.
Read-only audit завершён: exact size, pinned LFS/Git object identities, gzip CRC и
TAR/TSV membership прошли для обеих локалей. Источники внесены в ledger только после
этой проверки. Детерминированный 300-text raw slice и QA/VAD-ready manifests созданы:
`289` RU и `212` KK bona-fide rows.

Pinned Silero V4 fixed-profile synthesis сформировал strict paired candidate layers:
`214` RU pairs (`428` assets) и `152` KK pairs (`304` assets). Их manifests проверены
с ledger и SHA-256 assets, а builder исключил overlaps со Stage-B train/dev, calibration
и прежними frozen finals. Подробные receipts, hashes и ограничения —
`docs/data_sources_fleurs_2026-08-10.md` и
`docs/data_sources_silero_v4_2026-08-11.md`. Не fit calibration и не запускать
final inference, пока не появится independent, проверяемый mixed layer.
