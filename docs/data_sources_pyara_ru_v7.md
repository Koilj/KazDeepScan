# PyAra v7 — Russian binary research protocol

## Локально проверенный archive

- Официальная dataset page: <https://www.kaggle.com/datasets/alep079/pyara/versions/7>.
- Artifact: `archive.zip`, локально: `/home/ruslan/Downloads/archive.zip`.
- Размер ZIP: `28 092 611 663` байта; SHA-256:
  `dadf5b795adbd6d635e74f4f9662c3e9a425c88bd76f26731f9e6adbad278b91`.
- Полная проверка `unzip -tqq` прошла 9 августа 2026 года.
- Dataset card version 7 указывает `CC-BY-NC-SA-4.0`, русский bona-fide/spoof WAV,
  annotation label и synthesis algorithm. Использование в этом проекте ограничено
  `owner_authorized_personal_research`; коммерческое использование запрещено лицензией.

Archive содержит ровно `201 778` WAV и `final_dataset.tsv`, без directory/symlink members:

- `73 583` `Real` / bona-fide rows;
- `128 195` `Fake` / spoof rows: `alg_1` (`46 505`), `alg_2` (`10 065`),
  `alg_3` (`11 165`), `alg_4` (`25 412`), `alg_5` (`35 048`);
- TSV и ZIP paths полностью совпадают; text и duration заполнены для всех строк.

## Критичное ограничение split

PyAra не содержит проверяемого speaker ID. Intake не подменяет его возрастом/полом и не
выдаёт local split за speaker-disjoint. `speaker_pseudo_id` содержит только opaque
source-record key, а `text_hash` строится из нормализованного transcript: одинаковый текст
никогда не оказывается в разных split. Это **text-leakage-safe research protocol**, но не
speaker-independent benchmark и не основание для product scoring или calibration.

## Published slices

Raw slice `data/manifests/pyara_ru_v7_research_500.csv` выбирается детерминированно:
`250` bona-fide и по `50` spoof rows от каждого из пяти algorithms (`500` WAV всего).
После local text-group split: train/dev/test `392/63/45`.

После decode, QA и WebRTC VAD опубликован ready manifest
`data/manifests/pyara_ru_v7_research_500_ready.csv`:

- `481` mono PCM S16LE WAV, `16 000` Гц;
- исключены `19` raw rows: `18` `insufficient_speech`, `1` `signal_too_quiet`;
- все исключения записаны в `data/manifests/pyara_ru_v7_research_500_rejections.json`;
- ready split: train/dev/test `376/61/44`; labels `232` bona-fide / `249` spoof.

`kds validate-manifest --license-ledger` и `kds validate-assets` прошли для raw и ready
manifest-ов.

## B0 research smoke baseline

На RTX 5060 Ti обучен `models/b0-pyara-research-500.pt`: 3 эпохи, лучший dev loss `0.5084`.
Отдельная holdout test проверка на `44` rows дала loss `0.4053`, accuracy `0.8864`.
Это только проверка полного data/training/evaluation контура. Малый test, отсутствие
speaker-disjoint guarantee и некоммерческая лицензия запрещают использовать эти веса или
метрики как product release, external benchmark либо calibration basis.

## Воспроизведение

Все output paths должны быть новыми:

```bash
uv run python scripts/ingest_pyara_ru_v7.py \
  --archive /home/ruslan/Downloads/archive.zip \
  --output-manifest data/manifests/pyara_repro.csv \
  --slice-name research-repro

KDS_FFMPEG_BINARY="$PWD/.tools/ffmpeg/bin/ffmpeg" \
KDS_FFPROBE_BINARY="$PWD/.tools/ffmpeg/bin/ffprobe" \
uv run python scripts/preprocess_manifest.py \
  --input-manifest data/manifests/pyara_repro.csv \
  --output-manifest data/manifests/pyara_repro_ready.csv \
  --rejection-report data/manifests/pyara_repro_rejections.json \
  --allow-rejections --license-ledger data/licenses/license_ledger.csv --data-root data
```

Archive, extracted audio and B0 checkpoint are not committed to Git.
