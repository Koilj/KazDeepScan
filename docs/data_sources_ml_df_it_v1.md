# ML-DF v1 — Italian cross-lingual OOD

## Назначение и границы

- Официальный record: <https://zenodo.org/records/17098081>.
- Artifact: `dataset_IT.7z`; опубликованный размер: `1 485 098 719` байт;
  опубликованный MD5: `c3ce93f9566605e0a5ad2e3cda099d7d`.
- Отдельный официальный `metadata.zip` имеет MD5
  `25cc69e8d9234a22c1f38222e0bfdebf`.
- Лицензия record: `CC-BY-4.0`; в record указано происхождение от MLS с `CC-BY-4.0`.

Итальянская разметка содержит `8 000` bona-fide записей и по `2 000` записей для
`VITS`, `ZMM-TTS`, `LVC-VC`, `DDDM-VC`. В ней только два source speaker pseudo-ID.
Поэтому этот источник **нельзя** использовать для Russian/Kazakh B0 train/dev/test или
выдавать за speaker-disjoint target-language evaluation. Его допустимая роль — только
изолированный cross-lingual OOD slice с `split=ood` и `language=other`.

## Безопасный intake

До публикации любого slice скрипт:

1. проверяет размер и MD5 обоих official artifacts;
2. требует точного списка файлов из metadata и полное совпадение с members 7z archive;
3. проверяет CRC, точный распакованный размер release (`2 290 807 586` байт) и отказывается
   от symlink/неожиданных путей;
4. извлекает в staging directory только выбранные WAV. Поскольку archive solid-сжатый,
   декодер безопасно ограничен точным размером проверенного release, а сам выбранный OOD slice
   ограничен 2 GiB;
5. проверяет, что каждый WAV имеет `16 000` Гц, фиксирует SHA-256 и строит manifest,
   где fake rows содержат family/name/version/voice provenance.

Транскрипт не распространяется. `text_id` содержит только source content pseudo-ID, а не
текст; он нужен для консервативного контроля повторов content.

## Планируемый малый slice

Детерминированный initial slice опубликован как
`data/manifests/ml_df_it_v1_ood_200.csv`: `100` bona-fide плюс `25` записей на каждый из
четырёх fake generators. Это `200` записей, баланс классов `100/100`, но не замена
target-language test protocol. Archive локально проверен 9 августа 2026: exact size,
official MD5 и SHA-256
`e4155164722998c334de06a85ddfcb051720e3a8ba0673ea2d9751f5eef5ecec` внесены в ledger
со статусом `verified`; CRC и member whitelist успешно прошли перед extraction.

Воспроизведение требует новых output paths, поскольку ingestion не перезаписывает results:

```bash
uv run python scripts/ingest_ml_df_it_ood.py \
  --archive data/raw/ml_df/dataset_IT.7z \
  --metadata-archive data/raw/ml_df/metadata.zip \
  --output-manifest data/manifests/ml_df_it_v1_ood_200_repro.csv \
  --slice-name ood-200-repro

kds validate-manifest data/manifests/ml_df_it_v1_ood_200_repro.csv \
  --license-ledger data/licenses/license_ledger.csv \
  --require-ood-generator
kds validate-assets data/manifests/ml_df_it_v1_ood_200_repro.csv --audio-root data
```

Архив и extracted audio не попадают в Git.
