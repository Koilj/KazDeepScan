# Common Voice Scripted Speech 24.0 — Russian

## Локально проверенный archive

- Официальная datasheet: <https://dev.mozilladatacollective.com/datasets/cmj8l8ct700o5nlovbdnv58yr>.
- Artifact: `cv-corpus-24.0-2025-12-05-ru.tar.gz`.
- Локальный путь (не хранится в Git):
  `/home/ruslan/Downloads/cv-corpus-24.0-2025-12-05-ru.tar.gz`.
- Размер: `7 008 716 262` байта.
- SHA-256: `9a2ed32a0574f74f505cd7740a599f0b9edc9f52ba1e7d6624b66f258db4c0ea`.
- `gzip --test` прошёл 9 августа 2026 года.
- Лицензия datasheet: `CC0-1.0`; project usage scope:
  `owner_authorized_personal_research`.

Страница источника запрещает попытки установить личность говорящих и повторное размещение или
распространение dataset. `client_id` в TSV уже является предоставленным источником хешированным
pseudo-ID; он используется только как opaque ключ group split и не деанонимизируется.

## Archive и raw slice

Безопасный intake до чтения metadata или extraction повторно сверяет exact размер и SHA-256
archive с pinned source identity. Затем он принимает только ожидаемые 10 TSV и MP3 по пути
`ru/clips/<name>.mp3`, не вызывает `tar.extractall`, отклоняет symlink/неожиданные members и
публикует slice только после полного прохода archive в staging directory.

Проверенный archive содержит `201 326` MP3 и все ожидаемые metadata files. Из официальных
`train/dev/test.tsv` выбран детерминированный `first-250`: по 250 записей из каждого исходного
split, всего `750` MP3. Raw manifest:
`data/manifests/common_voice_ru_v24_first_250.csv`.

Локальный splitter объединяет связанные `parent_group_id`, `speaker_pseudo_id` и `text_hash`,
поэтому гарантия отсутствия leakage не зависит от исходного source split. Для raw slice он дал
`train=575`, `dev=82`, `test=93`.

## Ready slice

После decode, QA и WebRTC VAD опубликован ready manifest:
`data/manifests/common_voice_ru_v24_first_250_ready.csv`.

- `654` mono PCM S16LE WAV, `16 000` Гц;
- `96` raw клипов исключены только как `insufficient_speech`;
- все исключения находятся в
  `data/manifests/common_voice_ru_v24_first_250_rejections.json`;
- final ready split: `train=491`, `dev=79`, `test=84`;
- `kds validate-manifest --license-ledger` и `kds validate-assets` прошли для raw и ready
  manifest-ов.

## Воспроизведение

```bash
uv run python scripts/ingest_common_voice_ru_v24.py \
  --archive /home/ruslan/Downloads/cv-corpus-24.0-2025-12-05-ru.tar.gz \
  --output-manifest data/manifests/common_voice_ru_v24_first_250.csv \
  --slice-name first-250 --limit-per-source-split 250

KDS_FFMPEG_BINARY="$PWD/.tools/ffmpeg/bin/ffmpeg" \
KDS_FFPROBE_BINARY="$PWD/.tools/ffmpeg/bin/ffprobe" \
uv run python scripts/preprocess_manifest.py \
  --input-manifest data/manifests/common_voice_ru_v24_first_250.csv \
  --output-manifest data/manifests/common_voice_ru_v24_first_250_ready.csv \
  --rejection-report data/manifests/common_voice_ru_v24_first_250_rejections.json \
  --allow-rejections \
  --license-ledger data/licenses/license_ledger.csv \
  --data-root data
```

Каждый output должен быть новым: ingestion, preprocessing и manifest writer отказываются
перезаписывать имеющиеся results.

## Pre-extraction screen для следующего RU final candidate

Новый full-archive metadata-only screen не выбирает и не materializes клипы. После pinned
size/SHA-256 validation он проверил все `10 261` официальных RU `test` records (`2 075` client
groups) против `12 313` configured-role rows и `39 850` строк из `85` manifest-файлов по
`sample_id`, `text_hash`, `parent_group_id` и `speaker_pseudo_id`. Любой direct overlap исключает
весь client group: остаётся `6 211` records / `1 443` groups. Receipt:
`data/manifests/common_voice_ru_v24_full_test_metadata_exposure_screen_v1.json`, SHA-256
`f862ae667195c733c7deb6bf25f304a6287890ca87d4dc0ee7cb5e06aa6f46b3`.

Это не selection и не разрешение на extraction, synthesis или inference: следующий contract
обязан отдельно frozen bind size/seed и exact survivors до создания raw assets. Historical
first-250 intake, Stage-D `73` selection, уже scored `55` pairs и `18` rejected partners не
являются допустимым резервом или backfill для нового blind candidate.

Для fixed Silero V5.5 / `eugene` выполнен следующий literal-text screen без lexical rewrite:
`data/manifests/common_voice_ru_v24_full_test_silero_v5_5_literal_text_screen_v1.json`,
SHA-256 `4356c3ecbf3a9b68dd7a5d5f4e2ed9347d9c6f105d63d558bfc03dd1403b23d0`. Он потребляет
ровно `6 211` metadata survivors, требует exact equality после whitespace-only wrapper gate и
исключает весь client group при одном incompatible text. `113` direct incompatible records
(только неподдержанные кавычки или glyph `−`) исключили `106` groups / `611` records; остаются
`5 600` records / `1 337` groups. Это не изменение или оценка транскриптов и не audio operation.
