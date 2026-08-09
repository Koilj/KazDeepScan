# Источник bona-fide: KSC / OpenSLR SLR102

## Решение и происхождение

Первый скачиваемый bona-fide источник — **Kazakh Speech Corpus (KSC), OpenSLR SLR102**.
Страница OpenSLR указывает около 332 часов, более 153 000 размеченных записей и лицензию
CC BY 4.0.

- Страница источника: <https://www.openslr.org/102/>.
- Зеркало загрузки: <https://openslr.trmal.net/resources/102/ISSAI_KSC_335RS_v1.1_flac.tar.gz>.
- Лицензия: CC BY 4.0. Атрибуцию нужно сохранять в data card и model card.
- Размер, объявленный зеркалом: `19 092 377 812` байт.
- Дата изменения на зеркале: `2022-02-14T11:38:18Z`.

Подробная учётная запись находится в `data/licenses/license_ledger.csv`.

## Проверенный локальный archive

9 августа 2026 проверен файл
`/home/ruslan/Downloads/ISSAI_KSC_335RS_v1.1_flac.tar.gz`:

- размер в точности совпадает с ожидаемым: `19 092 377 812` байт;
- `gzip --test` проходит;
- SHA-256:
  `a200aa3ab6b0284a7241ac357951fa5422f6fea855a30c1ab2fa1559c3f0d149`;
- проверка whitelist структуры TAR обнаружила 153 853 FLAC, 153 853 paired transcript и
  ровно три metadata CSV.

В release есть полный связанный layout:

```text
ISSAI_KSC_335RS_v1.1_flac/
├── Audios_flac/<uttID>.flac
├── Transcriptions/<uttID>.txt
└── Meta/{train,dev,test}.csv
```

Следовательно, внешний metadata root для этого archive не нужен. Не восстанавливайте
transcript, split, speaker или code-switch по имени FLAC. В частности, `deviceID` не является
speaker ID; при отсутствии проверяемой utterance-level code-switch разметки manifest хранит
`unknown`.

## Безопасный intake

`scripts/ingest_ksc_slr102.py` перед извлечением проверяет точный размер, gzip/TAR-структуру,
парность всех audio/transcript IDs и наличие ровно трёх `Meta` CSV. Он потоково извлекает
только выбранные записи во временную directory, публикует результат только после проверки
конца archive, не заменяет уже существующие assets/manifest и никогда не вызывает
`tar.extractall`. Исходные KSC split сохраняются как `train`, `dev`, `test`.

Пример создания отдельного slice:

```bash
.tools/uv/uv run python scripts/ingest_ksc_slr102.py \
  --archive /home/ruslan/Downloads/ISSAI_KSC_335RS_v1.1_flac.tar.gz \
  --output-manifest data/manifests/ksc_slice.csv \
  --slice-name first-250 \
  --limit-per-split 250

.tools/uv/uv run kds validate-manifest data/manifests/ksc_slice.csv
.tools/uv/uv run kds validate-assets data/manifests/ksc_slice.csv --audio-root data
```

Уже создан и проверен `data/manifests/ksc_first_250.csv`: по 250 bona-fide записей из
каждого исходного split, всего 750.

После реального decode, QA и WebRTC VAD создан отдельный
`data/manifests/ksc_first_250_ready.csv`: 731 mono PCM S16LE WAV 16 кГц (`train=244`,
`dev=242`, `test=245`). Raw manifest сохранён без изменений. Отдельный
`data/manifests/ksc_first_250_rejections.json` содержит 19 исключений: 18 с недостаточной
длительностью речи и 1 слишком тихую запись. Готовый manifest и все его SHA-256 прошли
проверку; его `codec=wav` соответствует фактическим WAV assets.

## История ошибочного файла

Файл размером `19 167 494 356` байт не соответствовал опубликованному размеру и был отвергнут
intake. По указанию пользователя он удалён вместе со случайным неполным файлом. Не пытайтесь
«исправить» подобный archive через `truncate`: скачайте его заново и сначала проверьте размер,
gzip CRC и структуру.

## Ограничения

KSC содержит только bona-fide казахскую речь. Он не является fake-набором, не покрывает
русский язык и не может быть единственным train-источником anti-spoof модели. Его нельзя
смешивать со spoof-only corpus без симметричного проектирования канала и group-aware split.
