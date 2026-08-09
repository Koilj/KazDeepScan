# Контракт манифеста данных

Каждая строка CSV описывает один audio asset. В Git хранятся только манифесты, но не
исходное аудио, согласия или модельные веса.

Обязательный заголовок:

```text
sample_id,relative_path,sha256,split,label,language,code_switch,parent_group_id,source_name,source_license,rights_basis,speaker_pseudo_id,text_id,text_hash,duration_s,generator_family,generator_name,generator_version,voice_id,clone_consent_id,device,capture_route,original_sr,codec,augmentation_chain,augmentation_seed,created_at
```

`relative_path`, `sha256`, `duration_s` и `codec` описывают именно asset, на который указывает
строка. `original_sr` сохраняет исходную частоту дискретизации до нормализации, чтобы не
терять provenance. Поэтому после preprocessing `codec=wav`, а `original_sr` может отличаться
от 16 000.

Значения `split`: `train`, `dev`, `test`, `ood`; `label`: `bonafide`, `spoof`; `language`:
`ru`, `kk`, `mixed`, `other`. Значение `other` допустимо только для cross-lingual `ood`,
а не для target-language train/dev/test. Значение `code_switch` — `true`, `false` либо
`unknown`: последнее
обязательно использовать, когда источник не даёт проверяемой utterance-level разметки, а не
подменять неизвестное значение `false`. Для `spoof` обязательны семейство, имя, версия
генератора и `voice_id`.
Для `bonafide` эти поля должны быть пустыми: это предотвращает двусмысленное происхождение.

Проверка по умолчанию запрещает попадание одной `parent_group_id`, псевдонима диктора или
хеша текста в разные split-ы. При `--require-ood-generator` все fake-записи хотя бы одного
семейства генератора должны находиться только в `ood`; это доказывает, что в оценке есть
unseen-generator сценарий.

`rights_basis` содержит основание использования (например, идентификатор согласия или
проверенную лицензию), а `source_license` — ссылку либо неизменяемый текст версии лицензии.
Эти поля не заменяют юридическую проверку, но делают её аудируемой.

Если источник не распространяет транскрипт, `text_id` и `text_hash` могут содержать только
предоставленный source content pseudo-ID и его hash. Такой ID не следует выдавать за текст;
он используется исключительно как консервативный ключ, не позволяющий разделить один source
content между split-ами.

`data/licenses/license_ledger.csv` — реестр источников до попадания их audio assets в
processing или обучение. Запись с одобренным статусом `verified` либо
`owner_authorized_personal_research` обязана содержать размер и SHA-256 проверенного
archive. Второй статус фиксирует решение владельца проекта только для личного исследования;
он не отменяет ограничений лицензии, datasheet, privacy law или contributor consent.
Записи с любым иным статусом (например, `scope_confirmation_required`) могут фиксировать
кандидата, но не разрешают его использование. При передаче `--license-ledger` команда
проверяет, что каждый `source_name` манифеста есть в реестре и имеет одобренный статус.

`relative_path` — POSIX-путь от явно переданного data-root к нормализованному аудиофайлу,
например `processed/kk/9f/sample.flac`. Абсолютные пути, обратные слеши и компоненты `..`
запрещены: манифест не может выйти за пределы data-root и остаётся переносимым между
машинами. Это обязательное поле: без ссылки на asset нельзя безопасно и воспроизводимо
собрать обучающий Dataset.

## Локальные проверки

```bash
# Схема, provenance, leakage и разрешение каждого источника из ledger.
kds validate-manifest data/manifests/slice.csv \
  --license-ledger data/licenses/license_ledger.csv

# Файлы, symlink-границы и SHA-256.
kds validate-assets data/manifests/slice.csv --audio-root data

# Создать новый CSV с hash-детерминированным split по компонентам parent group, speaker и text.
kds assign-splits data/manifests/input.csv data/manifests/slice.csv --seed 20260808
```

`assign-splits` не перезаписывает файл и по умолчанию не меняет заранее выделенный `ood`.
Он объединяет записи в компоненты по `parent_group_id`, `speaker_pseudo_id` и `text_hash`,
так что любое значение, которое validator считает leakage, получает один split. Если
предвыделенный `ood` пересекается с train/dev/test по одному из этих ключей, команда
останавливается. После назначения обязательно проверить итоговый CSV через
`validate-manifest`; сам split не заменяет отдельный OOD protocol для нового
генератора/канала.

## Нормализация для обучения

`scripts/preprocess_manifest.py` принимает только manifest, прошедший проверку всех SHA-256,
и создаёт новые mono PCM WAV 16 кГц по пути `processed/<sha-prefix>/<sha>.wav`. Выходной
manifest создаётся лишь если **все** переданные assets прошли decode, QA и VAD; скрипт никогда
не перезаписывает audio asset или manifest. Пример:

```bash
KDS_FFMPEG_BINARY="$PWD/.tools/ffmpeg/bin/ffmpeg" \
KDS_FFPROBE_BINARY="$PWD/.tools/ffmpeg/bin/ffprobe" \
uv run python scripts/preprocess_manifest.py \
  --input-manifest data/manifests/ksc_raw.csv \
  --output-manifest data/manifests/ksc_processed.csv \
  --license-ledger data/licenses/license_ledger.csv \
  --data-root data
```

Для документированного исключения непригодных записей допустим только явный режим с новым
JSON-отчётом. Он сохраняет ready assets и создаёт manifest только из них; причины каждого
исключения остаются в отчёте. Не используйте этот режим, чтобы скрыть дисбаланс split или
ослабить QA-пороги:

```bash
uv run python scripts/preprocess_manifest.py \
  --input-manifest data/manifests/ksc_raw.csv \
  --output-manifest data/manifests/ksc_processed.csv \
  --rejection-report data/manifests/ksc_rejections.json \
  --allow-rejections \
  --license-ledger data/licenses/license_ledger.csv \
  --data-root data
```
