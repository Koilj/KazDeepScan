# KazDeepScan — статус проекта

**Обновлено:** 9 августа 2026

Этот файл — рабочая точка продолжения. После каждого завершённого этапа обновляйте разделы
«Сделано», «Проверено», «Текущий этап» и «Дальше».

## Цель MVP

Локально исследовать признаки синтезированности русской, казахской и смешанной речи по
аудиофайлу. Результат текущего этапа — воспроизводимые research metrics, а не риск-оценка,
идентификация человека, доказательство мошенничества или развёрнутый product/API сервис.

## Сделано

- Созданы Python-пакет, конфигурации, CLI, `uv.lock` и аудио-конвейер: безопасный decode,
  16 кГц mono PCM, QA, WebRTC VAD и окна 4.04 с.
- Реализованы строгий CSV manifest, проверка provenance/прав/split leakage, validator assets
  (включая SHA-256 и защиту от symlink escape), group-aware split и Dataset для
  preprocessed audio.
- Реализованы B0 log-Mel CNN, B0 train/eval-контур, XLS-R + SLS foundation, агрегация logit и
  temperature scaling. Это фундамент: обученной и калиброванной продуктовой модели пока нет.
- Добавлен FastAPI-каркас с health/readiness и безопасной потоковой загрузкой. Он не выдаёт
  score без обученного калибруемого scorer.
- Исправлены sync health/readiness handlers FastAPI: они стали `async`, поэтому API tests не
  зависают в Python 3.13 / AnyIO окружении.
- Исправлен `.gitignore`: игнорируется только корневой каталог model artifacts `/models/`,
  а исходники `src/kds/models/` теперь попадут в первый Git commit.
- Добавлена проверка `license_ledger.csv`: `kds validate-manifest --license-ledger`,
  preprocessing и B0 training требуют, чтобы каждый `source_name` манифеста был записан в
  ledger с одобренным статусом и с SHA-256 archive. Статус
  `owner_authorized_personal_research` допускает только личное обучение и не отменяет
  внешние ограничения источника; кандидаты без одобренного статуса не могут случайно попасть
  в processing или обучение.
- Введён явный `kds validate-training-protocol` и обязательный `--purpose` для B0 training.
  Policy ledger отделяет факт наличия pseudo-ID от проверяемой семантики speaker/voice group.
  `product` жёстко требует commercial-clean verified sources, group provenance `verified`,
  binary train/dev/test/OOD, unseen OOD generator и отсутствия повторного spoof voice между
  split-ами; выбранный режим и protocol report сохраняются в checkpoint.
- Проект зафиксирован в Git: первый проверенный commit `fbd2a53` отправлен в
  `git@github.com:Koilj/KazDeepScan.git`, ветка `main` отслеживает `origin/main`.
- Проведён повторный source review для product corpus. KazakhTTS2 — единственный
  условно-пригодный казахский bona-fide component: у него пять явных speaker groups, 271.7 ч,
  CC BY 4.0 и документированный consent process, но отсутствуют spoof class и достаточное
  число speaker groups для независимого binary benchmark. Архивы 35.7 GB не скачивались.
  MCSKL, YO-CPT-ru и SpeechFake не прошли source review для product use. Подробности —
  `docs/product_corpus_source_review_2026-08-09.md`.
- Добавлен `kds validate-consent-registry` для будущего собственного consented corpus. Он
  принимает только локальный pseudonymous export без PII и требует active consent с явными
  scopes на product training, synthetic derivatives и commercial deployment; `data/consents/`
  исключён из Git.
- По уточнению владельца проекта scope изменён с потенциального product на **personal
  research**. Лицензии, provenance, integrity checks и ограничения каждого источника остаются
  обязательными; снят только ложный блок на использование research-only источников в личном
  эксперименте. Product review и consent-registry сохранены как архивные fail-closed
  материалы, а не как текущая дорожная карта.
- Исправлен `GroupSplitter`: он назначает один split всей связной компоненте по
  `parent_group_id`, `speaker_pseudo_id` и `text_hash`, а не только по parent group. Это
  предотвращает text/speaker leakage при local re-split Common Voice.
- Реализован Common Voice Russian v24 intake: точный размер, gzip CRC, tar whitelist,
  потоковое atomic extraction выбранных MP3 и manifest без de-identification `client_id`.
- Локально проверен archive
  `/home/ruslan/Downloads/cv-corpus-24.0-2025-12-05-ru.tar.gz`: размер `7 008 716 262` байта,
  gzip CRC проходит, SHA-256
  `9a2ed32a0574f74f505cd7740a599f0b9edc9f52ba1e7d6624b66f258db4c0ea`.
- Archive содержит `201 326` MP3 и все `10` ожидаемых TSV. Создан raw Common Voice slice
  `first-250`: по 250 записей из исходных train/dev/test, всего `750` MP3. После local
  leakage-safe split: `train=575`, `dev=82`, `test=93`. Manifest:
  `data/manifests/common_voice_ru_v24_first_250.csv`.
- После реального decode, QA и WebRTC VAD создан ready manifest
  `data/manifests/common_voice_ru_v24_first_250_ready.csv`: `654` mono PCM S16LE WAV на
  `16 000` Гц (`train=491`, `dev=79`, `test=84`). Исключены `96` raw-клипов, все с речью
  менее 2.5 с; они зафиксированы в
  `data/manifests/common_voice_ru_v24_first_250_rejections.json`.
- Реализован ML-DF v1 Italian intake для **cross-lingual OOD только**: проверяются exact
  archive size/MD5, распакованный размер, CRC, 7z whitelist и metadata-to-audio
  correspondence; selected WAV публикуются атомарно с SHA-256. Формат `language=other`
  разрешён контрактом только для `split=ood`.
- Локально проверен `data/raw/ml_df/dataset_IT.7z`: размер `1 485 098 719` байт, official
  MD5 `c3ce93f9566605e0a5ad2e3cda099d7d`, SHA-256
  `e4155164722998c334de06a85ddfcb051720e3a8ba0673ea2d9751f5eef5ecec`. CRC и точный
  layout `16 000` WAV + три ожидаемых metadata/directory members прошли проверку.
- Создан ML-DF Italian OOD slice `data/manifests/ml_df_it_v1_ood_200.csv`: `200` WAV,
  `100` bona-fide и `100` spoof (`25` на каждый из VITS, ZMM-TTS, LVC-VC, DDDM-VC). Все
  assets находятся в `data/raw/ml_df_it_v1/slices/ood-200/`; manifest содержит generator
  provenance и data-relative paths.
- Проверен официальный RuASD Russian shard `ruasd-000000.tar`: `999 813 120` байт и
  SHA-256 `956efb0e1281ada0dcee6f2ed9498c454552be88b3e9784e52e70c3ef4dfcd67` совпали с
  Hugging Face revision. В нём `985` безопасных JSON/WAV пар, но только raw TTS fake:
  `306` ElevenLabs и `679` TeraTTS.
- Реализован RuASD intake, который допускает только paired fake `ood` rows, проверяет
  checksum/archive layout/metadata и извлекает WAV атомарно. Создан Russian fake-only
  stress slice `data/manifests/ruasd_ru_v1_shard000000_ood_100.csv`: по `50` samples
  ElevenLabs и TeraTTS. Он явно запрещён как B0 train/dev/test источник.
- В `~/Downloads/RuASD` найден полный локальный RuASD release: `250` TAR artifacts
  (`000000`–`000249`). Добавлены pinned official catalog
  `data/licenses/ruasd_v1_artifact_catalog.csv`, read-only safe audit и
  `scripts/audit_ruasd_collection.py`; все `250/250` SHA-256 совпали с pinned official
  catalog, а metadata pass подтвердил `585 353` JSON/WAV pairs без extraction. У всех
  `147 097` raw bona-fide rows speaker неизвестен, у raw fake известны только
  `4 750/228 266`, поэтому full release не является speaker/voice-disjoint binary protocol.
- Реализован `scripts/ingest_ruasd_research.py` для full RuASD personal-research binary
  slice: он использует только raw real/TTS rows, по умолчанию повторно сверяет весь pinned
  collection, распределяет равные классы с coverage source subset/model strata, не копирует
  транскрипты в manifest и сохраняет text-leakage-safe split. Исходники, synthetic TAR tests,
  documentation и ledger entry добавлены. Реально создан raw slice `research-2000`: `1 000`
  bona-fide + `1 000` spoof WAV, после deterministic split `train=1 564`, `dev=220`,
  `test=216`; все `2 000/2 000` raw asset SHA-256 прошли validation.
- После QA/VAD создан ready RuASD manifest на `1 814` WAV (`816` bona-fide, `998` spoof;
  `train=1 417`, `dev=202`, `test=195`). Отдельный report содержит все `186` rejections:
  `152` insufficient speech, `32` too quiet и `2` collision с уже существующими processed
  assets, которые не были перезаписаны. Готовые `1 814/1 814` assets, ledger и research
  training protocol успешно проверены.
- Добавлены `bonafide_accuracy`, `spoof_accuracy` и `balanced_accuracy` в B0 train/eval
  reports. Для single-class набора balanced accuracy честно возвращается `null`.
- B0 `models/b0-ruasd-full-research-2000.pt` обучен пять эпох на RTX 5060 Ti c `--purpose
  research`. Его in-source RuASD test balanced accuracy — `0.8808`; ML-DF OOD — `0.9262`,
  PyAra holdout — `0.7273`, KSC bona-fide recall — `0.9673`. Это не product metrics и не
  свидетельство speaker/voice-independent robustness. Подробный разбор —
  `docs/research_b0_ruasd_full_2000.md`.
- Локально проверен вручную скачанный PyAra v7 `archive.zip`: `28 092 611 663` байта,
  SHA-256 `dadf5b795adbd6d635e74f4f9662c3e9a425c88bd76f26731f9e6adbad278b91`, полный ZIP
  CRC успешно. Archive содержит `73 583` real и `128 195` fake WAV от пяти algorithms.
- Реализован PyAra intake с checksum, ZIP whitelist, exact TSV-to-WAV membership и atomic
  selected extraction. Создан text-leakage-safe **research-only** binary raw slice
  `pyara_ru_v7_research_500.csv` (`250/250` labels), затем ready manifest на `481` WAV;
  `19` QA/VAD rejections записаны отдельно. Источник не раскрывает speaker IDs, поэтому
  split не выдаётся за speaker-disjoint.
- На RTX 5060 Ti обучен B0 research smoke checkpoint `models/b0-pyara-research-500.pt`
  (3 эпохи); добавлен read-only `scripts/evaluate_b0.py`. Local PyAra test accuracy `0.8864`,
  но ML-DF OOD `0.5208`, RuASD fake-only OOD `0.3800`: это явное доказательство отсутствия
  переносимости, а не product metric.
- ML-DF и RuASD OOD manifests прошли тот же preprocessing: соответственно `192/200` и
  `100/100` ready WAV; все QA/VAD rejections зафиксированы в отдельных JSON reports.
- Реализован безопасный intake KSC / SLR102: точный размер, gzip CRC, whitelist структуры
  TAR, потоковое извлечение только выбранного slice в staging directory и запрет
  `tar.extractall`. `code_switch=unknown` честно допускается при отсутствии разметки.
- Чистый KSC archive проверен в
  `/home/ruslan/Downloads/ISSAI_KSC_335RS_v1.1_flac.tar.gz`: размер `19 092 377 812` байт,
  gzip CRC проходит, SHA-256
  `a200aa3ab6b0284a7241ac357951fa5422f6fea855a30c1ab2fa1559c3f0d149`.
- Архив содержит `153 853` пар `Audios_flac/<uttID>.flac` +
  `Transcriptions/<uttID>.txt` и ровно `Meta/train.csv`, `Meta/dev.csv`, `Meta/test.csv`.
  Внешний metadata root для этого release не требуется.
- Создан bona-fide KSC raw slice `first-250`: по 250 записей из исходных train/dev/test,
  всего 750 аудио и транскрипций. Manifest: `data/manifests/ksc_first_250.csv`.
- Установлены `ffmpeg 8.0.1` и `ffprobe 8.0.1`. После реального decode, QA и WebRTC VAD
  создан отдельный ready manifest `data/manifests/ksc_first_250_ready.csv`: `731` mono PCM
  S16LE WAV на `16 000` Гц (`train=244`, `dev=242`, `test=245`). Raw manifest не менялся.
- Исключены `19` raw-записей: `18` с речью менее 2.5 с и `1` слишком тихая. Все IDs и
  причины зафиксированы в `data/manifests/ksc_first_250_rejections.json`; строки не были
  молча удалены.
- Batch preprocessing сначала записывает WAV в staging directory. По умолчанию одна ошибка
  отменяет публикацию всего набора. Явный режим `--allow-rejections` требует нового JSON
  rejection report и только тогда публикует прошедшие QA/VAD assets.
- Исправлен preprocessing manifest: поле `codec` теперь описывает фактический asset из
  `relative_path` (`wav`), а `original_sr` сохраняет частоту исходного файла.
- Проведён read-only review следующих источников. Подробности и решения по применимости — в
  `docs/data_source_review_2026-08-09.md`: Common Voice Russian v24 затем локально проверен
  и разрешён владельцем только для personal research; ASVspoof 2021 отклонён из-за отсутствия
  лицензии в official record; ASVspoof 5 отклонён из-за неочищенных прав на individual
  contents.
- По явному указанию пользователя удалены старый oversized KSC archive из
  `data/raw/ksc_slr102/` и случайный неполный файл из `data/raw/ksc_slr102_clean/`.
  Проверенный archive в `~/Downloads` не перемещался и не изменялся.

## Проверено

Проверка 9 августа 2026:

- `ruff check src tests scripts services` — успешно;
- `mypy src scripts services` — успешно, 34 source files;
- `pytest -q` — 40 тестов успешно;
- KSC archive: точный ожидаемый размер, `gzip --test`, whitelist структуры и парность всех
  `153 853` audio/transcript IDs — успешно;
- `kds validate-manifest data/manifests/ksc_first_250.csv` — `750` строк, успешно;
- `kds validate-assets data/manifests/ksc_first_250.csv --audio-root data` — `750/750`
  исходных FLAC совпадают с SHA-256;
- `kds validate-manifest data/manifests/ksc_first_250_ready.csv` — `731` строка, успешно;
- `kds validate-assets data/manifests/ksc_first_250_ready.csv --audio-root data` — `731/731`
  WAV существуют и совпадают с SHA-256;
- `kds validate-manifest data/manifests/common_voice_ru_v24_first_250.csv --license-ledger
  data/licenses/license_ledger.csv` — `750` строк, успешно;
- `kds validate-assets data/manifests/common_voice_ru_v24_first_250.csv --audio-root data` —
  `750/750` MP3 существуют и совпадают с SHA-256;
- `kds validate-manifest data/manifests/common_voice_ru_v24_first_250_ready.csv
  --license-ledger data/licenses/license_ledger.csv` — `654` строки, успешно;
- `kds validate-assets data/manifests/common_voice_ru_v24_first_250_ready.csv --audio-root
  data` — `654/654` WAV существуют и совпадают с SHA-256;
- `ffprobe` Common Voice ready asset подтвердил mono `pcm_s16le`, `16 000` Гц;
- `ffprobe` ready asset подтвердил mono `pcm_s16le`, `16 000` Гц; manifest содержит
  `codec=wav`;
- целевые тесты KSC intake и preprocessing — успешно.
- Повторная проверка после аудита: `ruff`, `mypy` (35 source files), `pytest` (45 tests),
  SHA-256 обоих KSC manifest-ов и `kds validate-manifest --license-ledger` для обоих
  manifest-ов прошли успешно.
- `kds validate-manifest data/manifests/ml_df_it_v1_ood_200.csv --license-ledger
  data/licenses/license_ledger.csv --require-ood-generator` — `200` строк, успешно;
- `kds validate-assets data/manifests/ml_df_it_v1_ood_200.csv --audio-root data` —
  `200/200` WAV существуют и совпадают с SHA-256;
- `ffprobe` ML-DF asset подтвердил mono `pcm_s16le`, `16 000` Гц;
- `kds validate-manifest data/manifests/ruasd_ru_v1_shard000000_ood_100.csv
  --license-ledger data/licenses/license_ledger.csv --require-ood-generator` — `100`
  строк, успешно;
- `kds validate-assets data/manifests/ruasd_ru_v1_shard000000_ood_100.csv --audio-root
  data` — `100/100` WAV существуют и совпадают с SHA-256;
- `ffprobe` RuASD asset подтвердил mono `pcm_s16le`, `44 100` Гц;
- `kds validate-manifest` и `kds validate-assets` с ledger прошли для PyAra raw (`500/500`)
  и ready (`481/481`) manifest-ов;
- `kds validate-manifest` и `kds validate-assets` с ledger прошли для ML-DF ready
  (`192/192`) и RuASD ready (`100/100`) manifest-ов;
- `ffprobe` PyAra asset подтвердил mono `pcm_s16le`, `16 000` Гц;
- проверка после protocol gate implementation: `ruff check src tests scripts services`,
  `mypy src scripts services` (`46` source files) и `pytest -q` (`64` tests) — успешно;
  все пять ready manifest-ов повторно прошли `kds validate-manifest --license-ledger`.
- реальная проверка нового gate на PyAra ready manifest: `--purpose research` прошёл
  (`train=376`, `dev=61`, `test=44`); `--purpose product` корректно остановлен из-за
  отсутствующего OOD, research-only/non-commercial policy и unknown speaker/voice provenance.
- проверка после candidate review и consent-registry implementation: `ruff check src tests
  scripts services`, `mypy src scripts services` (`47` source files) и `pytest -q` (`69`
  tests) — успешно.
- full RuASD integrity audit: `250/250` archive size и SHA-256 совпали с official pinned
  catalog; safe TAR walk подтвердил `585 353` exact JSON/WAV pairs, без extraction и без
  изменения файлов в `~/Downloads/RuASD`.

Ранее успешно прошли CUDA smoke test B0, real M4A inspection и XLS-R + SLS GPU smoke test.

## Текущий этап

Русский PyAra research protocol, full RuASD binary protocol, KSC/Common Voice bona-fide layers
и OOD layers подготовлены. Новый RuASD B0 заметно лучше прежнего PyAra-only smoke baseline, но
PyAra spoof recall остаётся `0.4545`, а все RuASD metrics non-speaker-disjoint. Поэтому модель
можно использовать только для воспроизводимого local research comparison; model/API score и
calibration не включаются. Common Voice Russian v24 разрешён владельцем проекта только для
личного исследования; сохраняются внешний запрет на идентификацию и re-hosting.

## Дальше

1. Спроектировать explicit source-mixed research matrix: train/dev/test и evaluation source
   должны быть disjoint по исходному corpus, а не только по sample ID. Не использовать
   `ruasd-000000` fake-only shard как OOD после обучения на full RuASD.
2. Повторить B0 на source-mixed protocol и отдельно сравнить class recall/balanced accuracy с
   текущими single-source checkpoints. Не усреднять ML-DF, PyAra, KSC и within-source RuASD
   metrics в одну «общую accuracy».
3. При необходимости добавить KazakhTTS2, MCSKL, YO-CPT-ru или SpeechFake только после
   отдельного local artifact/terms audit; personal-research scope не отменяет лицензию,
   provenance или ограничение конкретного source. Для каждого нового source обновить ledger и
   создать отдельный reproducible intake, а не смешивать файлы вручную.

Повторный preprocessing с документированным исключением QA/VAD-rejections возможен только
так (каждый output и отчёт должны быть новыми):

```bash
.tools/uv/uv run python scripts/preprocess_manifest.py \
  --input-manifest data/manifests/ksc_first_250.csv \
  --output-manifest data/manifests/ksc_new_ready.csv \
  --rejection-report data/manifests/ksc_new_rejections.json \
  --allow-rejections --data-root data
```

## Блокеры и ограничения

- PyAra допускает personal-research binary smoke baseline по `CC-BY-NC-SA-4.0`, но не даёт
  speaker IDs. Его local metrics не являются independent evaluation.
- Независимые OOD результаты B0 слабые (ML-DF `0.5208`, RuASD fake-only `0.3800`), поэтому
  прежнего PyAra-only checkpoint больше не являются current baseline. Новый RuASD checkpoint
  всё ещё не даёт основания включать risk score, calibration или API inference: PyAra spoof
  recall только `0.4545`, а RuASD OOD shard не независим от full RuASD training source.
- Полный RuASD (250 TAR artifacts, около 250 GB) находится в `~/Downloads/RuASD` и не
  перемещается. Он пригоден для personal-research binary training, но `speakers` неизвестен
  для raw bona-fide rows и spoof voice provenance unknown: split не speaker/voice-disjoint.
- Загрузка разрешённых источников размером до 2 ГБ может выполняться без отдельного
  подтверждения. Если ожидаемый размер файла или набора файлов превышает 2 ГБ, остановить
  загрузку до её начала: пользователь скачивает данные самостоятельно, после чего работа
  продолжается с локальной копией. Это операционное правило не отменяет обязательную
  предварительную проверку лицензии, scope и provenance.
- KSC — только bona-fide казахская речь; `deviceID` не является speaker ID, поэтому не может
  использоваться как surrogate для speaker-disjoint split.
- Нельзя скачивать или использовать голоса, TTS-веса и corpus без записи лицензии/согласия.
- Git-история и GitHub remote уже настроены; крупные завершённые этапы нужно фиксировать
  отдельными проверенными commit-ами.
