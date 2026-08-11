# KazDeepScan — статус проекта

**Обновлено:** 11 августа 2026

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
- Реализованы B0 log-Mel CNN, B0 train/eval-контур, XLS-R + SLS foundation и реальный
  research Stage A, агрегация logit и temperature scaling. Это фундамент: обученной и
  калиброванной продуктовой модели пока нет.
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
- Добавлен архивный `kds validate-consent-registry` для ранее рассматривавшегося собственного
  consented corpus. Он
  принимает только локальный pseudonymous export без PII и требует active consent с явными
  scopes на product training, synthetic derivatives и commercial deployment; `data/consents/`
  исключён из Git.
- По уточнению владельца проекта scope изменён с потенциального product на **personal
  research**. Лицензии, provenance, integrity checks и ограничения каждого источника остаются
  обязательными; снят только ложный блок на использование research-only источников в личном
  эксперименте. Product review и consent-registry сохранены как архивные fail-closed
  материалы, а не как текущая дорожная карта.
- Владелец проекта окончательно исключил сбор и запись голосов людей. Новые данные разрешено
  получать только из готовых источников после проверки прав/provenance либо создавать локально
  text-to-speech генераторами с закреплёнными весами и встроенными голосами, без reference audio,
  voice cloning и имитации голоса конкретного человека.
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
- Реализована versioned source-mixed research matrix: отдельный JSON contract и
  `kds validate-source-matrix` требуют binary train/dev/test, непересекающиеся исходные
  `source_name`, обычное отсутствие sample/SHA-256/group/text leakage и разрешённое usage из
  ledger.
  `scripts/train_b0_matrix.py` выбирает checkpoint только на dev и оценивает final test один
  раз после этого. Реальная matrix v1: RuASD train (`1 417`), PyAra dev (`61`) и ML-DF OOD
  final test (`192`); ML-DF остаётся cross-lingual OOD, а не меняет смысл своего manifest.
- Обучены три B0 source-mixed checkpoints (seeds `20260810`–`20260812`). ML-DF final-test
  balanced accuracy составила соответственно `0.7340`, `0.8460` и `0.8579`, с bona-fide
  recall от `0.4681` до `0.7872`. Это подтвердило чувствительность к source/dev protocol;
  calibration, model release и API score по-прежнему запрещены. Полный отчёт —
  `docs/research_b0_source_mixed_v1.md`.
- В B0 evaluation добавлены exact correct counts и 95% Wilson intervals для accuracy и
  каждого class recall. Все три source-mixed checkpoint повторно оценены на ML-DF; interval
  описывает только конечные 192 clips и не выдаётся за robustness claim.
- LRLspoof исключён из проекта после read-only audit revision
  `793f667a579756602193bb5b783ba16e80bcb7e6`: `36 000` Kazakh spoof paths не компенсируют
  spoof-only состав и единый sequential ~452 GB gzip/tar release, который не позволяет
  загрузить только Kazakh. Не скачивать, не добавлять в ledger, не запрашивать shards и не
  смешивать с KSC. Audit сохранён только как обоснование:
  `docs/data_sources_lrlspoof_2026-08-10.md`.
- Реализован воспроизводимый KSC-derived Kazakh TTS stress source v1. Он берёт только KSC
  `test` transcripts, фиксирует локальные Piper (шесть voice profiles) и MMS/VITS (один
  profile) artifacts по revision/size/SHA-256, синтезирует только после проверки KSC text hash
  и публикует raw/ready/rejection manifests без перезаписи существующих WAV. После QA и
  безопасного exact-asset merge финальный paired test содержит `921` KSC bona-fide + `921`
  synthetic clips: `475` MMS/VITS и `446` Piper. Это controlled personal-research source, не
  speaker-independent corpus и не product/calibration data. Protocol —
  `docs/data_sources_ksc_derived_kk_v1.md`.
- Source-mixed matrix v2 использует RuASD train (`1 417`), PyAra dev (`61`) и этот
  source-disjoint Kazakh final test (`1 842`). Три B0 запуска не ошиблись на synthetic часть
  двух закреплённых family, но KSC bona-fide recall составил лишь `0.6417`, `0.7904`, `0.8252`
  по seeds. Это фиксирует сильную чувствительность к bona-fide source; model release, score и
  calibration по-прежнему запрещены. Полный отчёт —
  `docs/research_b0_source_mixed_v2_kk.md`.
- Реализована третья независимая Kazakh generator family: KazEmoTTS (Grad-TTS + HiFi-GAN),
  а не дополнительный Piper/VITS voice. Official source revision и два checkpoint archive
  суммарно `248 377 435` bytes закреплены по size/SHA-256; ZIP CRC, SHA-256 извлечённых
  checkpoint и `torch.load(weights_only=True)` прошли до synthesis. Runtime не использует
  legacy training binary extension и не принимает reference audio, поэтому local voice
  cloning отсутствует.
- Создан новый fresh KSC `test` slice, text/sample-disjoint с v1: `450` raw rows → `402`
  QA-accepted bona-fide. KazEmoTTS создал `402` clips, из которых `359` прошли QA/VAD, а `43`
  rejection записаны с причинами. Frozen paired final test
  `data/manifests/ksc_derived_kk_v2_kazemotts_test_359.csv` содержит `718` assets (`359/359`)
  и одну hash-identical bona-fide/spoof пару на каждый text. Это personal-research stress test,
  не speaker-independent/product data.
- Зафиксирована и реально провалидирована source-mixed matrix v3: RuASD train (`1 417`),
  PyAra dev (`61`) и новый KazEmoTTS final test (`718`). Final test пока не использовался для
  training, epoch selection, threshold или calibration.
- Реализована четвёртая независимая Kazakh generator family: Spark-TTS Kazakh (Qwen2.5 LLM +
  BiCodec), а не дополнительный VITS/Piper/Grad-TTS voice. Pinned essential bundle
  (`1 861 074 893` bytes) и controlled GPU runtime verified; reference audio/wav2vec2 route
  отсутствует. Fresh KSC v3 selection text/sample-disjoint с v1/v2: `450` raw → `387`
  bona-fide ready; Spark-TTS `387` raw → `381` ready, а шесть `insufficient_speech` rejection
  сохранены. Frozen paired final test
  `data/manifests/ksc_derived_kk_v3_sparktts_test_381.csv` имеет `762` assets (`381/381`) и
  ровно одну hash-identical pair на текст. Это только personal-research stress test.
- Зафиксирована source-mixed matrix v4: RuASD train (`1 417`), PyAra dev (`61`) и Spark-TTS
  final test (`762`). `kds validate-source-matrix` и полный static/unit набор прошли; final
  test не использовался для training, epoch selection, threshold или calibration.
- Реализована пятая независимая Kazakh generator family: eSpeak NG 1.52.0 Kazakh `kk`,
  rule-based formant synthesis, а не дополнительный neural voice. Official source и Ubuntu
  runtime packages суммарно `29 248 111` bytes закреплены по size/SHA-256. Они извлекаются
  path/type/size-safe во temporary directory без system install; runtime принимает только text
  через stdin и не содержит reference-audio/cloning route. Fresh KSC v4 text/sample-disjoint с
  v1–v3: `450` raw → `407` bona-fide ready; сохранены `43` rejections (12 insufficient speech,
  31 refusal to reuse a processed asset). eSpeak NG `407` raw → `358` ready, `49`
  insufficient-speech rejections сохранены. Frozen paired final test
  `data/manifests/ksc_derived_kk_v4_espeakng_test_358.csv` имеет `716` assets (`358/358`) и
  ровно одну hash-identical pair на текст. Это только personal-research stress test.
- Зафиксирована и реально провалидирована source-mixed matrix v5: RuASD train (`1 417`),
  PyAra dev (`61`) и eSpeak NG final test (`716`). Final test не использовался для training,
  epoch selection, threshold или calibration.
- Реализован строгий XLS-R+SLS Stage A contract и CUDA/BF16 runner. Plan фиксирует SHA-256
  RuASD train (`1 417`), source-disjoint PyAra dev (`61`), license ledger, локальные
  XLS-R-300M config/weights, SLS-head, optimizer и output paths. Encoder полностью frozen и
  удерживается в eval mode; runner не принимает final manifests, выбирает epoch только по
  `dev_loss`, сохраняет только head state и отказывается перезаписывать результаты.
- На RTX 5060 Ti выполнен Stage A v1: 3 эпохи, batch `16`, BF16, полный run `42.71` с,
  VRAM peak `2.78` GB allocated. Выбран epoch 3 с PyAra dev loss `0.37577`, accuracy `0.8197`
  и balanced accuracy `0.8429`; head state SHA-256
  `1370b9b81e0c61f0ced94d29fdcc15e28ba28f5240ef724683b8bfd0cdb490e6`. Frozen final
  evaluation и calibration не выполнялись. Отчёт —
  `docs/research_xlsr_sls_stage_a_v1.md`.
- Полный PyAra v7 archive в `/home/ruslan/Downloads/archive.zip` использован для нового
  независимого Stage B dev slice. При выборе заранее исключены `500` старых record IDs и
  `499` text hashes; после QA/VAD осталось `973` rows. Строгий builder удалил ещё три text
  groups, пересекавшихся с RuASD train. Итоговый manifest содержит `970` rows (`474/496`
  labels), не пересекается со старым PyAra dev или RuASD train по доступным sample, asset,
  text и parent-group keys. Отсутствие speaker IDs по-прежнему явно ограничивает protocol.
- Реализован hash-pinned XLS-R+SLS Stage B plan/runner. Он проверяет Stage A plan, checkpoint
  и canonical head-state receipts, размораживает только blocks `16`–`23`, использует отдельные
  encoder/head learning rates, BF16, gradient checkpointing и exact example-normalized
  accumulation. Runner не принимает final manifests и атомарно сохраняет только head и
  encoder tail без перезаписи.
- Stage B v1 выполнен на RTX 5060 Ti за `532.32` с: 15 эпох, physical batch `4`, accumulation
  `8`. Минимальный fresh PyAra dev loss `0.15236` получен на epoch 7; dev accuracy `0.9381`,
  balanced accuracy `0.9393`, bona-fide recall `0.9895`, spoof recall `0.8891`. Checkpoint
  SHA-256 `18c967a8881404140ccda04fc6234079ac4b2802425e4111f3fef59bef505c32`;
  frozen final inference и calibration не выполнялись. Отчёт —
  `docs/research_xlsr_sls_stage_b_v1.md`.
- Проведён актуальный поиск дополнительных готовых данных. XMAD — сильный multilingual
  кандидат, но русская часть использует Common Voice/M-AILABS, уже представленные в RuASD,
  поэтому без отдельного speaker/source audit она не принята как Stage B dev. Common Voice
  Kazakh содержит только bona-fide, а LRLspoof Kazakh остаётся spoof-only в неселективном
  архиве. Решения закреплены в `docs/data_source_review_2026-08-10_stage_b.md`.
- Подготовлен новый frozen **calibration-only** PyAra dev для будущего XLS-R Stage B final
  protocol. До извлечения исключены 1 500 старых PyAra record IDs и 1 482 text hashes;
  `1 000` raw rows дали `978` QA/VAD-ready rows, затем один text overlap с historical
  RuASD/Stage-A/Stage-B role был удалён. Итоговый `977`-row manifest (`478/499`) —
  `data/manifests/pyara_ru_v7_stage_b_calibration_dev_1000_ready_v2.csv`, SHA-256
  `3704f4da82ff02066260fbbd472952771bf721b3890f9f8bfac71dbd403fa879`.
  Builder сохраняет SHA-256 всех historical/candidate/published manifests и полный exclusion
  receipt. Набор не использовался для обучения, выбора epoch, calibration fit или final score.
- Полные локальные FLEURS `ru_ru` и `kk_kz` закреплены на revision
  `4683b04af03d2d9549064c7d72060a9a94bb6046`: проверены audio LFS SHA-256, TSV Git blob IDs,
  gzip CRC и TAR/TSV membership. Созданы 300-row text-group-disjoint raw bona-fide test
  candidates для каждого языка. После QA/VAD готовы `289` RU и `212` KK WAV; все `99` quiet
  rejections записаны отдельно. Это независимые bona-fide candidates. К этому
  base применён pinned Silero V4 Cyrillic FastPitch+HiFi-GAN fixed-profile generator:
  135 text-incompatible rows честно отклонены до synthesis, `366/366` raw clips прошли
  QA/VAD. Опубликованы strict paired candidate layers `214` RU pairs (`428` assets,
  SHA-256 `8afd2d461495eb4b16c6e4de89ea731536d41ff73d7ea7152bfb1ea20e23ec1b`) и
  `152` KK pairs (`304` assets, SHA-256
  `e23f1ce80dbc866eafd4fe1f488f2f55c5705bef1bd9b12e33ce80632328dac2`). Builder
  подтвердил нулевое пересечение с Stage-B train/dev, calibration и frozen final
  manifests по sample/asset/text keys. Это ещё не общий final layer. Подробности —
  `docs/data_sources_fleurs_2026-08-10.md` и
  `docs/data_sources_silero_v4_2026-08-11.md`.
- KSC2 локально прошёл полный read-only multipart audit: все 10 частей (`80 809 122 212` bytes),
  gzip CRC и safe TAR layout; global SHA-256
  `43d1ee6725d737a438125a13997a0abde4159de84ef17d1706fe7921e8632cbe`. Ledger и source lock
  закреплены. `crowdsourced` и `tts` консервативно исключены как potentially legacy KSC/
  KazakhTTS2; `Test` в новых component paths содержит `6 023` paired candidates. Повторная
  проверка статьи KSC2 уточнила priority source paths: authors report test CS rates `8.6%`
  podcasts, `6.3%` television programs (`talkshow`), `3.1%` radio; `tv_news` имеет лишь
  `1.3%`. Archive и official recipe не дают per-row результата, поэтому KSC2 пока не создаёт
  mixed final layer. Новый fail-closed builder повторно stream-проверил весь archive и
  опубликовал `2 632` pending-only candidates из ровно `Test/podcasts` (`1 547`),
  `Test/talkshow` (`383`) и `Test/radio` (`702`):
  `data/manifests/ksc2_test_mixed_annotation_v1.csv`, SHA-256
  `5b80ee0b6d9d80907ed77a9c1e21821a6324e9621f3cb96ad1daab17982acc20`.
  Все rows остаются `language=unknown`, `code_switch=unknown`. Полный source report и
  границы evidence review — `docs/data_sources_ksc2_2026-08-10.md`.
- Для KSC2 packet v1
  добавлен machine-readable lock `packet → receipt → source lock` и воспроизводимый
  `scripts/publish_ksc2_ai_mixed_review.py`. Он публикует только явно сохранённый
  single-AI semantic positive list с explicit Russian/Kazakh transcript-token evidence: `32`
  narrow mixed rows (`22` podcasts, `4` radio, `6` talkshow), SHA-256
  `63257415ad744bf3095e28f6dede7e6e53e9608587258170cdd924651f5075e2`.
  Непросмотренные и неоднозначные `2 600` rows не получают никакого automatic label.
  Evidence имеет single-AI provenance и не является binary training manifest.

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
- source-mixed matrix v1 прошла реальную CLI validation; checkpoint содержит source-matrix
  report, seed, best dev loss и final-test class metrics;
- после реализации matrix: `ruff check src tests scripts services`, `mypy src scripts
  services` (`51` source files) и `pytest -q` (`77` tests) — успешно;
- повторно проверен старый RuASD-only checkpoint на том же ML-DF ready manifest: balanced
  accuracy `0.9262` (bona-fide recall `0.8830`, spoof recall `0.9694`). Это сохранено только
  как protocol comparison, не как общая product metric.
- после добавления Wilson intervals: `ruff check src tests scripts services`, `mypy src
  scripts services` (52 source files) и `pytest -q` (80 tests) — успешно; все три
  source-mixed checkpoint переоценены с сохранением exact correct counts.
- KSC-derived v1 final manifest прошёл `kds validate-manifest` и `kds validate-assets` с
  ledger: `1 842/1 842` assets, ровно `921/921` labels и одна bona-fide/spoof пара на каждый
  `text_id`; source-mixed v2 прошла `kds validate-source-matrix`. Все три v2 checkpoint
  повторно оценены с class, generator-family и voice strata. После реализации `ruff check src
  tests scripts services` и `mypy src scripts services` прошли для `58` source files, `pytest
  -q` — `85` tests успешно.
- KazEmoTTS v2: `kds validate-manifest` и `kds validate-assets` с ledger успешно для `402`
  bona-fide ready, `359` spoof ready и final `718`-row paired manifest; source-mixed matrix v3
  прошла `kds validate-source-matrix`. Все final pairs имеют одинаковый `text_hash` и ровно
  две метки.
- Spark-TTS v3: `kds validate-manifest` и `kds validate-assets` с ledger успешно для `387`
  raw clips, `381` spoof-ready и final `762`-row paired manifest; все `381` final pair имеют
  одинаковый `text_hash` и ровно две метки. `ruff check src tests scripts services`, `mypy src
  scripts services` (`62` source files) и `pytest -q` (`95` tests) успешно прошли.
- eSpeak NG v4: `kds validate-manifest` и `kds validate-assets` с ledger успешно для `407`
  raw clips, `358` spoof-ready и final `716`-row paired manifest; все `358` final pair имеют
  одинаковый `text_hash` и ровно две метки. После полного v5 update `ruff check src tests scripts
  services`, `mypy src scripts services` (`64` source files) и `pytest -q` (`97` tests) успешно
  прошли.
- Frozen unseen-generator suite v1 прошёл реальную validation трёх final family; aggregate и
  provenance strata для будущих final evaluation собираются одним loader traversal. После этого
  `ruff check src tests scripts services`, `mypy src scripts services` (`66` source files) и
  `pytest -q` (`101` tests) успешно прошли.
- Реализован отдельный fail-closed runner для **будущего** unseen-generator protocol:
  `scripts/train_b0_unseen_suite.py`. Строгий run-plan заранее фиксирует SHA-256 suite, ledger и
  всех manifests, B0 config, seed, окно, optimizer parameters, device и output paths. Runner
  выбирает epoch только по dev, хеширует выбранный state до final evaluation, собирает class/
  family/voice strata каждого final за один проход, атомарно без перезаписи публикует каждый из
  checkpoint и JSON receipt. Suite v1 этим runner не переоценивается. Контракт —
  `docs/frozen_b0_run_plan.md`.
- После добавления runner: `ruff check src tests scripts services`, `mypy src scripts services`
  (`68` source files) и `pytest -q` (`106` tests) успешно прошли; реальный suite v1 повторно
  прошёл contract validation без model inference. SHA-256 assets всех пяти его полных input
  manifests повторно проверены: `4 491/4 491` файлов совпадают.
- XLS-R Stage A plan preflight на реальной RTX подтвердил CUDA 12.8, BF16, pinned XLS-R SHA-256,
  source/label/language contract и `1 478/1 478` train/dev assets. Batch `16` profile прошёл с
  peak `2 776 097 792` allocated / `3 198 156 800` reserved bytes. После train read-only audit
  подтвердил минимальный dev loss на выбранном epoch, canonical head state hash, отсутствие
  encoder state и final results в checkpoint, `frozen_final_evaluation_performed=false` и
  `calibrated=false`.
- После Stage A и обновления scope: `ruff check src tests scripts services`, `mypy src scripts
  services` (`71` source file), `pytest -q` (`113` tests) и `git diff --check` прошли успешно.
- Stage B preflight подтвердил CUDA 12.8/BF16, pinned base/head receipts, нулевой доступный
  leakage и `2 387/2 387` assets. Batch profile подтвердил blocks `16`–`23`, `100 769 792`
  trainable encoder и `660 251` head parameters, peak `2 902 534 656` allocated bytes.
  Post-run read-only audit загрузил checkpoint с `weights_only=True`, пересчитал canonical hash
  всех 139 trainable tensors, подтвердил epoch 7 как argmin dev loss и отсутствие frozen final
  результатов/калибровки.
- После завершения Stage B: `ruff check src tests scripts services`, `mypy src scripts
  services` (`75` source files), весь `pytest -q` (`129` tests) и `git diff --check` прошли
  успешно; trailing whitespace в обновлённой документации отсутствует.
- New calibration-dev builder: Ruff и mypy без ошибок; профильные unit tests (`3/3`) прошли.
  `kds validate-manifest` с ledger и `kds validate-assets` подтвердили `977/977` published
  calibration assets.
- После FLEURS/KSC2 update полный текущий worktree прошёл `ruff check src tests scripts
  services`, `mypy src scripts services` (`92` source files), `pytest -q` и `git diff --check`.
  `pytest --collect-only` обнаруживает актуальные `144` tests.
- New FLEURS intake: Ruff, mypy и `3/3` unit tests прошли; отдельная real validation с ledger
  и asset SHA-256 подтвердила `289/289` RU и `212/212` KK ready assets.
- New KSC2 multipart audit: Ruff, mypy и `2/2` unit tests прошли. Реальный streaming audit
  подтвердил `645 860/645 860` audio-to-transcript pairs и один orphan transcript, который
  запрещён к future selection; повторно-suffixed TV-news pairs учитываются корректно.
- Silero V4 FLEURS builder и runtime: `ruff check src tests scripts services`, `mypy src
  scripts services`, полный `pytest -q` и `git diff --check` прошли без ошибок. Реальная
  ledger/schema/asset validation подтвердила `428/428` RU и `304/304` KK assets.
- KSC2 annotation packet builder: Ruff и mypy без ошибок, `5/5` profile tests прошли.
  Реальный run повторно сверил SHA-256 полного multipart archive, а post-publication
  audit подтвердил `2 632/2 632` extracted FLAC hashes, pending-only fields и точные
  source component counts.
- KSC2 single-AI mixed review: профильные tests подтвердили locked 32-row positive-only output,
  immutable candidate packet, published CSV/receipt и evidence обоих языков. Полные `ruff check`,
  `mypy`, `pytest -q` (`144` tests) и `git diff --check` прошли без ошибок.
- KSC2 mixed bona-fide candidate builder проверил 32 source FLAC и выполнил QA/VAD. Опубликованы
  `32` raw и `31` ready rows; единственная запись `09_01_296` исключена только за
  `insufficient_speech`, receipt сохранён. Технический Silero V4 smoke-test на пяти mixed
  transcripts и двух fixed profiles создал `10/10` non-empty, QA/VAD-ready WAV. Он не заявляет
  intelligibility или сохранение языковых segments и не создаёт spoof manifest.
- KSC2 input-pinned Silero candidate: все `31` ready mixed transcripts прошли strict text
  contract и сгенерированы одним fixed `kz_M1` profile. QA/VAD оставил `30` spoof WAV;
  `09_03_368` исключён как `insufficient_speech`. Публикация содержит `30` exact text-hash
  bona-fide/spoof pairs (`60` rows), raw/ready manifests и полный rejection accounting. Spoof
  `language=mixed` означает подтверждённый input transcript, а не acoustic language certificate;
  write-once pair-lock связывает каждую пару с RU/KK evidence tokens и обеими audio SHA-256.
  Слой остаётся research-only candidate без final/calibration/inference use.
- Для этих нерасширяемых 30 KSC2/Silero pairs создан отдельный строгий XLS-R exploratory runner
  и hash-pinned plan. Он закрепляет checkpoint, Stage-B receipt, encoder, ledger, candidate,
  pair-lock, relevant implementation и runtime lock; первый forward создаёт write-once execution
  receipt. Preflight повторно проверил 60 assets, после чего frozen Stage-B v1 один раз оценён
  без обучения, calibration или threshold selection. Raw-boundary recall: bona-fide `29/30`,
  spoof `29/30`; все 30 pair-level results сохранены в
  `docs/research_xlsr_sls_stage_b_ksc2_mixed_exploratory_30.md`. Это exploratory stress-test,
  не final quality, и он не изменяет calibration/final protocol.
- Завершён углублённый read-only audit независимого Russian spoof source. Подходящего release не
  найдено. MLAAD — multilingual synthetic-only, gated, около `159 GB`, `CC-BY-NC-4.0` и содержит
  `reference_speaker`; его актуальная документация допускает speaker-reference audio для части
  multi-speaker generation. Поэтому он excluded pending provenance, а не intake candidate. XMAD
  остаётся convenience package с spoof-only M-AILABS component и не образует binary RU layer.
  Ничего не добавлено в ledger и не скачано. Подробности —
  `docs/russian_spoof_source_search_2026-08-11.md`.
- Пользователь предоставил OpenSTT `tts_russian_addresses_rhvoice_4voices`; его archive и
  manifest прошли новый read-only streaming integrity audit без extraction. Receipt
  `data/licenses/openstt_rhvoice_v1_artifact_audit_receipt.json` (SHA-256
  `0d594e86f6985b5498284aeb56f49461c581d634668b5472907c739e46a63f3c`) подтверждает official
  MD5/size, `1,741,838` physical OPUS/TXT pairs, `1,628,561` unique paths и byte-identical
  SHA-256 verification `226,554` repeated members. Это **не** source acceptance: manifest
  лишён per-row voice/reference provenance, source spoof-only и `CC-BY-NC`. Более того, frozen
  Stage-A train manifest содержит `25` spoof rows с `generator_name=rhvoiceTTS`, то есть тот же
  RHVoice generator family. Independent/unseen-generator gate окончательно не пройден; exact
  asset overlap не может изменить это решение. Нет extraction, ledger, QA, training, calibration
  или final use.
- ToneSpeak `Vikhrmodels/ToneSpeak` прошёл отдельный source-level intake как small public Russian
  spoof-only research source. Exact Hugging Face revision
  `d40f94cd5c7dcf756a8c59a1c465b834220bec56`, seven artifact hashes and every one of `6,998`
  embedded `24 kHz` MP3 закреплены в `tone_speak_ru_v1_artifact_lock.json` и receipt
  `c14d3f0fd38e6ee8675a78b08b627aa43ca618bde52be9c1f90cec8d71996908`. Frozen Stage-A manifest
  имеет zero OpenAI/ten-voice markers. Ledger intentionally permits only personal research and
  records spoof voices as `source_provided`: no per-row generation log, training, calibration
  или final inference exists. Один immutable 100-asset validation OOD candidate прошёл narrow
  two-review acoustic gate (`100/100` WAV) и ровно один hash-pinned frozen XLS-R run (`88/100`
  spoof recall at raw zero boundary); это не меняет research-only scope или отсутствие bona-fide
  class.
- PhoneSpoof окончательно исключён владельцем проекта: его paper описывает Russian TTS через
  реальный telephone channel, но доступ к data требует обращения к авторам для non-commercial use,
  а public page не даёт archive, licence text, hash или per-row provenance. Не связываться с
  авторами, не скачивать и не добавлять его в ledger. STC Spoofing и XMAD также исключены: первый
  строит target voices на `30 s`–`3 h` речи людей, второй применяет voice conversion с original
  speaker voice как reference.
- Реализован и опубликован fail-closed acoustic language-preservation gate v1 для тех же frozen
  30 KSC2/Silero pairs. Immutable `60`-asset listening packet повторно закрепляет actual WAV
  SHA-256, pair-lock и explicit RU/KK transcript evidence; SHA-256 packet —
  `225f5cfe70eb422ef4c5cf131c81537eefea3a4bc9eabd24ace4df87af620421`. Gate требует два
  раздельных pseudonymous acoustic review на каждый asset и публикует `not_eligible` при любом
  отсутствии/несогласии. Два заполненных 60-row review files с разными IDs прошли validator:
  `60/60` assets, `30/30` pairs имеют две записи `pass` и все три обязательных `yes`. Receipt
  SHA-256 `3585cda150e09a40a57bee50f3209e02e836b86738027c224902ae98d98eed01`; это narrow
  acoustic confirmation, не final/product status. Контракт —
  `docs/ksc2_mixed_acoustic_language_gate_v1.md`.
- Создана отдельная controlled **Russian** formant generator family: eSpeak NG `ru`, без новых
  скачиваний, reference audio или voice cloning. Она использует только `75` ready FLEURS RU text
  groups, которые text-disjoint с frozen 214-pair FLEURS/Silero candidate. Exact runtime lock,
  full FLEURS transcript revalidation и `75/75` raw → `75/75` QA/VAD-ready WAV прошли; paired
  candidate содержит `75` exact pairs (`150` assets), SHA-256
  `65b12c43df18fcf7d1ea6b38d2e951cc505963241f5d821092b035b56c9a3af8`.
  Он не является найденным external Russian-only benchmark и не используется для calibration,
  detector inference или final claim. Immutable RU acoustic packet на `150` WAV / `75` pairs,
  SHA-256 `7aecbebbd91444fe75bf261c15715862a7bb194bb501e8862d01b3f02059ff40`, получил две
  complete 150-row reviews с разными pseudo-ID. Strict evaluator подтвердил `150/150` asset
  `pass` (`75/75` pairs); receipt SHA-256
  `ec9f6129391687ef12799a8a6b06ad9a6dfd308a3defba1c9cda09a81e2380ea`. Это narrow acoustic
  confirmation, а не source independence, calibration, final или product status. Подробности —
  `docs/data_sources_espeakng_ru_2026-08-11.md`.

Ранее успешно прошли CUDA smoke test B0, real M4A inspection и XLS-R + SLS GPU smoke test.

## Текущий этап

Русский PyAra research protocol, full RuASD binary protocol, KSC/Common Voice bona-fide layers,
OOD layers и frozen source-mixed matrices v1–v5 подготовлены. KazEmoTTS — третья, Spark-TTS
Kazakh — четвёртая, eSpeak NG formant — пятая independent family; все Kazakh final test не имеют
corpus overlap с train/dev. Existing v2 результаты различаются по seed главным образом из-за KSC
bona-fide recall
(`0.6417`–`0.8252`). Модель годится только для воспроизводимого local research comparison;
model/API score и calibration не включаются. Common Voice Russian v24 разрешён владельцем
проекта только для личного исследования; сохраняются внешний запрет на идентификацию и
re-hosting.

Отдельный frozen unseen-generator suite v1 реализован и провалидирован. Он фиксирует RuASD
train и PyAra dev, затем проверяет 718 KazEmoTTS, 762 Spark-TTS и 716 eSpeak NG final assets.
Контракт разрешает общий `source_name=ksc_slr102` только как corpus provenance, но требует
нулевое cross-final overlap по sample ID, asset SHA-256 и text hash; каждая spoof family не
встречается в train/dev. Новый B0 checkpoint с seed `20260817` выбран только по PyAra dev
(best epoch 4) и затем оценён один раз на каждом final: balanced accuracy `0.9178` KazEmoTTS,
`0.9331` Spark-TTS, `0.9399` eSpeak NG. Это finite research result без calibration/API score;
KazEmoTTS остаётся class-only stratum, так как его final test не повторяется ради недостающей
voice таблицы. Полный report — `docs/research_b0_unseen_generator_suite_v1.md`.

Для следующего независимого suite подготовлен воспроизводимый execution path с обязательным
preflight. Готового run-plan v2 намеренно нет: его можно создать только вместе с новыми, ранее
не раскрытыми final assets и зафиксировать до первого запуска.

XLS-R+SLS Stage B v1 завершён как train/dev-этап. Новый fresh PyAra dev вырос с 61 до 970
clips и не повторяет доступные IDs/assets/text groups Stage A dev или RuASD train. Выбран epoch
7 с balanced accuracy `0.9393`, но checkpoint нельзя выдавать за готовую модель: PyAra не
публикует speaker IDs, наблюдается сильное колебание dev по эпохам, record-level calibration и
frozen final inference в исходном train/dev run не выполнялись. Позднейший 30-pair exploratory
stress-test отделён отдельным plan и не меняет этот статус. Артефакт содержит SLS-head и blocks
`16`–`23` и требует неизменный pinned XLS-R base.

Новый PyAra calibration dev подготовлен отдельно от обоих epoch-selection dev. Его 977 assets
и receipts заморожены, но temperature fit ещё не выполнен — корректно сначала получить новый
unseen RU/KK/mixed final contract. FLEURS теперь закрывает candidate binary RU и KK: Silero V4
создал `214` и `152` проверяемые pairs, соответственно. Они остаются неиспользованными final
assets. KSC2 — наиболее обоснованный будущий mixed bona-fide candidate, но не является mixed
final source: paper указывает на `Test/podcasts`, `Test/talkshow`, `Test/radio` как приоритетные
component paths, а archive не публикует per-row code-switch result. Hash-pinned packet из
`2 632` rows уже готов; узкий 32-row single-AI transcript evidence layer опубликован, из него
QA/VAD готовы `31` bona-fide candidates, а остальные rows намеренно остаются unknown. Для 30
из них создан exact paired Silero candidate и ровно один exploratory Stage-B inference; mixed
status synthetic WAV по-прежнему относится только к input transcript. Narrow 30-pair layer теперь
прошёл acoustic language-preservation gate. Controlled 75-pair FLEURS RU/eSpeak candidate добавляет
независимую generator family и также прошёл narrow RU acoustic gate: `150/150` WAV имеют два
`pass/yes/yes` decision. Полноценный acoustic-validated mixed binary final layer всё ещё
отсутствует. ToneSpeak теперь даёт independently sourced Russian **spoof-only** research
candidate: public Apache-2.0 revision, all `6,998` MP3 и source-card generator/voice fields
прошли local integrity audit, а Stage-A manifest не содержит OpenAI/его ten voice markers.
Однако это ещё не final/product source: per-row generation provenance only source-provided и
нет bona-fide class. Из ToneSpeak `validation` теперь published один hash-pinned balanced
`100`-asset spoof-only OOD candidate (`10` voice IDs × `10`), all `100/100` raw MP3 → ready
16-kHz WAV прошли QA/VAD и no project manifest text-hash collision найден против 60 valid
existing manifests. Write-once acoustic packet и две 100-row review forms завершены: strict
evaluator подтвердил `100/100` WAV с двумя `pass/yes/yes` decisions; report v2 SHA-256
`8525df3980210c8e2b4dd827859e0d0c7b1ecb74f512ab6a5eaa628cbeb55df6` pins both review CSV.
Это narrow auditory confirmation exact assets, не final/product result. После отдельного
hash-pinned preflight выполнен ровно один frozen XLS-R OOD run: `88/100` spoof recall на raw
zero boundary, `12` raw bonafide predictions; no training, calibration, threshold fitting или
binary metric. Полный receipt — `docs/research_xlsr_sls_stage_b_tone_speak_ru_ood_100.md`.
OpenSTT RHVoice прошёл artifact-integrity gate,
но не проходит independent/unseen-generator gate из-за `rhvoiceTTS` в Stage-A train; он не
является исключением из этого ограничения.

Для fixed 30 pairs acoustic language-preservation gate завершён: `60/60` bona-fide/synthetic
WAV имеют два формально независимых pseudonymous `pass` review. Ни ASR, ни input transcript
не подменяли эти decisions. Gate устранил только один узкий provenance blocker и не открывает
final/product protocol без bona-fide counterpart, acoustic review конкретных final assets и
остальных ранее зафиксированных условий. Новый 75-pair FLEURS RU/eSpeak layer закрывает только
generator-diversity gap в personal research и уже прошёл свой narrow acoustic gate.

## Дальше

1. Stage B v1 больше не менять и не переобучать. Calibration dev уже создан, но до любого
   final inference нужны нераскрытый binary mixed layer и отдельный hash-pinned final-run plan.
   RU/KK FLEURS+Silero candidates уже созданы и теперь запрещены для adaptation, threshold или
   calibration. KSC2 packet v1 уже ограничен `Test/podcasts`, `Test/talkshow`, `Test/radio`
   и исключает `crowdsourced`/`tts`; immutable original не менять. Использовать только
   опубликованный 31-row QA/VAD-ready single-AI evidence и нерасширяемый 30-pair input-pinned
   Silero candidate как narrow research assets; уже выполненный exploratory run не повторять и
   не расширять его LID/character/component-path heuristic. Перед final/calibration/API всё ещё
   нужен explicit language-preservation quality gate: текущий Silero layer подтверждает input
   text и WAV/QA status, но не acoustic content. Для narrow 30-pair layer два review уже
   завершены и не повторяются. Новый 75-pair FLEURS RU/eSpeak candidate уже прошёл свой narrow
   two-review RU acoustic-preservation gate; не расширять этот result на другие WAV и не считать
   его final quality. ToneSpeak — новый independent Russian spoof-only **research** source,
   имеет один frozen 100-asset validation OOD candidate, прошедший narrow two-review acoustic
   gate. Он не заменяет bona-fide counterpart и не проходит product gate, поскольку
   voice/generation provenance only source-provided. Отдельный hash-pinned plan уже выполнил
   единственный OOD run; не повторять его и нельзя подбирать по нему threshold, calibration или
   architecture. OpenSTT RHVoice прошёл local
   artifact-integrity audit, но уже
   исключён из этой роли: RHVoice family присутствует в frozen Stage-A train. MLAAD не
   принимать:
   gated 159-GB `CC-BY-NC` release с возможным reference-speaker provenance противоречит текущим
   требованиям; XMAD не подменяет его, поскольку Russian M-AILABS component spoof-only. Искомый
   release обязан документировать text-only fixed-voice TTS, immutable revision, права,
   generator и per-row provenance.
2. Frozen unseen-generator v1 уже выполнен один раз: не менять contract, не повторять эти final
   tests и не использовать новые Piper voices. Для следующего независимого run сначала создать
   новый suite с ранее не раскрытыми final assets, затем строгий run-plan по
   `docs/frozen_b0_run_plan.md`, выполнить `--validate-only` и только после этого один final run.
   Не подбирать threshold, calibration, augmentation или architecture по final assets и не
   сводить ML-DF, PyAra, KSC и RuASD к одной «accuracy».
3. LRLspoof — исключённый вариант, не возвращаться к нему: 452 GB sequential archive без
   selective Kazakh download и spoof-only состав не соответствуют этому проекту. Не скачивать,
   не добавлять в ledger, не запрашивать shards и не смешивать с KSC.
4. Собственные записи людей не собирать. Расширять данные только двумя путями: искать готовые
   RU/KK/mixed bona-fide или spoof releases с проверяемыми правами и provenance либо создавать
   text-only synthetic clips новыми архитектурными TTS family с pinned model hashes, встроенными
   публичными voices и полным QA/rejection ledger. Reference audio, voice cloning и имитация
   конкретного человека исключены.

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
  speaker IDs. Fresh 970-row dev улучшает объём, но остаётся selection set, а не independent
  final evaluation или calibration basis.
- Source-mixed B0 пока не даёт основания включать risk score, calibration или API inference:
  ML-DF balanced accuracy имеет большой seed spread (`0.7340`–`0.8579`), а bona-fide recall
  на первом запуске лишь `0.4681`. RuASD OOD shard не независим от full RuASD training source.
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
