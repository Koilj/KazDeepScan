# KazDeepScan

KazDeepScan — локальный personal-research проект для исследования признаков
синтезированности русской, казахской и смешанной речи. Это не идентификация говорящего,
не доказательство мошенничества и не развёрнутый сервис оценки риска.

Текущее реализованное состояние включает проверяемый фундамент данных, обученный research
checkpoint и строго ограниченный evaluation-контур:

- безопасная проверка размера, MIME-типа, фактического контейнера и длительности;
- нормализация через `ffmpeg` в mono PCM WAV, 16 кГц;
- измерение уровня, клиппинга и DC offset;
- выделение речи WebRTC VAD, отказ при менее чем 2.5 с речи;
- детерминированные окна 4.04 с с шагом 2.0 с;
- CSV-манифест с проверкой прав, происхождения и leakage между split-ами.
- KSC / OpenSLR SLR102 downloader с проверкой размера и gzip CRC, а также ingestion только из
  чистого архива; metadata и транскрипции берутся из его `Meta/` + `Transcriptions/`;
- Common Voice Russian v24 intake с pinned размером и SHA-256 archive, tar whitelist,
  атомарным MP3 slice extraction и leakage-safe split по связанным client/text группам;
- ML-DF v1 Italian intake с archive/CRC/metadata whitelist и изолированным cross-lingual
  OOD manifest; он не используется как Russian/Kazakh training или calibration data;
- RuASD: fake-only shard для OOD и full-release intake для personal-research binary slices;
  оба проверяют SHA-256/TAR/JSON-WAV pairing, а full release берёт только raw recordings;
- read-only audit полного локального RuASD release: pinned catalog всех 250 TAR artifacts,
  безопасная проверка JSON/WAV layout и контроль доступности source speaker/voice metadata;
  full release не предоставляет проверяемые groups для speaker-disjoint protocol;
- PyAra v7 personal-research intake с ZIP/TSV-WAV checks, text-leakage-safe binary slice и
  B0 smoke baseline; источник не даёт speaker IDs;
- явный training-protocol gate: текущий рабочий режим — `research`; строгая ветка `product`
  остаётся только неактивным fail-closed validator для возможного будущего изменения scope;
- local consent-registry validator сохранён только как архивный инструмент. Текущая дорожная
  карта не предусматривает запись голосов людей, voice cloning или сбор consented corpus;
  PII и соглашения не создаются и никогда не хранятся в Git.
- B0 foundation и реальные XLS-R + SLS Stage A/B: hash-pinned CUDA/BF16 runners сначала
  обучают SLS-head, затем размораживают только последние восемь XLS-R blocks. V2 использует
  исправленный RuASD train (`1 471`) и новый 969-row PyAra dev без доступного
  sample/asset/text/group overlap;
- отдельный write-once Stage-B v2 contract fitted temperature только на disjoint PyAra
  calibration и один раз выполнил раздельный confirmatory RU/KK/mixed run. Это не pooled
  product score: mixed assets ранее видел checkpoint v1, а KK acoustic gate был завершён только
  после раскрытия результата;
- для exact 304-asset FLEURS KK/Silero layer завершён отдельный post-inference two-review gate:
  packet не содержал predictions, обе полные формы прошли строгую проверку, `304/304` assets
  получили `pass`; статус уже раскрытой метрики при этом не повышен;
- historical Stage-C source inventory v2 зафиксировал тогдашние 55 RU и 197 KK fresh text
  groups; после завершённых selection/QA/inference pinned FLEURS RU больше не имеет usable
  unevaluated capacity для следующего final. Второй disjoint KSC2 semantic pass довёл historical
  fresh QA-ready mixed слой с 1 до 58 groups;
- до synthesis заморожена all-eligible selection policy: 55 RU, 60 KK и 58 mixed groups без
  model-based отбора и backfill. Bona-fide QA оставил 50/60/58 rows; пять тихих RU recordings
  получили accounted rejection, combined ready слой содержит 168 rows;
- character-inventory normalization заморожена до повторного synthesis и хранит отдельный hash
  фактически произнесённого текста, не меняя исходные text IDs/hashes. Нормализованный run создал
  168/168 WAV; audio QA оставил 167 и одну mixed строку отклонил без backfill. Опубликован
  167-pair candidate (`50 RU / 60 KK / 57 mixed`) с нулевым exact sample/audio/text overlap
  против 11 869 prior configured rows;
- абсолютная architecture novelty заменена на доказуемый exact checkpoint/runtime gate, потому
  что historical RuASD manifests не содержат architecture IDs. ISSAI KazakhTTS2 Male2
  Tacotron2 + ParallelWaveGAN прошёл rights/artifact/config/exposure и CUDA technical-smoke
  checks без cloning/reference audio. Exact route новый, но Male2 speaker alias уже встречался
  через Piper, поэтому speaker-independence не заявляется. Два pre-inference listening review
  одобрили `kk`, `ru` и `mixed` только для подготовки нового candidate. Обе full-asset 167-row
  формы прошли до inference (`167/167` exact synthetic WAV); затем ровно один Stage-C GPU run
  оценил 50 RU, 60 KK и 57 mixed pairs. Это asset-level-blind research evidence, не source- или
  speaker-independent result; повтор run и tuning по его final errors запрещены;
- отдельный Stage-D RU слой завершён строго по frozen Common Voice v24 текстам: новый exact
  checkpoint/runtime route Dialogs-RU VITS2, fixed `Masha`/`neutral`, прошёл rights/artifact и
  route-exposure lock. Это только personal-research route: model repository не содержит
  отдельного `LICENSE`, поэтому OpenRAIL declaration и dataset license не являются broad
  commercial clearance. Generic historical `vits2TTS` не позволяет заявлять architecture-family
  novelty;
- Stage D создал ровно 73 synthetic WAV без замены текстов; technical QA оставил 55 binary pairs
  и отклонил 18 пар как `insufficient_speech` без resynthesis/backfill. Две независимые
  full-asset review формы прошли для всех 110 exact assets до inference; затем immutable plan
  выполнил ровно один GPU run. Это отдельный RU result, не source-, speaker- или
  architecture-independent evidence; повтор run и tuning по его ошибкам запрещены;
- отдельная v3 ветка завершена по изолированному governance contract: train RuASD (1 471),
  Stage-A PyAra dev (61), fresh Stage-B dev (969), calibration PyAra (976) и exact Stage-D final
  (55 pairs) проверены на отсутствие leakage. Только train получает детерминированную
  label-agnostic symmetric channel/codec/replay augmentation; Stage A выбрал epoch 3, Stage B —
  epoch 4 только по своим dev loss. После write-once preflight выполнен один v3 final run;
  Stage-D v2 logits/errors не загружались. Поскольку те же exact pairs уже оценивались v2,
  этот v3 result не называется blind/unseen и не может использоваться для дальнейшей настройки;
- новый Silero V5.5 RU `eugene` закреплён только как exact text-only checkpoint/runtime route:
  hash-pinned package, source, ZIP/dispatcher audit и fail-closed wrapper запрещают reference
  audio, cloning, random profile, SSML и `voice_path`. Route audit видит 0 exact V5.5/eugene
  rows среди 18 605 historical spoof rows, но 1 265 legacy Silero rows исключают architecture-,
  vendor- и speaker-independence claims. Новый bona-fide pre-QA selection уже заморожен,
  materialized, literal-text bound и синтезирован в raw 75-row layer; technical QA оставил 42
  spoof rows и exact 42-pair candidate заморожены. Две 84-row forms с distinct pseudonymous
  reviewer IDs прошли strict technical acoustic gate: `84/84` exact assets получили по два
  `pass/yes/yes/yes/yes` решения. Subsequent exposure audit against 30 configured research
  contracts / 17 manifests / 12,313 prior rows found `0/0/0` sample/audio/text overlap. One
  immutable XLS-R+SLS Stage-B v2 contract now pins that evidence, a fixed 976-row calibration,
  the selected checkpoint and write-once paths. Its 1,060-asset CUDA/BF16 preflight and exactly
  one final GPU run completed: `84/84` accuracy and balanced accuracy `1.0000`, with 42/42 pairs
  correct. Это fixed source-linked layer, not source/speaker/vendor/architecture-independent
  evidence or product quality. Its final report has a known immutable `detector_inference_performed`
  metadata defect; the reconciliation receipt binds the actual one-time execution. Distinct IDs
  сами не доказывают organizational independence;
- перед любым выбором нового RU final набора весь full Common Voice `test` прошёл historical
  exposure screen, затем fixed V5.5 literal-text gate без lexical rewrite. Из `10 261` source
  records остаётся `5 600` в `1 337` client groups. Из них отдельный immutable receipt выбрал
  `80` exact clip metadata — по одной записи на group, без audio/model-based selection и без
  backfill. Technical decode/QA/VAD извлёк все 80 MP3 и оставил 75 ready WAV; пять
  `insufficient_speech` rejections учтены без replacement. Это всё ещё не synthetic/audio-review
  или model result;
- explicit source-mixed research matrix и B0 runner, который не допускает overlap исходных
  corpus между train/dev/final-test и проверяет обычный sample/SHA-256/group/text leakage поверх
  этого;
- узкий KSC2 mixed evidence layer из single-AI semantic transcript review: каждая из 32 строк
  содержит explicit Russian/Kazakh token evidence; остальные 2 600 candidates остаются unknown;
- отдельный Stage-C semantic-review delta добавил 59 disjoint explicit decisions: 57 прошли
  QA/VAD, 2 получили accounted rejection; после исключения 30 прежних exposed rows доступны 58
  fresh mixed groups, а 2 541 непросмотренная строка остаётся unknown;
- из этого evidence подготовлен QA/VAD-ready KSC2 bona-fide candidate: 31 из 32 строк; 1
  rejection сохранён. Silero V4 technical smoke-test создал 10 технически готовых WAV, но пока
  не создаёт acoustic language-quality claim. Отдельный input-pinned research candidate содержит
  30 QA/VAD-ready exact bona-fide/spoof pairs и явно отмечает эту границу provenance;
- FastAPI health/readiness/upload scaffold, который не выдаёт score без обученного,
  калиброванного model release.

Есть personal-research B0 checkpoints на PyAra, full RuASD и двух source-mixed matrix. На трёх
зафиксированных seed source-mixed ML-DF cross-lingual test дал balanced accuracy 73.40–85.79%.
Новый controlled Kazakh KSC-derived test дал 82.08–91.26%, но bona-fide recall только
64.17–82.52%: это существенные ложные срабатывания и сильная source/seed sensitivity. Ни один
источник не даёт проверяемый speaker-disjoint binary split. Отчёты:
[RuASD](docs/research_b0_ruasd_full_2000.md),
[source-mixed v1](docs/research_b0_source_mixed_v1.md) и
[source-mixed v2 Kazakh](docs/research_b0_source_mixed_v2_kk.md). XLS-R+SLS Stage B v2 выбрал
epoch 5 по fresh PyAra dev loss `0.17405`, accuracy `0.9216` и balanced accuracy `0.9229`;
полный train/dev отчёт — [XLS-R+SLS Stage B v2](docs/research_xlsr_sls_stage_b_v2.md).
Отдельный calibrated confirmatory run дал balanced accuracy RU `0.9800`, KK `1.0000`, mixed
`0.9333`, но его ограничения не позволяют включать model/API risk score. Отдельный 30-pair
KSC2/Silero exploratory stress-test проведён с write-once plan и без calibration/threshold
selection; это не final quality. Полный pair-level report —
[здесь](docs/research_xlsr_sls_stage_b_ksc2_mixed_exploratory_30.md).

Отдельный Stage-D Dialogs-RU VITS2 / Masha-neutral layer дал balanced accuracy `0.9727`
(`107/110`; 55 exact pairs), bona-fide recall `0.9455` и spoof recall `1.0000`. Это один
предварительно зафиксированный RU research run с fixed boundary `0.5`, а не основание для
подстройки v2/v3. Полный rights, QA, review и inference receipt —
[Stage D Dialogs-RU](docs/stage_d_dialogs_ru_vits2_intake_2026-08-13.md).

Независимо обученный XLS-R+SLS v3 на том же immutable Stage-D наборе дал `107/110` и balanced
accuracy `0.9727`, с bonafide recall `0.9636`, spoof recall `0.9818` и `52/55` полностью верных
пар. Его checkpoint был выбран только по v3 Stage-A/Stage-B dev loss, а temperature — только по
disjoint calibration. Exact final assets уже были известны через v2, поэтому результат —
governed confirmatory RU evidence, не новый blind test. Полный contract и receipt —
[XLS-R+SLS v3 Stage-D](docs/research_xlsr_sls_v3_stage_d_dialogs_ru_v1.md).

## Требования

- Python 3.11–3.13 (зафиксирован CPython 3.13: для него есть готовый WebRTC VAD wheel);
- `ffmpeg` и `ffprobe` в `PATH` либо их пути в `KDS_FFMPEG_BINARY` и
  `KDS_FFPROBE_BINARY`;
- для рабочего VAD — пакет `webrtcvad-wheels`.

Для воспроизводимой установки используется [uv](https://docs.astral.sh/uv/). На Linux с
NVIDIA проект фиксирует CUDA 12.8-сборки PyTorch в `uv.lock`; не заменяйте их случайной
CPU-сборкой.

```bash
uv sync --all-extras --locked
uv run pytest
uv run ruff check src tests
uv run mypy src
```

В текущей рабочей машине локальная, не отслеживаемая Git сборка FFmpeg находится в
`.tools/ffmpeg`. Чтобы использовать её с CLI, передайте пути явно:

```bash
export KDS_FFMPEG_BINARY="$PWD/.tools/ffmpeg/bin/ffmpeg"
export KDS_FFPROBE_BINARY="$PWD/.tools/ffmpeg/bin/ffprobe"
uv run kds inspect-audio /safe/path/recording.m4a --mime-type audio/mp4
```

## Команды

```bash
# Проверить один локальный файл. MIME передаётся только как дополнительная проверка.
kds inspect-audio /safe/path/recording.m4a --mime-type audio/mp4

# Проверить схему, отсутствие data leakage и статус источников в license ledger.
kds validate-manifest data/manifests/slice.csv \
  --license-ledger data/licenses/license_ledger.csv

# Проверить, что каждый asset существует и совпадает с зафиксированным SHA-256.
kds validate-assets data/manifests/slice.csv --audio-root data

# Детерминированно назначить train/dev/test по компонентам group/speaker/text.
kds assign-splits data/manifests/input.csv data/manifests/slice.csv --seed 20260808

# Создать bona-fide slice KSC только из чистого архива.
uv run python scripts/ingest_ksc_slr102.py \
  --archive /home/ruslan/Downloads/ISSAI_KSC_335RS_v1.1_flac.tar.gz \
  --output-manifest data/manifests/ksc_slice.csv \
  --slice-name first-250 --limit-per-split 250

# Создать русский bona-fide slice из проверенного локального Common Voice v24 archive.
uv run python scripts/ingest_common_voice_ru_v24.py \
  --archive /home/ruslan/Downloads/cv-corpus-24.0-2025-12-05-ru.tar.gz \
  --output-manifest data/manifests/common_voice_ru_v24_first_250.csv \
  --slice-name first-250 --limit-per-source-split 250

# Создать сбалансированный personal-research binary slice из полного локального RuASD.
# По умолчанию заново проверяются SHA-256 всех 250 pinned TAR; --skip-sha256 допустим
# только после документированного полного audit этого же неизменённого набора.
uv run python scripts/ingest_ruasd_research.py \
  --archive-dir /home/ruslan/Downloads/RuASD \
  --output-manifest data/manifests/ruasd_ru_v1_full_research_2000_v2.csv \
  --selection-receipt data/manifests/ruasd_ru_v1_full_research_2000_v2_selection_receipt.json \
  --slice-name research-2000-v2 --limit-per-label 1000 --min-per-stratum 1 \
  --selected-at-utc 2026-08-12T00:00:00Z

# Проверить B0 checkpoint на отдельном manifest без калибровки.
uv run python scripts/evaluate_b0.py \
  --checkpoint models/b0-pyara-research-500.pt \
  --manifest data/manifests/ml_df_it_v1_ood_200_ready.csv \
  --audio-root data \
  --license-ledger data/licenses/license_ledger.csv \
  --split ood --device cuda

# Для финального OOD-набора потребовать целое семейство генератора вне train/dev/test.
kds validate-manifest data/manifests/ood.csv --require-ood-generator

# Проверить, что полный manifest годится только для заявленной цели.
kds validate-training-protocol data/manifests/slice.csv \
  --license-ledger data/licenses/license_ledger.csv --purpose research

# Проверить, что train/dev/final test используют разные исходные corpus.
# ML-DF остаётся отмеченным в своём manifest как OOD: это cross-lingual stress-test,
# а не русскоязычная «общая accuracy».
kds validate-source-matrix configs/research/source_mixed_v1.json \
  --license-ledger data/licenses/license_ledger.csv

# Опубликовать narrow KSC2 single-AI mixed evidence review. Скрипт не запускает LID,
# ASR или эвристику: он воспроизводит только явно сохранённые token-level review decisions.
uv run python scripts/publish_ksc2_ai_mixed_review.py \
  --packet data/manifests/ksc2_test_mixed_annotation_v1.csv \
  --packet-receipt data/licenses/ksc2_test_mixed_annotation_v1_receipt.json \
  --packet-lock data/licenses/ksc2_test_mixed_annotation_v1_packet_lock.json \
  --reviewed-at 2026-08-11T00:00:00Z \
  --output-csv data/manifests/ksc2_test_mixed_ai_review_v1.csv \
  --output-receipt data/licenses/ksc2_test_mixed_ai_review_v1_receipt.json

# Обучить B0 с source-disjoint train/dev и один раз оценить final test после выбора epoch по dev.
uv run python scripts/train_b0_matrix.py \
  --matrix configs/research/source_mixed_v1.json \
  --audio-root data \
  --license-ledger data/licenses/license_ledger.csv \
  --output models/b0-source-mixed-research-v1.pt --device cuda

# v2 использует тот же source-disjoint protocol, но frozen KSC-derived Kazakh final test.
# Не использовать его для подбора epoch, threshold или calibration.
kds validate-source-matrix configs/research/source_mixed_v2_kk.json \
  --license-ledger data/licenses/license_ledger.csv
uv run python scripts/train_b0_matrix.py \
  --matrix configs/research/source_mixed_v2_kk.json \
  --audio-root data \
  --license-ledger data/licenses/license_ledger.csv \
  --output models/b0-source-mixed-research-v2-kk.pt --device cuda

# v4 — отдельный frozen Spark-TTS Kazakh final test. Его нельзя использовать
# для выбора epoch, threshold или calibration.
kds validate-source-matrix configs/research/source_mixed_v4_sparktts.json \
  --license-ledger data/licenses/license_ledger.csv

# v5 — отдельный frozen eSpeak NG Kazakh formant final test. Его нельзя использовать
# для выбора epoch, threshold или calibration.
kds validate-source-matrix configs/research/source_mixed_v5_espeakng.json \
  --license-ledger data/licenses/license_ledger.csv

# Проверить все три frozen Kazakh final test как unseen generator family
# относительно фиксированных RuASD train и PyAra dev. Команда не обучает модель.
kds validate-unseen-generator-suite configs/research/unseen_generator_ood_v1.json \
  --license-ledger data/licenses/license_ledger.csv

# Stage C: воспроизвести exact-route exposure audit до synthesis/detector inference.
uv run python scripts/audit_stage_c_generator_route.py \
  --model-lock configs/research/kazakhtts_tacotron2_pwg_v1_models.json \
  --manifest-directory data/manifests \
  --fixed-voice-alias ISSAI_KazakhTTS2_M2 \
  --audited-at 2026-08-12T00:00:00Z \
  --output data/manifests/fresh_suite_stage_c_generator_route_gate_v2.json

# Smoke v1 и two-listener language gate уже выполнены; write-once report не пересоздавать.
# Команда ниже сохранена только как исторический receipt. Gate разрешил сбор candidate,
# но не detector inference.
uv run python scripts/kazakhtts_stage_c_acoustic_gate.py evaluate \
  --packet data/manifests/fresh_suite_stage_c_kazakhtts_acoustic_gate_packet_v1.csv \
  --reviewer-1 data/manifests/fresh_suite_stage_c_kazakhtts_acoustic_review_reviewer_1.csv \
  --reviewer-2 data/manifests/fresh_suite_stage_c_kazakhtts_acoustic_review_reviewer_2.csv \
  --evaluated-at 2026-08-12T00:00:00Z \
  --output-report data/manifests/fresh_suite_stage_c_kazakhtts_acoustic_gate_report_v1.json

# Stage C full-asset gate: packet и обе формы уже созданы, prepare повторно не запускать.
# После двух независимых reviews строго проверить 167 exact synthetic assets. Во всех
# прошедших строках нужны pass/yes/yes/yes/no; любой иной ответ оставляет suite blocked.
uv run python scripts/prepare_fresh_suite_stage_c_acoustic_gate.py evaluate \
  --packet data/manifests/fresh_suite_stage_c_kazakhtts_full_acoustic_gate_packet_v1.csv \
  --reviewer-1 data/manifests/fresh_suite_stage_c_kazakhtts_full_acoustic_review_reviewer_1.csv \
  --reviewer-2 data/manifests/fresh_suite_stage_c_kazakhtts_full_acoustic_review_reviewer_2.csv \
  --evaluated-at 2026-08-12T00:00:00Z \
  --output-report data/manifests/fresh_suite_stage_c_kazakhtts_full_acoustic_gate_report_v1.json

# После full-asset pass Stage-C XLS-R plan выполняется только один раз. Сначала
# validate-only проверяет pinned assets, права, gate, exposure audit и CUDA/BF16 без inference.
uv run python scripts/evaluate_xlsr_fresh_suite_stage_c.py \
  --plan configs/research/xlsr_sls_stage_b_v2_fresh_suite_stage_c_v1.json \
  --audio-root data --validate-only

# Run `xlsr-sls-stage-b-v2-fresh-suite-stage-c-v1` уже завершён; не запускайте команду без
# --validate-only повторно. Execution lock и report намеренно блокируют повторные inference.
# Calibration выполнялась только на pinned PyAra role; threshold и pooled metric не выбирались.

# Stage D Dialogs-RU также уже завершён. Не запускайте его synthesis, preflight или inference:
# write-once outputs намеренно запрещают повтор. Его versioned receipts и результат находятся в
# docs/stage_d_dialogs_ru_vits2_intake_2026-08-13.md; локальные WAV, model bundle и artifacts/
# не отслеживаются Git.

# v3 training и его Stage-D final также уже завершены. Не запускать Stage A/B v3, preflight или
# inference повторно: immutable plan, execution lock и результат описаны в
# docs/research_xlsr_sls_v3_stage_d_dialogs_ru_v1.md.

# Исторический синтаксис preflight для XLS-R+SLS Stage A v2.
# Этот plan уже выполнен: write-once guard теперь ожидаемо отклонит и validate/profile/train.
# Для нового эксперимента нужны новый run_id и новые output paths.
uv run python scripts/train_xlsr_sls_stage_a.py \
  --plan configs/research/xlsr_sls_stage_a_v2.json \
  --audio-root data --validate-only

# Исторический синтаксис отдельного CUDA/VRAM profile до обучения:
uv run python scripts/train_xlsr_sls_stage_a.py \
  --plan configs/research/xlsr_sls_stage_a_v2.json \
  --audio-root data --profile-only

# Исторический синтаксис preflight для Stage B v2. Plan уже выполнен, поэтому write-once
# guard ожидаемо отклонит повторный validate/profile/train. Новый experiment требует
# новых run_id, dev/calibration manifests и output paths.
uv run python scripts/train_xlsr_sls_stage_b.py \
  --plan configs/research/xlsr_sls_stage_b_v2.json \
  --audio-root data --validate-only

# Исторический синтаксис CUDA/VRAM profile до обучения:
uv run python scripts/train_xlsr_sls_stage_b.py \
  --plan configs/research/xlsr_sls_stage_b_v2.json \
  --audio-root data --profile-only

# Confirmatory v1 уже выполнен и write-once outputs запрещают повтор. Его точное pinned
# implementation tree сохранено в commit 52d6e6b; --validate-only для исторического plan
# следует выполнять в отдельном checkout этого commit. Для нового experiment требуется
# новый plan с новыми output paths и честным disclosure test-set history.
uv run python scripts/evaluate_xlsr_research_final.py \
  --plan configs/research/xlsr_sls_stage_b_v2_research_final_v1.json \
  --audio-root data --validate-only

```

Все результаты CLI содержат только технические метаданные. Команды не отправляют аудио по
сети и не создают его постоянных копий.

Подробные контракты находятся в [docs/audio_pipeline.md](docs/audio_pipeline.md),
[docs/data_contract.md](docs/data_contract.md), [docs/threat_model.md](docs/threat_model.md) и
[docs/frozen_b0_run_plan.md](docs/frozen_b0_run_plan.md). XLS-R Stage A/B описаны в
[docs/xlsr_sls.md](docs/xlsr_sls.md); подготовка отдельного Stage-B calibration dev и границы
следующего final protocol — в [docs/xlsr_stage_b_final_preparation.md](docs/xlsr_stage_b_final_preparation.md).
Выполненный calibrated confirmatory receipt — в
[docs/research_xlsr_sls_stage_b_v2_research_final_v1.md](docs/research_xlsr_sls_stage_b_v2_research_final_v1.md).
KSC2 single-AI mixed evidence review и ограничения альтернативных LID/ASR sources — в
[docs/ksc2_mixed_ai_review_v1.md](docs/ksc2_mixed_ai_review_v1.md); готовый bona-fide candidate
и результат технического TTS smoke-test — в
[docs/ksc2_mixed_candidate_v1.md](docs/ksc2_mixed_candidate_v1.md). Fresh-source capacity и
Stage-C contract зафиксированы в
[docs/fresh_research_suite_stage_c_source_review_2026-08-12.md](docs/fresh_research_suite_stage_c_source_review_2026-08-12.md),
а завершённый Stage-D RU contract — в
[docs/stage_d_dialogs_ru_vits2_intake_2026-08-13.md](docs/stage_d_dialogs_ru_vits2_intake_2026-08-13.md),
а v3 governance/training/final receipt — в
[docs/research_xlsr_sls_v3_stage_d_dialogs_ru_v1.md](docs/research_xlsr_sls_v3_stage_d_dialogs_ru_v1.md).
Проверенный и ровно один раз оценённый в research-only scope Silero V5.5 route и immutable pre-QA
selection нового Common Voice RU slice описаны в
[docs/silero_v5_5_ru_eugene_intake_2026-08-13.md](docs/silero_v5_5_ru_eugene_intake_2026-08-13.md)
и [pre-QA selection receipt](docs/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_selection_v1.md).
Technical materialization этого slice описана в
[отдельном receipt](docs/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_materialization_v1.md),
а literal-text binding всех 75 ready rows — в
[text-binding receipt](docs/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_text_binding_v1.md),
а one-WAV-per-text synthesis — в
[synthesis receipt](docs/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_synthesis_v1.md), а его
42-row technical-QA layer — в
[QA receipt](docs/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_spoof_technical_qa_v1.md), а
84-asset immutable pair lock — в
[pairing receipt](docs/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_pairing_v1.md), а completed
two-review technical gate — в
[acoustic-gate receipt](docs/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_acoustic_gate_v1.md),
а его project-exposure audit — в
[versioned receipt](data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_candidate_project_exposure_v1.json),
а prepared evaluation contract — в
[V5.5 contract receipt](docs/research_xlsr_sls_stage_b_v2_common_voice_ru_v24_silero_v5_5_eugene_v1.md),
а immutable report-status correction — в
[execution reconciliation receipt](data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_execution_reconciliation_v1.json).
Новый VoxForge Russian bona-fide source прошёл byte-locked GPL/transcript intake и strict
pre-extraction project-exposure screen: `6,412` WAV, `194` source-provided contributor groups,
`81` canonical text groups и `0/0/0/0/0` sample/two-text/group overlap. Затем metadata-only
selection заморозил `81` records с уникальными text и conservative contributor groups; raw WAV
не извлекались. Новый Qwen3-TTS CustomVoice Q8_0 / fixed `aiden` route прошёл local
six-artifact/CUDA verification и exact-route audit: `0` overlaps в `18,764` historical spoof
rows (`59` manifests), включая `0` legacy Qwen identifiers и `0` `aiden` aliases. Это не
Russian-native/speaker/architecture-independence claim и не synthesis: `aiden` документирован
как English token, поэтому обязательны source QA и full acoustic/language review. Первый
UtrobinTTS candidate остаётся отклонённым (76 unversioned historical rows). Ограничения и
literal-text binding завершён: `79/79` ready rows снова связаны с archive prompts без хранения
transcript, lexical rewrite или reselection; два `signal_too_quiet` rejects остаются исключёнными.
Следующий one-shot synthesis gate описан в
[VoxForge text-binding receipt](docs/voxforge_ru_mdc_qwen3_tts_customvoice_pre_qa_text_binding_v1.md).
