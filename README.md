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
- Common Voice Russian v24 intake с tar whitelist, атомарным MP3 slice extraction и
  leakage-safe split по связанным client/text группам;
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
- для следующего asset-level-blind suite опубликован source inventory v2: полный pinned FLEURS
  release оставляет 55 RU и 197 KK fresh text groups, из которых сейчас QA-ready 60 KK; второй
  disjoint KSC2 semantic pass довёл fresh QA-ready mixed слой с 1 до 58 groups;
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
  одобрили `kk`, `ru` и `mixed` только для подготовки нового candidate. Full-asset packet и две
  167-row fail-closed формы теперь готовы; detector inference запрещён до их полного pass;
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

# Suite v1 уже оценён и повторно не запускается. Следующее имя plan — только шаблон:
# файл создаётся вместе с новым suite и ранее не раскрытыми final assets. Сначала
# проверить такой заранее зафиксированный run-plan без обучения и final inference:
uv run python scripts/train_b0_unseen_suite.py \
  --plan configs/research/unseen_generator_b0_run_v2.json \
  --audio-root data --validate-only

# После успешного preflight новый protocol выполняется ровно один раз той же командой
# без --validate-only. Seed, model/training config, входные SHA-256 и output пути
# берутся только из plan, а не из аргументов запуска.

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
[docs/ksc2_mixed_candidate_v1.md](docs/ksc2_mixed_candidate_v1.md). Следующий source/rights gate
и точная fresh-asset ёмкость зафиксированы в
[docs/fresh_research_suite_stage_c_source_review_2026-08-12.md](docs/fresh_research_suite_stage_c_source_review_2026-08-12.md).
