# KazDeepScan

KazDeepScan — локальный personal-research проект для исследования признаков
синтезированности русской, казахской и смешанной речи. Это не идентификация говорящего,
не доказательство мошенничества и не развёрнутый сервис оценки риска.

Текущий реализованный этап — проверяемый фундамент для данных, аудио и безопасного intake:

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
- local consent-registry validator сохранён как архивный инструмент; PII и сами соглашения
  никогда не хранятся в Git.
- B0 и XLS-R + SLS tensor/training foundations, record-level logit aggregation и temperature
  scaling;
- explicit source-mixed research matrix и B0 runner, который не допускает overlap исходных
  corpus между train/dev/final-test и проверяет обычный sample/SHA-256/group/text leakage поверх
  этого;
- FastAPI health/readiness/upload scaffold, который не выдаёт score без обученного,
  калиброванного model release.

Есть personal-research B0 checkpoints на PyAra, full RuASD и source-mixed матрице. На трёх
зафиксированных seed source-mixed ML-DF cross-lingual test дал balanced accuracy от 73.40% до
85.79%; наблюдались серьёзные ложные срабатывания на bona-fide речи. Ни один источник не даёт
проверяемого speaker-disjoint binary split. Отчёты:
[RuASD](docs/research_b0_ruasd_full_2000.md) и
[source-mixed v1](docs/research_b0_source_mixed_v1.md). Поэтому ни модель, ни API не выдают
вероятность риска или калиброванный score.

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
  --output-manifest data/manifests/ruasd_ru_v1_full_research_2000.csv \
  --slice-name research-2000 --limit-per-label 1000 --min-per-stratum 1

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

# Обучить B0 с source-disjoint train/dev и один раз оценить final test после выбора epoch по dev.
uv run python scripts/train_b0_matrix.py \
  --matrix configs/research/source_mixed_v1.json \
  --audio-root data \
  --license-ledger data/licenses/license_ledger.csv \
  --output models/b0-source-mixed-research-v1.pt --device cuda

```

Все результаты CLI содержат только технические метаданные. Команды не отправляют аудио по
сети и не создают его постоянных копий.

Подробные контракты находятся в [docs/audio_pipeline.md](docs/audio_pipeline.md),
[docs/data_contract.md](docs/data_contract.md) и [docs/threat_model.md](docs/threat_model.md).
