# KazDeepScan — актуальная схема реализации

**Версия:** 2.0

**Проверено:** 12 августа 2026

**Scope:** локальное personal research по детектированию синтезированной русской, казахской и
смешанной речи.

Этот документ описывает фактически реализованную систему и её обязательные границы. История
экспериментов находится в `docs/`, оперативное состояние — в `PROJECT_STATUS.md`, дальнейшая
работа — в `План реализации.md`.

## 1. Назначение и границы

KazDeepScan сейчас является воспроизводимым исследовательским контуром, а не готовым сервисом
оценки риска. Он умеет проверять источники и аудио, строить защищённые manifests, обучать
XLS-R+SLS на CUDA и выполнять hash-pinned evaluation.

Текущий scope не включает:

- идентификацию человека или доказательство мошенничества;
- запись голосов участников, reference audio, voice cloning или имитацию конкретного человека;
- product/commercial claim на research-only данных;
- публикацию общего RU+KK+mixed score;
- выдачу `risk_score` через API по исследовательскому checkpoint;
- повторное выполнение уже завершённых write-once планов.

В репозитории сохранён fail-closed product validator, но он не является разрешением на
deployment и не меняет текущий personal-research scope.

## 2. Реализованный поток

```text
локальный архив / dataset revision
  -> source review + artifact lock + license ledger
  -> безопасный intake и raw manifest
  -> SHA-256 / schema / leakage / rights validation
  -> decode, QA, VAD, mono PCM WAV 16 kHz
  -> ready manifest + rejection receipt
  -> Dataset и окна 64 600 samples
  -> Stage A: SLS head, frozen XLS-R
  -> Stage B: head + XLS-R blocks 16–23
  -> отдельная calibration role
  -> отдельные RU / KK / mixed evaluation layers
  -> immutable report и execution lock
```

FastAPI-каркас расположен рядом, но намеренно отделён от исследовательского checkpoint: пока
нет разрешённого model release, `/readyz` возвращает `503`, а `/v1/analyze` не выдаёт score.

## 3. Неизменяемые инженерные правила

1. **Права до обработки.** Источник сначала появляется в license ledger; experiment использует
   минимальный frozen ledger snapshot, а не изменяемый общий CSV.
2. **Точные bytes.** Архивы, manifests, checkpoints, код plan-runner и reports закрепляются
   SHA-256.
3. **Write once.** Runner не перезаписывает plan, checkpoint, execution lock или report.
4. **Разделение ролей.** Train, model-selection dev, calibration и evaluation не смешиваются.
5. **Leakage gate.** Проверяются sample ID, asset SHA, parent group, доступные speaker/voice keys
   и text hash. Неизвестный ID не выдаётся за verified speaker provenance.
6. **Производные рядом с родителем.** Оригинал, TTS по тому же тексту и кодированные варианты
   не могут расходиться по split-ам.
7. **Final не обучает.** Ошибки и logits завершённого evaluation не используются для изменения
   checkpoint, threshold, calibration или architecture.
8. **GPU fail closed.** XLS-R training/final runner требует CUDA и BF16; CPU fallback для
   незаметно другого эксперимента запрещён.
9. **Без pooled claim.** RU, KK и mixed публикуются отдельно, с ограничениями каждого слоя.
10. **Крупные данные вне Git.** В Git находятся manifests/receipts, но не raw audio и model
    weights. Manifest не заменяет резервную копию исходного архива.

## 4. Основные компоненты репозитория

| Область | Реализация |
| --- | --- |
| Audio | `src/kds/audio/`: media validation, FFmpeg decode, QA, WebRTC VAD, waveform/windows |
| Data | `src/kds/data/`: manifests, assets, ledger, split, preprocessing и source-specific intake |
| Models | `src/kds/models/`: B0 и `XlsrSlsClassifier` |
| Training | `src/kds/training/`: B0, Stage-A/Stage-B plans и runners |
| Evaluation | `src/kds/eval/`: aggregation, metrics, calibration, acoustic gates, final protocols |
| Scripts | `scripts/`: auditable entrypoints без скрытого изменения планов |
| API | `services/api/`: health/readiness/upload scaffold без model score |
| Contracts | `configs/research/`, `data/manifests/`, `data/licenses/frozen/` |
| Evidence | `docs/research_*.md`, локальные ignored `models/` и `artifacts/` |

Актуальный manifest contract описан в [docs/data_contract.md](docs/data_contract.md), audio QA —
в [docs/audio_pipeline.md](docs/audio_pipeline.md), XLS-R — в
[docs/xlsr_sls.md](docs/xlsr_sls.md).

## 5. Текущие роли данных XLS-R v2

| Роль | Источник | Строки | Назначение |
| --- | --- | ---: | --- |
| Train | RuASD v2 ready, split `train` | 1 471 | Stage A/B training |
| Stage-A dev | исторический PyAra ready, split `dev` | 61 | выбор Stage-A epoch |
| Stage-B dev | fresh PyAra v3 | 969 | выбор Stage-B epoch |
| Calibration | disjoint PyAra calibration v3 | 976 | только temperature scaling |
| RU evaluation | FLEURS/eSpeak NG | 150 / 75 pairs | confirmatory RU layer |
| KK evaluation | FLEURS/Silero V4 | 304 / 152 pairs | confirmatory KK layer |
| Mixed evaluation | KSC2/Silero V4 | 60 / 30 pairs | confirmatory mixed layer |

RuASD v2 raw содержит 2 000 строк; после decode/QA/VAD опубликовано 1 815 ready WAV. Выборка,
rejections и допустимое переиспользование ранее нормализованных exact bytes закреплены receipts.

Все эти источники разрешены только в зафиксированном research scope. Ни RuASD, ни PyAra, ни
FLEURS не дают достаточного verified speaker provenance для заявления speaker-independent
quality.

## 6. Модель и обучение

`XlsrSlsClassifier` использует локально закреплённый `facebook/wav2vec2-xls-r-300m`:

```text
waveform 16 kHz
  -> XLS-R hidden states
  -> обучаемая softmax-смесь слоёв
  -> masked attentive statistics pooling
  -> MLP
  -> один raw spoof logit на окно 4.04 s
```

Stage A обучает только SLS head. Stage B загружает точный Stage-A state, оставляет blocks 0–15
замороженными и обучает head вместе с blocks 16–23. Применяются BF16, gradient checkpointing,
physical batch 4 и gradient accumulation 8.

V2 выполнен на NVIDIA GeForce RTX 5060 Ti 16 GB:

| Этап | Выбранный epoch | Dev loss | Accuracy | Balanced accuracy |
| --- | ---: | ---: | ---: | ---: |
| Stage A v2 | 3 | 0.20181 | 0.90164 | 0.89945 |
| Stage B v2 | 5 | 0.17405 | 0.92157 | 0.92287 |

Точные планы и hashes: [Stage A v2](docs/research_xlsr_sls_stage_a_v2.md) и
[Stage B v2](docs/research_xlsr_sls_stage_b_v2.md).

## 7. Calibration и confirmatory evaluation

Temperature scaling fitted только на 976-row calibration role. Порог не подбирался: решение
оставлено на calibrated probability `0.5`. Температура `T=1.29954`; NLL и Brier улучшились, ECE
незначительно ухудшился, поэтому calibration нельзя описывать как безусловно улучшенную.

| Слой | Balanced accuracy | Главное ограничение |
| --- | ---: | --- |
| RU | 0.9800 | exact bytes прошли review, но нет verified speaker independence |
| KK | 1.0000 | exact bytes прошли gate только после inference |
| Mixed | 0.9333 | эти exact assets ранее видел checkpoint v1 |

Это confirmatory research evidence, а не blind project-level final и не product metric. Полный
immutable receipt: [research_xlsr_sls_stage_b_v2_research_final_v1.md](docs/research_xlsr_sls_stage_b_v2_research_final_v1.md).
Поздний KK gate подтвердил audibility, соответствие казахскому тексту и отсутствие явных
дефектов для `304/304` exact assets, но не изменил уже раскрытую метрику или её статус.

ToneSpeak остаётся отдельным RU spoof-only OOD исследованием (`88/100` spoof recall). У него нет
bona-fide counterpart, поэтому он не превращается в binary final source.

## 8. API и возможный model release

Текущий API допустимо использовать только для проверки upload/decode/QA и состояния процесса:

- `GET /healthz` — процесс жив;
- `GET /readyz` — `503`, пока model release отсутствует;
- `POST /v1/analyze` — безопасно обрабатывает файл, но не выдаёт исследовательский score.

Чтобы включить inference, недостаточно подключить существующий `.pt`. Нужны новый разрешённый
release contract, отдельная product/research-demo calibration policy, model card, latency/security
tests и явное решение владельца о допустимом UX. До этого поведение fail closed сохраняется.

## 9. Воспроизводимость завершённого запуска

Точное implementation tree уже выполненного final plan сохранено в Git-коммите `52d6e6b`.
Новые изменения проекта не должны переписывать этот plan: исторический `--validate-only`
выполняется в отдельном checkout указанного коммита, а final inference не повторяется.

## 10. Архитектура следующего исследования

Следующий meaningful experiment должен получить новый `run_id` и выполнить такой порядок:

1. проверить права, provenance и точные revisions новых источников;
2. выбрать ранее не использованные assets и новую generator family;
3. сделать raw/ready manifests, rejection reports и frozen ledger snapshot;
4. завершить acoustic review до model inference;
5. проверить отсутствие project exposure и обычного leakage;
6. закрепить код, checkpoint, calibration policy и outputs новым plan;
7. выполнить один preflight и один GPU-run;
8. опубликовать отдельные метрики и ограничения, не адаптируя модель по результатам.

Конкретная последовательность и критерии остановки находятся в
[План реализации.md](План%20реализации.md).

Source/rights gate от 12 августа 2026 года закрепил exact fresh capacity, но остановил переход к
artifact lock: IMS Toucan использует FastSpeech-2-like/FastPitch/HiFi-GAN route и не считается
новой family относительно Silero V4. Детали и write-once inventory:
[fresh_research_suite_stage_c_source_review_2026-08-12.md](docs/fresh_research_suite_stage_c_source_review_2026-08-12.md).
