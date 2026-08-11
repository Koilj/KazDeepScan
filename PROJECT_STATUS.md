# KazDeepScan — текущий статус

**Обновлено:** 12 августа 2026

**Scope:** personal research, без записи голосов людей, voice cloning и product/API score.

**Состояние:** XLS-R+SLS v2 обучен и один раз проверен; следующий blocker — независимый новый
evaluation suite, а не дополнительное обучение на старых данных.

Этот файл намеренно краткий. Архитектура описана в
[KazDeepScan_implementation_blueprint.md](KazDeepScan_implementation_blueprint.md), следующие
действия — в [План реализации.md](План%20реализации.md), детальные receipts — в `docs/`.

## Что готово

- Безопасный audio pipeline: проверка media, FFmpeg decode, QA, WebRTC VAD, 16 kHz mono WAV и
  окна 4.04 s.
- Строгие manifests, SHA-256 asset validation, license ledger/snapshots, group-aware split и
  leakage checks.
- Source-specific intake и аудиты RU/KK/mixed datasets; raw audio и weights исключены из Git.
- B0 research baselines и source-mixed stress tests.
- RuASD v2: 2 000 raw строк, 1 815 ready WAV; train split для XLS-R — 1 471 строка.
- Раздельные PyAra роли: Stage-A dev 61, Stage-B dev 969, calibration 976.
- Stage A v2 и Stage B v2 обучены на NVIDIA RTX 5060 Ti с CUDA/BF16.
- Temperature scaling выполнен только на calibration role; threshold не подбирался.
- Один write-once confirmatory RU/KK/mixed run завершён.
- FastAPI health/readiness/upload scaffold работает fail closed и не выдаёт model score.

## Актуальные результаты

| Этап / слой | Результат | Статус |
| --- | ---: | --- |
| Stage A v2 dev balanced accuracy | 0.89945 | model-selection dev |
| Stage B v2 dev balanced accuracy | 0.92287 | model-selection dev |
| RU FLEURS/eSpeak balanced accuracy | 0.9800 | confirmatory research |
| KK FLEURS/Silero balanced accuracy | 1.0000 | нет pre-inference two-review gate |
| Mixed KSC2/Silero balanced accuracy | 0.9333 | exact assets ранее видел checkpoint v1 |
| ToneSpeak RU spoof recall | 88/100 | отдельный spoof-only OOD observation |

Общая pooled RU+KK+mixed accuracy намеренно не рассчитывается. Это не product quality и не
speaker-independent result: используемые источники не дают достаточного verified speaker
provenance.

Calibration v3: `T=1.29954`, NLL `0.15976 -> 0.15424`, Brier `0.04955 -> 0.04830`, ECE
`0.06444 -> 0.06497`. ECE немного ухудшился, поэтому нельзя утверждать улучшение всех аспектов
calibration.

## Зафиксированные ограничения

- Завершённые Stage A/B/final/ToneSpeak plans не повторять и не изменять.
- Final logits и ошибки не использовать для training, architecture, threshold или calibration.
- Если выполнить KK acoustic review сейчас, он подтвердит только качество exact bytes, но не
  сделает уже раскрытый результат blind задним числом.
- Mixed layer является holdout для checkpoint v2, но не project-level blind.
- ToneSpeak подходит только как independent RU spoof-only personal-research source; binary
  counterpart отсутствует.
- YO-CPT-ru/kk не подходят: это крупные YouTube-derived bona-fide TTS-pretraining corpora без
  spoof-класса и с unresolved rights/privacy provenance. Dusha — human emotion corpus, не spoof.
- Research checkpoint нельзя подключать к API risk score.

## Локальные данные

| Оставить | Можно убрать только после проверенного backup |
| --- | --- |
| `RuASD/` — 234 GiB | KSC SLR102 archive — 18 GiB |
| `KSC2/` — 76 GiB | Common Voice RU archive — 6.6 GiB |
| PyAra `archive.zip` — 27 GiB | |
| `FLEURS/` — 4.6 GiB | |

Ничего автоматически не удалялось. Подробности:
[docs/local_raw_dataset_retention_2026-08-12.md](docs/local_raw_dataset_retention_2026-08-12.md).

## Следующие действия

1. Сделать и проверить off-machine backup; затем отдельно решить удаление двух неиспользуемых
   текущим v2 архивов.
2. Провести две независимые acoustic review текущих KK exact assets без просмотра predictions;
   статус метрики не повышать.
3. Найти или создать новую разрешённую TTS architecture family и заранее проверить права,
   provenance и voices.
4. Собрать fresh, ранее не inferred RU/KK/mixed suite; acoustic gates завершить до inference.
5. Закрепить новый ledger snapshot, manifests, implementation hashes и output paths.
6. Выполнить один preflight и один GPU run. Только затем решать, нужен ли model v3.

Полный порядок и критерии остановки: [План реализации.md](План%20реализации.md).

## Проверка и воспроизводимость

- Ruff: успешно.
- mypy: 117 source files без ошибок.
- pytest: 180 tests успешно.
- Final preflight: 3 991 asset bindings.
- Точное implementation tree выполненного final plan: Git commit `52d6e6b`.
- Scope clarification: `b1368c9`.
- Final plan SHA-256: `1dfc3ca866607191385b33b85a1ee67cb3981099c6fc836aef720c6c2610d4fc`.

Для исторического `--validate-only` нужен отдельный checkout `52d6e6b`; final inference повторять
нельзя. Предыдущая подробная версия этого файла остаётся доступна через
`git show b1368c9:PROJECT_STATUS.md`.

## Главные receipts

- [RuASD v2](docs/research_ruasd_full_v2.md)
- [XLS-R Stage A v2](docs/research_xlsr_sls_stage_a_v2.md)
- [XLS-R Stage B v2](docs/research_xlsr_sls_stage_b_v2.md)
- [Calibrated RU/KK/mixed run](docs/research_xlsr_sls_stage_b_v2_research_final_v1.md)
- [ToneSpeak RU OOD](docs/research_xlsr_sls_stage_b_tone_speak_ru_ood_100.md)
- [External RU spoof-source search](docs/russian_spoof_source_search_2026-08-11.md)
- [License-ledger snapshots](docs/license_ledger_snapshots.md)
