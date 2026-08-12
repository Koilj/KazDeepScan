# KazDeepScan — текущий статус

**Обновлено:** 12 августа 2026

**Scope:** personal research, без записи голосов людей, voice cloning и product/API score.

**Состояние:** XLS-R+SLS v2 обучен и один раз проверен; post-inference two-review KK acoustic
gate завершён для всех 304 exact assets. Для следующего blind research suite выполнен fresh-source
inventory, но новая разрешённая TTS architecture family пока не найдена.

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
- Post-inference KK acoustic gate завершён: две полные формы, 608 решений и `304/304` exact
  assets с итогом `pass`; write-once receipt опубликован.
- Fresh-source inventory повторно проверил полный pinned FLEURS release и KSC2 evidence:
  доступны 55 RU release-level groups, 60 QA-ready + 137 ещё не обработанных KK groups и только
  1 QA-ready mixed row; 2 600 KSC2 candidates требуют semantic review.
- FastAPI health/readiness/upload scaffold работает fail closed и не выдаёт model score.

## Актуальные результаты

| Этап / слой | Результат | Статус |
| --- | ---: | --- |
| Stage A v2 dev balanced accuracy | 0.89945 | model-selection dev |
| Stage B v2 dev balanced accuracy | 0.92287 | model-selection dev |
| RU FLEURS/eSpeak balanced accuracy | 0.9800 | confirmatory research |
| KK FLEURS/Silero balanced accuracy | 1.0000 | exact bytes прошли post-inference two-review gate |
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
- Выполненный KK acoustic gate подтверждает только качество exact bytes и не делает уже
  раскрытый результат blind задним числом.
- Mixed layer является holdout для checkpoint v2, но не project-level blind.
- ToneSpeak подходит только как independent RU spoof-only personal-research source; binary
  counterpart отсутствует.
- YO-CPT-ru/kk не подходят: это крупные YouTube-derived bona-fide TTS-pretraining corpora без
  spoof-класса и с unresolved rights/privacy provenance. Dusha — human emotion corpus, не spoof.
- IMS Toucan отклонён как новая family: official architecture — FastSpeech-2-like с
  FastPitch-style conditioning и HiFi-GAN, а такая route уже присутствует через Silero V4.
- Research checkpoint нельзя подключать к API risk score.

## Следующие действия

1. Найти или создать действительно новую разрешённую TTS architecture family с RU/KK/mixed
   text-only route и заранее проверить права, provenance и fixed voice profile.
2. Только после успешного TTS gate закрепить selection policy и собрать fresh suite из уже
   посчитанной ёмкости; acoustic gates завершить до inference.
3. Закрепить новый ledger snapshot, manifests, implementation hashes и output paths.
4. Выполнить один preflight и один GPU run. Только затем решать, нужен ли model v3.

Полный порядок и критерии остановки: [План реализации.md](План%20реализации.md).

## Проверка и воспроизводимость

- Ruff: успешно.
- mypy: 121 source files без ошибок.
- pytest: 187 tests успешно.
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
- [KK acoustic gate receipt](docs/fleurs_kk_silero_v4_acoustic_gate_v1.md)
- [Stage C source review и fresh inventory](docs/fresh_research_suite_stage_c_source_review_2026-08-12.md)
- [External RU spoof-source search](docs/russian_spoof_source_search_2026-08-11.md)
- [License-ledger snapshots](docs/license_ledger_snapshots.md)
