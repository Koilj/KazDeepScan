# KazDeepScan — текущий статус

**Обновлено:** 13 августа 2026

**Scope:** personal research, без записи голосов людей, voice cloning и product/API score.

**Состояние:** XLS-R+SLS v2 обучен и проверен на двух write-once research protocols. Для нового
167-pair RU/KK/mixed Stage-C candidate через exact KazakhTTS route завершены synthesis, QA,
exposure audit, two-review full-asset gate, immutable plan и один GPU inference run. Результат
asset-level blind, но не source- или speaker-independent; checkpoint не используется в API.

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
- Fresh-source inventory v2 повторно проверил полный pinned FLEURS release и KSC2 evidence:
  доступны 55 RU release-level groups, 60 QA-ready + 137 ещё не обработанных KK groups и 58
  QA-ready mixed groups; 2 541 KSC2 candidates остаются без semantic review.
- Для mixed Stage C опубликован disjoint 59-row single-AI semantic-review delta: 57 rows прошли
  QA/VAD, 2 отклонены с полным accounting. Это устранило статистически непригодный размер 1,
  но не заменяет будущий human acoustic gate.
- Selection policy заморожена до synthesis: все `173` доступных groups (`55 RU / 60 KK / 58
  mixed`) без metric-based отбора и backfill. После bona-fide QA готовы `50 / 60 / 58`; пять RU
  rows отклонены как `signal_too_quiet`, combined ready manifest содержит 168 rows.
- ISSAI KazakhTTS2 Male2 Tacotron2 + ParallelWaveGAN принят как `unseen_exact_generator_route`:
  122 908 306 outer artifact bytes, все required inner hashes/CRC/configs и local CUDA smoke
  проверены; reference audio и cloning запрещены.
- Exposure audit охватил 46 manifests / 17 657 spoof rows: exact route overlap `0`, но 312
  сохранённых строк используют тот же Male2 speaker alias через Piper. Speaker independence не
  заявляется.
- Три smoke WAV (KK official support, RU/mixed conditional) прошли два независимых listening
  review: `kk`, `ru` и `mixed` одобрены для подготовки candidate. Receipt SHA-256
  `946c3a3a59fdd437553c2fe8e93d4ade157e718cf67505abb1216c02cbc82a73`; detector inference
  всё ещё запрещён.
- Первая массовая попытка KazakhTTS явно отклонила 72/168 исходных surface forms. До повторного
  synthesis заморожен character-inventory normalizer: изменены 60 KK и 12 RU текстов, исходные
  `text_id/text_hash` сохранены, detector/metric-based решений не было.
- Нормализованный run создал 168/168 WAV. Generated-asset QA принял 167 и отклонил одну mixed
  строку как `insufficient_speech`; без regeneration/backfill опубликованы 50 RU, 60 KK и 57
  mixed exact pairs (`334` candidate assets).
- Project-exposure audit сравнил candidate с 15 manifests / 11 869 rows из 21 research config:
  overlap по `sample_id`, audio SHA-256 и `text_hash` равен `0/0/0`.
- Две независимые full-asset формы заполнены и строго проверены: `334` review rows,
  `167/167` exact synthetic WAV получили `pass`; receipt SHA-256
  `9a12f235072ce5ae4c3bd6bb0616a804a710abea74f3c3b387cebe12baf8153c`.
- New Stage-C XLS-R plan закрепляет candidate, full acoustic gate, project-exposure audit, frozen
  five-source license snapshot, calibration role, implementation hashes и новые write-once paths.
  Preflight подтвердил `1 310` asset bindings (`976` calibration, `100 RU`, `120 KK`, `114 mixed`).
- Единственный Stage-C GPU inference run завершён: все `334` final assets получили по одному
  logit, execution-lock SHA-256
  `7d692542e6397c64ac0379eca34564d16f1fd670d5deb93e7a9a940695c2ccdb`, report SHA-256
  `2cb7198a6ec03f2c6424748dde3263731d87ff6b0b557f59b0463b7f74cc5e32`.
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
| Fresh Stage-C RU balanced accuracy | 0.9000 | 50 new exact pairs, asset-level blind |
| Fresh Stage-C KK balanced accuracy | 0.9500 | 60 new exact pairs, asset-level blind |
| Fresh Stage-C mixed balanced accuracy | 0.8070 | 57 new exact pairs, asset-level blind |

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
- Абсолютная architecture novelty больше не используется: historical RuASD manifests не дают
  architecture IDs. Проверяется exact checkpoint/runtime route; component и voice overlaps
  раскрываются отдельно. IMS Toucan не выбран, KazakhTTS route принят.
- Research checkpoint нельзя подключать к API risk score.

## Следующие действия

1. Не повторять Stage-C run и не использовать его final errors для tuning.
2. Решить отдельно, нужен ли model v3; если да, готовить новый train/dev/calibration contract и
   прежде нераскрытый final suite.

Полный порядок и критерии остановки: [План реализации.md](План%20реализации.md).

## Проверка и воспроизводимость

- Ruff: успешно.
- mypy: 142 source files без ошибок.
- pytest: 213 tests успешно.
- Final preflight: 3 991 asset bindings.
- Stage-C preflight: 1 310 asset bindings; один GPU inference run, `334` exact final predictions.
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
- [Stage C KazakhTTS pre-inference language gate](docs/fresh_suite_stage_c_kazakhtts_acoustic_gate_v1.md)
- [KSC2 mixed Stage-C semantic evidence v2 delta](docs/ksc2_mixed_ai_review_v2_delta.md)
- [Stage C frozen selection и bona-fide materialization](docs/fresh_suite_stage_c_selection_v1.md)
- [Stage C KazakhTTS normalized candidate и full-asset gate](docs/fresh_suite_stage_c_kazakhtts_candidate_v1.md)
- [XLS-R Stage-C asset-level-blind evaluation](docs/research_xlsr_sls_stage_b_v2_fresh_suite_stage_c_v1.md)
- [External RU spoof-source search](docs/russian_spoof_source_search_2026-08-11.md)
- [License-ledger snapshots](docs/license_ledger_snapshots.md)
