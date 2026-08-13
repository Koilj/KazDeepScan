# KazDeepScan — текущий статус

**Обновлено:** 13 августа 2026

**Scope:** personal research, без записи голосов людей, voice cloning и product/API score.

**Состояние:** XLS-R+SLS v2 и отдельная v3 ветка завершены на write-once research protocols.
v3 использовал изолированные train / Stage-A dev / Stage-B dev / calibration roles, симметричную
train-only augmentation, выбрал Stage-A epoch 3 и Stage-B epoch 4 только по dev loss, затем
провёл один final GPU run на неизменяемых 55 Common Voice/Dialog-RU парах. Stage-D v2
logits/errors не загружались. Exact final assets уже были оценены v2, поэтому v3 — не blind
test; результаты не source-, speaker- или architecture-family-independent и checkpoint не
используется в API.

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
- Historical fresh-source inventory v2 повторно проверил полный pinned FLEURS release и KSC2
  evidence. После завершённых Stage-C selection/QA/inference pinned FLEURS RU больше не имеет
  usable unevaluated groups; historical KSC2 inventory всё ещё указывает 137 не обработанных KK
  groups и 2 541 candidates без semantic review, но они не заменяют новый RU bona-fide source.
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
- Dialogs-RU VITS2 / fixed Masha-neutral принят только как `unseen_exact_generator_route`:
  pinned model revision и 15 required files проверены по size/SHA-256; local wrapper fail closed
  использует `torch.load(weights_only=True)`, не принимает reference audio и фиксирует
  speaker/emotion IDs. Audit 53 historical spoof manifests / 18 422 rows не нашёл exact-route
  или Masha-alias overlap; generic RuASD `vits2TTS` исключает claim новой architecture family.
- Silero V5.5 RU / fixed `eugene` принят только как новый exact route для будущего RU
  personal-research candidate: `v5_5_ru.pt` (`145420684` bytes) и source archive pinned по
  SHA-256, local wrapper разрешает только literal Russian text → built-in `eugene` at 48 kHz и
  запрещает reference audio, cloning, random profile, SSML и `voice_path`. Audit 56 historical
  manifests / 18 605 spoof rows нашёл `0` exact V5.5/eugene overlap, но `1 265` legacy Silero
  rows исключают architecture-, vendor- и speaker-independence claims. Literal-text binding
  проверил все 75 ready rows against pinned archive без rewrite; synthesis, review и detector
  inference ещё не начаты.
- Full Common Voice RU v24 `test` metadata screen до extraction сравнил `10 261` records / `2 075`
  client groups с `12 313` configured-role rows и `39 850` rows в `85` manifest files. Строгое
  whole-client-group exclusion оставляет `6 211` records / `1 443` groups; это только capacity
  receipt без selection, audio extraction, synthesis, QA/review или inference.
- Fixed V5.5 literal-text gate без lexical rewrite проверил ровно эти `6 211` metadata survivors:
  `113` records с неподдержанными кавычками/glyph `−` taint `106` full client groups. После
  strict group exclusion доступны `5 600` records / `1 337` groups. Immutable pre-QA receipt
  выбрал из них `80` exact records: seeded two-stage rule берёт одну запись на отдельную client
  group, без audio/duration, detector/model output, metrics или final errors. `80/80` sample IDs,
  client groups и text hashes уникальны. Exact extraction и normal decode/QA/VAD опубликовали
  `75` ready WAV; пять `insufficient_speech` rejections полностью учтены без backfill. Exact
  literal-text binding закрепил все 75 ready rows без rewrite. Synthesis, acoustic review и
  inference не начаты.
- Stage D строго привязал 73 frozen Common Voice RU текста, создал ровно 73 synthetic WAV и без
  backfill сохранил 55 binary pairs / 110 assets после 18 `insufficient_speech` rejects.
  Проектный exposure audit против 23 configs / 12 203 rows дал `0/0/0` sample/audio/text overlap.
  Обе full-asset acoustic review формы прошли до inference; preflight проверил 1 086 bindings,
  после чего выполнен ровно один GPU run.
- v3 governance receipt проверил 3 587 assets и нулевые pairwise overlap между 1 471-row RuASD
  train, 61-row Stage-A dev, 969-row Stage-B dev, 976-row calibration и 110-row final roles.
  Augmentation `symmetric-channel-codec-replay-simulation-v1` применяется только на train и
  выводит параметры без label. Stage A v3 выбрал epoch 3 (`dev_loss=0.42729345`), Stage B v3 —
  epoch 4 (`dev_loss=0.18516177`).
- v3 final plan SHA-256 `0f2d63826b728ea07b3bb418901230aa304b0b629d61bdb63162033865f26252`
  успешно прошёл write-once preflight на 1 086 assets и выполнил один inference. Temperature
  fitted только на 976-row calibration (`T=0.79692465`); threshold фиксирован на `0.5`.
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
| Stage-D RU Dialogs-RU balanced accuracy | 0.9727 | 55 fixed Common Voice/Dialog-RU pairs; one run |
| v3 Stage-B dev loss | 0.18516 | epoch 4; only v3 checkpoint-selection role |
| v3 Stage-D RU balanced accuracy | 0.9727 | 107/110; same 55 pairs previously evaluated by v2 |

Общая pooled RU+KK+mixed accuracy намеренно не рассчитывается. Это не product quality и не
speaker-independent result: используемые источники не дают достаточного verified speaker
provenance.

Calibration confirmatory v2: `T=1.29954`, NLL `0.15976 -> 0.15424`, Brier `0.04955 -> 0.04830`, ECE
`0.06444 -> 0.06497`. ECE немного ухудшился, поэтому нельзя утверждать улучшение всех аспектов
calibration.

Отдельная v3 calibration: `T=0.79692`, NLL `0.17983 -> 0.17593`, Brier `0.05744 -> 0.05766`,
ECE `0.08754 -> 0.07823`. Улучшение NLL/ECE не является основанием менять фиксированный порог
или повторять final run.

## Зафиксированные ограничения

- Завершённые Stage A/B/final/ToneSpeak plans не повторять и не изменять.
- Завершённые Stage C и Stage D synthesis, preflight и inference не повторять и не изменять.
- Не повторять Stage A v3, Stage B v3, v3 preflight или v3 final inference; не менять их
  checkpoint/report hashes и не заменять/не добавлять Stage-D пары.
- Final logits и ошибки не использовать для training, architecture, threshold или calibration.
- Exact Stage-D pairs уже получили v2 predictions, поэтому v3 result нельзя называть blind или
  unseen. v2 logits/errors не использовались для v3 решений, но это не отменяет history набора.
- Выполненный KK acoustic gate подтверждает только качество exact bytes и не делает уже
  раскрытый результат blind задним числом.
- Mixed layer является holdout для checkpoint v2, но не project-level blind.
- ToneSpeak подходит только как independent RU spoof-only personal-research source; binary
  counterpart отсутствует.
- YO-CPT-ru/kk не подходят: это крупные YouTube-derived bona-fide TTS-pretraining corpora без
  spoof-класса и с unresolved rights/privacy provenance. Dusha — human emotion corpus, не spoof.
- Абсолютная architecture novelty больше не используется: historical RuASD manifests не дают
  architecture IDs. Проверяется exact checkpoint/runtime route; component и voice overlaps
  раскрываются отдельно. IMS Toucan не выбран, KazakhTTS и Dialogs-RU exact routes приняты.
- Dialogs-RU model repository в pinned revision не содержит собственного `LICENSE`. Его model-card
  OpenRAIL declaration и pinned dataset license допускают только зафиксированный personal-research
  scope, а не broad commercial clearance.
- Silero V5.5 RU имеет CC-BY-NC-SA-4.0: route ограничен personal research, не авторизован для
  product/commercial use и не является доказательством новой architecture/vendor/speaker family.
- Research checkpoint нельзя подключать к API risk score.

## Следующие действия

1. Не повторять Stage-C/Stage-D/v3 runs и не использовать их final errors для tuning, выбора
   checkpoint, temperature, threshold или augmentation.
2. Новый RU route, historical exposure, literal-text screen и `80`-record pre-QA selection уже
   закреплены; exact extraction/QA/VAD оставил `75` ready rows. Следом bind literal texts только
   для них, затем выполнить один fixed-profile synthesis WAV per text без replacement/backfill.
   Старые 55 Stage-D/v3 пар, 73-row selection и их rejections нельзя переиспользовать как
   «новый blind» тест.
3. API/product track не начинать без отдельного commercial-rights, privacy, verified-speaker,
   deployment и product-calibration contract.

Полный порядок и критерии остановки: [План реализации.md](План%20реализации.md).

## Проверка и воспроизводимость

- Ruff: успешно.
- mypy: успешно.
- pytest: успешно.
- Final preflight: 3 991 asset bindings.
- Stage-C preflight: 1 310 asset bindings; один GPU inference run, `334` exact final predictions.
- Stage-D preflight: 1 086 asset bindings; один GPU inference run, `110` exact final predictions.
- v3 Stage-D preflight: 1 086 asset bindings; один GPU inference run, `110` exact final
  predictions; report SHA-256 `9f99a7bed878ecfc561831e5130ac92dd5c8b73cb9965fc51e8a4d00d66e50e3`.
- Точное implementation tree исторического v2 final plan: Git commit `52d6e6b`.
- Scope clarification: `b1368c9`.
- Historical v2 final plan SHA-256: `1dfc3ca866607191385b33b85a1ee67cb3981099c6fc836aef720c6c2610d4fc`.
- v3 final plan SHA-256: `0f2d63826b728ea07b3bb418901230aa304b0b629d61bdb63162033865f26252`.

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
- [Stage-D Common Voice RU precheck](docs/stage_d_common_voice_ru_precheck_v1.md)
- [Stage-D Dialogs-RU VITS2 / Masha-neutral](docs/stage_d_dialogs_ru_vits2_intake_2026-08-13.md)
- [XLS-R+SLS v3 Stage-D governed evaluation](docs/research_xlsr_sls_v3_stage_d_dialogs_ru_v1.md)
- [Silero V5.5 RU / eugene route intake](docs/silero_v5_5_ru_eugene_intake_2026-08-13.md)
- [Common Voice RU full-test metadata exposure screen](data/manifests/common_voice_ru_v24_full_test_metadata_exposure_screen_v1.json)
- [Common Voice RU / Silero V5.5 literal-text screen](data/manifests/common_voice_ru_v24_full_test_silero_v5_5_literal_text_screen_v1.json)
- [Common Voice RU / Silero V5.5 immutable pre-QA selection](docs/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_selection_v1.md)
- [Common Voice RU / Silero V5.5 pre-QA materialization](docs/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_materialization_v1.md)
- [Common Voice RU / Silero V5.5 literal-text binding](docs/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_text_binding_v1.md)
- [External RU spoof-source search](docs/russian_spoof_source_search_2026-08-11.md)
- [License-ledger snapshots](docs/license_ledger_snapshots.md)
