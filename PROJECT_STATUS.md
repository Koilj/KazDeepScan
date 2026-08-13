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
  проверил все 75 ready rows against pinned archive без rewrite, а exact CPU route создал один
  raw 48 kHz WAV на каждый text. Technical QA оставил 42 spoof WAV и учёл 33
  `insufficient_speech` rejects; immutable pairing locked exactly 42/42 pairs. Two completed
  84-row forms with distinct pseudonymous IDs passed the strict technical acoustic gate: every
  exact asset has two `pass/yes/yes/yes/yes` decisions, gate-report SHA-256
  `cb9604a6a2c41fa16ce6e0c8c1947e44c0d0d21d626b88ccbf90673c872c3631`. Gate был обязательным
  предикатом immutable evaluation contract. Его project-exposure audit covered 30 configured research
  files, 17 referenced manifests and 12,313 prior rows; sample/audio/text overlap is `0/0/0`,
  receipt SHA-256 `6071deb2f60ca914e475611addf81ef2cf81b485c2b9b86826c5a135c0cca3ff`. The immutable
  V5.5 evaluation plan SHA-256 `cdf3fcbb496006478e575c024963cca497854dae1ce17775e58d95ae4d74cadf`
  pins the passing evidence, V2 checkpoint, 976-row calibration, frozen license ledger and
  write-once paths. Its one preflight validated 1,060 assets with CUDA/BF16 and no logits,
  SHA-256 `3df30bd5a70bcb471d57db4a85765658396e6a2491ab9937731018e37e4206f3`. Exactly one final
  GPU run then gave `84/84` accuracy, balanced accuracy `1.0000` and `42/42` fully correct pairs;
  execution SHA-256 is `0bad2d746c2036068e1c6ae1d160b294ab22a351b3a56f00867825b1fdb28015`;
  report SHA-256 is `06744ecd71244791efe431c6473b59c8c9962f7b27273ed919495365bbe48991`. It is a fixed
  source-linked personal-research layer, not independent generalization, and distinct IDs do not
  themselves prove organizational independence. The immutable final report has a known erroneous
  top-level `detector_inference_performed=false` inherited from preflight; its execution lock,
  result rows and report status prove the run. Reconciliation receipt SHA-256
  `0a301c3e4f14d7a5ea048b0cfda7bfae7eda83b6913b787b34d10fc7c079c5b4` records this without
  changing the report or allowing a rerun.
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
  literal-text binding закрепил все 75 ready rows без rewrite, а V5.5/eugene synthesis создал
  один raw WAV per bound text. Synthetic technical QA оставил 42 rows и учёл 33 rejects без
  resynthesis/backfill; immutable candidate locked matching 42 bona fide/spoof pairs. Completed
  acoustic gate passed all 84 exact assets, and project-exposure audit against all 30 then-current
  configs / 17 referenced manifests / 12,313 rows found `0/0/0` sample/audio/text overlap.
  Subsequent immutable contract completed exactly one GPU inference run; rerun и post-hoc tuning
  запрещены.
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
- VoxForge Russian / Mozilla Data Collective прошёл source-level intake и metadata-only selection:
  exact
  `3,795,197,539`-byte archive SHA-256
  `7372c6f8d067b8d1651995ad8306b673acaf2cde705ee51295152b96c93de557`, GPL-3.0-or-later
  notices/text, `644` submissions, `6,412` transcript-bound mono 48 kHz WAV и `194`
  source-provided contributor groups. На metadata-only screen raw WAV не извлекались; `81` canonical prompt texts
  прошли strict pre-extraction screen: all `6,412` records / `194` contributor groups survive;
  overlap `0/0/0/0/0` against both transcript layers, sample and group keys in 31 configs / 90
  manifests. Immutable metadata-only selection затем зафиксировал `81` records с unique text и
  conservative contributor groups; raw aliases не versioned. Receipts SHA-256
  `0e8bd5c7d1e02bedc235adcb3bdb7ed3bc7efdd0ff7637339460e3f43c38272f`,
  `275367a9738bfcc017315cfb3799078c0c3ab1981a318098b0849eaf7893dffe` and
  `18dc659ce30a6eaec03cdc27b74e709e066d556b142b063fc2a48b7c4fc1224f`. Qwen3-TTS CustomVoice
  Q8_0 / fixed `aiden` теперь принят как exact text-only route: six locked files
  (`1,610,363,823` bytes), local CUDA runtime health и audit `59` manifests / `18,764` spoof
  rows дали `0` exact, legacy-identifier и `aiden` alias overlap. `aiden` documented as English,
  so no Russian-native, speaker- or architecture-independence claim. The one frozen 81-WAV
  VoxForge materialization then passed archive rebinding and technical decode/QA/VAD: `79` ready,
  `2` `signal_too_quiet` rejects, no replacement. Literal-text binding revalidated the same `79`
  prompt/group/hash rows without text persistence or rewrite. A non-candidate CUDA smoke passed
  with one temporary 24 kHz mono Russian WAV. The completed one-shot Qwen run then created `79/79`
  24 kHz mono raw WAVs with `0` failed attempts; technical decode/quality/WebRTC-VAD QA retained
  `79/79` 16 kHz mono PCM-16 ready spoof rows with `0` rejects, reuse, resynthesis, replacement or
  backfill. Pairing, full acoustic/language review and detector inference remain prohibited.
  UtrobinTTS remains rejected: `76` historical spoof rows carry its unversioned identifier.
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
2. Новый RU route прошёл завершённые selection/materialization/text binding/synthesis/technical
   QA/pair lock, 84-asset technical acoustic gate, zero-overlap project-exposure audit, immutable
   evaluation contract, no-logit preflight and its exactly one final GPU inference run. Не
   повторять его, не менять candidate/reviews/checkpoint/calibration/boundary and не использовать
   final errors для tuning. Старые 55 Stage-D/v3 пар, 73-row selection и их rejections нельзя
   переиспользовать как «новый blind» тест.
3. Для VoxForge RU создать ровно один Qwen3-TTS CustomVoice / `aiden` WAV per `79` bound text и
   выполнить technical synthetic QA; два source rejects не заменять и не backfill. Failed
   synthesis/QA rows также не regenerate. До отдельных pair, full acoustic/language review и
   immutable evaluation receipts не запускать detector; UtrobinTTS нельзя использовать как
   backfill.
4. API/product track не начинать без отдельного commercial-rights, privacy, verified-speaker,
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
- V5.5/eugene preflight: 1 060 asset bindings; no logits/inference; receipt SHA-256
  `3df30bd5a70bcb471d57db4a85765658396e6a2491ab9937731018e37e4206f3`.
- V5.5/eugene final: exactly one GPU run, `84/84` correct / balanced accuracy `1.0000`; execution
  lock SHA-256 `0bad2d746c2036068e1c6ae1d160b294ab22a351b3a56f00867825b1fdb28015`, report SHA-256
  `06744ecd71244791efe431c6473b59c8c9962f7b27273ed919495365bbe48991`.
- V5.5/eugene execution reconciliation: final report's inherited
  `detector_inference_performed=false` is a documented immutable status defect; receipt SHA-256
  `0a301c3e4f14d7a5ea048b0cfda7bfae7eda83b6913b787b34d10fc7c079c5b4`.
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
- [VoxForge Russian source-level intake](docs/data_sources_voxforge_ru_mdc_2026-08-13.md)
- [VoxForge Russian pre-extraction exposure screen](data/manifests/voxforge_ru_mdc_2026_05_metadata_exposure_screen_v1.json)
- [VoxForge Russian frozen pre-QA selection](docs/voxforge_ru_mdc_pre_qa_selection_v1.md)
- [VoxForge Russian accepted Qwen3-TTS CustomVoice route](docs/voxforge_ru_mdc_qwen3_tts_customvoice_route_review_2026-08-13.md)
- [VoxForge Russian pre-QA materialization](docs/voxforge_ru_mdc_pre_qa_materialization_v1.md)
- [VoxForge Russian Qwen literal-text binding](docs/voxforge_ru_mdc_qwen3_tts_customvoice_pre_qa_text_binding_v1.md)
- [VoxForge Russian Qwen CUDA smoke](docs/voxforge_ru_mdc_qwen3_tts_customvoice_cuda_smoke_v1.md)
- [VoxForge Russian Qwen one-shot synthesis](docs/voxforge_ru_mdc_qwen3_tts_customvoice_pre_qa_synthesis_v1.md)
- [VoxForge Russian Qwen technical QA](docs/voxforge_ru_mdc_qwen3_tts_customvoice_pre_qa_technical_qa_v1.md)
- [VoxForge Russian rejected UtrobinTTS route review](docs/voxforge_ru_mdc_utrobinmv_vits_route_review_2026-08-13.md)
- [Silero V5.5 RU / eugene route intake](docs/silero_v5_5_ru_eugene_intake_2026-08-13.md)
- [Common Voice RU full-test metadata exposure screen](data/manifests/common_voice_ru_v24_full_test_metadata_exposure_screen_v1.json)
- [Common Voice RU / Silero V5.5 literal-text screen](data/manifests/common_voice_ru_v24_full_test_silero_v5_5_literal_text_screen_v1.json)
- [Common Voice RU / Silero V5.5 immutable pre-QA selection](docs/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_selection_v1.md)
- [Common Voice RU / Silero V5.5 pre-QA materialization](docs/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_materialization_v1.md)
- [Common Voice RU / Silero V5.5 literal-text binding](docs/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_text_binding_v1.md)
- [Common Voice RU / Silero V5.5 pre-QA synthesis](docs/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_synthesis_v1.md)
- [Common Voice RU / Silero V5.5 spoof technical QA](docs/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_spoof_technical_qa_v1.md)
- [Common Voice RU / Silero V5.5 immutable pairing](docs/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_pairing_v1.md)
- [Common Voice RU / Silero V5.5 acoustic gate](docs/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_acoustic_gate_v1.md)
- [Common Voice RU / Silero V5.5 project-exposure audit](data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_candidate_project_exposure_v1.json)
- [XLS-R Stage-B v2 / Common Voice RU Silero V5.5 contract](docs/research_xlsr_sls_stage_b_v2_common_voice_ru_v24_silero_v5_5_eugene_v1.md)
- [External RU spoof-source search](docs/russian_spoof_source_search_2026-08-11.md)
- [License-ledger snapshots](docs/license_ledger_snapshots.md)
