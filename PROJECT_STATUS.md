# KazDeepScan — текущий статус

**Обновлено:** 15 августа 2026

**Source release:** KazDeepScan v1.0 Research (`1.0.0-research`, tag
`v1.0.0-research`). Python distribution metadata сохранено как `0.1.0`, потому что
`pyproject.toml` и `uv.lock` являются hash-pinned inputs завершённых write-once runs.

**Scope:** завершённый personal-research toolkit, без записи голосов людей, voice cloning и
product/API risk score. Raw audio и model weights не входят в Git/release. Отдельный post-v1.0
user-audio route возвращает только явно некалиброванный research signal.

**Состояние:** текущий v1.0 Research plan завершён. XLS-R+SLS v2 и отдельная v3 ветка завершены
на write-once research protocols. Реализация XLS-R+SLS model v4 начата по отдельному
[плану](docs/kazdeepscan_v4_implementation_plan.md). Capacity/integrity часть Gate A завершена с
`proceed_24k`: повторно проверены exact RuASD/Common Voice/KSC2 bytes, current project history,
лицензии и локальные KK TTS-family. Канонический v2 role contract заморозил `28 800`
metadata-only train candidates (`7 200` на cell, включая `1 200` QA reserve), исключил
historical sample/text collisions и разделил source/TTS-family roots между v4-ролями. Audio
materialization извлёк `21 600` RuASD/KSC2 assets; exact raw-audio gate допустил `21 598` и
отклонил две historical TeraTTS collisions. Decode/QA/VAD и exact/near-audio gate оставил
`18 930` eligible source rows и заморозил по `5 000` RU bona-fide, RU spoof и KK bona-fide.
Принято `proceed_20k_balanced`; KK spoof создаётся через четыре train-only TTS-family. Все
`7 200` frozen KSC2 text inputs (`4 × 1 800`) извлечены и exact-проверены;
synthesis contract и resumable local runner завершили четыре routes по `1 800/1 800` raw WAV
(`1 500` target + `300` reserve на family) без runtime reject. Hash-pinned общий audio QA/leakage
gate обработал все `7 200` rows: `6 200` eligible, `1 000` rejects только
`insufficient_speech`, а frozen KK spoof train содержит ровно `5 000` (`4 × 1 250`) строк.
Historical/within-pool exact и near-audio intersections равны `0`. Completed write-once receipt
содержит слишком широкое `training_authorized=true`; новый governance receipt сохраняет исходные
bytes и устанавливает effective boundary: разрешены только combined `20 000` manifest и отдельный
training-contract preflight. Write-once assembler уже заморозил combined `20 000` manifest с
четырьмя cells по `5 000`; `4 604` shared KK text hashes разрешены только внутри одной train
role. Full training contract и no-training preflight завершены: hashes всех `21 917` selected
assets, runtime lock и CUDA/BF16 проверены без forward pass. One-batch tail-unfreeze profile
прошёл без OOM (`2.91` GB peak allocated) и без artifacts. Единственный write-once training
run завершён за `1 869.94` s: tail-unfreeze epoch 2 выбран по macro RU/KK dev loss
`0.08414228`, selected state и ignored checkpoint hash зафиксированы. Metadata-only
calibration-input gate затем заморозил `81` новых VoxForge source identities / `81` new
contributor groups и повторно проверил pinned eSpeak RU route. Последующий write-once
materialization/audio-isolation contract извлёк `81` source WAV, retained `79` после QA/VAD,
синтезировал ровно `79` new text-only eSpeak WAV и заморозил `73` exact RU pairs; все шесть
synthetic rejects — `insufficient_speech`, без replacement/backfill. Current-history screen
охватил `84,605` unique hashes (`84,213` fingerprinted, `392` ML-DF exact-only). Checkpoint,
calibration и новый final не запускались; speaker independence не заявлена. Детали:
Isolated dev-input contract выполнен на CUDA: PyAra `969` rows и `474` frozen KSC
SLR102/Silero V4 pairs образуют combined dev `1 917` rows. Из `600` KSC candidates QA оставил
`571` source и `535` spoof rows; target достигнут только по predeclared reserve. Historical и
within-pool exact/near-audio intersections равны нулю. Единственный разрешённый write-once
training run уже завершён; его checkpoint нельзя повторно обучать или перезаписывать.
[capacity](docs/artifacts/v4/gate_a_2026-08-14.md) и
[selection](docs/artifacts/v4/train_candidate_selection_2026-08-14.md),
[source raw materialization](docs/artifacts/v4/source_raw_materialization_2026-08-14.md) и
[source decode/QA](docs/artifacts/v4/source_decode_qa_2026-08-14.md),
[KK spoof texts](docs/artifacts/v4/kk_spoof_text_materialization_2026-08-14.md) и
[synthesis plan](docs/artifacts/v4/kk_spoof_synthesis_plan_2026-08-14.md),
[MMS synthesis](docs/artifacts/v4/xlsr_sls_model_v4_kk_spoof_kk_mms_kaz_v1_synthesis_v1.json) и
[KazEmoTTS synthesis](docs/artifacts/v4/xlsr_sls_model_v4_kk_spoof_kk_kazemotts_v1_synthesis_v1.json),
[Piper synthesis](docs/artifacts/v4/xlsr_sls_model_v4_kk_spoof_kk_piper_issai_high_v1_synthesis_v1.json) и
[SparkTTS synthesis](docs/artifacts/v4/xlsr_sls_model_v4_kk_spoof_kk_sparktts_v1_synthesis_v1.json),
[common audio gate](docs/artifacts/v4/kk_spoof_audio_gate_2026-08-15.md) и
[его reconciliation](docs/artifacts/v4/xlsr_sls_model_v4_kk_spoof_audio_gate_governance_v1.json),
[combined train manifest](docs/artifacts/v4/combined_train_manifest_2026-08-15.md) и
[isolated dev-input receipt](docs/artifacts/v4/isolated_dev_inputs_2026-08-15.md) и
[full training contract](docs/artifacts/v4/v4_training_contract_2026-08-15.md) и
[calibration-input gate](docs/artifacts/v4/calibration_inputs_2026-08-15.md) и
[calibration materialization/isolation](docs/artifacts/v4/calibration_materialization_2026-08-15.md).
Текущий frozen calibration ledger намеренно запрещает temperature fitting; нужен новый explicit
rights decision до любого checkpoint-scoring/calibration contract.
v3 использовал изолированные train / Stage-A dev / Stage-B dev / calibration roles, симметричную
train-only augmentation, выбрал Stage-A epoch 3 и Stage-B epoch 4 только по dev loss, затем
провёл один final GPU run на неизменяемых 55 Common Voice/Dialog-RU парах. Stage-D v2
logits/errors не загружались. Exact final assets уже были оценены v2, поэтому v3 — не blind
test; результаты не source-, speaker- или architecture-family-independent и checkpoint не
используется в API.

Для будущих unevaluated layers принят трёхуровневый evidence policy: основной независимый слой,
external source/generator-family holdout с непроверенным TTS training-data overlap и same-family
sensitivity test. Denis source/current-exposure/frozen-selection/bona-fide-QA и official VoxCPM2
artifact/source/history/runtime gates вместе с одним non-candidate CUDA smoke завершены. Denis
оставил minimum `64/79` bona-fide layer без backfill. Последующий frozen VoxCPM2 run создал
`64/64` raw spoof WAV, но synthetic QA/VAD сохранил только `53` и отклонил `11` как
`insufficient_speech`. Minimum `60` не достигнут: route остановлен до pairing/reviews/exposure/
detector inference.

Post-v1.0 local user inference v1 завершён отдельным contract и не меняет frozen evaluation:
exact Git-ignored B0 checkpoint проверяется по file/state SHA-256, CPU scoring работает только для
external user audio вне project data/model roots, а CLI/API требуют явный research-only
acknowledgment. Output называется `uncalibrated_spoof_score`, не содержит `risk_score` и всегда
фиксирует `calibrated=false`, `probability_claim=false`, `fraud_claim=false`,
`product_grade=false`. Existing `services.api.main:app` остаётся `model_unavailable`.

Локальная очистка 14 августа 2026 удалила только воспроизводимые caches, exact duplicate первого
RuASD TAR, неиспользуемый `ffplay` и Italian ML-DF OOD media bytes. ML-DF не является RU/KK v4
данными; его completed historical manifests/reports сохранены, evaluation не перезапускался.
Основные RU/KK archives, synthetic-family weights, frozen checkpoints и protected
`/home/ruslan/Downloads/269-lockdown/` не затронуты.

Этот файл намеренно краткий. Архитектура описана в
[KazDeepScan_implementation_blueprint.md](KazDeepScan_implementation_blueprint.md), следующие
действия — в [План реализации.md](План%20реализации.md), новые подробные receipts — в
`docs/artifacts/`.

## Что готово

- Безопасный audio pipeline: проверка media, FFmpeg decode, QA, WebRTC VAD, 16 kHz mono WAV и
  окна 4.04 s.
- Строгие manifests, SHA-256 asset validation, license ledger/snapshots, group-aware split и
  leakage checks. Separate license-ledger CSV gate fail closed при non-exact header и extra row fields; все `24`
  mutable-ledger строки загружаются без обрезания comma-containing `notes`, frozen snapshots
  не изменены.
- Source-specific intake и аудиты RU/KK/mixed datasets; raw audio и weights исключены из Git.
- Denis 1.0 source intake/current pre-selection exposure завершены: exact archive `109,594,943`
  bytes / SHA-256 `75e2c6…ccf9b`, `1,150` paired unique
  texts/audio, `1,150` complete Ogg/Opus decodes и `1,143` rows `>=2.5` s. Direct sample/audio/
  three-text-hash overlap равен нулю по `35` configs и `95` manifest files. Human source новый,
  но corpus single-speaker, а `12` unique historical `ru_RU-denis-medium` samples (`11` train,
  `1` dev) делают speaker lineage likely exposed. Route остаётся external holdout с
  `TTS training-data overlap unverified`, не speaker-independent/robust evidence. Metadata-only
  target заморозил `79` exact rows с category balance `27/26/26`; decode/QA/VAD оставил `64`
  ready (`23/17/24`) и `15` `insufficient_speech` rejects (`4/9/2`) без reuse/replacement/
  backfill. Minimum 60 пройден, target 79 не достигнут.
  Official OpenBMB VoxCPM2 exact model revision `bffb3d…` и source commit `ee8161…` закреплены:
  `9` files / `4,960,731,703` bytes, `577` contiguous BF16 safetensors tensors, `316`-member
  AudioVAE ZIP и `312`-tensor weights-only state, tokenizer/source safety pass. History screen
  связал `95` manifests / `40,682` rows / `19,001` spoof rows и нашёл `0` VoxCPM rows, поэтому
  generator family новый для проекта. Exact official `uv.lock` установлен frozen в isolated
  Python 3.12 (`160` distributions). После versioned pre-inference duplicate-`streaming` failure
  без WAV corrected wrapper выполнил ровно один actual CUDA call внутри network namespace:
  `0` network attempts, no reference/prompt/LoRA/normalizer/denoiser/retry, mono `48 kHz`,
  `161,280` frames. Smoke не является listening/acoustic evidence и не повторяется;
  training-data overlap/default voice identity не подтверждены.
  Frozen 64-row candidate contract впоследствии исполнен один раз: один model load, `64/64`
  successful generation attempts, raw duration `276.00` s, без network/retry/resynthesis/backfill.
  Единственный normal decode/technical-QA/VAD pass дал `53` ready mono PCM16 `16 kHz` rows
  (`251.52` s) и `11` final `insufficient_speech` rejects. Предустановленный minimum `60` не
  достигнут; technical receipt имеет status `stop_below_minimum_60`, pair lock и detector
  inference запрещены.
  MCSKL остаётся blocked из-за `78/73` participant и `CC BY` / `CC BY-NC-SA` конфликтов;
  VoxCPM2-KZ-Darwin не раскрывает provenance Kazakh LoRA/base достаточно и после RU VoxCPM2
  является same-family sensitivity, не новым generator-family layer.
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
  backfill. Exact text-hash/text-ID pairing then froze `79` pairs / `158` assets without metric
  selection. The immutable full `158`-asset acoustic/language packet and two completed forms then
  passed the fail-closed gate: `158/158` assets have two `pass/yes/yes/yes/yes` decisions from
  distinct pseudonymous reviewer IDs. Report SHA-256
  `bf7a6d84c7ecb71462c128b70f68d8f939ebece916fc11e87c6b9ac8afd26029`; it confirms only
  exact-byte acoustic/language criteria and records no detector inference. The subsequent current
  project-exposure audit pinned `33` configs / `18` referenced manifests / `12,397` prior rows and
  found `0/0/0` sample/audio/text overlap; receipt SHA-256
  `8696bfbea8d9f59451881bcf6ee875ff235c2e281c7b9ddbe5be4ecf74804a72`. It also did not
  authorize inference. A separate immutable Stage-B v2 contract is now prepared: fixed 976-row
  PyAra calibration, `0.5` boundary, three-source frozen ledger and fresh output paths; plan
  SHA-256 `9e36b5d6a35cfa0b796ff24e62f3bfa78667d0b1d9da993f1863a2fe61c421cc`. Static validation
  found zero calibration/final overlap in all five leakage fields. The single no-logit preflight
  then validated all `1,134` assets and CUDA/BF16; receipt SHA-256
  `4f9a56ab8de8fdb876d64c408032c468492c5ca813a34ad6bd846dd543312e5b`. It performed no
  training, threshold selection or detector inference. Exactly one lock-before-logits run then
  fitted temperature only on PyAra (`T=1.299543`) and evaluated the frozen layer: `146/158`,
  balanced accuracy `0.9241`, bona-fide recall `74/79`, spoof recall `72/79`, fully correct pairs
  `67/79`. Execution/report SHA-256 are
  `286c4e680defb01217b138da604096d29a7615d632bab18cff7653ac867e3c94` and
  `7da4df67f756addbc4bcd21868e294a62a150985479f30aad7e8f93bdbc96dff`; the report correctly
  records inference as performed. Repeat run and final-error tuning are prohibited.
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
| VoxForge RU / Qwen Aiden balanced accuracy | 0.9241 | 146/158; 79 fixed pairs; exactly one run |

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
- Не повторять VoxForge/Qwen preflight или inference; не менять его exact pairs, reviews,
  exposure receipt, contract, calibration/boundary или artifacts и не использовать 12 final
  errors для tuning/replacement.
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
- Research checkpoint нельзя подключать к product API risk score. Разрешён только отдельный
  uncalibrated user-audio contract без probability/fraud/product claims.

## Действия после v1.0 Research

Для завершения v1.0 дополнительные ML runs, sources или detector tests не требуются. Следующие
пункты относятся только к будущей v1.x/v2 development и не являются долгом текущего release.

1. Не повторять Stage-C/Stage-D/v3 runs и не использовать их final errors для tuning, выбора
   checkpoint, temperature, threshold или augmentation.
2. Новый RU route прошёл завершённые selection/materialization/text binding/synthesis/technical
   QA/pair lock, 84-asset technical acoustic gate, zero-overlap project-exposure audit, immutable
   evaluation contract, no-logit preflight and its exactly one final GPU inference run. Не
   повторять его, не менять candidate/reviews/checkpoint/calibration/boundary and не использовать
   final errors для tuning. Старые 55 Stage-D/v3 пар, 73-row selection и их rejections нельзя
   переиспользовать как «новый blind» тест.
3. Для VoxForge RU сохранить `79` exact pairs, completed forms, gate/exposure/completion receipts,
   immutable contract, preflight, execution lock и report без изменений. Не повторять inference,
   не использовать 12 final errors для tuning/replacement и не использовать UtrobinTTS как
   backfill. Новый research layer требует genuinely new source/route и отдельный contract.
4. Denis source/current-exposure/selection/QA и official VoxCPM2 artifact/source/history/runtime/
   smoke gates завершены:
   сохранить exact hashes, pre-inference failure, one-shot smoke receipt, likely speaker-lineage
   и unverified TTS-training-overlap disclosures без изменений; smoke не повторять. Exact 79-row
   selection, 64 ready rows и 15 rejects не менять и не backfill. Отдельный immutable 64-row
   literal/canonical binding и one-attempt synthesis contract завершён до candidate WAV; receipt
   SHA-256 `943a9595968996f29da1a13f213e28419fc2c7b5215df790e4d4c440528f2b7b`.
   One-shot synthesis и normal synthetic QA/VAD также завершены: `64/64` raw generated,
   `53` ready, `11` `insufficient_speech`, no retry/replacement/backfill. Frozen minimum `60`
   не достигнут, поэтому status `stop_below_minimum_60` сохранять без pair lock, reviews или
   detector inference. Для нового evaluation layer нужен отдельный genuinely new source/route и
   новый pre-outcome contract; эти 11 rows нельзя пересинтезировать или заменять.
5. API/product track не начинать без отдельного commercial-rights, privacy, verified-speaker,
   deployment и product-calibration contract.
6. Local user inference применять только к внешним пользовательским файлам с явным
   acknowledgment. Не передавать ему frozen project assets, не трактовать score как вероятность
   или fraud verdict и не использовать пользовательские результаты для tuning.

Полный порядок и критерии остановки: [План реализации.md](План%20реализации.md).

## Проверка и воспроизводимость

- Ruff: успешно.
- mypy: успешно.
- pytest: `352 passed`, `0 failed`, `1 skipped`; optional ToneSpeak Parquet tests выполняются с
  exact Linux test overlay `pyarrow==22.0.0`, не изменяющим исторический `uv.lock`.
- Denis pre-QA: current exposure v2 SHA-256 `d140918a60d437f41d209b57803058179bb1d8cfd7ae8e7db217788d0b9841cb`;
  selection/materialization receipts SHA-256 `5e9dd93290eece14f738cab06e665d61a47d0e79cb5e1730198574471b2fc37c` /
  `c36fc8bcc60c16d5d2493c4bf8b77719f32ca3d9da9ba15d51054b9ee16d5386`; exact asset
  validation `79/79` raw и `64/64` ready.
- Denis × VoxCPM2 binding: exact `64` ready rows, receipt SHA-256
  `943a9595968996f29da1a13f213e28419fc2c7b5215df790e4d4c440528f2b7b`, row fingerprint
  `b28d1ff99bc50b5dc6879b75a7dee018cef3a0767508cfde2fc660f9156204c0`.
- Denis × VoxCPM2 synthesis/QA: raw/synthesis/ready/rejection/QA SHA-256
  `45c8d5c9fb4d9f9bd9b5745add9b6e738111928b2b5c42a8779e030377195362` /
  `b827ba8208d4d44fdaeefaabeaa841355ed580aa253b261dead766a3a16ee83b` /
  `f90a634b80364a3a70046cf66354dbc7c11459f15a375b1e3a61c1f440e3028a` /
  `38c4da79e2bd0a50168fabb1817f866c6dacbbcd657c8ee18e6846a45e058ecb` /
  `ca46362313f50f79043dd559f8d739185b51d8cb0dc9dcc0f5dc659e5b02951c`;
  `64` raw, `53` ready, `11` final rejects, detector inference не выполнялся.
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

- [KazDeepScan v1.0 Research release contract](docs/kazdeepscan_v1_0_research_release.md)
- [KazDeepScan v1.0 Research machine receipt](data/releases/kazdeepscan_v1_0_research_release_receipt.json)
- [Local user-audio research inference v1](docs/research_user_audio_inference_v1.md)
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
- [VoxForge Russian Qwen immutable pair lock](docs/voxforge_ru_mdc_qwen3_tts_customvoice_pre_qa_pairing_v1.md)
- [VoxForge Russian Qwen acoustic/language gate](docs/voxforge_ru_mdc_qwen3_tts_customvoice_pre_qa_acoustic_gate_v1.md)
- [VoxForge Russian Qwen project-exposure audit](docs/voxforge_ru_mdc_qwen3_tts_customvoice_candidate_project_exposure_v1.md)
- [VoxForge Russian Qwen immutable XLS-R+SLS contract](docs/research_xlsr_sls_stage_b_v2_voxforge_ru_mdc_qwen3_tts_customvoice_aiden_v1.md)
- [VoxForge Russian Qwen one-time execution completion](data/manifests/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_evaluation_completion_v1.json)
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
- [External holdout policy и VoxCPM2 candidate review](docs/external_holdout_policy_and_voxcpm2_candidates_2026-08-14.md)
- [Denis 1.0 source intake и exposure screen](docs/data_sources_denis_1_0_mdc_2026-08-14.md)
- [Denis frozen selection и bona-fide QA/VAD](docs/denis_1_0_mdc_pre_qa_materialization_v1.md)
- [Official OpenBMB VoxCPM2 artifact/source/history gate](docs/data_sources_voxcpm2_official_2026-08-14.md)
- [Denis × official VoxCPM2 immutable 64-row text binding](docs/denis_1_0_mdc_voxcpm2_pre_qa_text_binding_v1.md)
- [Denis × official VoxCPM2 one-shot synthesis и technical QA](docs/denis_1_0_mdc_voxcpm2_pre_qa_synthesis_and_technical_qa_v1.md)
