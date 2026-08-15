# KazDeepScan v1.0 Research

KazDeepScan — локальный personal-research проект для исследования признаков
синтезированности русской, казахской и смешанной речи. Это не идентификация говорящего,
не доказательство мошенничества и не развёрнутый сервис оценки риска.

**Source release:** `KazDeepScan v1.0 Research` (`1.0.0-research`), Git tag
`v1.0.0-research`. Текущий personal-research scope завершён и может использоваться как
воспроизводимый toolkit для аудита audio/manifests/licenses, preprocessing, frozen research
protocols и локального fail-closed API scaffold. Raw datasets, model weights и готовый risk
scorer в Git release не входят.

Python distribution metadata намеренно остаётся `0.1.0`: её содержащие `pyproject.toml` и
`uv.lock` уже закреплены SHA-256 в завершённых write-once run plans. Release identity доступен
через `kds --version`, OpenAPI и Git tag без нарушения этих historical contracts.

Реализация расширенной research-модели v4 начата по отдельному
[плану XLS-R+SLS v4](docs/kazdeepscan_v4_implementation_plan.md). Capacity/integrity часть Gate A
завершена с решением `proceed_24k`. Канонический v2 role contract и metadata-only train pool
также заморожены: `28 800` кандидатов, по `7 200` на каждую `RU/KK × bona-fide/spoof` cell,
с нулевым historical sample/text overlap и раздельными source/TTS-family roots между v4-ролями.
Source materialization извлёк `21 600` RuASD/KSC2 assets; exact raw-audio gate допустил `21 598`
и учёл две historical TeraTTS collisions. Decode/QA/VAD и audio leakage gate заморозил по
`5 000` RU bona-fide, RU spoof и KK bona-fide строк и принял `proceed_20k_balanced`; KK spoof
synthesis создал `7 200/7 200` raw WAV из exact-проверенных frozen KSC2 texts. Four-route
hash-pinned runner завершил по `1 800/1 800` WAV на Piper, MMS, KazEmoTTS и SparkTTS (`1 500`
target + `300` reserve на family), без runtime reject. Общий decode/QA/VAD/leakage gate
обработал все `7 200` WAV, оставил `6 200` eligible и заморозил ровно `5 000` KK spoof строк
(`4 × 1 250`); exact/near-audio intersections равны нулю. Его write-once receipt содержит
слишком широкое `training_authorized=true`; отдельная reconciliation фиксирует корректную
границу. Разрешённый assembler собрал combined `20 000` manifest с четырьмя balanced cells по
`5 000`; `4 604` shared KK text hashes закреплены только как within-train property. Отдельный
full training contract и его no-training preflight завершены: все `21 917` selected assets,
runtime и CUDA/BF16 проверены без forward pass. One-batch tail-unfreeze capacity profile прошёл
без OOM и без artifacts. Единственный write-once training run завершён: выбран tail-unfreeze
epoch 2 по macro RU/KK dev loss `0.08414228`; checkpoint находится вне Git и hash-bound в
versioned report. Отдельный metadata-only calibration-input gate затем заморозил `81` новых
VoxForge exact source identities с `81` новыми contributor groups и pinned eSpeak RU route.
Следующий write-once materialization/audio-isolation contract materialized `81` source WAV,
retained `79` source-ready, created exactly `79` text-only eSpeak WAV and froze `73` exact RU
pairs after QA/VAD and full current-history exact/near-audio screen. Новый
[RU calibration contract](docs/artifacts/v4/v4_ru_calibration_contract_2026-08-15.md) уже
hash-bind-ит selected checkpoint и эти pairs для одного temperature-only run. Write-once preflight
проверил все `146` assets и checkpoint SHA, после чего единственный run fit-нул RU temperature
`0.72535688`: NLL/ECE улучшились, Brier вырос, поэтому это смешанный calibration diagnostic, а не
claim об улучшении модели. Read-only [аудит готовности final](docs/artifacts/v4/v4_final_readiness_2026-08-15.md)
исключил все ранее inferred exact assets; final inference не запускался и потребует нового
immutable input/materialization contract. Такой [metadata-only contract](docs/artifacts/v4/v4_final_inputs_contract_2026-08-15.md)
уже frozen, но selection, materialization и final inference ещё не выполнялись.
Historical VoxForge text overlap раскрыт, но v4 train/dev sample/text/group intersections равны
нулю; speaker independence не заявлена.
Isolated dev-input contract уже выполнен: historical PyAra dev (`969` rows) и `474` frozen
KSC SLR102/Silero V4 KK pairs образуют `1 917` dev rows. Среди `600` KSC candidates source QA
оставил `571`, Silero QA — `535`; freeze использует только заранее объявленный reserve и не
использует detector feedback. Exact/near-audio intersections с history и внутри pool равны нулю.
Подробности:
[Gate A capacity](docs/artifacts/v4/gate_a_2026-08-14.md),
[frozen train candidates](docs/artifacts/v4/train_candidate_selection_2026-08-14.md) и
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
[calibration materialization/isolation](docs/artifacts/v4/calibration_materialization_2026-08-15.md) и
[RU calibration contract](docs/artifacts/v4/v4_ru_calibration_contract_2026-08-15.md) и
[RU calibration report](docs/artifacts/v4/xlsr_sls_model_v4_ru_calibration_v1.json).

Быстрый старт из clone:

```bash
uv sync --all-extras --locked
uv run kds --version
uv run kds validate-manifest data/manifests/denis_1_0_mdc_voxcpm2_official_pre_qa_ready_v1.csv \
  --license-ledger data/licenses/license_ledger.csv
uv run uvicorn services.api.main:app --host 127.0.0.1 --port 8000
```

`GET /healthz` должен вернуть `200`. `GET /readyz` и готовый `POST /v1/analyze` без отдельно
авторизованного scorer должны вернуть `503 model_unavailable`; это обязательная safety boundary,
а не поломка release. Полный release contract и ограничения находятся в
[KazDeepScan v1.0 Research release](docs/kazdeepscan_v1_0_research_release.md).

Ожидаемая строка версии: `kds 0.1.0 (KazDeepScan v1.0 Research)`.

## Локальный research inference пользовательского аудио

После tagged v1.0 добавлен отдельный opt-in contract
[`b0-user-audio-local-research-v1`](configs/inference/b0_user_audio_local_research_v1.json).
Он не вызывает frozen evaluation runners, не читает evaluation manifests, не создаёт execution
locks и не изменяет завершённые результаты. Старый `services.api.main:app` остаётся fail-closed
без product scorer.

Контур использует локальный Git-ignored checkpoint
`models/b0-unseen-generator-suite-v1.pt` только read-only. Contract требует exact SHA-256
`7b620af0c7e20788550b432c1d428b4e29e0a9c57cedc2fa549687c46b200539`; веса не входят в Git и
не скачиваются автоматически. Сначала проверьте наличие exact checkpoint:

```bash
.venv/bin/kds validate-research-inference
```

Затем передайте собственный файл, находящийся вне project roots `data/`, `models/`,
`artifacts/`, `checkpoints/`:

```bash
export KDS_FFMPEG_BINARY="$PWD/.tools/ffmpeg/bin/ffmpeg"
export KDS_FFPROBE_BINARY="$PWD/.tools/ffmpeg/bin/ffprobe"

.venv/bin/kds research-infer /absolute/path/to/user-audio.wav \
  --mime-type audio/wav \
  --acknowledge-research-only
```

`uncalibrated_spoof_score` — sigmoid от агрегированного raw logit, а не вероятность. Поля
`calibrated`, `probability_claim`, `fraud_claim` и `product_grade` всегда `false`; output содержит
явное предупреждение. `bonafide_like`/`spoof_like` означает только сторону зафиксированной
нулевой research boundary. Training-data overlap пользовательского файла не проверен, а модель
не является speaker-independent.

Отдельный local research API запускается только с явным contract:

```bash
export KDS_RESEARCH_INFERENCE_CONTRACT="$PWD/configs/inference/b0_user_audio_local_research_v1.json"
export KDS_FFMPEG_BINARY="$PWD/.tools/ffmpeg/bin/ffmpeg"
export KDS_FFPROBE_BINARY="$PWD/.tools/ffmpeg/bin/ffprobe"

.venv/bin/uvicorn \
  kds.serving.research_api:create_research_app_from_environment \
  --factory --host 127.0.0.1 --port 8001
```

```bash
curl -F "audio=@/absolute/path/to/user-audio.wav;type=audio/wav" \
  -F "acknowledge_research_only=true" \
  -F "confirm_external_user_audio=true" \
  http://127.0.0.1:8001/v1/research/analyze
```

API требует отдельное подтверждение, что upload является внешним пользовательским файлом, а не
frozen project asset. Он сохраняет upload только во временном private directory и применяет лимиты
`50 MiB / 10 min`, удаляет файл после запроса и не возвращает `risk_score`. Полный contract,
ограничения и QA описаны в
[local user-audio research inference v1](docs/research_user_audio_inference_v1.md).

Текущее реализованное состояние включает проверяемый фундамент данных, обученный research
checkpoint и строго ограниченный evaluation-контур:

- безопасная проверка размера, MIME-типа, фактического контейнера и длительности;
- нормализация через `ffmpeg` в mono PCM WAV, 16 кГц;
- измерение уровня, клиппинга и DC offset;
- выделение речи WebRTC VAD, отказ при менее чем 2.5 с речи;
- детерминированные окна 4.04 с с шагом 2.0 с;
- CSV-манифест с проверкой прав, происхождения и leakage между split-ами;
- fail-closed license-ledger CSV gate: non-exact header и extra row fields отклоняются, а
  поля с запятыми обязаны быть CSV-quoted; текущий mutable ledger strict-loads все
  `24/24` строки, frozen snapshots не изменены;
- KSC / OpenSLR SLR102 downloader с проверкой размера и gzip CRC, а также ingestion только из
  чистого архива; metadata и транскрипции берутся из его `Meta/` + `Transcriptions/`;
- Common Voice Russian v24 intake с pinned размером и SHA-256 archive, tar whitelist,
  атомарным MP3 slice extraction и leakage-safe split по связанным client/text группам;
- ML-DF v1 Italian intake с archive/CRC/metadata whitelist и изолированным cross-lingual
  OOD manifest; он не используется как Russian/Kazakh training или calibration data. Локальные
  ML-DF archive/raw/processed bytes удалены 14 августа 2026 как непригодные для будущего RU/KK
  v4 train; historical manifests/reports сохранены, а завершённый OOD run не перезапускается;
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
- VoxForge RU / fixed Qwen CustomVoice `aiden` route завершён по write-once protocol: `79` exact
  pairs прошли full two-review gate и current exposure audit, immutable Stage-B v2 contract и
  no-logit preflight. Единственный CUDA/BF16 run дал `146/158`, balanced accuracy `0.9241`,
  bona-fide recall `74/79`, spoof recall `72/79` и `67/79` fully correct pairs. Это
  source-linked personal-research evidence; rerun и tuning по final errors запрещены;
- для следующих исследований введены уровни evidence вместо требования «идеального» готового
  набора: основной independent layer, external source/generator-family holdout с явным
  `TTS training-data overlap unverified` и same-family sensitivity test. Минимум снижен до `60`
  готовых пар при цели `79`; cloning-capable TTS допустим только в offline text-only default
  voice contract без reference/prompt audio, normalizer, denoiser и retry. Denis 1.0 source
  intake/current exposure screen пройден: `1,150` exact pairs, `1,143` rows `>=2.5` s и zero direct
  sample/audio/text overlap. Но source single-speaker и likely speaker-lineage exposed через `12`
  unique historical `ru_RU-denis-medium` samples (`11` train, `1` dev). Official VoxCPM2
  artifact/source/history gate также пройден: exact `9` model files / `4,960,731,703` bytes,
  official source commit, safetensors, AudioVAE weights-only state и tokenizer/source code
  проверены; `0` VoxCPM rows среди `40,682` historical rows подтверждают новую для проекта
  generator family. Isolated Python 3.12 frozen-lock runtime и единственный non-candidate CUDA
  smoke также завершены: one actual call, `0` network attempts, no reference/prompt audio,
  normalizer, denoiser или retry; output `48 kHz` mono. Предшествующий duplicate-`streaming`
  interface failure зафиксирован и произошёл до generation/WAV. Frozen Denis metadata selection
  затем закрепил target `79` с category balance `27/26/26`; normal bona-fide QA/VAD оставил
  minimum layer `64` и отклонил `15` `insufficient_speech` rows без backfill. Training-data
  overlap остаётся unverified. Отдельный 64-row literal/canonical binding и one-shot synthesis
  contract затем были исполнены без отклонений от protocol: один model load, `64/64` generation
  calls, `0` failures и `0` network attempts. Synthetic decode/QA/VAD сохранил только `53` rows
  и отклонил `11` как `insufficient_speech`; заранее замороженный minimum `60` не достигнут,
  поэтому route остановлен без pair lock, reviews или detector inference. MCSKL/VoxCPM2-KZ-Darwin
  остаётся blocked review и после RU route считается той же family;
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
uv pip install --require-hashes \
  -r requirements/kazdeepscan-v1.0-research-test-linux.txt
.venv/bin/kds --version
.venv/bin/pytest
.venv/bin/ruff check src tests scripts services
.venv/bin/mypy src scripts services
```

Exact `pyarrow==22.0.0` overlay предназначен для CPython 3.11–3.13 на Linux x86_64 и включает
ToneSpeak Parquet tests. На другой platform основной locked environment остаётся usable, но эти
optional tests будут module-level skipped до установки совместимого `pyarrow>=18,<23`.

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

# ML-DF OOD evaluation исторически завершён и не перезапускается. Локальные media bytes
# удалены; versioned manifest и reports сохранены только как immutable research evidence.
# Подробный cleanup receipt: docs/local_storage_cleanup_2026-08-14.md.

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
selection заморозил `81` records с уникальными text и conservative contributor groups; на этом
этапе raw WAV не извлекались. Новый Qwen3-TTS CustomVoice Q8_0 / fixed `aiden` route прошёл local
six-artifact/CUDA verification и exact-route audit: `0` overlaps в `18,764` historical spoof
rows (`59` manifests), включая `0` legacy Qwen identifiers и `0` `aiden` aliases. Это не
Russian-native/speaker/architecture-independence claim и не synthesis: `aiden` документирован
как English token, поэтому обязательны source QA и full acoustic/language review. Первый
UtrobinTTS candidate остаётся отклонённым (76 unversioned historical rows). Ограничения и
literal-text binding завершён: `79/79` ready rows снова связаны с archive prompts без хранения
transcript, lexical rewrite или reselection; два `signal_too_quiet` rejects остаются исключёнными.
Non-candidate CUDA smoke подтвердил 24 kHz mono output fixed Qwen route. One-shot synthesis затем
сохранил `79/79` unique 24 kHz mono fixed-`aiden` WAV (`0` failures), а normal decode/quality/VAD
QA сохранил `79/79` mono 16 kHz PCM-16 ready spoof assets (`0` rejects, no replacement/backfill),
после чего immutable lock зафиксировал `79` exact text-matched pairs / `158` assets. Две
independent full-asset формы затем прошли fail-closed acoustic/language gate: `158/158` exact WAV
получили по два решения `pass/yes/yes/yes/yes`. Это exact-byte acoustic evidence, а не
Russian-native/speaker/organizational-independence claim и не detector inference. Completed gate и
его ограничения описаны в
[VoxForge Qwen acoustic-gate receipt](docs/voxforge_ru_mdc_qwen3_tts_customvoice_pre_qa_acoustic_gate_v1.md);
последующий project-exposure audit pinned `33` current configs / `18` referenced manifests /
`12,397` prior rows и нашёл `0/0/0` sample/audio/text overlap. Он сам не запускал и не разрешал
inference; полный audit описан в
[VoxForge Qwen exposure receipt](docs/voxforge_ru_mdc_qwen3_tts_customvoice_candidate_project_exposure_v1.md).
Последующий immutable evaluation contract закрепил Stage-B v2 checkpoint, disjoint 976-row PyAra
calibration, fixed `0.5` boundary, three-source frozen ledger и новые write-once paths; plan
SHA-256 `9e36b5d6a35cfa0b796ff24e62f3bfa78667d0b1d9da993f1863a2fe61c421cc`. Contract receipt —
[здесь](docs/research_xlsr_sls_stage_b_v2_voxforge_ru_mdc_qwen3_tts_customvoice_aiden_v1.md).
После этого ровно один no-logit preflight проверил `1,134` assets и CUDA/BF16; local receipt
SHA-256 `4f9a56ab8de8fdb876d64c408032c468492c5ca813a34ad6bd846dd543312e5b`. Единственный
write-once inference run затем дал `146/158` correct, balanced accuracy `0.9241`, recall
`74/79` bona-fide и `72/79` spoof, `67/79` fully correct pairs. Execution/report SHA-256:
`286c4e680defb01217b138da604096d29a7615d632bab18cff7653ac867e3c94` /
`7da4df67f756addbc4bcd21868e294a62a150985479f30aad7e8f93bdbc96dff`. Repeat run и tuning
по final errors запрещены; это не source/speaker/vendor/architecture independence или product
quality.

Политика силы будущих доказательств, проверка Denis/VoxCPM2 и условный MCSKL/KZ route описаны в
[external holdout policy/source review](docs/external_holdout_policy_and_voxcpm2_candidates_2026-08-14.md).
Завершённый exact Denis archive/source-exposure intake и ограничения speaker lineage
зафиксированы в [source receipt](docs/data_sources_denis_1_0_mdc_2026-08-14.md), frozen 79-row
selection и `64/79` bona-fide QA/VAD — в
[materialization receipt](docs/denis_1_0_mdc_pre_qa_materialization_v1.md), а exact
VoxCPM2 artifact/source/history gate — в
[model/source receipt](docs/data_sources_voxcpm2_official_2026-08-14.md). Последующий
[immutable 64-row text binding](docs/denis_1_0_mdc_voxcpm2_pre_qa_text_binding_v1.md) уже
закрепил literal/collapse-whitespace/NFKC hashes, exact runtime/program hashes, один model load,
ровно одну attempt на каждый ready text и будущие write-once synthesis/QA paths. Plaintext не
сохранён. Последующий [one-shot synthesis и technical-QA receipt](docs/denis_1_0_mdc_voxcpm2_pre_qa_synthesis_and_technical_qa_v1.md)
зафиксировал `64/64` successful raw WAV и один normal synthetic decode/QA/VAD pass: `53` ready,
`11` `insufficient_speech`, reuse/retry/replacement/backfill `0/false/false/false`. Это ниже
неизменяемого minimum `60`, поэтому `stop_below_minimum_60` окончателен для этого frozen route:
pairing, acoustic review и detector inference не выполнялись и не разрешены.
