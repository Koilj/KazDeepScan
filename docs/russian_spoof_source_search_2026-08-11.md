# Поиск независимого Russian spoof source — 11 августа 2026

**Обновлённый итог:** готовый независимый *external Russian-only binary* release, совместимый
с текущим final/product protocol, по-прежнему **не найден**. Но один маленький public
Russian **spoof-only** release — ToneSpeak — прошёл source-level personal-research intake:
exact revision, license declaration, artifacts и embedded audio structure закреплены локально.
Это не подменяет отсутствующий bona-fide counterpart, product provenance или acoustic gate.

Как controlled personal-research alternative создан отдельный local text-only Russian eSpeak NG
generator family на ранее неиспользованных `75` FLEURS RU text groups. Это не меняет вывод поиска:
локальный derivative не называется external release. Его собственный narrow acoustic gate уже
пройден, но это не создаёт external source independence. Полный receipt —
`docs/data_sources_espeakng_ru_2026-08-11.md`.

## ToneSpeak — принят только как research-only Russian spoof source

[Vikhrmodels/ToneSpeak](https://huggingface.co/datasets/Vikhrmodels/ToneSpeak/tree/d40f94cd5c7dcf756a8c59a1c465b834220bec56)
публикует `6,998` Russian MP3 with `text`, `text_description` и per-row `voice_name` на
commit `d40f94cd5c7dcf756a8c59a1c465b834220bec56`, card license `Apache-2.0`. Авторская
card declares GPT-4.1 mini for text/prompt generation and GPT-4o mini TTS for audio;
all ten names correspond to built-in OpenAI TTS voices. Local streaming audit without MP3
extraction verified all five LFS payloads, `6,298` train + `700` validation rows, all
`24 kHz` MP3, zero duplicate audio payloads and zero normalized-text groups shared across
the source splits. Lock and receipt:

- `data/licenses/tone_speak_ru_v1_artifact_lock.json`;
- `data/licenses/tone_speak_ru_v1_artifact_audit_receipt.json`, SHA-256
  `c14d3f0fd38e6ee8675a78b08b627aa43ca618bde52be9c1f90cec8d71996908`.

Frozen Stage-A manifest has zero OpenAI generator/version markers and zero of the ten
ToneSpeak voice IDs. Therefore this is a plausible *unseen-generator Russian spoof*
candidate for a future **research-only** protocol, unlike OpenSTT RHVoice. It has a new
`license_ledger.csv` row only at `personal_research/research_only`, with
`spoof_voice_group_provenance=source_provided`.

It still cannot be called final quality: the card has no independent per-row API log/model
snapshot/reference-audio proof and it has no bona-fide class. A single locked, validation-only,
balanced 100-row **OOD candidate** now exists with raw/ready receipts; it never entered
train/dev/calibration/final. Its 100-asset two-review acoustic gate is complete: all `100/100`
locked WAVs received two `pass/yes/yes` decisions. One separately hash-pinned frozen-checkpoint
OOD run is also complete: fixed-boundary spoof recall is `88/100`, with no threshold fitting or
binary metric. This confirms only Russian audibility and lexical preservation for these exact
bytes plus a narrow model observation; it is not binary quality or product eligibility. Full
details, pinned review hashes, plan and receipt are in
`docs/research_xlsr_sls_stage_b_tone_speak_ru_ood_100.md`.

## Повторная external-проверка — результат не изменился

12 августа дополнительно проверены предложенные YO-CPT-ru, YO-CPT-kk и Dusha. Оба YO-CPT —
YouTube-derived bona-fide TTS-pretraining corpora, а не spoof releases; они имеют unresolved
per-record copyright/privacy provenance, face-derived persona/identity fields и огромный объём
(около 1.01 TB RU и 99.95 GB KK). Не скачивать. Dusha — human Russian emotion corpus, не spoof
source; crowd archive 28 GB сейчас не нужен, podcast audio сам publisher не распространяет по
licensing причине. Решения: `data_sources_yocpt_ru_2026-08-12.md`,
`data_sources_yocpt_kk_2026-08-12.md`, `data_sources_dusha_2026-08-12.md`.

Повторный read-only search в этот же день не нашёл пригодного **independent Russian-only** release.
Актуальная [RuASD dataset card](https://huggingface.co/datasets/lab260/RuASD) описывает `37`
Russian-capable TTS/voice-cloning systems и binary Russian corpus, однако этот exact source уже
раскрыт в Stage-B train; он не может стать независимым final spoof source. Новое описание не
даёт отдельного неиспользованного immutable Russian-only subset с проверенной text-only provenance.

Актуальная [LRLspoof dataset card](https://huggingface.co/datasets/lab260/LRLspoof) подтверждает
`MIT` license, но также `1,304,455` **spoof-only** utterances на `66` языках, единый multi-part
tarball `452 GB` и отключённый dataset viewer. Следовательно, это multilingual, не Russian-only
release; кроме отсутствия binary counterpart, он остаётся несовместим с локальным ограничением
acquisition более `2 GB` и безопасной selective intake. Его не добавлять в ledger и не скачивать.

## OpenSTT RHVoice — artifact integrity passed, source ещё не принят

Обнаружен отдельный Russian TTS layer `tts_russian_addresses_rhvoice_4voices` из official
[OpenSTT repository](https://github.com/snakers4/open_stt), который заархивирован на commit
`b8c2c48c8c778234d2801266b44d532ecc66a1bc`. Official catalogue описывает `1,741,838`
Russian address utterances (`754` h) от `4` TTS voices. [Azure Open Datasets card](https://learn.microsoft.com/en-us/azure/open-datasets/dataset-open-speech-text)
публикует отдельные archive и manifest, а также указывает `CC-BY-NC`; commercial use возможен
только по отдельному agreement с авторами.

| Artifact | Exact official evidence |
| --- | --- |
| `archives/tts_russian_addresses_rhvoice_4voices.tar.gz` | `13,862,699,423` bytes; Last-Modified `2020-05-04`; MD5 `2bdd0e26d972f60a0e54dafeef642264` |
| `manifests/tts_russian_addresses_rhvoice_4voices.csv` | `220,255,453` bytes; Last-Modified `2020-05-05`; MD5 `628c2974eeb2edfba4a560445d9dc628` |

Пользователь предоставил локальные копии обоих artifacts. Read-only аудит
`src/kds/data/openstt_rhvoice.py` повторно сверил точный размер и MD5, затем без extraction
потоково прошёл весь gzip/TAR и сверил его с headerless CSV. Результат закреплён в
`data/licenses/openstt_rhvoice_v1_artifact_audit_receipt.json` (SHA-256
`0d594e86f6985b5498284aeb56f49461c581d634668b5472907c739e46a63f3c`):

- `1,741,838` CSV rows и ровно столько же OPUS плюс TXT TAR members;
- `1,628,561` unique paired paths; `113,277` exact duplicate rows на `88,897` paths;
- archive содержит то же число повторов. Все `226,554` повторённых OPUS/TXT members были
  прочитаны потоково и совпали с первым вхождением byte-for-byte по SHA-256;
- unsafe paths, links, неподходящие типы members, лишние paths, недостающие или избыточные
  occurrences fail-closed отклоняются. Ничего не извлекалось, не добавлялось в manifest,
  ledger, QA, train, calibration или model inference.

Это снимает только риск повреждённого/неполного download. Source всё ещё **не** является
принятым final source и не записывается в `license_ledger.csv` по следующим причинам:

- published manifest связывает только `audio path`, `text path` и duration. Он не публикует
  per-row RHVoice voice ID, SHA-256 или доказательство, что каждая строка создана без reference
  audio/voice conversion. «4 TTS voices» — source-level, а не row-level provenance;
- source spoof-only. Даже после intake ему потребуется independent RU bona-fide counterpart,
  SHA/text overlap audit против RuASD/PyAra/FLEURS/KSC2 и отдельный acoustic gate;
- `CC-BY-NC` разрешает только current personal-research scope, не final/product use.

Этот candidate **не проходит** independent/unseen-generator gate frozen Stage-B protocol:
Stage-A plan обучал frozen head на `ruasd_ru_v1_full_research_2000_ready.csv`, где `25` train
spoof WAV имеют `generator_name=rhvoiceTTS`. Это тот же RHVoice generator family, что явно
назван в OpenSTT artifact. Full RuASD audit также перечисляет OpenSTT среди bona-fide upstream
sources. Поэтому exact asset/text overlap уже не может вернуть ему статус independent final
source; extraction и отдельный overlap audit для intake не запускаются.

OpenSTT может быть рассмотрен лишь в будущем как явно **seen-generator, research-only** stress
source после отдельного решения владельца. Это не разрешает extraction для dataset intake, новую
запись ledger, QA, обучение, calibration или финальный run. Поиск независимого Russian-only
spoof source должен продолжаться вне RHVoice family.

## Остальные results continued search — нового final candidate нет

Повторная проверка актуальных public releases также не дала альтернативы:

- [HIR-SDD-raw](https://huggingface.co/datasets/marsianin500/HIR-SDD-raw) — слой human
  annotations над уже известными `MLAAD`, `ASVspoof5`, `LibriSeVoc`, `DFADD` и M-AILABS,
  а не independently generated Russian spoof audio release. В отображаемых rows audio language
  отмечен как `en`; Russian comments annotators не образуют Russian speech provenance. Не
  скачивать и не использовать как RU final source.
- [SpeechFake](https://arxiv.org/abs/2507.21463) формально охватывает 46 языков, но его
  train/dev состоят только из English/Chinese; остальные 37 языков объединены в один test
  bucket. Для multilingual part bona-fide upstream — Common Voice. Это не Russian-only release
  и не устраняет известный Common Voice/RuASD overlap risk.
- [SynHate](https://arxiv.org/abs/2506.06772) содержит Russian rows, но это четырёхклассовый
  hate-speech dataset на `37` языках, построенный поверх MuTox/ADIMA. Его fake audio создано
  MMS-1B TTS; frozen Stage-A train уже содержит `19` `mms_tts_rus` spoof rows. Значит не
  проходит neither Russian-only, nor independent/unseen-generator gate.
- [XMAD-Bench](https://aclanthology.org/2026.findings-eacl.162/) заявляет cross-domain split,
  но остаётся multilingual package с ранее установленным Russian M-AILABS spoof-only component
  и VC provenance. Он не заменяет independent RU binary layer.
- [PolyGlotFake](https://arxiv.org/abs/2405.08838) — multilingual multimodal set, который
  прямо использует TTS, voice cloning и lip-sync. Он противоречит запрету reference/target-voice
  synthesis и не является audio-only Russian final source.

Следовательно, помимо ToneSpeak research-only source после исключения PhoneSpoof и RHVoice
family подходящего public independent Russian-only **binary/final** source по прежним условиям
не найден. Ничего из этой секции не добавлено в ledger, manifests или runs.

## PhoneSpoof — исключён по решению владельца

[PhoneSpoof](https://sigport.org/documents/phonespoof-new-dataset-spoofing-attack-detection-telephone-channel)
описывает Russian TTS attacks, прошедшие реальный телефонный канал; в paper перечислены Russian
Google, Yandex и STC subsets. Это единственный найденный внешний Russian-focused resource помимо
RuASD/OpenSTT, который потенциально добавляет channel-shift.

Однако footnote paper требует для non-commercial access обращаться к авторам. Public page не
публикует archive URL, immutable revision, size/checksum, licence text, manifest или per-row
generator/voice/reference provenance. Владелец проекта отклонил такой путь: не отправлять запрос,
не скачивать и не добавлять PhoneSpoof в ledger.

## STC Spoofing — исключён: прямое voice-cloning provenance

Старый STC Spoofing database действительно русскоязычный, но paper прямо описывает TTS voices,
построенные на `30` seconds–`3` hours speech каждого target speaker, и доступ выдаётся авторами
для non-commercial use. Это reference-audio/target-voice synthesis, несовместимая с политикой
проекта; не запрашивать и не принимать даже как research source.

## MLAAD Russian subset — заблокирован, не кандидат для intake

[MLAAD](https://huggingface.co/datasets/mueller91/MLAAD) — multilingual (не Russian-only)
synthetic-only release с Russian среди заявленных языков. Он внешне отличается от текущих RuASD
и PyAra, но пока не удовлетворяет обязательным условиям проекта:

- доступ к файлам требует аутентифицированного Hugging Face account и согласия на gated access;
  immutable file index и Russian row metadata не доступны для локальной проверки без этого шага;
- опубликованный объём полного release — около `159 GB`, что превышает правило проекта: наборы
  более `2 GB` не загружаются агентом; пользователь может предоставить локальную копию только
  после отдельного одобрения source;
- licence — `CC-BY-NC-4.0`: официальный dataset card прямо ограничивает его
  non-commercial academic research, поэтому он несовместим с product/commercial целью;
- `meta.csv` содержит `reference_speaker` (поле введено в v8). В актуальном описании MLAAD
  multi-speaker generation использует выбранный baseline audio как speaker reference; следовательно,
  часть строк может быть reference-audio/voice-cloning provenance. Это запрещено текущей политикой
  проекта даже для нового local synthetic source.

Поэтому MLAAD не получает статус «приоритетного кандидата». Теоретически можно рассмотреть лишь
отдельный **research-only** Russian subset, если пользователь законно предоставит metadata-only
export или локальную копию и audit докажет для каждой строки text-only fixed-voice TTS без
reference audio, voice cloning и overlap. До такого доказательства статус — **excluded pending
provenance**, а не `pending intake`.

M-AILABS `ru_RU` сам по себе является bona-fide corpus, а не недостающим spoof source. Связывать
его с MLAAD заранее нельзя: MLAAD распространяет только synthetic audio и рекомендует брать
authentic counterpart отдельно; необходимость binary balance не снимает запрет на неясный
reference-speaker provenance.

## XMAD Russian / M-AILABS — недостаточен

[XMAD-Bench](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/XMAD) — удобный
multilingual package/metadata, но не независимый Russian-only release. Его Russian composition
опирается на Common Voice и M-AILABS; M-AILABS rows опубликованы **spoof-only**, без real class.
Кроме того, пакет имеет `CC-BY-NC-SA-4.0`. Он не создаёт готовый binary RU final layer и не
снимает ни overlap audit, ни проверку generator/reference provenance. Более того, official paper
описывает generation как TTS с последующим voice conversion, использующим voice original speaker
как reference. Не принимать XMAD в ledger или использовать как «замену MLAAD».

## Проверенные исключения

- RuASD и PyAra уже присутствуют в Stage-B train/dev: это не независимый final spoof source для
  данного checkpoint.
- LRLspoof исключён ранее: spoof-only, один sequential archive около `452 GB`, без безопасного
  selective Russian acquisition.
- PolyGlotFake и найденные Kaggle/reupload варианты не подходят: в найденных описаниях есть
  voice-cloning/непроверяемое происхождение или нет official immutable artifact, полной license и
  metadata для overlap audit.

## Следующий безопасный путь

1. ToneSpeak immutable **research-only** OOD candidate уже прошёл completed narrow acoustic
   gate и ровно один hash-pinned exploratory frozen-checkpoint run. Не повторять этот plan,
   не менять candidate и не использовать результат для threshold fitting, calibration, training
   или final claim.
2. До любого final claim получить vetted independent Russian bona-fide counterpart; даже
   completed ToneSpeak acoustic gate не создаёт binary final protocol.
3. Продолжать read-only поиск источника с stronger per-row generator/provenance для
   product-capable binary protocol. Принимать лишь text-only fixed-voice TTS без reference
   audio, voice cloning, voice conversion или имитации конкретного человека.
4. До загрузки любого другого source проверить размер; при размере более `2 GB` остановиться и
   ждать локальную копию пользователя после отдельного одобрения.

Источники: [MLAAD dataset card](https://huggingface.co/datasets/mueller91/MLAAD),
[MLAAD v10 paper](https://arxiv.org/abs/2401.09512),
[M-AILABS source and licence](https://github.com/i-celeste-aurora/m-ailabs-dataset),
[XMAD dataset card](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/XMAD) и
[XMAD paper](https://arxiv.org/abs/2506.00462).
