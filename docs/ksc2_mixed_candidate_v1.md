# KSC2 mixed bona-fide candidate v1

**Статус:** опубликован 11 августа 2026 года как narrow research candidate. Это не binary
training/final-test layer и не разрешение выполнять detector inference.

## Подготовка bona-fide стороны

Вход — hash-pinned 32-row single-AI transcript-review evidence
[`ksc2_test_mixed_ai_review_v1.csv`](../data/manifests/ksc2_test_mixed_ai_review_v1.csv).
`scripts/build_ksc2_mixed_bonafide_candidate.py` проверил SHA-256 всех исходных FLAC,
прочитал их metadata, нормализовал аудио до WAV и применил действующий QA/VAD contract.

| Artifact | Rows | SHA-256 |
| --- | ---: | --- |
| `data/manifests/ksc2_test_mixed_ai_review_v1_raw.csv` | 32 | `5bcdcf1a9ea57a3d533154734367fdaf871225e6e540b01230fc522a45aab8fe` |
| `data/manifests/ksc2_test_mixed_ai_review_v1_ready.csv` | 31 | `e3a881ba8b0d0c4e26dccaefce17ad7cd2fcd38c3890ec265047d4b045055770` |
| `data/manifests/ksc2_test_mixed_ai_review_v1_rejections.json` | 1 rejection | `b897b51d410cdfaaf0e57b0be1fe8cce37de3691b82e61af715b682d2f8f0f09` |

`ksc2_v1:Test/podcasts/09_01_296` не попала в ready layer: VAD обнаружил недостаточную
длительность речи. Это не изменение её linguistic evidence и не отрицательная language label;
исходная запись и причина исключения сохранены в rejection receipt. Все 31 ready WAV прошли
`validate-manifest`, license-ledger validation и SHA-256 asset validation.

Каждая ready row имеет `label=bonafide`, `language=mixed`, `code_switch=true` и явное
provenance до review CSV, KSC2 archive и source lock. `speaker_pseudo_id=ksc2_v1:unknown`,
поэтому слой не заявляет speaker-disjointness.

## Silero V4 technical smoke test

Чтобы не генерировать сразу весь spoof слой, `scripts/smoke_ksc2_mixed_silero_v4.py` выбрал
детерминированно 5 reviews (`Test/podcasts`, `Test/radio`, `Test/talkshow`) и сгенерировал по
одному WAV fixed `b_ru` и `kz_M1` profile. Все 10 outputs являются непустыми 48 kHz WAV и
прошли текущий signal QA/VAD (`technical_status=ready`). Полный список input/output SHA-256,
durations, profiles и технических статусов —
`data/manifests/ksc2_mixed_silero_v4_smoke_v1_report.json`, SHA-256
`3b9660e4443678748f5052e7e055e3ca32cd48fd96a41898b040028f9f912e53`.

Это только технический результат. Signal QA/VAD не измеряет intelligibility и не доказывает,
что fixed RU или KK profile корректно произнёс каждый русский и казахский segment. Поэтому
smoke WAV не имеют spoof label, не имеют manifest и не используются как training/test assets.

## Input-pinned Silero candidate

После технического smoke-test создан ограниченный text-derived spoof candidate через один fixed
`kz_M1` profile без reference audio/voice cloning. Отдельный source
`ksc2_mixed_v1_silero_v4` добавлен в license ledger: он закрепляет тот же проверенный Silero
model lock, но сохраняет CC-BY-NC-SA research-only ограничение.

| Artifact | Rows | SHA-256 |
| --- | ---: | --- |
| `data/manifests/ksc2_mixed_v1_silero_v4_raw.csv` | 31 raw spoof | `a42e8206f6c7d1deaf6da510fe242edd809f05d43d9cac69f49f74001f3ad742` |
| `data/manifests/ksc2_mixed_v1_silero_v4_ready.csv` | 30 ready spoof | `6d0f72dfa05418608378d7e5d1f5c3e58b322213253bd89af454f4de40ee33eb` |
| `data/manifests/ksc2_mixed_v1_silero_v4_text_rejections.json` | 0 | `f9352a5e72790162119e6e39940db9e859bad2547976034cf18b0b0b6bb1678f` |
| `data/manifests/ksc2_mixed_v1_silero_v4_audio_rejections.json` | 1 | `88d02822fa1a805a00ea080078fc1440ca075047615c8e483676e93eeeb9b37c` |
| `data/manifests/ksc2_mixed_v1_silero_v4_candidate_30.csv` | 30 exact pairs / 60 rows | `dafa33d424da8efab19d849afeeeb11c279d2e6c10ff187f3a57cada76c6c4a8` |
| `data/licenses/ksc2_mixed_v1_silero_v4_pair_lock.json` | 30 pair-to-evidence mappings | `9e46e56ebc082620f9317cf41e93783a2d2aab691306bb9bf45b309954f5b7bf` |

Все 31 input transcripts прошли strict Silero text contract. Один generated WAV, соответствующий
KSC2 review row `09_03_368`, исключён VAD как `insufficient_speech`; в candidate остались только
30 одинаковых `text_hash` bona-fide/spoof pairs. Manifest и asset validation прошли для raw,
ready и paired artifacts.

Pair-lock дополнительно хранит для каждой пары `annotation_id`, component, input evidence tokens
обоих языков и SHA-256 обоих WAV; он проверяет, что candidate ровно совпадает с ready pairs после
всех accounted rejections. Поэтому связь с исходной explicit transcript evidence не теряется при
переходе к synthetic asset.

`language=mixed`, `code_switch=true` у spoof rows означает provenance намеренно поданного
подтверждённого mixed transcript. Поле `augmentation_chain` прямо содержит
`language_provenance=intended_input_text_only`. Это **не** acoustic language-preservation
certificate: waveform QA/VAD подтверждает лишь техническую пригодность сигнала.

## Следующая граница

Нужен явный language-preservation quality gate для смешанного synthetic текста. Он должен
проверять содержание каждого RU/KK segment и публиковать все rejections; component path,
waveform quality или успешный запуск TTS его заменить не могут.

После создания candidate был разрешён ровно один изолированный exploratory inference уже frozen
XLS-R+SLS Stage-B checkpoint, с отдельным hash-pinned plan, execution lock и без training,
calibration или выбора threshold. Полный результат на всех 30 парах находится в
[`research_xlsr_sls_stage_b_ksc2_mixed_exploratory_30.md`](research_xlsr_sls_stage_b_ksc2_mixed_exploratory_30.md).
Он не меняет статус candidate: `candidate_30` всё ещё запрещён для frozen final test,
calibration, API/product score и любого quality claim. Нынешние frozen assets и Stage-B
calibration не меняются.
