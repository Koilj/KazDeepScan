# Подготовка final и calibration для XLS-R+SLS Stage B

**Статус на 11 августа 2026:** calibration dev подготовлен и заморожен. Новый final
RU/KK/mixed layer **ещё не создан**, поэтому final inference, temperature fit, threshold
selection и API score запрещены.

Единственный проведённый запуск на 30 KSC2/Silero mixed pairs — отдельный
[exploratory stress-test](research_xlsr_sls_stage_b_ksc2_mixed_exploratory_30.md), а не
исключение из final protocol. У synthetic WAV есть лишь input-transcript provenance, acoustic
language-preservation gate не пройден; этот результат не может использоваться для temperature,
threshold, model selection или API.

## Отдельный calibration dev

Fresh PyAra Stage-B dev уже выбирал epoch 7 и не может использоваться для temperature scaling.
Для калибровки подготовлен новый детерминированный PyAra v7 slice с seed `20260821`:

| Артефакт | Строк | SHA-256 |
| --- | ---: | --- |
| Raw manifest | 1 000 (500/500) | `3de5f1187c25bc84947e138d69ce2949e742a4ed86d36d9cd00e70112f8e9d07` |
| QA/VAD candidate | 978 | `a2386bae340d198dc3b64d7b64bbc1090c4ad856022029b7ada35f709c7129ad` |
| Frozen calibration dev | 977 (478 bona-fide / 499 spoof) | `3704f4da82ff02066260fbbd472952771bf721b3890f9f8bfac71dbd403fa879` |
| QA/VAD rejection report | 22 rows | `3e23546988aa078e2c4cc797db18c49a93d7b52e5befcc758dd4a9b4539c12aa` |
| Stage-B exclusion receipt | 1 row | `d8f60f3e34b9e8133002fc26047727b304ddcf3ed6cb9aa27f553a72df9d2c4d` |

Raw selection excluded every sample ID and text hash in both earlier PyAra manifests:
`pyara_ru_v7_research_500.csv` and `pyara_ru_v7_fresh_dev_1000.csv`.
After QA/VAD, `build_xlsr_stage_b_calibration_dev.py` additionally checked all ordinary
leakage keys (`sample_id`, asset SHA-256, parent group, speaker pseudo-ID and text hash) against:

- full ready RuASD research manifest;
- old Stage-A PyAra manifest;
- chosen Stage-B PyAra epoch-selection dev.

One candidate (`pyara_ru_v7:alg_2_398`) shared a text hash with historical data and was
excluded. The published manifest passed schema/license validation and independent SHA-256 asset
validation for all `977` files.

The manifest is a **calibration-only input**. It is not used to retrain Stage B, select epoch,
choose a threshold or inspect a final result. As with every PyAra slice, absence of source
speaker IDs means this is record/text-disjoint under available metadata, not a
speaker-independent calibration guarantee.

## Required final contract

The final run plan may be created only after all three previously unobserved binary layers exist:

| Layer | Required language | Required evidence before pinning |
| --- | --- | --- |
| `ru` | Russian | bona-fide and spoof; no train/dev source overlap or ordinary key overlap |
| `kk` | Kazakh | bona-fide and spoof; new assets and a generator family absent from Stage-B train/dev |
| `mixed` | Kazakh–Russian code-switching | bona-fide and spoof in each record; `language=mixed`, `code_switch=true` |

For every layer, publish raw manifest, ready manifest, QA/VAD rejection report, asset hashes and
license-ledger entry before writing the final-run plan. The plan must pin its own JSON bytes,
Stage-B checkpoint/report, calibration manifest, ledger and every final manifest. It must reject
an existing output path, fit `TemperatureScaler` only on the frozen calibration dev, and then
read each final asset exactly once. Metrics must remain separate by language and generator
family; there is no pooled product "accuracy".

Until the three rows above are concrete and hash-pinned, a template with placeholder hashes is
not a frozen plan and must not be executed.

## KSC2 source audit

KSC2 was manually supplied and completed the required read-only multipart audit. The
[official ISSAI corpus page](https://issai.nu.edu.kz/kz-speech-corpus/) states CC-BY-4.0,
while the [pinned Hugging Face dataset card](https://huggingface.co/datasets/issai/Kazakh_Speech_Corpus_2/tree/cececbec1049f93f34a7421552500da01971ead8)
still displays `mit`; the project records the licensor's primary CC-BY-4.0 statement and retains
the discrepancy. The complete integrity receipt and component policy are in
`docs/data_sources_ksc2_2026-08-10.md`.

The ten parts (`80 809 122 212` bytes) passed streaming SHA-256, gzip CRC and safe TAR validation
without concatenating or extracting them. The archive has `645 860` FLAC files; each has a paired
transcript, with one extra `Train/radio` transcript excluded from any future selection. To prevent
legacy overlap, every `crowdsourced` and `tts` component is reserved/excluded. Test
`parliament`, `podcasts`, `radio`, `talkshow` and `tv_news` provide `6 023` paired candidates
before text-overlap and QA/VAD filtering.

KSC2 remains bona-fide only. A повторная проверка статьи уточнила приоритет: её Table 1
указывает test code-switch rates `8.6%` для `podcasts`, `6.3%` для television programs
(`talkshow` в archive) и `3.1%` для `radio`; `tv_news` — отдельный component с `1.3%`.
Авторы определяли rate по словам обоих языков, включая intra-word cases. Но archive и
official recipe не публикуют этот результат для отдельных записей. Component-level rate не
оправдывает setting `language=mixed` или `code_switch=true` on individual samples. На основе
hash-pinned packet из `Test/podcasts`, `Test/talkshow`, `Test/radio` выполнен narrow single-AI
semantic transcript review: опубликовано 32 positive rows с explicit Russian/Kazakh token
evidence, а остальные 2 600 candidates намеренно остаются `unknown`. Этот output не является
KDS manifest и не может быть передан detector-у: до любого final inference по-прежнему нужны
QA/VAD, overlap audit, независимая spoof-половина и отдельный frozen binary protocol.
