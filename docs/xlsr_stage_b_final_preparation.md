# Подготовка final и calibration для XLS-R+SLS Stage B

**Статус на 12 августа 2026:** новый disjoint calibration v3 и строгий confirmatory
RU/KK/mixed research plan для Stage B v2 подготовлены. Его единственный GPU run выполнен;
temperature fit и раздельные results закреплены в
[`research_xlsr_sls_stage_b_v2_research_final_v1.md`](research_xlsr_sls_stage_b_v2_research_final_v1.md).
API/product score остаётся запрещён.

Единственный проведённый запуск на 30 KSC2/Silero mixed pairs — отдельный
[exploratory stress-test](research_xlsr_sls_stage_b_ksc2_mixed_exploratory_30.md), а не
исключение из final protocol. Позднее те же 30 pairs прошли narrow two-review acoustic gate,
но v1 run уже раскрыл их detector results. Поэтому в v2 они помечены как previously exposed
confirmatory layer, а не blind project-level final.

## Отдельный calibration dev

Stage-B epoch-selection dev нельзя повторно использовать для temperature scaling. Исходный
детерминированный PyAra calibration candidate дополнительно отфильтрован против RuASD v2 и
нового Stage-B dev v3:

| Артефакт | Строк | SHA-256 |
| --- | ---: | --- |
| Frozen calibration dev v3 | 976 (478 bona-fide / 498 spoof) | `7fe2c89be7f02eea1abdfc369fc9c16185b0a395fa2526b5b2c58e361fa8fa31` |
| Stage-B v2 exclusion receipt | 1 text-overlap row | `dd1144eb1df6d72686cea86d438c0a9f4fd1e6f5b36624b687b14ebdea06d31b` |

Raw selection excluded every sample ID and text hash in both earlier PyAra manifests:
`pyara_ru_v7_research_500.csv` and `pyara_ru_v7_fresh_dev_1000.csv`.
After QA/VAD, `build_xlsr_stage_b_calibration_dev.py` additionally checked all ordinary
leakage keys (`sample_id`, asset SHA-256, parent group, speaker pseudo-ID and text hash) against:

- full ready RuASD research manifest;
- old Stage-A PyAra manifest;
- chosen Stage-B PyAra epoch-selection dev.

V2 сначала исключил старый overlap, а v3 удалил ещё `pyara_ru_v7:alg_5_34535`, чей text hash
появился в новых v2-ролях. Published manifest прошёл schema/license и asset validation для всех
`976` files.

The manifest is a **calibration-only input**. It is not used to retrain Stage B, select epoch,
choose a threshold or inspect a final result. As with every PyAra slice, absence of source
speaker IDs means this is record/text-disjoint under available metadata, not a
speaker-independent calibration guarantee.

## Выполненный confirmatory contract

Plan использовал три binary слоя с явно разным evidence status:

| Layer | Required language | Required evidence before pinning |
| --- | --- | --- |
| `ru` | Russian | FLEURS/eSpeak, 75 pairs; two-review acoustic gate; ранее не inferred |
| `kk` | Kazakh | FLEURS/Silero, 152 pairs; ранее не inferred, acoustic review отсутствует |
| `mixed` | Kazakh–Russian code-switching | KSC2/Silero, 30 pairs; gate пройден, assets раскрыты v1 run |

Plan закрепил checkpoint/report, immutable ledger snapshot, manifests/evidence/implementation и
write-once outputs. Он fitted `TemperatureScaler` только на calibration v3, не подбирал threshold
и прочитал каждый final asset один раз. Метрики остаются раздельными; pooled product accuracy
отсутствует. KK acoustic gate и новый blind mixed layer — задачи будущего более сильного
протокола, а не скрытые предположения текущего результата.

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
