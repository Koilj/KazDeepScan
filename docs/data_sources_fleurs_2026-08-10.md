# Google FLEURS RU/KK — intake и QA

**Статус:** source audit, raw intake, QA/VAD и paired Silero V4 candidate layers
завершены 10–11 августа 2026 года. FLEURS — независимый CC-BY-4.0 bona-fide
источник для personal research, но сам по себе не binary benchmark и не основание
запускать общий Stage-B final inference.

## Источник и integrity

Pinned release: `google/fleurs` revision
`4683b04af03d2d9549064c7d72060a9a94bb6046`. Для `ru_ru` и `kk_kz` локально
проверены все три TAR archive и три TSV: exact byte size, audio LFS SHA-256,
TSV Git blob SHA-1, gzip CRC и exact TAR-to-TSV membership.

| Locale | Rows train/dev/test | Unique normalized test texts | Source lock |
| --- | --- | ---: | --- |
| `ru_ru` | 2 562 / 356 / 775 | 344 | `fleurs_ru_ru_v1_artifact_lock.json` |
| `kk_kz` | 3 200 / 369 / 856 | 349 | `fleurs_kk_kz_v1_artifact_lock.json` |

FLEURS TSV не использует CSV quoting: `"` может быть literal character в
transcript. Intake использует strict TSV parser без RFC-4180 интерпретации
кавычек. Публичного speaker ID нет; gender и prompt ID не используются как
speaker surrogate. Prompt ID применяется только как text-group provenance.

## Published bona-fide candidates

Из `test` deterministic seed `20260824` выбрал по одному recording на каждый
из 300 text groups. До selection были исключены `sample_id` и `text_hash` всех
исторических project manifests. Raw assets извлечены atomically в ignored
`data/raw/`; raw manifest, ready manifest и rejection report не перезаписывают
существующие outputs.

| Layer | Raw | Ready | QA/VAD rejection | Ready SHA-256 |
| --- | ---: | ---: | ---: | --- |
| RU | 300 | 289 | 11 `signal_too_quiet` | `1840c29796289acf5faafa36bb14a8117ea2b977ff5228d700e68ac7b28da5ba` |
| KK | 300 | 212 | 88 `signal_too_quiet` | `ca7dbe5cbb577a18376e68b2491f5433e761c4837603fc61988792a2be5cdd29` |

Все `289` RU и `212` KK ready assets прошли manifest/license/asset SHA-256
validation. Слабые KK recordings не удалялись и не заменялись скрыто: они
зафиксированы в отдельном rejection report. Поэтому текущий KK base layer не
дотягивается до первоначальной цели 300 clips. Базовый selection был заморожен до
synthesis; совместимость текста с новым generator проверяется отдельно и полностью
учитывается ниже.

## Paired Silero V4 RU/KK candidates

К неизменяемым FLEURS ready rows применена одна новая text-only family: Silero V4
Cyrillic FastPitch + HiFi-GAN с fixed profiles. Пакет и исходный repository закреплены
в `configs/research/silero_v4_cyrillic_v1_models.json`; доступен только personal-research
scope из-за `CC-BY-NC-SA-4.0` модели. Runtime допускает только встроенные `b_ru` для
RU и `kz_M1`/`kz_M2`/`kz_F1`/`kz_F2`/`kz_F3` для KK. `random`, external `voice_path`,
reference audio и voice cloning отсутствуют из CLI и запрещены adapter-ом.

Неподдерживаемые в модели digits/Latin/mixed notation не удаляются молча: до synthesis
записаны 135 explicit text rejections (75 RU, 60 KK). Synthesis сформировал 366 raw WAV;
общий QA/VAD принял все 366, audio rejection report пуст. Final builder доказал, что
каждая исходная строка исключена либо по text report, либо образует одну hash-identical
по тексту bona-fide/spoof пару, и отклонил пересечения с Stage-B train/dev, calibration и
прежними frozen finals по `sample_id`, asset SHA-256 и `text_hash`.

| Layer | Пары | Все assets | Manifest SHA-256 |
| --- | ---: | ---: | --- |
| RU FLEURS + Silero V4 | 214 | 428 | `8afd2d461495eb4b16c6e4de89ea731536d41ff73d7ea7152bfb1ea20e23ec1b` |
| KK FLEURS + Silero V4 | 152 | 304 | `e23f1ce80dbc866eafd4fe1f488f2f55c5705bef1bd9b12e33ce80632328dac2` |

Raw and ready spoof manifests have SHA-256
`39fd20db4b65703a0a76448e72470153a2d1542218ba76c5c236ffcf2b6db15d` and
`04a84895241944017310ae7b74ea2d16812c30f17f2b1027ea85e045ca6ed7b2` respectively.
Полный model/runtime receipt и воспроизводимые команды находятся в
`docs/data_sources_silero_v4_2026-08-11.md`.

## Ограничения

- FLEURS предоставляет bona-fide read speech; созданные Silero V4 pairs — отдельные
  candidate layers, но не общий final result.
- Source не имеет public speaker ID: no sample/text leakage доказан, но
  speaker-independence не заявляется.
- RU и KK layers не являются `mixed`: `code_switch=false`; поэтому они не закрывают
  обязательный mixed row Stage-B final contract, а calibration/final inference ещё запрещены.
