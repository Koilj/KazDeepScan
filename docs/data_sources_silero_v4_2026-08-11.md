# Silero V4 Cyrillic — fixed-profile FLEURS synthesis receipt

**Статус:** 11 августа 2026 года создана и проверена одна новая text-only spoof
family для уже замороженных FLEURS RU/KK bona-fide candidates. Это personal-research
candidate material; не общий final result и не product asset.

## Источник, pin и scope

- Repository: [snakers4/silero-models, commit
  `d9355348e2781dc8fa25a135d1602c530afae24c`](https://github.com/snakers4/silero-models/tree/d9355348e2781dc8fa25a135d1602c530afae24c).
- Hosted package: `v4_cyrillic.pt`, `35 431 527` bytes, SHA-256
  `5e3862319e13883ea105cd4db835273c7febde62ff82d98d1ccf596607f8673f`.
- Pinned source archive: `789 187` bytes, SHA-256
  `cca6d3e6e34e03f9fe30c4e33ee2de8e89aa384f95bc0f3143c51af7a72765aa`.
- Lock: `configs/research/silero_v4_cyrillic_v1_models.json`, SHA-256
  `b1cd921e743c24fb5be6129973c10b35cc0b728e6e37bf061d10328bdcdfc629`.
- Лицензия модели и source archive: `CC-BY-NC-SA-4.0`; следовательно, все
  производные clips и final candidates остаются только personal research.

Статическая проверка package включала размер/hash, ZIP CRC, safe member paths/types
и ограничение распакованного объёма. До runtime проверен исходный wrapper и dispatcher;
реальный package включает FastPitch acoustic model и HiFi-GAN vocoder. Это новая
acoustic family относительно используемых ранее TTS routes, но HiFi-GAN vocoder
является общим архитектурным компонентом с KazEmoTTS. Этот факт не скрыт и не даёт
оснований считать оба набора полностью независимыми по каждой внутренней компоненте.

## Безопасная voice policy

Использованы только фиксированные профили, встроенные в модель: `b_ru` для Russian
и `kz_M1`, `kz_M2`, `kz_F1`, `kz_F2`, `kz_F3` для Kazakh. Adapter и CLI не принимают
reference audio или voice path. В частности, route `random`, который загружает внешний
voice tensor через `voice_path`, заблокирован до исполнения. Не выполнялись voice cloning,
имитация конкретного человека или загрузка пользовательского голоса.

## Входы и честное отсеивание текста

Перед synthesis каждый FLEURS row повторно привязан к full audited release по
`sample_id`, `text_id` и `text_hash`. Текст не перефразируется: разрешена только
минимальная нормализация известной пунктуации; unsupported lexical characters вызывают
explicit rejection. Это сохранило provenance и не превратило изъятый digit/Latin/mixed
fragment в иной текст.

| Этап | RU | KK | Всего |
| --- | ---: | ---: | ---: |
| Ready FLEURS base | 289 | 212 | 501 |
| Text rejection до synthesis | 75 | 60 | 135 |
| Silero raw spoof | 214 | 152 | 366 |
| Audio QA/VAD rejection | 0 | 0 | 0 |
| Silero ready spoof | 214 | 152 | 366 |

Text-rejection report:
`data/manifests/fleurs_ru_kk_v1_silero_v4_test_text_rejections.json`, SHA-256
`9aca2d636555337c3b1c448456e5631f5caefeedf469dd0aba0461e5b9962909`.
Audio-rejection report:
`data/manifests/fleurs_ru_kk_v1_silero_v4_test_audio_rejections.json`, SHA-256
`3a924ab56c7d858ce3beb90f517cf9f84a852386912599388d8004b56686321b`.

Raw manifest has SHA-256
`39fd20db4b65703a0a76448e72470153a2d1542218ba76c5c236ffcf2b6db15d`; ready manifest
has SHA-256 `04a84895241944017310ae7b74ea2d16812c30f17f2b1027ea85e045ca6ed7b2`.

## Paired candidate publication

`scripts/build_fleurs_silero_v4_final.py` accepts a language only after accounting
for every base/raw exclusion. It proves exact text identity within each pair and refuses
any `sample_id`, asset SHA-256 or `text_hash` collision with supplied historical
manifests. The real build excluded Stage-B train/dev, calibration dev and all three
previous frozen Kazakh final families.

| Candidate manifest | Pairs | Assets | SHA-256 |
| --- | ---: | ---: | --- |
| `fleurs_ru_v1_silero_v4_test_214.csv` | 214 | 428 | `8afd2d461495eb4b16c6e4de89ea731536d41ff73d7ea7152bfb1ea20e23ec1b` |
| `fleurs_kk_v1_silero_v4_test_152.csv` | 152 | 304 | `e23f1ce80dbc866eafd4fe1f488f2f55c5705bef1bd9b12e33ce80632328dac2` |

`kds validate-manifest --license-ledger` and `kds validate-assets --audio-root data`
passed for both manifests and every listed asset. These layers are deliberately not
fed to XLS-R Stage B: the required bona-fide/spoof `mixed` layer is still absent, so
temperature fit, threshold selection, detector inference and API score remain forbidden.
