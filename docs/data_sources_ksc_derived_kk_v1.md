# KSC-derived Kazakh TTS stress source v1

## Назначение и строгая граница

Это **local personal-research test layer**, а не публичный corpus, product dataset,
speaker-independent benchmark или основание для calibration/API score. Он нужен, чтобы
добавить к source-mixed matrix казахский binary stress-test, когда готовый открытый Kazakh
binary corpus отсутствует локально.

В нём bona-fide и spoof clips построены на одних и тех же проверенных KSC transcripts. Это
контролирует lexical content, но не делает слой независимым по speaker, каналу или тексту.
KSC не публикует пригодный speaker ID, а Piper/MMS model card не доказывает legal group
provenance synthetic voices для product use. Поэтому source никогда не переводится в
`product_allowed` и не участвует в калибровке.

## Зафиксированные модели

Все runtime artifacts объявлены в
[`configs/research/ksc_derived_kk_v1_models.json`](../configs/research/ksc_derived_kk_v1_models.json).
Downloader разрешает только эти HTTPS URLs, revision, expected size и SHA-256, скачивает
273 106 664 bytes целиком во временный каталог и публикует bundle только после повторной
проверки. Запуск synthesis снова сверяет каждый local file; model ID нельзя заменить
аргументом CLI.

| TTS family | Пин | Голоса | Права и ограничение |
| --- | --- | ---: | --- |
| Piper neural TTS | `piper-tts 1.6.0`, voice revision `ea046e8458f6acd997706d6e6066a022b42f6fb1` | 6 | Voice card: Kazakh_TTS, CC BY 4.0; Piper runtime GPL-3.0-or-later |
| Meta MMS/VITS | `transformers 5.14.1`, revision `5c8a1d86e6a952f78f9c5b0f5d3090c19d00ad63` | 1 | model card: CC BY-NC 4.0; только personal research |

Piper model card называет шесть `speaker_id`; MMS model card — один Kazakh checkpoint. Это
семь **TTS profiles**, но только две TTS family. Их нельзя выдавать за семь независимых
human speakers или семь независимых generator architecture.

## Protocol

1. Безопасный KSC intake извлекает deterministic `test`-only base slice, сохраняя исходные
   KSC audio, transcript и `text_hash`.
2. `synthesize_ksc_derived_kk.py` сначала проверяет KSC asset SHA-256, затем передаёт text в
   локальные Piper/MMS. Он отказывается от transcript, если canonical UTF-8 text не совпадает
   с manifest hash.
3. Synthetic rows получают собственный `source_name=ksc_derived_kk_v1`, точные model/voice
   fields, локальный device и MMS seed. Они публикуются сначала как raw spoof-only manifest.
4. Обычный audio preprocessing выполняет 16 kHz mono PCM/VAD/QA с отдельным rejection report.
   `build_ksc_derived_kk_test.py` сопоставляет каждый готовый spoof clip ровно с одной KSC
   bona-fide строкой по `text_id` и снова проверяет hash/assets/ledger.

Назначение profiles сбалансировано сначала по двум family, затем по Piper voices. До QA из
`968` готовых KSC base rows назначены `484` MMS + `484` Piper. Это баланс synthetic
profiles, не доказательство real-world prevalence TTS.

## Опубликованный срез и QA

Результат намеренно сохраняет и входные, и промежуточные manifests, поэтому каждое
исключение можно проверить, а не принять на веру.

| Этап | Bona-fide KSC | Synthetic spoof | Что произошло |
| --- | ---: | ---: | --- |
| test-only raw intake | 1 000 | — | Детерминированный исходный slice из KSC `test` |
| первый preprocessing | 889 | — | 32 реальных QA/VAD-rejection и 79 collision с уже существующими неизменяемыми WAV |
| безопасный merge | 968 | — | Добавлены только 79 byte-identical уже проверенных rows; ничего не перезаписывалось |
| raw synthesis | — | 968 | 484 MMS + 484 Piper profiles |
| synthetic QA и merge | — | 921 | 47 реальных QA/VAD-rejection; один exact ранее проверенный output безопасно переиспользован |
| финальный paired test | 921 | 921 | Ровно одна bona-fide и одна spoof строка на каждый `text_id` |

Финальная spoof-часть: `475` MMS/VITS и `446` Piper. В Piper по голосам: F1 `70`, F2 `73`,
F3 `71`, M2 `75`, Raya `77`, Iseke `80`. Rejection JSON и raw/ready manifests находятся в
`data/manifests/ksc_derived_kk_v1_*`; audio и модели намеренно не versioned в Git.

`921` примера каждого класса достаточны, чтобы измерить этот **конкретный** контролируемый
слой: даже около 50% 95%-интервал Wilson имеет полу-ширину примерно 3.2 процентных пункта.
Этого недостаточно для общего вывода о казахской речи или для продукта: здесь только две
generator family, парные тексты и отсутствуют проверяемые speaker groups KSC.

## Воспроизводимый запуск

```bash
# Пин models уже проверен до этой команды; download <= 2 GB.
uv run --extra ml --extra synthesis python scripts/download_research_tts_models.py \
  --model-lock configs/research/ksc_derived_kk_v1_models.json \
  --model-root models/research/ksc_derived_kk_v1

# Создать KSC test-only base, затем normalise его обычным preprocessing script.
uv run python scripts/ingest_ksc_slr102.py \
  --archive /path/to/ISSAI_KSC_335RS_v1.1_flac.tar.gz --data-root data \
  --output-manifest data/manifests/ksc_derived_kk_v1_base_raw_1000.csv \
  --slice-name derived-v1-base-1000 --source-splits test --limit-per-split 1000 \
  --seed 20260810 --created-at 2026-08-10T00:00:00Z

# После QA не перезаписывать collision: безопасно объединить только exact уже проверенные WAV.
uv run python scripts/merge_ksc_derived_kk_base_ready.py \
  --raw-manifest data/manifests/ksc_derived_kk_v1_base_raw_1000.csv \
  --new-ready-manifest data/manifests/ksc_derived_kk_v1_base_ready_1000.csv \
  --reusable-ready-manifest data/manifests/ksc_first_250_ready.csv \
  --preprocess-rejections data/manifests/ksc_derived_kk_v1_base_rejections_1000.json \
  --output-manifest data/manifests/ksc_derived_kk_v1_base_ready_968.csv \
  --output-rejections data/manifests/ksc_derived_kk_v1_base_rejections_968.json \
  --data-root data --license-ledger data/licenses/license_ledger.csv

uv run --extra ml --extra synthesis python scripts/synthesize_ksc_derived_kk.py \
  --base-manifest data/manifests/ksc_derived_kk_v1_base_ready_968.csv \
  --transcript-root data/raw/ksc_slr102/slices/derived-v1-base-1000 \
  --model-lock configs/research/ksc_derived_kk_v1_models.json \
  --model-root models/research/ksc_derived_kk_v1 \
  --license-ledger data/licenses/license_ledger.csv --data-root data \
  --output-manifest data/manifests/ksc_derived_kk_v1_spoof_raw_968.csv \
  --slice-name derived-v1-spoof-968 --limit 968 --seed 20260810 \
  --created-at 2026-08-10T00:00:00Z --mms-device cuda
```

Do not run this command against a source that lacks paired KSC transcripts, an approved ledger
entry, all model hashes or write-new output paths. The scripts explicitly reject these cases.

После preprocessing synthetic manifest объединяется тем же
`merge_ksc_derived_kk_base_ready.py` (с `--source-name ksc_derived_kk_v1 --label spoof
--source-description 'Derived Kazakh TTS spoof'`), а окончательный binary manifest создаётся
только `scripts/build_ksc_derived_kk_test.py`. Общий `kds validate-training-protocol` нельзя
применять к этому test-only manifest: он справедливо требует train и dev. Для этого слоя
правильная проверка — `kds validate-source-matrix configs/research/source_mixed_v2_kk.json`.

## Как расширять и создавать данные самостоятельно

Для следующего независимого слоя практичная исследовательская цель — 3–5 **архитектурно
разных** generator family, по 300–500 прошедших QA spoof clips на family, с равноценными
bona-fide clips в тех же условиях. Это даёт 1 500–2 500 clips каждого класса при пяти
family. Для обучения, а не только оценки, полезнее стремиться к 5 000+ каждого класса,
распределив их по устройствам, каналам и условиям записи. Это проектная рекомендация, а не
универсальный норматив.

Дальнейшее расширение выполняется без записи голосов людей:

1. Bona-fide брать только из готовых RU/KK/mixed releases после проверки лицензии, provenance,
   archive hash и доступных speaker/device groups. Не подменять неизвестный speaker ID.
2. Для собственной synthetic части использовать только text-to-speech с встроенными публичными
   model voices; reference audio, voice cloning и имитация конкретного человека исключены.
   Тексты и source assets разделить по train/dev/test **до** генерации.
3. Для каждой независимой TTS family записать exact model revision, checkpoint hash,
   runtime, voice ID, параметры, seed и разрешённую лицензию. Не считать несколько голосов
   одного checkpoint независимыми family.
4. Прогнать одинаковые decode, 16 kHz mono, VAD и QA; публиковать raw manifest, ready
   manifest и полный rejection report. Проверять SHA-256 assets, отсутствие symlink escape,
   дубликатов и text/group leakage.
5. Заморозить test и отдельный unseen-generator OOD до обучения. Не подбирать порог,
   preprocessing или checkpoint по итогам test/OOD, а отчитываться раздельно по class,
   family, voice и условиям с confidence intervals.

Пока эти условия не выполнены, derived source остаётся personal-research stress test, а не
dataset для выдачи risk score или product calibration.
