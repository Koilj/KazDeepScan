# KazEmoTTS: третий независимый Kazakh generator family

## Решение

**KazEmoTTS (Grad-TTS + HiFi-GAN) принят только для personal research** как третья
generator family после Piper neural TTS и Meta MMS/VITS. Это не новый Piper speaker и не
другая настройка VITS: acoustic generator основан на diffusion Grad-TTS, а waveform создаёт
отдельный HiFi-GAN vocoder. Сравнение family относится к цепочке генерации, а не к числу
голосов или людей.

Авторы из ISSAI описывают `54 760` казахских audio-text pairs от трёх narrators в шести
эмоциональных состояниях и прямо указывают CC BY 4.0 для датасета и модели. Ни dataset, ни
оригинальные voice recordings не добавляются в этот проект; используется только официально
опубликованный checkpoint без local voice cloning. Источники: [статья KazEmoTTS](https://arxiv.org/abs/2404.01033),
[official source repository](https://github.com/IS2AI/KazEmoTTS) и [official checkpoint
links](https://github.com/IS2AI/KazEmoTTS/tree/0db250b2ebd95c0de7f8fb7ccba9fd65d4815115).

## Зафиксированные bytes и проверка

Lock [`kazemotts_kk_v1_models.json`](../configs/research/kazemotts_kk_v1_models.json)
фиксирует revision исходников, URL, размер и SHA-256 каждого скачиваемого файла. Суммарный
download — `248 377 435` bytes, поэтому он разрешён проектным лимитом 2 GiB.

| Artifact | Size | SHA-256 | Назначение |
| --- | ---: | --- | --- |
| `source.tar.gz` | 4 439 716 | `2b0e…d4bf4` | GitHub source revision `0db250b…e5115` |
| `pt_10000.zip` | 192 062 769 | `f157…43ae` | Grad-TTS checkpoint |
| `pre_trained_hf.zip` | 51 874 950 | `74e9…5431` | HiFi-GAN checkpoint |

До включения в ledger и synthesis были выполнены: archive SHA-256, ZIP CRC обоих archives,
allowlist extraction ровно нужных regular members, SHA-256 извлечённых checkpoint (`209 375 007`
и `55 824 433` bytes) и `torch.load(..., weights_only=True)`. Последний пункт существенен:
обычный legacy pickle load запрещён.

У upstream source есть extension для training alignment, рассчитанное на Python 3.9 и
`distutils`. В inference Grad-TTS alignment не вызывается. Поэтому runtime не собирает и не
использует upstream бинарник: он извлекает в temporary directory только exact allowlist source
files и создаёт fail-closed shim, который немедленно остановит случайную training-path. Это
сохраняет compatibility с Python 3.13 без выполнения непроверенного binary artifact.

GPU smoke test на RTX 5060 Ti с pinned bytes создал не пустой mono PCM S16LE WAV: `22 050` Hz,
`2.496` s. Это проверка runtime, а не оценка качества речи или детектора.

## Профили и границы

Runtime фиксирует 18 controls: `M1`/`F1`/`M2` × `angry`, `surprise`, `fear`, `happy`,
`neutral`, `sad`. Они равномерно назначаются выбранным KSC texts, а seed записывается в
manifest. Все 18 controls — **одна** `gradtts_hifigan_emotional_tts` family. Имена `M1`/`F1`/`M2`
не используются как доказательство identity или verified voice groups, и не дают
speaker-independent protocol.

Новый corpus должен брать свежий KSC `test` subset, не пересекающий frozen v1 text/sample
selection. Сначала публикуется только spoof raw manifest, затем обычный decode/QA/VAD и
paired binary test. Его результаты нельзя объединять с matrix v2, подбирать по нему epoch,
threshold или calibration.

## Фактический frozen v2 набор

Для нового source были заранее отобраны `450` KSC `test` записей, text/sample-disjoint с
frozen v1. После обычного decode/QA/VAD готовы `402` bona-fide WAV. KazEmoTTS синтезировал
по одному clip на каждый из них; тот же QA/VAD пропустил `359`, а `43` клипа записаны с
причинами в [`ksc_derived_kk_v2_kazemotts_rejections_402.json`](../data/manifests/ksc_derived_kk_v2_kazemotts_rejections_402.json).
Финальный manifest [`ksc_derived_kk_v2_kazemotts_test_359.csv`](../data/manifests/ksc_derived_kk_v2_kazemotts_test_359.csv)
содержит `359` строгих KSC/KazEmoTTS пар (`718` assets). Все `718` файлов, права в ledger и
pair text hashes прошли validation. Это frozen final test для matrix v3, не validation set.

## Воспроизведение

```bash
# Downloads exact source/checkpoint archives atomically; the bundle must not already exist.
uv run python scripts/download_research_tts_models.py \
  --model-lock configs/research/kazemotts_kk_v1_models.json \
  --model-root models/research/kazemotts_kk_v1

# Requires only local KSC audio/transcripts and the verified bundle.  It does not send text or
# audio to a provider and never accepts reference audio for cloning.
uv run --extra ml --extra kazemotts python scripts/synthesize_ksc_kazemotts.py \
  --base-manifest data/manifests/ksc_derived_kk_v2_base_ready_450.csv \
  --transcript-root data/raw/ksc_slr102/slices/derived-v2-base-450 \
  --model-lock configs/research/kazemotts_kk_v1_models.json \
  --model-root models/research/kazemotts_kk_v1 \
  --license-ledger data/licenses/license_ledger.csv --data-root data \
  --output-manifest data/manifests/ksc_derived_kk_v2_kazemotts_raw_402.csv \
  --slice-name kazemotts-402 --limit 402 --seed 20260813 \
  --created-at 2026-08-10T00:00:00Z --device cuda
```

`source_name=ksc_derived_kk_v2_kazemotts` становится eligible только после успешной локальной
проверки lock; policy остаётся `research_only`. Ни один generated WAV, model archive или
transcript не versioned в Git.
