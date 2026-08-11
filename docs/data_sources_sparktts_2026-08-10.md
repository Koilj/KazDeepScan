# Spark-TTS Kazakh: четвёртая независимая generator family

## Решение и граница применения

**Spark-TTS-Kazakh (LLM + BiCodec controlled TTS)** принят только для personal research как
четвёртая generator family после Piper, Meta MMS/VITS и KazEmoTTS Grad-TTS + HiFi-GAN. Upstream
архитектура генерирует speech tokens LLM на базе Qwen2.5 и затем восстанавливает waveform через
BiCodec; это не вариант VITS, Piper или Grad-TTS. Model card declares Kazakh (`kk`), 16 kHz и
CC-BY-NC-SA-4.0. Upstream inference source is Apache-2.0. Источники:
[model card](https://huggingface.co/ErnarBahat/Spark-TTS-Kazakh/tree/bd0572a9bf744cb4cf9ab1c7a67bd182fde0c367),
[official Spark-TTS source](https://github.com/SparkAudio/Spark-TTS/tree/2f1ea9082400547242641f5271b6f941c9f439d1)
and [Spark-TTS paper](https://arxiv.org/abs/2503.01710).

В model card доступен режим voice cloning, но он запрещён этим проектом. Intake допускает только
official **controlled generation**: фиксированные gender/pitch/speed labels и KSC text; путь не
принимает, не читает и не кодирует reference audio. Его 12 virtual controls не являются
заявлением об identity, consent или independent voice groups.

## Почему не скачан wav2vec2

Обычный upstream `BiCodecTokenizer` безусловно инициализирует `wav2vec2-large-xlsr-53`, хотя
controlled branch не вызывает audio tokenization: LLM создаёт semantic и global tokens сам, а
BiCodec только их detokenizes. Этот checkpoint весит `1 269 737 156` bytes; с ним комплект был
бы около 3.13 GB и нарушил бы проектный лимит 2 GiB.

Local adapter не патчит веса или generation semantics. Он извлекает из pinned Apache source
только exact Python closure BiCodec detokenization, загружает LLM and BiCodec из verified
`safetensors` с `local_files_only=True`, `trust_remote_code=False` и не импортирует
`audio_tokenizer.py`. Таким образом, отсутствие wav2vec2 — не неполная clone-capable
installation, а fail-closed отсутствие самого reference-audio path. GPU smoke подтвердил
non-empty 16 kHz PCM WAV (9.400 s) без reference audio.

## Закреплённые bytes

Lock [`sparktts_kk_v1_models.json`](../configs/research/sparktts_kk_v1_models.json) pins the
Kazakh model revision `bd0572a9bf744cb4cf9ab1c7a67bd182fde0c367` and upstream source revision
`2f1ea9082400547242641f5271b6f941c9f439d1`. Exact source and all eight required model files
sum to `1 861 074 893` bytes, under the `2 147 483 648` byte limit.

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| Spark-TTS source archive | 6 034 176 | `b435…9c52` |
| BiCodec `model.safetensors` | 625 518 756 | `e994…67ec` |
| LLM `model.safetensors` | 1 222 491 632 | `9058…58c3` |
| configs + tokenizer files | 7 030 329 | individually pinned in lock |

All nine local files passed exact size/SHA-256 verification. The model bundle and all generated
WAV remain ignored by Git. `ksc_derived_kk_v3_sparktts` is marked `research_only` in the license
ledger; NC-SA terms rule out product use.

## Dataset protocol

The final Spark-TTS paired test uses a new KSC `test` selection whose sample IDs and text hashes
are disjoint from both frozen v1 and KazEmoTTS v2. It is processed as: raw KSC → normal QA/VAD
ready KSC → spoof-only Spark raw → normal QA/VAD → exact paired final manifest. The final test
must remain untouched by epoch choice, threshold selection and calibration.

The fresh KSC selection has `450` raw rows, of which `387` passed bona-fide QA. Spark-TTS
generated exactly `387` raw clips; `381` passed the same QA/VAD gate and `6` were rejected only
as `insufficient_speech` (the per-item reasons are in
`data/manifests/ksc_derived_kk_v3_sparktts_rejections_387.json`). The frozen final manifest
[`ksc_derived_kk_v3_sparktts_test_381.csv`](../data/manifests/ksc_derived_kk_v3_sparktts_test_381.csv)
contains `762` assets, balanced `381` bona-fide / `381` spoof. Every pair has exactly the same
`text_hash`; neither selection nor rejected rows were replaced with text from an earlier test.

```bash
uv run --extra ml --extra sparktts python scripts/synthesize_ksc_sparktts.py \
  --base-manifest data/manifests/ksc_derived_kk_v3_base_ready_450.csv \
  --transcript-root data/raw/ksc_slr102/slices/derived-v3-base-450 \
  --model-lock configs/research/sparktts_kk_v1_models.json \
  --model-root models/research/sparktts_kk_v1 \
  --license-ledger data/licenses/license_ledger.csv --data-root data \
  --output-manifest data/manifests/ksc_derived_kk_v3_sparktts_raw_387.csv \
  --slice-name sparktts-387 --limit 387 --seed 20260815 \
  --created-at 2026-08-10T00:00:00Z --device cuda
```
