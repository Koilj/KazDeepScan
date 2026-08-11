# eSpeak NG Kazakh: пятая независимая generator family

## Решение и граница применения

**eSpeak NG 1.52.0 Kazakh (`kk`)** принят только как compact personal-research stress source.
Это rule-based formant synthesis, а не нейросеть и не дополнительный voice предыдущих Piper,
MMS/VITS, Grad-TTS + HiFi-GAN или LLM + BiCodec family. Official
[language table](https://github.com/espeak-ng/espeak-ng/blob/4870adfa25b1a32b4361592f1be8a40337c58d6c/docs/languages.md)
lists `kk` as Kazakh; the [upstream README](https://github.com/espeak-ng/espeak-ng/tree/4870adfa25b1a32b4361592f1be8a40337c58d6c)
states formant synthesis and GPL-3.0-or-later licensing.

The intentionally simpler formant sound is a limitation, not a quality claim. It is useful here
because it adds a materially distinct synthesis mechanism. It is never presented as natural
speech, a real person, speaker-independent evidence or product data.

The adapter accepts only verified KSC text on standard input. It has no reference-audio argument,
does not read audio during synthesis, and has no cloning path. Twelve `speed/pitch/amplitude`
controls are deterministic runtime settings, not speakers or voice identities.

## Закреплённые bytes и local runtime

Lock [`espeakng_kk_v1_models.json`](../configs/research/espeakng_kk_v1_models.json) pins official
eSpeak NG source revision `4870adfa25b1a32b4361592f1be8a40337c58d6c` and the Ubuntu
`1.52.0+dfsg-5build1` x86_64 runtime packages. All six artifacts have exact size/SHA-256 checks
and total `29 248 111` bytes, well below the 2 GiB intake cap.

| Artifact | Size | SHA-256 |
| --- | ---: | --- |
| eSpeak NG source archive | 17 739 803 | `bb4338…0a23` |
| `espeak-ng.deb` | 2 014 882 | `01ed5a…2454c` |
| `libespeak-ng1.deb` | 211 318 | `73f017…b7251` |
| `libpcaudio0.deb` + `libsonic0.deb` | 20 670 | individually pinned |
| `espeak-ng-data.deb` | 9 261 438 | `8304eb…f56a` |

No system package is installed. On every run, the already verified Debian payloads are extracted
into a temporary directory. The extractor rejects absolute/traversing names, devices, hard links,
unexpected member types and more than 64 MiB unpacked data. It then runs only the extracted binary
with the extracted libraries and `kk` language data. A local smoke test produced a non-empty mono
16-bit PCM WAV at its native 22 050 Hz; the shared preprocessing contract resamples accepted clips.

## Dataset protocol

Fresh KSC `test` selection v4 excludes every sample ID and `text_hash` in KSC base raw v1, v2 and
v3 before extraction. It has zero sample and text overlap with those sources: `450` raw rows →
`407` QA-accepted bona fide rows. Forty-three base rejections are retained (12
`insufficient_speech`, 31 refusal to reuse a pre-existing processed asset); no replacement text is
drawn from an earlier final test.

eSpeak NG generated all `407` selected texts. `358` clips passed the same QA/VAD stage and `49`
`insufficient_speech` rejections are retained in
`data/manifests/ksc_derived_kk_v4_espeakng_rejections_407.json`. Frozen final
[`ksc_derived_kk_v4_espeakng_test_358.csv`](../data/manifests/ksc_derived_kk_v4_espeakng_test_358.csv)
has `716` assets, exactly `358` bona fide / `358` spoof. Each pair has one identical `text_hash`;
the final test is prohibited for epoch choice, threshold selection and calibration.

```bash
uv run python scripts/download_research_tts_models.py \
  --model-lock configs/research/espeakng_kk_v1_models.json \
  --model-root models/research/espeakng_kk_v1

uv run python scripts/synthesize_ksc_espeakng.py \
  --base-manifest data/manifests/ksc_derived_kk_v4_base_ready_450.csv \
  --transcript-root data/raw/ksc_slr102/slices/derived-v4-base-450 \
  --model-lock configs/research/espeakng_kk_v1_models.json \
  --model-root models/research/espeakng_kk_v1 \
  --license-ledger data/licenses/license_ledger.csv --data-root data \
  --output-manifest data/manifests/ksc_derived_kk_v4_espeakng_raw_407.csv \
  --slice-name espeakng-407 --limit 407 --seed 20260817 \
  --created-at 2026-08-10T00:00:00Z
```
