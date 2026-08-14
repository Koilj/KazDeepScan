# XLS-R+SLS model v4 — KK spoof synthesis plan v1

**Status:** frozen pre-synthesis contract. Final-contract preflight успешно проверил complete
hash-pinned local bundles и controls всех четырёх маршрутов на local CUDA: Piper (`6`), MMS
(`1`), KazEmoTTS (`18`) и SparkTTS (`12`) profiles. Все four invocations returned
`preflight_ok` with `1,800` pending tasks each and created neither WAV nor journal. Training
remains forbidden until all four routes complete and the shared decode/QA/leakage gate freezes
exactly 5,000 KK spoof rows.

Machine contract:
[xlsr_sls_model_v4_kk_spoof_synthesis_v1.json](../../configs/research/v4/xlsr_sls_model_v4_kk_spoof_synthesis_v1.json)
(SHA-256 4735fa343ee3a97809fab8b4c5d4854963be8680e778c40b3e35e97041e971f9).

The contract binds the frozen v2 candidate CSV, selection governance, source decode receipt,
exact KSC2 text inventory/receipt, current license ledger, v4 runner/contract code and the
three pre-existing model locks. It also binds the historical low-level adapter code used by each
route. Inputs are verified by SHA-256 before a model is loaded.

| Train-only route | Family | Attempted | Frozen target | Reserve | Declared controls |
| --- | --- | ---: | ---: | ---: | ---: |
| kk-piper-issai-high-v1 | Piper neural TTS | 1,800 | 1,500 | 300 | 6 speaker controls × 300 |
| kk-mms-kaz-v1 | MMS VITS | 1,800 | 1,500 | 300 | 1 default control |
| kk-kazemotts-v1 | Grad-TTS + HiFi-GAN | 1,800 | 1,500 | 300 | 18 speaker/emotion controls × 100 |
| kk-sparktts-v1 | controlled LLM + BiCodec | 1,800 | 1,500 | 300 | 12 virtual controls × 150 |

Each invocation loads one route in a fresh process, runs only local text-to-speech, and uses
HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1. It accepts no reference/prompt audio, cloning,
network, detector score, logit, or new-dataset input. The Spark route retains its already pinned
fixed structural retry policy only; no retry may use detector feedback.

WAVs are content-addressed below the new ignored v4 raw namespace. An fsynced append-only local
journal records the frozen task, profile/control, actual seed, exact WAV hash, duration and any
terminal failure. A successful WAV is never overwritten or regenerated; an orphaned WAV without a
terminal journal record blocks safely for manual review. A route publishes its manifest, full
success/failure inventory and receipt only after all 1,800 terminal records exist.

The next permitted action after all four route receipts is one common canonical decode/16 kHz
PCM/QA/VAD and historical/current audio-leakage screen. It may choose only the announced target
then reserve order; it may not use detector outcomes.
