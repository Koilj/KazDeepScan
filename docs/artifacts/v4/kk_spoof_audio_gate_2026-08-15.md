# XLS-R+SLS model v4 — KK spoof common audio gate v1

**Status:** frozen pre-QA contract. Все четыре local text-only routes уже завершили `7,200/7,200`
raw WAV без runtime reject, но ни один из них ещё не объявлен QA-ready или eligible for training.

Machine contract:
[xlsr_sls_model_v4_kk_spoof_audio_gate_v1.json](../../configs/research/v4/xlsr_sls_model_v4_kk_spoof_audio_gate_v1.json)
(SHA-256 c0ff8ef9ca9c51fab11fb06e22df938498cda7c5ca5d165b2f9efa43cc1eee0c).

The contract hash-binds all four route receipts, raw manifests and complete synthesis inventories;
the original synthesis plan; the frozen source decode receipt/inventory; the existing historical
fingerprint inventory; current license ledger; and the exact gate runner/audio primitives.

Each raw asset is decoded exactly once to mono PCM-16 WAV at 16 kHz in a new ignored namespace.
The gate applies the existing QualityPolicy, WebRTC VAD, decoded-exact SHA-256 screen and the
gain/padding-tolerant spectral near-audio fingerprint screen against both all current v4 source
rows and the frozen historical inventory. A decode/quality failure or exact collision is rejected;
a near match is held as `pending_near_audio_review` and cannot enter the frozen train layer.

The only permitted selection is `target → reserve → selection_rank` within each declared TTS route.
It freezes exactly `1,250 × 4 = 5,000` eligible rows or fails closed. Detector/logit feedback,
resynthesis, new-dataset search, output overwrite and backfill outside the frozen reserve are
prohibited. A successful receipt is the first possible authorization to build the combined
20,000-row train manifest; it does not claim speaker independence.
