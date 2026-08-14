# XLS-R+SLS model v4 — KK spoof common audio gate v1

**Status:** completed exactly once. Common gate processed all `7,200` local text-only WAV,
retained `6,200` eligible rows and froze exactly `5,000` train rows (`1,250` per route). No raw
WAV, decoded output or completed receipt was overwritten or rerun.

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
prohibited.

## Completed accounting

| Route | Eligible | `insufficient_speech` rejects | Frozen |
| --- | ---: | ---: | ---: |
| KazEmoTTS | 1,456 | 344 | 1,250 |
| MMS | 1,602 | 198 | 1,250 |
| Piper | 1,510 | 290 | 1,250 |
| SparkTTS | 1,632 | 168 | 1,250 |
| **Total** | **6,200** | **1,000** | **5,000** |

All historical and within-pool decoded-exact collisions, and all historical and within-pool
near-audio review candidates, equal `0`. The exact output packet is:

- decode inventory: `7,200` rows, SHA-256
  `4eed7c7e3707404a182de66f90354362b3eaadd5d272194bd87bc9824387c1e7`;
- ready manifest: `6,200` rows, SHA-256
  `1a157cadb669ac6172032b05d78463dce71f3de37dafac293dbda7666567d2bf`;
- frozen KK spoof manifest: `5,000` rows, SHA-256
  `52d662b08b5b67e34d2f9795fd0b67abf83b24f80b9ddab4712720915e02b990`;
- completed gate receipt: SHA-256
  `0e2ddb4b40264237def45c10bebdc09e5aef49e15461df7e0bcd78632dfecf50`.

## Authorization boundary reconciliation

The immutable completed receipt accidentally records `claims.training_authorized=true` and a
state name that reads as actual training authorization. Its own `next_gate` correctly requires a
combined `20,000`-row manifest and a separate training contract first. The original receipt is
preserved byte-for-byte; the [machine governance and reconciliation]
(xlsr_sls_model_v4_kk_spoof_audio_gate_governance_v1.json) makes the effective boundary explicit:
only construction of the combined manifest and a no-training contract preflight are authorized.
Actual training, checkpoint selection, calibration and final inference remain not authorized, and
speaker independence is not claimed.
