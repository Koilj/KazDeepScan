# XLS-R base encoder provenance

| Field | Value |
|---|---|
| Model | `facebook/wav2vec2-xls-r-300m` |
| Purpose | SSL encoder only; not an anti-spoof classifier without the SLS head and training |
| License | Apache-2.0 |
| Input | mono waveform, 16 kHz |
| Local destination | `models/xlsr-300m/` (excluded from Git) |
| Revision | `1a640f32ac3e39899438a2931f9924c02f080a54` |

The model card at Hugging Face lists Apache-2.0, 128 languages, and 16 kHz input. The
downloaded encoder must be fine-tuned only on the approved data manifests. No untrained or
uncalibrated SLS output may be used by the API.

The local FP32 and bf16 forward smoke tests on RTX 5060 Ti passed on 8 August 2026. The loader
reports seven unexpected pretraining-only keys (`project_*`, `quantizer.*`); that is expected
because KazDeepScan loads the `Wav2Vec2Model` encoder and deliberately discards the pretraining
heads.
