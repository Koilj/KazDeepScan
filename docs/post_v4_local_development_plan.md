# KazDeepScan — post-v4 local development plan without new sources

**Status:** active roadmap; it does not authorize a new model run by itself.

**Scope:** development after the completed `xlsr-sls-model-v4` research cycle, using only
already available local code, checkpoints and source inventory. No new dataset, source archive,
TTS model, raw-audio collection, voice cloning or product scorer is in scope.

## 1. Non-negotiable boundary

The v4 training receipt, RU calibration receipt, final pair lock and one-time final evaluation
are complete and immutable. Do not use the v4 checkpoint or its final data for user-audio
scoring, repeat inference, retraining, threshold selection, calibration refit, backfill or
resynthesis. The canonical v4 result and its limits remain in
[the final evaluation receipt](artifacts/v4/final_reconciliation_evaluation_2026-08-16.md) and
[the model card](artifacts/v4/xlsr_sls_model_v4_model_card_2026-08-16.md).

This roadmap distinguishes three activities:

| Activity | Allowed now | Not an authorization for |
| --- | --- | --- |
| Local user-audio research check | Existing B0 CLI/API contract for an external user file | v4 scoring, a probability, fraud/identity/product decision or tuning from the result |
| Engineering maintenance | Tests, static checks, dependency/security review, documentation and fail-closed behavior | Changing frozen manifests, receipts, checkpoints or completed runs |
| Future local-only research proposal | Read-only eligibility inventory and a new contract proposal | Extraction, synthesis, training or inference before explicit authorization |

## 2. What is available for a user's own audio now

The implemented route is
[`b0-user-audio-local-research-v1`](../configs/inference/b0_user_audio_local_research_v1.json),
not v4/XLS-R. It loads the Git-ignored local checkpoint
`models/b0-unseen-generator-suite-v1.pt` read-only after exact SHA-256 validation and uses the
existing audio pipeline. It never reads frozen evaluation manifests and creates no execution lock,
report, dataset row or checkpoint.

The route accepts only an external file outside project `data/`, `models/`, `artifacts/` and
`checkpoints/` roots. It checks the real container, normalizes with FFmpeg to mono 16 kHz PCM,
runs quality checks and WebRTC VAD, and requires at least `2.5 s` of speech. Limits are `50 MiB`
and `10 min`. The temporary normalized audio is not saved by the CLI or API.

### 2.1 CLI: recommended local check

From the repository root, first validate the installed local checkpoint and contract:

```bash
export KDS_FFMPEG_BINARY="$PWD/.tools/ffmpeg/bin/ffmpeg"
export KDS_FFPROBE_BINARY="$PWD/.tools/ffmpeg/bin/ffprobe"

.venv/bin/kds validate-research-inference
```

It must return `"status": "ready"` with all of `calibrated`, `probability_claim`,
`fraud_claim` and `product_grade` set to `false`. Then submit only your own external file:

```bash
.venv/bin/kds research-infer /absolute/path/to/user-audio.wav \
  --mime-type audio/wav \
  --acknowledge-research-only
```

Use the MIME type that matches the actual file; supported containers include WAV, FLAC, MP3,
M4A/MP4 and OGG. A rejected file is an audio-quality/pipeline result, not a statement about a
person or authenticity.

### 2.2 Optional localhost API

Use the API only on a local trusted machine. It requires both explicit confirmations and removes
the uploaded bytes from its private temporary directory after the request:

```bash
export KDS_RESEARCH_INFERENCE_CONTRACT="$PWD/configs/inference/b0_user_audio_local_research_v1.json"
export KDS_FFMPEG_BINARY="$PWD/.tools/ffmpeg/bin/ffmpeg"
export KDS_FFPROBE_BINARY="$PWD/.tools/ffmpeg/bin/ffprobe"

.venv/bin/uvicorn \
  kds.serving.research_api:create_research_app_from_environment \
  --factory --host 127.0.0.1 --port 8001
```

In another terminal:

```bash
curl -F "audio=@/absolute/path/to/user-audio.wav;type=audio/wav" \
  -F "acknowledge_research_only=true" \
  -F "confirm_external_user_audio=true" \
  http://127.0.0.1:8001/v1/research/analyze
```

### 2.3 How to interpret the result

`uncalibrated_spoof_score` is the sigmoid of an aggregated raw logit. It is not a probability.
`bonafide_like` and `spoof_like` only identify the side of the fixed zero-logit research boundary.
The result must not be used for fraud, authenticity, consent, identity, moderation, employment,
access or other automated decisions. Training-data overlap for the user file is unverified, and
the model is not speaker-independent or commercially cleared. Do not retain user audio or copy
it into Git, manifests, receipts or training data.

The authoritative contract and implementation details are in
[local user-audio research inference v1](research_user_audio_inference_v1.md).

## 3. Development order without new sources

### Phase 0 — preserve the completed evidence

1. Keep v4 and all historical versioned receipts byte-stable.
2. Run the documented QA suite before code changes: Ruff, canonical mypy targets, pytest and
   `git diff --check`.
3. Keep raw audio, model weights, checkpoints, user uploads and generated outputs Git-ignored.

### Phase 1 — make the existing local user route operable

1. Run `validate-research-inference` before each local session.
2. Use only external user audio with explicit research-only acknowledgment.
3. Treat `status=ok` as successful technical processing, not model validation.
4. Record no user result in a frozen research artifact and never use it for model changes.

### Phase 2 — propose, but do not execute, a local-only future experiment

No source may be added. Before any extraction, synthesis, model loading or detector inference,
create a separate versioned proposal with a new `run_id` that:

1. inventories every candidate against the full current project history by sample ID, exact audio
   SHA-256, text hashes, groups and source/TTS family;
2. excludes all v4 train, dev, calibration and final assets, as well as every failed or
   write-once-rejected route;
3. names the evidence correctly: remaining material from an already used corpus can at most be an
   asset-level sensitivity layer, not an independent source-level final;
4. defines a no-outcome-based selection rule, immutable outputs and stopping conditions before
   any audio operation; and
5. obtains explicit owner authorization for the proposed scope before execution.

If this audit cannot produce all required separated roles from the existing inventory, the correct
outcome is `stop_local_capacity_exhausted`; it is not a reason to relax leakage, rights,
speaker-provenance or final-run rules.

### Phase 3 — product work remains blocked

A product/API risk score, calibrated probability, new deployment endpoint or use of v4 on user
audio requires a separate scope decision and contracts for rights, privacy, verified speaker
provenance, security, calibration and deployment. None is authorized by this plan.

## 4. Definition of success for this roadmap

The completed near-term result is a reproducible local technical check of an external user audio
file through the existing B0 research route, while v4 remains untouched. Any new research model
or evaluation is a separately named future cycle, not an extension or rerun of v4.
