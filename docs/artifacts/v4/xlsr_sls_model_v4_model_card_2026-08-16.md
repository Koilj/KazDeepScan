# XLS-R+SLS model v4 — research model card

**Version:** `xlsr-sls-model-v4-train-v1`
**Scope:** personal research only; not a product model, fraud detector, speaker verifier or risk score.

## Model and training

The model is XLS-R 300M with an SLS head (`attention_size=128`, `classifier_size=256`,
`dropout=0.2`). It was trained once on the frozen balanced `20,000`-asset RU/KK dataset (`5,000`
per language × label cell). The selected tail-unfreeze epoch was fixed by macro RU/KK dev loss,
not by final results. The ignored local checkpoint and selected state hashes are bound in the
[training receipt](v4_training_contract_2026-08-15.md).

## Data roles

Train, bilingual dev, RU calibration and final are separate immutable roles. The final contains
two independently reviewed sources per language: Common Voice/Qwen for RU and FLEURS/KazakhTTS
for KK. It is not evidence of verified speaker independence: FLEURS and the TTS routes do not
provide sufficient speaker/voice provenance for that claim.

## Evaluation and calibration

One immutable final run followed the locked `792` pairs and a no-logit preflight. It reports RU
and KK separately; the canonical results, confidence intervals and limits are in the
[final evaluation receipt](final_reconciliation_evaluation_2026-08-16.md). RU only reuses a
disjoint temperature calibration. KK remains uncalibrated, and a combined RU+KK headline is
intentionally absent.

## Prohibited uses and limits

- Do not use this model to make fraud, authenticity, identity, consent or moderation decisions.
- Do not interpret scores as calibrated KK probabilities or as a product-grade risk score.
- Do not retrain, backfill, tune a threshold, refit calibration or repeat final inference from
  this checkpoint/final role.
- The evaluation measures these exact source/TTS routes only; it does not establish
  source-independent, generator-independent or speaker-independent performance.
