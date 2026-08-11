# XLS-R+SLS Stage B — ToneSpeak Russian spoof-only OOD (100 WAV)

**Статус:** завершён 12 августа 2026 года. Это единственный write-once
**research-only spoof-only OOD evaluation** frozen Stage-B checkpoint. Это не binary
evaluation, не final quality, не calibration и не product result.

## Зафиксированный запуск

- run-plan: [`xlsr_sls_stage_b_v1_tone_speak_ru_ood_100.json`](../configs/research/xlsr_sls_stage_b_v1_tone_speak_ru_ood_100.json),
  SHA-256 `b655abcdf90b3d53cd2d32065e4296436dd3467a94cb5cce60f9e2e2c7a544b0`;
- frozen checkpoint: `models/xlsr-sls-stage-b-v1.pt`, SHA-256
  `18c967a8881404140ccda04fc6234079ac4b2802425e4111f3fef59bef505c32`;
- canonical Stage-B trainable state: `59ad0812e14d33abec00ba5225876de4c208efa9c8f8f9061e253e60df9d1089`;
- locked candidate manifest: SHA-256
  `45527f97ddbb5be65ea12ff8a8ed7b723e9d1135aff0cde71addd95366dc84a0`;
- acoustic gate report v2: SHA-256
  `8525df3980210c8e2b4dd827859e0d0c7b1ecb74f512ab6a5eaa628cbeb55df6`;
- execution lock: [`artifacts/...execution.json`](../artifacts/xlsr-sls-stage-b-v1-tone-speak-ru-ood-100.execution.json),
  SHA-256 `ba6ee2d39dd058132b1df5d51c1f6cd8a78035af366a29bf81e9bb5620a92365`;
- complete machine-readable report (all 100 sample-level logits):
  [`artifacts/...report.json`](../artifacts/xlsr-sls-stage-b-v1-tone-speak-ru-ood-100.report.json),
  SHA-256 `ee711761b91b9950b0cb04d13ea7b3e7f7a78078be5be0af25f32c71d7d7e96b`.

`--validate-only` before execution rechecked the plan, all implementation hashes, ledger,
source audit/lock, ready receipt, acoustic packet/report, every processed WAV SHA-256, checkpoint,
Stage-B receipt and encoder. Before the first forward pass the execution lock was written; its
existence forbids a second execution of this plan.

No training, weight updates, calibration, threshold fitting or threshold selection occurred.
The only classifier rule shown below is the frozen checkpoint default `raw_logit >= 0.0`; it is
not a chosen operating point or a probability.

## Result at the fixed raw boundary

| Metric | Result |
| --- | ---: |
| Records | `100` spoof / `0` bona-fide |
| Spoof recall | `88/100 = 88.0%` (95% Wilson `80.19–93.00%`) |
| Raw prediction counts | `88` spoof, `12` bona-fide |
| Raw-window BCE loss | `0.23256` |

No bona-fide recall, accuracy, balanced accuracy, EER, ROC/DET, calibrated probability or
binary quality metric is available: the locked set contains no bona-fide class.

## Per-voice descriptive slice

The ten `voice_id` values are source-card-provided fields. This table is descriptive only and
does not establish independent human speaker groups.

| Voice | Spoof recalled | Raw `bonafide` predictions |
| --- | ---: | ---: |
| `alloy` | `8/10` | `2` |
| `ash` | `10/10` | `0` |
| `ballad` | `5/10` | `5` |
| `coral` | `9/10` | `1` |
| `echo` | `10/10` | `0` |
| `fable` | `9/10` | `1` |
| `nova` | `10/10` | `0` |
| `onyx` | `10/10` | `0` |
| `sage` | `7/10` | `3` |
| `shimmer` | `10/10` | `0` |

The complete report names all 12 fixed-boundary false negatives and carries their audio and text
hashes. They are observations, not a reason to revise the raw boundary, select a threshold,
calibrate the model, alter architecture or change the candidate.

## Limits that remain in force

1. ToneSpeak is spoof-only; it cannot establish binary detection quality or substitute for a
   vetted independent Russian bona-fide counterpart.
2. Per-row model/voice provenance is only source-card-provided; there is no independent API log,
   model snapshot or reference-audio proof for every generated row.
3. The completed two-review gate establishes Russian audibility and lexical preservation only for
   these exact 100 WAV bytes. It does not prove product eligibility or future output language.
4. The fixed 100-row sample and its Wilson interval describe only this exploratory sample. They
   must not be combined with prior mixed-pair results or called final quality.
