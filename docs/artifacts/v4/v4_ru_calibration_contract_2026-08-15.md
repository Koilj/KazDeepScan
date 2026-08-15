# XLS-R+SLS model v4 — RU calibration contract v1

**Дата:** 15 августа 2026

## Решение

Новый immutable contract
[`xlsr_sls_model_v4_ru_calibration_v1`](../../../configs/research/v4/xlsr_sls_model_v4_ru_calibration_v1.json)
разрешает ровно один local CUDA/BF16 scoring pass и one-parameter temperature scaling только на
уже frozen `73` RU VoxForge/eSpeak exact-text pairs (`146` assets). Это отдельное узкое
research-only decision; immutable materialization ledger не меняется.

Новый two-source ledger
[`xlsr_sls_model_v4_calibration_v1.csv`](../../../data/licenses/frozen/xlsr_sls_model_v4_calibration_v1.csv)
имеет SHA-256 `853259916351b3ca8a4e2524c3aa1d0a7684dea461a5e48828e2cffa90f438d5`. Он опирается
на текущую VoxForge Russian card MDC (GPL-3.0, `Restrictions: N/A`, `Forbidden Usage: N/A`) и
оставляет обязательными Data Consumer Terms и applicable law. Разрешение ограничено personal
research; это не юридическое заключение и не разрешение product use.

Контракт hash-bind-ит:

- complete pair lock (`146` rows, `73` pairs), SHA-256
  `a8a367549f566222690ea199955e19b51315182fe329a30dc765e24edc5b5d71`;
- exact v4 training/materialization plans and receipts, plus frozen v4 train/dev manifests;
- selected tail-unfreeze epoch 2 state
  `3cfca24a3731d3f9e3c259dcea905be07aefc4fbf2fbefa98189696df01fbe4a` and ignored local
  checkpoint file SHA-256 `8be73165a4e6f65e966fa6d6a162fbb319d7089d1e8c1597c131e9ccb226852f`;
- local XLS-R 300M config/weights, CUDA/BF16 runtime and nine implementation files.

## Fail-closed controls

Runner [`run_v4_calibration.py`](../../../scripts/run_v4_calibration.py) refuses a changed byte,
an absent or non-`research_only` ledger entry, a non-complete pair, any overlap by sample/audio/
text/group with v4 train/dev, runtime mismatch, an altered checkpoint state or an output overwrite.
It first writes a no-logit preflight; execution then writes a lock before loading the checkpoint.

Only temperature scaling is permitted. Threshold selection, epoch/model/architecture/augmentation
selection, checkpoint mutation, pair replacement/backfill, detector feedback and final inference
are explicitly prohibited. The result, when produced, will be RU-only: no calibrated KK
probability and no verified speaker-independence claim are possible.

## Текущее состояние

Contract code and ledger are prepared and validated by targeted tests. No calibration preflight,
checkpoint load, calibration logit, temperature fit or final inference has occurred under this
contract yet. The next safe action is exactly one `--validate-only` preflight with the frozen
local audio root; only its successful write-once receipt may unlock the single calibration run.
