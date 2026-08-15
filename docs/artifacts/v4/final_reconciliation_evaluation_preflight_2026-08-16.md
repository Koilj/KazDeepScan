# XLS-R+SLS model v4 — final reconciliation evaluation preflight v1

**Дата:** 16 августа 2026
**Статус:** complete; exactly one final inference remains authorized and unexecuted.

No-logit preflight for
[`xlsr_sls_model_v4_final_reconciliation_evaluation_v1.json`](../../../configs/research/v4/xlsr_sls_model_v4_final_reconciliation_evaluation_v1.json)
completed successfully. Its local immutable receipt is
`artifacts/v4/xlsr-sls-model-v4-final-reconciliation-evaluation-v1.preflight.json`, SHA-256
`0ac5edf40b789b293d3dfc87bb645d24e157128b0472eade348ea66654ecd91e`.

It hash-validated all `1,584` frozen final WAV assets and the selected checkpoint without
loading model weights or producing a logit. The final has `792` pairs: `332` RU and `460` KK.
Detectable intersections with every frozen role are zero for `sample_id`, asset SHA-256,
`text_hash`, `parent_group_id` and `source_name`:

| Final versus role | Result |
| --- | --- |
| train | all five intersections `0` |
| dev | all five intersections `0` |
| calibration | all five intersections `0` |

Pinned CUDA/BF16 runtime is available on the RTX 5060 Ti. The preflight performed no checkpoint
load, calibration refit, threshold selection, detector feedback or final inference.
