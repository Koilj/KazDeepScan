# XLS-R+SLS model v4 — final reconciliation evaluation v1

**Дата:** 16 августа 2026
**Статус:** complete; final inference is permanently locked against repetition.

The one authorized CUDA/BF16 final pass completed after the immutable no-logit preflight. It
scored exactly `1,584` independently reviewed assets (`792` complete pairs): `332` RU and `460`
KK. The full machine receipt is
[`xlsr_sls_model_v4_final_reconciliation_evaluation_v1.json`](xlsr_sls_model_v4_final_reconciliation_evaluation_v1.json),
SHA-256 `f2c0639328a6a65b6648e60079735b852346b6825343913025b49f502548b5a0`.

| Locked layer | Fixed-zero-logit balanced accuracy | EER | Both assets correct per pair |
| --- | ---: | ---: | ---: |
| RU (`332` pairs) | `0.9744` | `0.0211` | `0.9488` (`315/332`) |
| KK (`460` pairs) | `0.8880` | `0.0891` | `0.7870` (`362/460`) |

RU bona-fide/spoof recalls are `0.9819`/`0.9669`; KK bona-fide/spoof recalls are
`0.9717`/`0.8043`. Wilson intervals, exact source/family/control strata and immutable per-asset
results are in the machine receipt. The RU temperature `0.7253568768501282` was reused from the
disjoint 73-pair calibration role without refit. KK is reported only as an uncalibrated score.

No pooled RU+KK headline, threshold selection, calibration refit, retraining, backfill,
resynthesis or second final inference is authorized. The final is research-only, not a product or
fraud-risk score; source/text/group/audio isolation is evidenced, but speaker independence is not
verified.
