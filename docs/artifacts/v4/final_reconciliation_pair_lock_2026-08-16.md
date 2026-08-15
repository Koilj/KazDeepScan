# XLS-R+SLS model v4 — final reconciliation pair lock v1

**Дата:** 16 августа 2026, `2026-08-16T01:30:00+06:00`
**Статус:** complete; final inference still forbidden.

Exact two-review pair lock создан из `1,584` review assets: `792` pairs, из них `332` RU и
`460` KK. Все assets получили `pass` от двух distinct reviewer identities.

Locked manifest:
[`xlsr_sls_model_v4_final_reconciliation_pairs_frozen_v1.csv`](../../../data/manifests/v4/xlsr_sls_model_v4_final_reconciliation_pairs_frozen_v1.csv),
SHA-256 `a6f57d65f866f8c14eeaa897ebaa2caa00dde9aa62f670738abdc733c759658b`.
Exact receipt:
[`xlsr_sls_model_v4_final_reconciliation_pair_lock_v1.json`](xlsr_sls_model_v4_final_reconciliation_pair_lock_v1.json),
SHA-256 `e7a2804f64a76c1207c7fc276308625e7ab3d0a4e15d1984e3217017c0f8fd2c`.

Pair lock не authorizes checkpoint loading, calibration, detector inference или final inference.
Любой final evaluation требует отдельного нового immutable contract.
