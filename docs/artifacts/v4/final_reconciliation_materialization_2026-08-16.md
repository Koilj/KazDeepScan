# XLS-R+SLS model v4 — final reconciliation publication v1

**Дата:** 16 августа 2026, `2026-08-16T01:00:00+06:00`
**Статус:** materialized; two independent reviews required; pair lock pending.

Publication-only reconciliation успешно обработал hash-bound evidence без extraction, synthesis,
decoder execution, detector checkpoint, calibration или inference. Full-history isolation вновь
применён к complete QA evidence.

| Metric | Result |
| --- | ---: |
| Raw assets audited | 1,994 |
| Individually eligible assets | 1,694 |
| Complete eligible pairs | 792 |
| RU complete pairs | 332 |
| KK complete pairs | 460 |
| Review packet assets | 1,584 |

Raw manifests сохраняют все `499` RU + `498` KK source и synthetic routes. Ready manifests
содержат только complete source/spoof pairs: непарные QA survivors исключены из review, но не
заменены и не восполнены. Exact receipt:
[`xlsr_sls_model_v4_final_reconciliation_v1.json`](xlsr_sls_model_v4_final_reconciliation_v1.json),
SHA-256 `61319e016e6e2c644330d18a03f1687db39de8b7e24bedb0329a1a5fd279cf35`.

Review packet SHA-256 `3f1f68f3a2c2445282bd807d87519e39fcb83e81b2b8019c10c4f42b9a7d6c7b`.
Два разных независимых reviewer должны заполнить exact versioned forms и отметить `pass` для
обеих сторон retained pair. Только затем отдельная команда вправе создать pair lock. Final
inference остаётся запрещённым.
