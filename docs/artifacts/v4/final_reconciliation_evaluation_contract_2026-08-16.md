# XLS-R+SLS model v4 — final reconciliation evaluation contract v1

**Дата:** 16 августа 2026
**Статус:** prepared; no-logit preflight required before the only authorized final run.

This contract implements the project-owner authorization received in the active session. It
binds the selected v4 checkpoint, the complete two-review pair lock and the existing disjoint RU
temperature receipt. It permits exactly one CUDA/BF16 inference pass on the frozen `1,584` final
assets (`792` pairs: `332` RU and `460` KK).

The source ledger from materialization remains immutable and is not reinterpreted. A separate
evaluation-only frozen ledger gives the reviewed assets `research_only` test permission for this
single owner-authorized run; it does not permit training, calibration refit, resynthesis, product
use or a second inference.

The required first command is:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_v4_final_reconciliation.py \
  --plan configs/research/v4/xlsr_sls_model_v4_final_reconciliation_evaluation_v1.json \
  --audio-root data --validate-only
```

Only a successful immutable preflight may be followed by the same command without
`--validate-only`. The report is split by RU and KK; it contains fixed-zero-logit classification,
pair accuracy, EER and available source/TTS/voice-control strata. RU reuses the already fitted
temperature `0.7253568768501282` without refit; KK remains an uncalibrated score, and a pooled
RU+KK headline is prohibited.

The contract binds
[`xlsr_sls_model_v4_final_reconciliation_evaluation_v1.json`](../../../configs/research/v4/xlsr_sls_model_v4_final_reconciliation_evaluation_v1.json),
SHA-256 `4df671a0ddd1b8afd8e8957dfd1a7ce3b8dbe60e696cb293238aea29f3504685`.
