# XLS-R+SLS model v4 — reconciliation pair-lock contract v1

Both independent review forms are complete, have distinct reviewer pseudo-identities and are
hash-bound in the canonical authorization:
[`xlsr_sls_model_v4_final_reconciliation_pair_lock_v1.json`](../../../configs/research/v4/xlsr_sls_model_v4_final_reconciliation_pair_lock_v1.json).

This narrow contract permits only exact pair locking from the already published review packet.
It accepts the reconciliation receipt's `technical_decode_qa_vad_reused_exactly=true` claim in
place of the legacy same-run QA flag; it does not alter QA evidence, audio, manifests or reviews.
Extraction, synthesis, resynthesis, calibration, detector loading/inference and final inference
remain forbidden.
