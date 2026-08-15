# XLS-R+SLS model v4 — final reconciliation contract v1

**Статус:** frozen; publication ещё не запускалась.

Этот contract исправляет только publication boundary failed salvage attempt. Он не извлекает и не
декодирует audio, не загружает TTS или detector и не синтезирует WAV. Вместо этого он exact-bind-ит
`997` existing source/spoof raw assets, `454`-event remaining KK journal и четыре complete
decode/QA journals, затем повторно применяет current-history isolation.

Raw manifests сохраняют полный audit scope. Ready manifests и review packet могут содержать только
text-identical пары, у которых обе source/spoof стороны individually eligible после QA/isolation.
QA reject не заменяется и не восполняется.

Canonical plan:
[`xlsr_sls_model_v4_final_reconciliation_v1.json`](../../../configs/research/v4/xlsr_sls_model_v4_final_reconciliation_v1.json),
SHA-256 `b23f83120a00a051b5e2b5df71c40fd137419cef4bd1e33bd69bcc46e1bdf6ba`.
После publication остаются два независимых acoustic/language review и pair lock. Final inference,
calibration и detector checkpoint loading всё ещё запрещены.
