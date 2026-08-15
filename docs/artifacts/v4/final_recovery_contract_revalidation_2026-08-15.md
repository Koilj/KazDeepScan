# XLS-R+SLS model v4 — recovery contract code revalidation v1

**Дата:** 15 августа 2026

## Статус

Первый recovery plan был создан metadata-only, но ещё **не проходил preflight** и не создавал
runtime namespace, raw audio, synthetic WAV, manifests или review outputs. До выполнения была
найдена статическая typing-проблема в новом decode adapter. Поэтому исходный plan сохранён
неизменным и не используется.

После code-only correction создан новый hash-pinned
[`revalidated plan`](../../../configs/research/v4/xlsr_sls_model_v4_final_recovery_materialization_v1_revalidated.json).
Он использует те же immutable `999` metadata rows, authorization и ledger; не расширяет scope,
не меняет original rank-1 reject и не даёт нового права на synthesis для уже attempted row.

Все проверки к моменту revalidation были metadata/code-only: no TTS synthesis, detector loading,
calibration, detector inference или final inference. Следующий разрешённый шаг — только
read-only preflight именно revalidated plan.
