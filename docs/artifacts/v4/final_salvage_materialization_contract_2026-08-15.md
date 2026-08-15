# XLS-R+SLS model v4 — final salvage materialization contract v1

**Статус:** frozen; materialization ещё не запускался.

После исчерпанного recovery one-shot этот независимый contract ограничивает scope ровно
`997` потенциальными парами: `499` RU и `498` KK. Он hash-bind-ит сохранённые `499` RU Qwen
и `271` KK KazakhTTS WAV вместе с append-only recovery journals; их разрешено только
проверить, canonicalize и изолировать, без пересинтеза.

KK ranks `272` (`№`) и `310` (`%`) записаны как permanent rejects в
[`v4_final_salvage_permanent_rejects_2026-08-15.json`](v4_final_salvage_permanent_rejects_2026-08-15.json).
Для replacement/backfill и повторного synthesis этих строк нет разрешения. Единственная новая
audio operation, которую допускает contract, — один KazakhTTS pass для ровно `227`
предварительно проверенных оставшихся KK rows.

Canonical plan:
[`xlsr_sls_model_v4_final_salvage_materialization_v1.json`](../../../configs/research/v4/xlsr_sls_model_v4_final_salvage_materialization_v1.json),
SHA-256 `b886674e97dfe9261d6d79c88e5a9a5ba03b5f7421b2fe0bc08e47276d782f11`.
Authorization и 997-row selection находятся в отдельных versioned artifacts; raw audio и
model weights по-прежнему Git-ignored.

До pair lock должны завершиться QA/VAD, full-history audio isolation и два независимых
acoustic/language review. Detector checkpoint loading, calibration, detector inference и final
inference запрещены этим contract.
