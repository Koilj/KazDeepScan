# XLS-R+SLS model v4 — final salvage materialization preflight v1

**Дата:** 16 августа 2026, `2026-08-16T00:05:00+06:00`
**Статус:** `ok`, `writes_performed=false`

Read-only preflight успешно проверил plan
`xlsr_sls_model_v4_final_salvage_materialization_v1` (SHA-256
`b886674e97dfe9261d6d79c88e5a9a5ba03b5f7421b2fe0bc08e47276d782f11`).

Проверки включали:

- exact Common Voice RU archive и pinned FLEURS KK release; `499` RU + `498` KK selected
  source identities и hashes локального partial source slice;
- exact append-only journals и WAV hashes для `499` RU Qwen и `271` KK KazakhTTS partial
  outputs, без пересинтеза;
- два permanent KK rejects, отсутствие replacement/backfill, а также весь remaining scope:
  `227` KK normalized texts проходят locked KazakhTTS token validation;
- Qwen/KazakhTTS model locks и local runtime load; output/runtime namespaces ещё отсутствуют;
- current-history audio-isolation inputs и fail-closed prohibitions.

Ни raw audio, ни synthetic WAV, canonical audio, manifests, review forms, pair lock, detector
checkpoint, calibration или final inference не создавались/не запускались. Следующий
разрешённый шаг — единственный `materialize` run этого salvage contract; после него нужны
независимые reviews, а не inference.
