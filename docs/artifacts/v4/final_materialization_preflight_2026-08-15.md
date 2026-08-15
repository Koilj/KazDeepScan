# XLS-R+SLS model v4 — final materialization preflight v1

**Дата:** 15 августа 2026, `2026-08-15T22:30:00+06:00`

## Результат

Read-only preflight завершён со статусом `ok`. Контракт
[`xlsr_sls_model_v4_final_materialization_v1`](../../../configs/research/v4/xlsr_sls_model_v4_final_materialization_v1.json),
SHA-256 `9d471f8e961a530a209fa2652344a0c12d4dfc28c03b4c28f5e14bd4bac2088a`,
вернул `writes_performed=false` и `final_inference_performed=false`.

Проверены exact Common Voice RU v24 archive, полный pinned FLEURS `kk_kz` release, frozen
metadata selection/ledger, оба local text-only TTS route и full current project history. History
screen охватил `120` manifest files / `140 470` rows / `84 917` unique audio hashes: `84 525`
имеют canonical near-audio fingerprints, `392` остаются exact-only. Journals завершённой
calibration materialization добавляют свои raw/canonical bindings; ни один current manifest hash
не остался без required exact/fingerprint coverage.

## Границы

Не созданы raw WAV, processed WAV, runtime namespace, final manifests, review forms, pair lock
или materialization receipt. TTS models были только проверены/loaded как часть preflight; не было
synthesis attempt. Detector checkpoint не загружался; calibration, detector inference и final
inference не выполнялись.

## Следующий разрешённый шаг

One-shot `materialize` из того же hash-pinned contract. Он повторит preflight перед созданием
единственных разрешённых `500` RU + `500` KK source assets и `500+500` text-only synthetic
attempts. После этого обязательны QA/VAD, isolation и две реальные независимые review forms;
final inference по-прежнему не разрешён.
