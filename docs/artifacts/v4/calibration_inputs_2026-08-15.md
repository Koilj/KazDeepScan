# XLS-R+SLS model v4 — RU calibration metadata-input gate v1

**Дата:** 15 августа 2026

## Решение

Metadata-only isolation gate завершён со статусом
`metadata_inputs_frozen_materialization_contract_required`. Он не выполнял calibration,
temperature fitting, checkpoint loading, detector/final inference, WAV extraction, synthesis,
QA, pairing или acoustic review.

Контракт
[`xlsr_sls_model_v4_calibration_inputs_v1`](../../../configs/research/v4/xlsr_sls_model_v4_calibration_inputs_v1.json)
имеет SHA-256 `a5074a95b78b5b53591a7c7c248eaeab76daf6ec0d549bb85063278c318d37db`.
Machine receipt:
[`xlsr_sls_model_v4_calibration_inputs_v1.json`](xlsr_sls_model_v4_calibration_inputs_v1.json),
SHA-256 `07a1b3026d0d68d0c22145128d4acae9f7517b21f7e754c301f81a4d94218daf`.

## Проверенные границы

- Training receipt `e4ded7e9…3a22f9` и ignored local checkpoint проверены без загрузки;
  file SHA-256 checkpoint — `8be73165…26852f`, selected state SHA-256 —
  `3cfca24a…01fbe4a`.
- Exact VoxForge archive (`3,795,197,539` bytes, SHA-256 `7372c6f8…93de557`) и source-audit
  receipt проверены before metadata read. Pinned Russian eSpeak NG `1.52.0` bundle и все шесть
  artifacts повторно прошли hash verification; модель не запускалась.
- Из `6,412` archive metadata records audit исключил все prior VoxForge exact sample IDs и
  `81` contributor groups, а также все text hashes, уже использованные historical
  `formant_rule_based_tts` outputs. Seeded maximum matching заморозил `81` fresh exact source
  identities с `81` distinct contributor groups.
- Между selected metadata и v4 `20,000`-row train / `1,917`-row dev overlap по
  `sample_id`, text hash, parent group и speaker pseudo-ID равен `0` во всех ячейках.

Historical VoxForge/Qwen layer уже содержит те же `81` prompt texts: это явно recorded как
non-v4 historical text overlap (`81` inventory text groups; `79` configured-role rows). Это не
разрешает reuse старых WAV, sample IDs или contributor groups; их overlap равен нулю. Claim
speaker independence по-прежнему запрещён.

Metadata selection CSV:
[`xlsr_sls_model_v4_calibration_voxforge_metadata_v1.csv`](../../../data/manifests/v4/xlsr_sls_model_v4_calibration_voxforge_metadata_v1.csv),
81 rows, SHA-256 `15d9cb11a23495e3bc42be7fcb2f266d272813a8668095df117f91c3285162e8`.
Он не является audio manifest и не содержит raw/decoded audio SHA-256.

## Следующий разрешённый шаг

Нужен отдельный immutable materialization-and-audio-isolation contract: новый frozen ledger для
VoxForge/eSpeak derivative, archive rebinding, exact raw hashes, one-shot text-only eSpeak
synthesis, decode/QA/VAD and exact/near-audio leakage gate, затем complete pair lock. До этого
temperature fitting, calibration, final inference и detector feedback остаются запрещены в рамках
этого metadata-only gate. Его required materialization/audio-isolation gate завершён позднее, а
отдельный [RU calibration contract](v4_ru_calibration_contract_2026-08-15.md) теперь допускает
только один temperature-only run; его собственный no-logit preflight уже завершён. Final inference
и feedback по-прежнему запрещены.
