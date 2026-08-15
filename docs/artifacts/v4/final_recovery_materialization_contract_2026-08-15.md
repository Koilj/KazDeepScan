# XLS-R+SLS model v4 — final recovery materialization contract v1

**Дата:** 15 августа 2026

## Статус

Заморожен отдельный metadata-only recovery contract
`xlsr-sls-model-v4-final-recovery-materialization-v1`. Он не изменяет исходный failed contract
или его ignored raw traces и не разрешает повтор первого Qwen вызова.

Контракт разрешает только новый one-shot materialization для `999` ранее не затронутых synthesis
rows: `499` RU и `500` KK. RU rank `1`
`common_voice_ru_v24:common_voice_ru_33348952` закреплён как irrecoverable reject: его текст,
audio, synthesis, replacement и backfill запрещены.

Новый Qwen adapter передаёт CrispASR абсолютный output path и hash-pinned как отдельный input;
historical wrapper и lock остаются неизменными. Recovery по-прежнему ограничен extraction,
text-only synthesis, QA/VAD, full-history audio isolation, двумя independent reviews и pair lock.
Checkpoint loading, calibration, detector inference и final inference запрещены.

## Versioned bindings

- [recovery authorization](v4_final_recovery_authorization_2026-08-15.json)
- [recovery metadata selection](../../../data/manifests/v4/xlsr_sls_model_v4_final_recovery_metadata_v1.csv)
- [recovery license ledger](../../../data/licenses/frozen/xlsr_sls_model_v4_final_recovery_materialization_v1.csv)
- [initial unused recovery plan](../../../configs/research/v4/xlsr_sls_model_v4_final_recovery_materialization_v1.json)
- [revalidated recovery plan](../../../configs/research/v4/xlsr_sls_model_v4_final_recovery_materialization_v1_revalidated.json)
- [code revalidation receipt](final_recovery_contract_revalidation_2026-08-15.md)
- [original failure receipt](final_materialization_attempt_failure_2026-08-15.md)

## Следующий безопасный шаг

Read-only [preflight](final_recovery_materialization_preflight_2026-08-15.md) revalidated plan
завершён без outputs. Следующий шаг — один write-once materialization только для `499+500`
не затронутых rows.
