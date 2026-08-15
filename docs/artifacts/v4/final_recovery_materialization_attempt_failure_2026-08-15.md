# XLS-R+SLS model v4 — final recovery materialization attempt failure v1

**Дата:** 15 августа 2026, `2026-08-15T23:45:00+06:00`

## Статус

Revalidated recovery one-shot attempt **не завершён** и не может быть повторён по этому
contract. No versioned raw/ready manifests, QA/VAD inventory, review packet/forms, pair lock или
materialization receipt не опубликованы. Detector checkpoint, calibration, detector inference и
final inference не выполнялись.

Контракт
[`xlsr_sls_model_v4_final_recovery_materialization_v1_revalidated`](../../../configs/research/v4/xlsr_sls_model_v4_final_recovery_materialization_v1_revalidated.json),
SHA-256 `fd7c5cc2849d486d6f739aaab963c64693896795991b468d7bb5860955e5b3a7`.

## Фактическая остановка

RU Qwen route завершил `499/499` planned/generated pairs. KK KazakhTTS route завершил
`271/271`, после чего rank `272`
`google_fleurs_kk_v1:16990681329536390054` остановил runner **до** journal `planned` event и до
WAV: frozen normalized text содержит `№` (`U+2116`), отсутствующий в pinned token list.

Read-only post-failure input audit проверил остальные `229` unattempted KK texts. Ещё один
неприемлемый frozen text — rank `310` `google_fleurs_kk_v1:15895861050511584755` с `%`; только
`227` remaining KK texts совместимы с этим exact pinned route. Rank `272` и rank `310` нельзя
переписывать или заменять; они требуют permanent reject в следующем contract.

## Сохранённые локальные traces

Все ниже Git-ignored, ещё не являются manifests и не допускаются к QA/review/inference без
отдельного salvage contract:

| Local slice | Files | Bytes | Ordered `(path,size,SHA-256)` aggregate SHA-256 |
| --- | ---: | ---: | --- |
| RU Common Voice source | 499 | 17,846,376 | `451ec7d52fcfe451c31577c76b4ff9473bf4996ce63a83dec958129d50570047` |
| KK FLEURS source | 500 | 421,864,520 | `7962d73bb5c83510c97c6a3de823648ee266ac3c3426ff1c3df7c5cf2d31aa87` |
| RU Qwen synthetic | 499 | 101,298,490 | `8a54efc25ad0570a5e5eac18f1a869d1e09b46379dd171bee47fcd9100cb6844` |
| KK KazakhTTS synthetic | 271 | 98,378,900 | `747c7096fe2414dd727d848bb861df1d4baf031f738e2c29d307b08cb1663e25` |

Append-only journals: RU Qwen `998` events (`499` planned + `499` generated), SHA-256
`7a1a07e68de87b3fceece714a18d2de9b271d9ce4ccc88994bbe15358aa40561`; KK KazakhTTS `542`
events (`271` planned + `271` generated), SHA-256
`b7a56951618a202d4e3a6419fa4e976417d2e6b3c47592a3452929c0c060e2ff`.

## Required recovery decision

The contract is exhausted. A new immutable salvage/remainder contract must (1) hash-bind and
QA/isolate the existing `499` RU and `271` KK synthetic traces without resynthesis, (2) make KK
ranks `272` and `310` permanent rejects without replacement/backfill, and (3) authorize exactly
one synthesis attempt only for the remaining `227` prevalidated KK rows. It must not run final
inference.
