# Immutable license-ledger snapshots

Начиная с XLS-R v2 frozen plans не закрепляют общий изменяемый
`data/licenses/license_ledger.csv`. `scripts/freeze_license_ledger.py` создаёт минимальный
write-once CSV только с реально используемыми `source_id`, сортирует строки и повторно загружает
результат строгим validator-ом.

| Snapshot | Источники | SHA-256 |
| --- | ---: | --- |
| `xlsr_sls_stage_ab_v2.csv` | RuASD + PyAra | `5ef01f6f648280e1eb6905be15a9921b78fc78d479d293c86a9f102234cc7477` |
| `xlsr_sls_stage_b_v2_research_final_v1.csv` | 8 train/dev/calibration/final sources | `3e656f636a01d21d96b6ee90e365ef63bab3c3e033367f85233d1b63cfbe7538` |

Snapshot нельзя обновлять на месте; новый protocol получает новое имя. Исторические v1 plans
закрепляли общий ledger по SHA-256, когда он ещё не был сохранён в Git как отдельный immutable
blob. Их exact прежние ledger bytes нельзя достоверно восстановить из текущего дерева, поэтому
старые hashes не переписываются и v1 receipt остаётся исторически честным, но не полностью
revalidatable. Model/checkpoint results от этого не становятся product evidence.
