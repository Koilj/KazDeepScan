# Immutable license-ledger snapshots

Начиная с XLS-R v2 frozen plans не закрепляют общий изменяемый
`data/licenses/license_ledger.csv`. `scripts/freeze_license_ledger.py` создаёт минимальный
write-once CSV только с реально используемыми `source_id`, сортирует строки и повторно загружает
результат строгим validator-ом.

| Snapshot | Источники | SHA-256 |
| --- | ---: | --- |
| `xlsr_sls_stage_ab_v2.csv` | RuASD + PyAra | `5ef01f6f648280e1eb6905be15a9921b78fc78d479d293c86a9f102234cc7477` |
| `xlsr_sls_stage_b_v2_research_final_v1.csv` | 8 train/dev/calibration/final sources | `3e656f636a01d21d96b6ee90e365ef63bab3c3e033367f85233d1b63cfbe7538` |
| `xlsr_sls_stage_b_v2_fresh_suite_stage_c_v1.csv` | PyAra + 6 Stage-C RU/KK/mixed sources | `22433b07d16c7cea61db1521ba60d4971977f943e71b0610ddc2bb2faef0eb05` |
| `xlsr_sls_stage_b_v2_stage_d_dialogs_ru_v1.csv` | PyAra + Common Voice RU + Dialogs-RU VITS2 | `d14cdd6fdd235fe2e511178ac4b3ba6aed5e632eb3656c0560d55e5bfbdb787c` |
| `xlsr_sls_v3_governance_v1.csv` | RuASD + PyAra + Common Voice RU + Dialogs-RU VITS2 | `9386aa3ace5b0b021c4af74312cf5bb910da0bb9d9537790ccd08601695c345f` |
| `xlsr_sls_stage_b_v2_common_voice_ru_v24_silero_v5_5_eugene_v1.csv` | PyAra + Common Voice RU + Silero V5.5/eugene | `6de966f1197626e15ac786a38f2abbd211f701a51979c7378b79fb611c9100e8` |
| `xlsr_sls_stage_b_v2_voxforge_ru_mdc_qwen3_tts_customvoice_aiden_v1.csv` | PyAra + VoxForge RU + Qwen/Aiden | `f23415c2e57995426e2562059accd24ede5b2da3abad449e99425a3f1c6f2f16` |

Snapshot нельзя обновлять на месте; новый protocol получает новое имя. Исторические v1 plans
закрепляли общий ledger по SHA-256, когда он ещё не был сохранён в Git как отдельный immutable
blob. Их exact прежние ledger bytes нельзя достоверно восстановить из текущего дерева, поэтому
старые hashes не переписываются и v1 receipt остаётся исторически честным, но не полностью
revalidatable. Model/checkpoint results от этого не становятся product evidence.

14 августа 2026 mutable ledger был исправлен: шесть legacy `notes` с запятыми получили
корректные CSV quotes. Новый separate gate отклоняет non-exact header и extra row fields
вместо молчаливого обрезания. Pinned semantic loader не изменён: это инвалидировало
бы completed contracts. Все listed snapshots и их SHA-256 остались без изменений.
