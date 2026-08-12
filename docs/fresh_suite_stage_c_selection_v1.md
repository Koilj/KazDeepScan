# Stage C fresh-suite selection and bona-fide materialization v1

**Статус:** selection заморожен и bona-fide сторона материализована 12 августа 2026 года.
Detector inference и массовый TTS synthesis на момент фиксации не выполнялись.

## Selection contract

Политика выбрала все доступные fresh groups из inventory v2, без использования detector
predictions, logits или метрик:

| Роль | До QA | После QA | Решение |
| --- | ---: | ---: | --- |
| RU FLEURS | 55 | 50 | 5 `signal_too_quiet`, без backfill |
| KK FLEURS | 60 | 60 | уже QA-ready до selection |
| KSC2 mixed | 58 | 58 | уже QA-ready до selection |
| Всего | 173 | 168 | метрики только раздельно |

Selection seed: `stage-c-v1-all-eligible-20260812`. Политика требует один source recording на
text group, запрещает post-selection backfill и сохраняет claims
`source_independent=false`, `speaker_independent=false`.

| Артефакт | SHA-256 |
| --- | --- |
| `fresh_suite_stage_c_selection_v1.json` | `dfd43c60d7b22bf0a93127f4f810be0b1900b66c0611aaf56e8f935a0d39e03d` |
| `fresh_suite_stage_c_ru_base_raw_v1.csv` | `f1a469db4860ef191bd9d89d8d0c0f67de98d11c6e35d12be612ea01fbac4c43` |
| `fresh_suite_stage_c_ru_base_ready_v1.csv` | `8b779f5f0d6d420dc67138cdfed1ce6b61256c2880cc3ef48f42a181bf7f0417` |
| `fresh_suite_stage_c_ru_base_rejections_v1.json` | `287a2b33f0d5b3597fbe16fd656639633ac0c895f1578c59d96eeb7cb9989066` |
| `fresh_suite_stage_c_kk_base_ready_v1.csv` | `7c829455543052856a6dcbde2f93596dfc5d8f7c019ae46d5ade0f2fcc012b34` |
| `fresh_suite_stage_c_mixed_base_ready_v1.csv` | `a02e7e4cba7277ac68200efbca833dc9c2969a1034d498b4e76ecc0612153760` |
| `fresh_suite_stage_c_base_ready_v1.csv` | `93c7319e5733f65a5af84bd73907ff4ee259e627ce0eaa187a1c7118713e6855` |
| `fresh_suite_stage_c_base_materialization_v1.json` | `522917f4a7aade4403099b3e4a678d9f9f6e2293c014635d605505a0cd59b6a2` |

Selection JSON хранит exact transcripts, IDs, text hashes, source members, готовые base-asset
hashes для KK/mixed, полный pinned FLEURS artifact set и 15 input bindings. Для RU selection был
зафиксирован до extraction/QA; поэтому пять тихих записей сохранены в rejection accounting и не
заменены другими строками.

## Граница следующего шага

Разрешён text-only synthesis ровно для 168 ready base rows через уже одобренный exact
KazakhTTS route. Любой generated WAV сначала проходит signal QA/VAD, а затем два независимых
full-asset acoustic reviews. До завершения этих gates detector inference остаётся запрещён.
