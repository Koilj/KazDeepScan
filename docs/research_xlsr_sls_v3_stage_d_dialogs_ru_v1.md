# XLS-R + SLS v3 — governed Stage-D Dialogs-RU evaluation

**Дата:** 13 августа 2026
**Run ID:** `xlsr-sls-v3-stage-d-dialogs-ru-v1`
**Scope:** personal research; не product/API score.

## Итог

v3 выполнил ровно один GPU inference после отдельного immutable governance contract,
Stage-A/Stage-B model selection и write-once preflight. На неизменяемых 55 Common Voice RU /
Dialogs-RU Masha-neutral парах получено `107/110` correct, balanced accuracy `0.972727`.

Это отдельный RU-layer result. Он не позволяет менять final-набор, выбирать checkpoint,
architecture, augmentation, temperature или threshold и не является claim product quality,
source-independence, speaker-independence или новой architecture family.

## Исправление governance до обучения

Первый черновой contract `xlsr-sls-v3-data-governance-v1` и его receipt сохранены как
отклонённые pre-training evidence: формулировка `checkpoint_selection=stage_b_dev_loss_only`
не описывала допустимый Stage-A выбор head по Stage-A dev. До training или detector inference
v1 был заменён на `xlsr-sls-v3-data-governance-v2` с точным правилом
`stage_a_and_stage_b_dev_loss_only`. Ни Stage-D v2 logits, ни errors не загружались ни при
проверке v2, ни в обучении/оценке v3.

## Замороженные роли и права

Locked ledger: `data/licenses/frozen/xlsr_sls_v3_governance_v1.csv`, SHA-256
`9386aa3ace5b0b021c4af74312cf5bb910da0bb9d9537790ccd08601695c345f`.
Он оставляет только четыре personal-research source entries: RuASD train, PyAra и exact
Dialogs-RU Masha-neutral route вместе с Common Voice RU final counterpart.

| Роль | Manifest | Строки | Назначение |
| --- | --- | ---: | --- |
| train | RuASD v1 full ready v2 | 1 471 (663 bona / 808 spoof) | Только обучение |
| Stage-A dev | PyAra 500 ready | 61 (26 / 35) | Только выбор Stage-A head |
| Stage-B dev | fresh PyAra v3 | 969 (474 / 495) | Только выбор Stage-B checkpoint |
| calibration | disjoint PyAra v3 | 976 (478 / 498) | Только temperature scaling |
| final Stage D | frozen Common Voice/Dialog-RU | 110 = 55 pairs | Только один v3 final inference |

Governance contract SHA-256: `cef05c3a5873ed7b31ba5e17c01317b24ce6d713bc7d220f729ea8c08792cfc5`.
Validated governance receipt SHA-256:
`1d60ddac74b038ec605f39abc820676593946aeea7c59458e5f11ff3d70851a0`.

Receipt проверил все 3 587 assets и нулевые pairwise overlap по `sample_id`, audio SHA-256,
`text_hash` и `parent_group_id` между train, обоими dev, calibration и final ролями. Для
calibration/final evaluator дополнительно проверяет `speaker_pseudo_id` непосредственно по
manifest-строкам.

Final содержит ровно тот же manifest SHA-256
`cd4ccc2be9fd0fa06bd3641f1e939499d93f111c5f6a5a636753adf97ac2891d`, тот же full-asset
two-review acoustic gate и тот же no-backfill pairing receipt из Stage D. Эти exact assets уже
получили v2 predictions, поэтому v3 не называют blind/unseen final. Contract запрещает
использовать v2 logits/errors для всех v3 решений и evaluator их не открывает.

## Training v3

В training role добавлена только детерминированная симметричная augmentation
`symmetric-channel-codec-replay-simulation-v1`: random gain `[-6, 6] dB`, resample/8-bit codec
simulation и delayed replay mix. Она применяется только при `mode=train`, одинаково для обеих
labels, параметры выводятся из `(seed namespace, policy ID, epoch, sample_id, component)` без
label; dev, calibration и final не augmented.

| Этап | План / выбор | Результат |
| --- | --- | --- |
| Stage A | `xlsr-sls-stage-a-v3`, 3 epochs, frozen encoder | epoch 3, Stage-A dev loss `0.4272934531`; head checkpoint SHA-256 `439b0c0daeef77e4d530d723dec9c5e37836c6b410d6f16eafe95e6bc6c4c6c5` |
| Stage B | `xlsr-sls-stage-b-v3`, 15 epochs, final 8 XLS-R blocks | epoch 4, fresh Stage-B dev loss `0.1851617722`; checkpoint SHA-256 `43f36f79d83b1b620a3095e5e98514309d2c72d736fc4f29c0ca9edcd3b92ecc` |

Stage-B selected trainable-state SHA-256:
`f2147364a8b33026c3a66fd59f871d002d460289e7a76d754d29f5b3b1bbbada`.
Модель загружалась с local XLS-R revision `1a640f32ac3e39899438a2931f9924c02f080a54` и
`weights_only=True` для trainable checkpoint. Сообщение Transformers о лишних quantizer/project
keys относится к task-head difference базовой Wav2Vec2 checkpoint и не меняет pinned encoder
weights или loaded SLS/tail state.

## One-time final evaluation

Immutable final plan: `configs/research/xlsr_sls_v3_stage_d_dialogs_ru_v1.json`, SHA-256
`0f2d63826b728ea07b3bb418901230aa304b0b629d61bdb63162033865f26252`.

Preflight проверил 1 086 bindings (`976` calibration + `110` final), CUDA BF16 и все pinned
files; receipt SHA-256 `36ca347d1a9d04bbc15be592737c6b87ad81d228997273657c7a9a1e1fef7df6`.
После него execution lock SHA-256
`bcbb69bdfe5376b2c9c9552e6f3eca57ceac06248b6a311f6a7941d2a4745b80` разрешил ровно один run.

Temperature была fitted только на calibration: `T=0.79692465`, NLL `0.17983271 → 0.17592850`,
Brier `0.05744414 → 0.05766366`, ECE-15 `0.08754022 → 0.07822767`. Порог не выбирался:
использован fixed calibrated probability `0.5`.

| Метрика final RU | Значение | 95% Wilson CI |
| --- | ---: | --- |
| Accuracy | 107/110 = 0.972727 | [0.922869, 0.990682] |
| Bonafide recall (Common Voice) | 53/55 = 0.963636 | [0.876765, 0.989970] |
| Spoof recall (Dialogs-RU) | 54/55 = 0.981818 | [0.903942, 0.996783] |
| Balanced accuracy | 0.972727 | — |
| Полностью верные пары | 52/55 = 0.945455 | [0.851469, 0.981277] |
| Final Brier / ECE-15 | 0.021770 / 0.053373 | — |

Final report SHA-256: `9f99a7bed878ecfc561831e5130ac92dd5c8b73cb9965fc51e8a4d00d66e50e3`.
Report, execution lock, preflight, model checkpoints и WAV остаются local ignored artifacts;
их paths и hashes закреплены plan/receipt, но сами bytes не добавляются в Git.

## Не менять и не повторять

- Не повторять Stage-A v3, Stage-B v3, v3 preflight или final inference.
- Не использовать final Stage-D v2/v3 errors для retraining, reranking, threshold или
  augmentation decisions.
- Не заменять, не дозаполнять и не re-synthesize 55 final pairs.
- Новый будущий result требует другого immutable contract и genuinely unevaluated final assets;
  API/product track требует отдельного commercial-rights и deployment review.
