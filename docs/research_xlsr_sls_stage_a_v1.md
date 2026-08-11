# XLS-R + SLS Stage A v1

## Назначение

Это воспроизводимый personal-research train/dev запуск, а не product model release. В Stage A
заморожен весь XLS-R-300M encoder; обучаются только смесь hidden states, attentive statistics
pooling и binary SLS classifier. Запись голосов людей, reference audio и voice cloning в этом
этапе не используются.

Runner не принимает final manifest и не загружает frozen unseen-generator suite. Поэтому
KazEmoTTS, Spark-TTS и eSpeak NG final assets не участвовали в обучении, выборе epoch,
threshold или calibration.

## Зафиксированный протокол

- plan: `configs/research/xlsr_sls_stage_a_v1.json`;
- plan SHA-256: `4cdb63f6490238f7c453afeab59f3c32bf0bdf1bbc5598e6d95e7847832f6d7e`;
- train: RuASD `train`, `1 417` rows (`637` bona-fide, `780` spoof);
- dev: PyAra `dev`, `61` rows (`26` bona-fide, `35` spoof);
- XLS-R revision: `1a640f32ac3e39899438a2931f9924c02f080a54`;
- XLS-R weights SHA-256:
  `d5e490574712ad0a6736923b9ed11d4cd51c78609c36205f704fc4e87b11d2e0`;
- head: attention `128`, classifier `256`, dropout `0.2`;
- train: seed `20260818`, 3 эпохи, batch `16`, BF16, AdamW, learning rate `1e-4`,
  weight decay `1e-4`, gradient clip `1.0`;
- выбор checkpoint: только минимальный PyAra `dev_loss`.

Train и dev принадлежат разным `source_name`. Их объединённый manifest прошёл проверки
sample/SHA-256/group/text leakage, источники разрешены ledger для personal research, все
`1 478/1 478` выбранных assets совпали с SHA-256.

## CUDA preflight

Запуск выполнен на NVIDIA GeForce RTX 5060 Ti (`16 616 521 728` bytes VRAM), PyTorch
`2.11.0+cu128`, CUDA runtime `12.8`, Transformers `5.14.1`; BF16 поддерживается. Одно
train- и одно dev-окно при batch `16` использовали пик `2 776 097 792` allocated и
`3 198 156 800` reserved bytes. После этого конфигурация не менялась.

Локальный checkpoint содержит стандартные pretraining quantizer/projector tensors. При
загрузке `Wav2Vec2Model` они ожидаемо отбрасываются; пропущенных encoder keys в load report
нет. Сам checkpoint закреплён полным SHA-256, поэтому другой набор весов не может быть принят
runner-ом молча.

## Результат

| Epoch | Train loss | Train balanced accuracy | Dev loss | Dev accuracy | Dev balanced accuracy |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.52915 | 0.7339 | 0.59246 | 0.7541 | 0.7857 |
| 2 | 0.27130 | 0.8843 | 0.37606 | 0.8361 | 0.8571 |
| 3 | 0.18874 | 0.9304 | **0.37577** | 0.8197 | 0.8429 |

Выбран epoch 3, потому что заранее закреплённым критерием был `dev_loss`. Нельзя постфактум
переключиться на balanced accuracy и выбрать epoch 2. На выбранном epoch dev bona-fide recall
равен `26/26 = 1.0`, spoof recall — `24/35 = 0.6857`. Dev содержит всего 61 запись, поэтому
эти числа имеют широкую неопределённость и не доказывают устойчивость к новым corpus,
генераторам, людям или каналам.

Полное обучение заняло `42.71` с. Сохранены только параметры SLS-head:

- `models/xlsr-sls-stage-a-v1-head.pt`: `2 650 306` bytes, SHA-256
  `46e8187551bb0d9defcb45eaae78f3e084fbff57667f593c3e696eea75a88a12`;
- `models/xlsr-sls-stage-a-v1-report.json`: `9 752` bytes, SHA-256
  `84fd0246962f108e7dfbffacd361e59d1d42ecbb67f63f67237db2c96487f23a`;
- canonical hash 11 tensors выбранного head state:
  `1370b9b81e0c61f0ced94d29fdcc15e28ba28f5240ef724683b8bfd0cdb490e6`.

Повторная read-only проверка подтвердила state hash, соответствие `best_epoch` минимуму
dev loss, отсутствие encoder state в checkpoint, `frozen_final_evaluation_performed=false` и
`calibrated=false`. Runner отказывается перезаписывать оба артефакта.

## Воспроизведение preflight

Готовый run повторно не запускать: output reservation остановит команду. Для нового run нужны
новый `run_id`, seed и output paths, после чего сначала выполняются read-only проверки:

```bash
uv run python scripts/train_xlsr_sls_stage_a.py \
  --plan configs/research/xlsr_sls_stage_a_v2.json \
  --audio-root data --validate-only

uv run python scripts/train_xlsr_sls_stage_a.py \
  --plan configs/research/xlsr_sls_stage_a_v2.json \
  --audio-root data --profile-only
```

Следующий модельный этап — отдельный hash-pinned Stage B: инициализировать head этим Stage A
checkpoint, разморозить только последние XLS-R blocks, заново выполнить CUDA profile и выбирать
состояние только по dev. Frozen final evaluation допустима лишь после окончательного выбора
архитектуры и отдельной предварительной регистрации.
