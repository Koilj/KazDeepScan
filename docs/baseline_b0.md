# B0: контрольный baseline

`B0LogMelCnn` преобразует waveform 16 кГц в log-Mel и выдаёт один сырой `spoof_logit`.
Это небольшой контрольный классификатор, цель которого — обнаружить ошибки в декодировании,
labels и split-ах до обучения XLS-R. Он не является продуктовой моделью.

До появления легального group-aware data slice веса модели случайны. Любой logit B0 до
обучения не имеет семантики риска, не должен проходить через sigmoid и не может появиться в
API-ответе. После обучения требуется отдельная record-level оценка и калибровка температуры.

Проверить тензорный тракт без данных можно так:

```bash
uv run python scripts/smoke_b0.py --device cuda
```

После появления подготовленного train/dev manifest запуск выглядит так:

```bash
uv run python scripts/train_b0.py \
  --manifest data/manifests/slice.csv \
  --audio-root data \
  --license-ledger data/licenses/license_ledger.csv \
  --purpose research \
  --output models/b0-slice.pt \
  --device cuda
```

`--purpose` обязателен: он не позволяет случайно представить research weights как product
checkpoint. Режим `product` требует отдельного полного binary protocol с verified
speaker/voice groups, legal commercial scope и independent OOD; текущие manifest-ы его не
проходят. Скрипт сохраняет выбранную цель и protocol report в checkpoint.

До обучения скрипт требует для каждого источника одобренный статус в license ledger,
проверяет SHA-256 всех train/dev assets, требует оба класса в каждом split и отказывается
перезаписать checkpoint. Статус `owner_authorized_personal_research` годится только для
личного исследования и не отменяет внешние условия источника. Выбор по `dev_loss` нужен
только для sanity baseline; он не заменяет независимый OOD test и record-level калибровку.

Отдельную некалиброванную проверку checkpoint можно выполнить так:

```bash
uv run python scripts/evaluate_b0.py \
  --checkpoint models/b0-slice.pt \
  --manifest data/manifests/ood.csv \
  --audio-root data \
  --license-ledger data/licenses/license_ledger.csv \
  --split ood --device cuda
```

Команда проверяет schema, ledger и SHA-256 files, но возвращает только loss/accuracy.
Она не вычисляет risk probability, threshold, EER или калибровку; fake-only OOD data не
позволяют измерить false-positive rate.
