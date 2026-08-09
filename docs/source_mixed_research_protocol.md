# Source-mixed research protocol v1

## Назначение

Обычная проверка manifest не допускает пересечение `sample_id`, SHA-256, parent group,
speaker pseudo-ID и text hash между split-ами. Этого недостаточно: B0 может научиться
артефактам конкретного corpus, его нормализации или цепочки записи. Поэтому эта матрица
жёстко разделяет исходные corpus между `train`, `dev` и финальным `test`.

Матрица задаётся JSON-файлом с версией схемы, ID протокола и тремя ролями. Каждая роль
содержит путь к неизменяемому source manifest, его исходный split и точный список ожидаемых
`source_name`. Валидатор проверяет:

- в каждой роли присутствуют оба класса;
- исходный split допустим для роли (`train` берёт только `train`, `dev` — только `dev`,
  `test` — `test` либо сохранённый `ood`);
- ни один `source_name` не участвует более чем в одной роли;
- полный source manifest и выбранные строки проходят обычные проверки leakage;
- статус license ledger одобрен, а declared use допускает personal research.

`scripts/train_b0_matrix.py` использует только роли `train` и `dev` во время обучения.
Лучшая эпоха выбирается только по loss на `dev`; роль `test` оценивается один раз после
выбора checkpoint. Метрики каждого источника остаются отдельными.

## Текущая матрица

`configs/research/source_mixed_v1.json` задаёт:

| Роль | Источник | Исходный split | Назначение |
| --- | --- | --- | --- |
| train | `ruasd_ru_v1_full` | `train` | Русский binary personal-research fit set. |
| dev | `pyara_ru_v7` | `dev` | Независимый от train corpus для выбора epoch. |
| test | `ml_df_it_v1` | `ood` | Итальянский cross-lingual binary stress-test. |

Это **не** полноценная русско-казахская benchmark-матрица. В локально разрешённых данных нет
третьего независимого русскоязычного binary corpus, а KSC и Common Voice содержат только
bona-fide речь. Поэтому ML-DF сохранён как OOD в собственном manifest и в отчётах должен
называться только cross-lingual stress-test. Его результат нельзя усреднять с RuASD, PyAra,
KSC или Common Voice и нельзя использовать для калибровки либо API score.

## Воспроизведение

```bash
.tools/uv/uv run kds validate-source-matrix configs/research/source_mixed_v1.json \
  --license-ledger data/licenses/license_ledger.csv

.tools/uv/uv run python scripts/train_b0_matrix.py \
  --matrix configs/research/source_mixed_v1.json \
  --audio-root data \
  --license-ledger data/licenses/license_ledger.csv \
  --output models/b0-source-mixed-research-v1.pt \
  --epochs 5 --device cuda
```

Новый source допускается только после отдельного terms/artifact audit, добавления в ledger,
отдельного intake и preprocessing. Его нельзя добавлять в матрицу ручным копированием WAV.
