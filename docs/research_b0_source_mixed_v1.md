# B0 source-mixed research v1

**Дата запуска:** 10 августа 2026.
**Scope:** только local personal research. Это не product benchmark, калиброванная
вероятность, risk score или доказательство синтетичности речи.

## Протокол

Матрица [`configs/research/source_mixed_v1.json`](../configs/research/source_mixed_v1.json)
задаёт три непересекающихся по `source_name` роли:

| Роль | Источник и выбранный исходный split | Размер | bona-fide / spoof |
| --- | --- | ---: | ---: |
| train | `ruasd_ru_v1_full`, `train` | 1 417 | 637 / 780 |
| dev | `pyara_ru_v7`, `dev` | 61 | 26 / 35 |
| final test | `ml_df_it_v1`, `ood` | 192 | 94 / 98 |

Валидатор проверил source isolation вместе с обычными проверками SHA-256, sample/group/text
leakage и license ledger. `ML-DF` остаётся OOD в собственном manifest: роль `final test`
означает только, что его не используют для обучения или выбора epoch. Это итальянский
cross-lingual stress-test, а не русскоязычная или казахская общая метрика.

Каждый запуск обучал B0 5 эпох на RTX 5060 Ti (CUDA 12.8, batch size 16). Checkpoint выбирался
только по минимальному `dev_loss` на PyAra, после чего ML-DF оценивался ровно один раз.

## Результаты

| Seed | Выбранная эпоха | Лучший dev loss | Dev balanced accuracy | ML-DF bona-fide recall | ML-DF spoof recall | ML-DF balanced accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20260810 | 5 | 0.5460 | 0.7654 | 0.4681 | 1.0000 | 0.7340 |
| 20260811 | 5 | 0.5891 | 0.8187 | 0.7021 | 0.9898 | 0.8460 |
| 20260812 | 2 | 0.5409 | 0.8038 | 0.7872 | 0.9286 | 0.8579 |

Balanced accuracy на финальном source-disjoint stress-test меняется от **0.7340 до 0.8579**.
На первом seed bona-fide recall составляет только 0.4681: почти половина настоящих записей
ML-DF помечается spoof. Разброс при неизменных данных и гиперпараметрах показывает, что
маленький dev set (61 записи) пока недостаточен для устойчивого выбора модели.

Для контекста прежний `b0-ruasd-full-research-2000.pt` на том же ML-DF manifest дал balanced
accuracy 0.9262 (bona-fide recall 0.8830, spoof recall 0.9694). Это не «регресс продукта»:
предыдущий checkpoint выбирался внутри RuASD, а новый — на независимом PyAra source. Разница
как раз демонстрирует, почему нельзя объявлять within-source/OOD число единой accuracy.

Отдельная диагностика checkpoint seed `20260810` на bona-fide-only слоях дала KSC test recall
0.7061 (`245` казахских записей) и Common Voice RU test recall 0.8095 (`84` русских записи).
В этих наборах нет spoof класса, поэтому balanced accuracy не определена; они не участвовали
в выборе epoch, калибровке или сравнении binary моделей.

## Решение

- Не включать calibration, model release или `POST /v1/analyze` score.
- Не усреднять эти результаты с RuASD/PyAra within-source, KSC или Common Voice.
- Не использовать `ruasd_ru_v1_shard000000` как OOD после training на full RuASD: это один
  исходный corpus и он запрещён текущей source matrix.

Чтобы улучшить доказательность, нужен отдельный лицензированно проверенный русско- или
казахскоязычный binary source с bona-fide и spoof классами. До его появления корректно
продолжать только reproducible research, включая повторение этой матрицы на заранее
зафиксированных seeds без tuning по ML-DF.
