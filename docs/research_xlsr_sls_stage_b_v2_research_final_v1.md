# Stage B v2 — calibrated confirmatory RU/KK/mixed research run

**Статус:** однократный GPU run выполнен 12 августа 2026. Это не product quality и не blind
project-level final.

Строгий plan SHA-256
`1dfc3ca866607191385b33b85a1ee67cb3981099c6fc836aef720c6c2610d4fc` закрепляет Stage-B
checkpoint/report, immutable 8-source ledger snapshot, четыре предшествующие роли, три evaluation
manifest, evidence receipts, implementation bytes и два write-once output path. Preflight проверил
`3 991` asset bindings и нулевой overlap доступных sample/SHA/group/speaker/text keys. Final
generator families отсутствуют в train/dev/calibration.

## Calibration

TemperatureScaler fitted только на PyAra calibration v3 (`976`, `478/498`), который не участвовал
ни в обучении, ни в выборе epoch. Threshold не подбирался; решение осталось фиксированным на
calibrated probability `0.5`.

| Показатель | До | После |
| --- | ---: | ---: |
| NLL | 0.15976 | 0.15424 |
| Brier | 0.04955 | 0.04830 |
| ECE, 15 bins | 0.06444 | 0.06497 |

Температура: `1.29954`. ECE немного ухудшился, поэтому нельзя утверждать, что улучшены все
аспекты calibration.

## Раздельные результаты

| Layer | Assets / pairs | Accuracy | bona-fide recall | spoof recall | Balanced accuracy | Both-correct pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RU FLEURS/eSpeak NG | 150 / 75 | 0.9800 | 0.9600 | 1.0000 | 0.9800 | 72/75 |
| KK FLEURS/Silero V4 | 304 / 152 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 152/152 |
| Mixed KSC2/Silero V4 | 60 / 30 | 0.9333 | 0.9000 | 0.9667 | 0.9333 | 26/30 |

Никакая pooled RU+KK+mixed accuracy не рассчитывается. Full pair/sample logits находятся только
в write-once report; execution lock SHA-256 —
`7fb88feb8df5e890e744498036744ac827ae5ebb411dadaf86b0b8f58fca0998`, report SHA-256 —
`bf58ba3e9f08fb8cf09b33b2cebf5ecdb09fd9d5ed73415c16729a5de56eca77`. Inference занял
`17.525` с, peak allocated VRAM `1 645 827 072` bytes.

## Обязательные ограничения

- RU и mixed locked assets имеют по два acoustic review на каждый WAV. Это подтверждает только
  audibility/lexical preservation конкретных bytes, а не speaker independence.
- KK Silero layer не имеет завершённого двухрецензентного acoustic language gate. Его идеальные
  метрики нельзя расширять до окончательного утверждения о казахской речи или generalization.
- Mixed assets уже использовались в exploratory run checkpoint v1. Для checkpoint v2 они были
  holdout относительно train/dev/calibration, но не являются новым blind project-level test.
- FLEURS не публикует speaker IDs; RuASD/PyAra также не дают достаточного verified speaker
  provenance. Результат остаётся personal research и не включает API/model release.
