# XLS-R+SLS Stage-C — asset-level-blind RU/KK/mixed research evaluation

**Статус:** однократный GPU inference run завершён 13 августа 2026 года.

**Scope:** personal research only. Это не product quality, не source-independent и не
speaker-independent result. Новый suite содержит exact assets, ранее не получавшие inference в
configured project roles, но его bona-fide sources уже использовались в проекте, а фиксированный
Male2 alias пересекается с 312 historical Piper rows через другой exact runtime route.

## До запуска

Full-asset acoustic gate получил две независимые полные формы: `334` решений для всех `167`
synthetic WAV, каждое `pass/yes/yes/yes/no`. Gate receipt SHA-256:
`9a12f235072ce5ae4c3bd6bb0616a804a710abea74f3c3b387cebe12baf8153c`.

Immutable plan
`configs/research/xlsr_sls_stage_b_v2_fresh_suite_stage_c_v1.json` имеет SHA-256
`bd6a5d5ccac63b10790c624d34f3aa3a9a429b81001fc5b4d2da70ba9840ad7a`. Его validate-only
preflight проверил `1 310` exact asset bindings: `976` calibration, `100 RU`, `120 KK` и
`114 mixed` final rows. Candidate exposure audit закрепляет zero overlap по `sample_id`, audio
SHA-256 и `text_hash` среди 15 prior manifests / 11 869 rows.

Execution lock был опубликован до первого final logit. Его SHA-256:
`7d692542e6397c64ac0379eca34564d16f1fd670d5deb93e7a9a940695c2ccdb`.
Повтор этого run запрещён write-once outputs.

## Calibration

Temperature scaling повторно вычислен только на pinned PyAra calibration role (`976` rows) до
чтения final assets. Threshold не подбирался: boundary остаётся calibrated probability `0.5`.

| Показатель | До | После |
| --- | ---: | ---: |
| Temperature | — | 1.29954 |
| NLL | 0.15976 | 0.15424 |
| Brier | 0.04955 | 0.04830 |
| ECE, 15 bins | 0.06444 | 0.06497 |

ECE немного вырос, поэтому temperature scaling не интерпретируется как однозначное улучшение
всей calibration.

## Раздельные final результаты

Все интервалы — Wilson 95% CI для соответствующей доли. Общий pooled RU+KK+mixed результат
намеренно не рассчитывался.

| Роль | Assets / pairs | Accuracy (95% CI) | Bona-fide recall | Spoof recall | Balanced accuracy | Оба assets пары верны |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RU | 100 / 50 | 0.9000 [0.826, 0.945] | 50/50 (1.0000) | 40/50 (0.8000) | 0.9000 | 40/50 (0.8000) |
| KK | 120 / 60 | 0.9500 [0.895, 0.977] | 60/60 (1.0000) | 54/60 (0.9000) | 0.9500 | 54/60 (0.9000) |
| mixed | 114 / 57 | 0.8070 [0.725, 0.869] | 42/57 (0.7368) | 50/57 (0.8772) | 0.8070 | 38/57 (0.6667) |

Каждый из `334` final assets получил ровно один logit: `167` bona-fide и `167` spoof, duplicate
`sample_id=0`. Inference занял `14.454` s; peak allocated VRAM — `1 645 827 072` bytes.

## Ограничения и дальнейшее решение

- Это новый **asset-level-blind** suite для проекта, а не новый independent source/speaker test.
- Exact checkpoint/runtime route KazakhTTS отсутствовал в prior spoof manifests; overlap voice
  alias `312` historical Piper rows раскрыт и остаётся ограничением.
- Не изменять XLS-R checkpoint, threshold, temperature, architecture или training recipe по
  ошибкам этого final suite. Новый model v3 можно рассматривать только по train/dev/calibration
  с отдельным заранее замороженным contract и ранее нераскрытыми test assets.
- Research checkpoint по-прежнему не подключается к API risk score.

Write-once report SHA-256:
`2cb7198a6ec03f2c6424748dde3263731d87ff6b0b557f59b0463b7f74cc5e32`.
