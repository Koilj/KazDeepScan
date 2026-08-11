# XLS-R+SLS Stage B — KSC2 mixed exploratory stress-test (30 pairs)

**Статус:** завершён 11 августа 2026 года. Это единственный write-once
**exploratory mixed stress-test**; это **не final quality**, не calibration и не product
evaluation.

## Зафиксированный запуск

- run-plan: [`xlsr_sls_stage_b_v1_ksc2_mixed_exploratory_30.json`](../configs/research/xlsr_sls_stage_b_v1_ksc2_mixed_exploratory_30.json), SHA-256 `4355cd3a8dd361ae5aebbceb9b4fc40c2d77ffc7a7a34142c947179545700070`;
- frozen checkpoint: `models/xlsr-sls-stage-b-v1.pt`, SHA-256
  `18c967a8881404140ccda04fc6234079ac4b2802425e4111f3fef59bef505c32`;
- canonical Stage-B state: `59ad0812e14d33abec00ba5225876de4c208efa9c8f8f9061e253e60df9d1089`;
- 30-pair candidate manifest: SHA-256
  `dafa33d424da8efab19d849afeeeb11c279d2e6c10ff187f3a57cada76c6c4a8`;
- pair lock: SHA-256 `9e46e56ebc082620f9317cf41e93783a2d2aab691306bb9bf45b309954f5b7bf`;
- execution lock: [`artifacts/...execution.json`](../artifacts/xlsr-sls-stage-b-v1-ksc2-mixed-exploratory-30.execution.json), SHA-256
  `effb371f9e568d1b21abd3390b3cad419a567624dd23162f1c1e4f01a9bee526`;
- complete machine-readable result: [`artifacts/...report.json`](../artifacts/xlsr-sls-stage-b-v1-ksc2-mixed-exploratory-30.report.json), SHA-256
  `e1ab67f5d075ade907bd7c067cf5afe31718394677b2c3571415faef6f7ecd0b`.

Preflight повторно сверил план, checkpoint, Stage-B receipt, XLS-R weights/config,
license ledger, implementation files, all 60 asset SHA-256, candidate manifest и pair-lock.
Inference использовал ровно одно фиксированное eval window в 4.04 секунды на record. До первого
model forward опубликован execution lock; его существование блокирует повтор этого run-plan.

Ни обучение, ни изменение весов, ни temperature fit, ни выбор threshold не выполнялись.
Для class recall показана только исходная classifier boundary `raw_logit >= 0.0` — это фиксированное
правило checkpoint, не выбранный operating point и не probability.

## Результат на raw boundary

| Metric | Result |
| --- | ---: |
| Records / pairs | `60 / 30` |
| Raw-decision accuracy | `58/60 = 96.67%` (95% Wilson `88.64–99.08%`) |
| Bona-fide recall | `29/30 = 96.67%` (95% Wilson `83.33–99.41%`) |
| Spoof recall | `29/30 = 96.67%` (95% Wilson `83.33–99.41%`) |
| Balanced recall | `96.67%` |
| Both records correct | `28/30` pairs |
| Exactly one record correct | `2/30` pairs |
| Neither record correct | `0/30` pairs |

`01_04_072` дал false positive на bona-fide (`+1.3047`), а synthetic counterpart of
`09_01_093` дал false negative (`−1.4844`). Это наблюдения данного узкого набора, а не повод
изменять boundary, calibration, architecture или набор.

## Все pair-level результаты

Логиты — некалиброванные raw logits. Формат ячейки: `logit / raw prediction / correctness`.

| # | KSC2 annotation | Component | Bona-fide | Spoof |
| ---: | --- | --- | --- | --- |
| 1 | `ksc2_v1:Test/podcasts/09_04_099` | podcasts | -8.1875 / bonafide / ✓ | 12.875 / spoof / ✓ |
| 2 | `ksc2_v1:Test/podcasts/09_04_024` | podcasts | -0.16016 / bonafide / ✓ | 14.812 / spoof / ✓ |
| 3 | `ksc2_v1:Test/radio/01_05_194` | radio | -2.7656 / bonafide / ✓ | 16.75 / spoof / ✓ |
| 4 | `ksc2_v1:Test/radio/01_04_072` | radio | 1.3047 / spoof / ✗ | 12.125 / spoof / ✓ |
| 5 | `ksc2_v1:Test/talkshow/01_02_093` | talkshow | -9.5625 / bonafide / ✓ | 11.25 / spoof / ✓ |
| 6 | `ksc2_v1:Test/radio/01_05_157` | radio | -1.1719 / bonafide / ✓ | 13.938 / spoof / ✓ |
| 7 | `ksc2_v1:Test/podcasts/09_01_292` | podcasts | -2.8281 / bonafide / ✓ | 12.812 / spoof / ✓ |
| 8 | `ksc2_v1:Test/podcasts/09_03_020` | podcasts | -1.9844 / bonafide / ✓ | 13.312 / spoof / ✓ |
| 9 | `ksc2_v1:Test/talkshow/01_02_274` | talkshow | -9.8125 / bonafide / ✓ | 13.688 / spoof / ✓ |
| 10 | `ksc2_v1:Test/podcasts/09_03_220` | podcasts | -3.25 / bonafide / ✓ | 10.875 / spoof / ✓ |
| 11 | `ksc2_v1:Test/podcasts/09_01_216` | podcasts | -4.0312 / bonafide / ✓ | 10.188 / spoof / ✓ |
| 12 | `ksc2_v1:Test/podcasts/09_01_095` | podcasts | -6.75 / bonafide / ✓ | 14.062 / spoof / ✓ |
| 13 | `ksc2_v1:Test/talkshow/01_02_109` | talkshow | -3.6406 / bonafide / ✓ | 11.25 / spoof / ✓ |
| 14 | `ksc2_v1:Test/podcasts/09_04_212` | podcasts | -5.6562 / bonafide / ✓ | 11.688 / spoof / ✓ |
| 15 | `ksc2_v1:Test/podcasts/09_03_251` | podcasts | -1.4219 / bonafide / ✓ | 11.625 / spoof / ✓ |
| 16 | `ksc2_v1:Test/talkshow/01_02_110` | talkshow | -8.8125 / bonafide / ✓ | 11 / spoof / ✓ |
| 17 | `ksc2_v1:Test/talkshow/01_02_277` | talkshow | -10.562 / bonafide / ✓ | 9.5 / spoof / ✓ |
| 18 | `ksc2_v1:Test/podcasts/09_04_072` | podcasts | -2.7812 / bonafide / ✓ | 13.938 / spoof / ✓ |
| 19 | `ksc2_v1:Test/podcasts/09_01_093` | podcasts | -12.062 / bonafide / ✓ | -1.4844 / bonafide / ✗ |
| 20 | `ksc2_v1:Test/podcasts/09_04_051` | podcasts | -7.4688 / bonafide / ✓ | 13.562 / spoof / ✓ |
| 21 | `ksc2_v1:Test/podcasts/09_03_204` | podcasts | -2.5781 / bonafide / ✓ | 11.812 / spoof / ✓ |
| 22 | `ksc2_v1:Test/talkshow/01_02_067` | talkshow | -0.60156 / bonafide / ✓ | 8 / spoof / ✓ |
| 23 | `ksc2_v1:Test/podcasts/09_01_251` | podcasts | -9.4375 / bonafide / ✓ | 11.812 / spoof / ✓ |
| 24 | `ksc2_v1:Test/radio/01_05_274` | radio | -7.6562 / bonafide / ✓ | 14 / spoof / ✓ |
| 25 | `ksc2_v1:Test/podcasts/09_04_074` | podcasts | -3.9062 / bonafide / ✓ | 11.25 / spoof / ✓ |
| 26 | `ksc2_v1:Test/podcasts/09_01_149` | podcasts | -11.188 / bonafide / ✓ | 14.938 / spoof / ✓ |
| 27 | `ksc2_v1:Test/podcasts/09_03_289` | podcasts | -3.7969 / bonafide / ✓ | 13.062 / spoof / ✓ |
| 28 | `ksc2_v1:Test/podcasts/09_01_151` | podcasts | -7.0312 / bonafide / ✓ | 12.312 / spoof / ✓ |
| 29 | `ksc2_v1:Test/podcasts/09_01_150` | podcasts | -7.5625 / bonafide / ✓ | 6.5938 / spoof / ✓ |
| 30 | `ksc2_v1:Test/podcasts/09_03_287` | podcasts | -2.4062 / bonafide / ✓ | 13.125 / spoof / ✓ |

Full result additionally contains two audio SHA-256 values and the explicit RU/KK token evidence
for every pair.

## Непреодолимые ограничения этого stress-test

1. Synthetic waveform inherited only the intended mixed input transcript. Signal QA/VAD и pair-lock
   не проверяют acoustic preservation русских и казахских segments.
2. Все synthetic records происходят от одного fixed Silero `kz_M1` profile; это не independent
   generator/voice generalization estimate.
3. KSC2 rows have `speaker_pseudo_id=unknown`; speaker-disjointness не заявляется.
4. Всего 30 пар. Wilson intervals описывают только этот конечный набор, а не ожидаемое качество
   в production.

Следствие: результат нельзя объединять с RU/KK, использовать для выбора threshold/temperature,
калибровать API или называть final quality. Для product/final по-прежнему требуется независимый
RU spoof layer, fully specified RU/KK/mixed binary protocol и acoustic language-preservation gate.
