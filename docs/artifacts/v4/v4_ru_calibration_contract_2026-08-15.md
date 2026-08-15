# XLS-R+SLS model v4 — RU calibration contract v1

**Дата:** 15 августа 2026

## Решение

Новый immutable contract
[`xlsr_sls_model_v4_ru_calibration_v1`](../../../configs/research/v4/xlsr_sls_model_v4_ru_calibration_v1.json)
разрешает ровно один local CUDA/BF16 scoring pass и one-parameter temperature scaling только на
уже frozen `73` RU VoxForge/eSpeak exact-text pairs (`146` assets). Это отдельное узкое
research-only decision; immutable materialization ledger не меняется.

Новый two-source ledger
[`xlsr_sls_model_v4_calibration_v1.csv`](../../../data/licenses/frozen/xlsr_sls_model_v4_calibration_v1.csv)
имеет SHA-256 `853259916351b3ca8a4e2524c3aa1d0a7684dea461a5e48828e2cffa90f438d5`. Он опирается
на текущую VoxForge Russian card MDC (GPL-3.0, `Restrictions: N/A`, `Forbidden Usage: N/A`) и
оставляет обязательными Data Consumer Terms и applicable law. Разрешение ограничено personal
research; это не юридическое заключение и не разрешение product use.

Контракт hash-bind-ит:

- complete pair lock (`146` rows, `73` pairs), SHA-256
  `a8a367549f566222690ea199955e19b51315182fe329a30dc765e24edc5b5d71`;
- exact v4 training/materialization plans and receipts, plus frozen v4 train/dev manifests;
- selected tail-unfreeze epoch 2 state
  `3cfca24a3731d3f9e3c259dcea905be07aefc4fbf2fbefa98189696df01fbe4a` and ignored local
  checkpoint file SHA-256 `8be73165a4e6f65e966fa6d6a162fbb319d7089d1e8c1597c131e9ccb226852f`;
- local XLS-R 300M config/weights, CUDA/BF16 runtime and nine implementation files.

## Fail-closed controls

Runner [`run_v4_calibration.py`](../../../scripts/run_v4_calibration.py) refuses a changed byte,
an absent or non-`research_only` ledger entry, a non-complete pair, any overlap by sample/audio/
text/group with v4 train/dev, runtime mismatch, an altered checkpoint state or an output overwrite.
It first writes a no-logit preflight; execution then writes a lock before loading the checkpoint.

Only temperature scaling is permitted. Threshold selection, epoch/model/architecture/augmentation
selection, checkpoint mutation, pair replacement/backfill, detector feedback and final inference
are explicitly prohibited. The result, when produced, will be RU-only: no calibrated KK
probability and no verified speaker-independence claim are possible.

## Write-once no-logit preflight

`PYTHONPATH=src .venv/bin/python scripts/run_v4_calibration.py --plan
configs/research/v4/xlsr_sls_model_v4_ru_calibration_v1.json --audio-root data --validate-only`
успешно записал ignored receipt
`artifacts/v4/xlsr-sls-model-v4-ru-calibration-v1.preflight.json`, SHA-256
`b2e6b7dc81266618ec45e38cb4b009f07bf0d2021a18d9025b4032f617c30c5f`.

Он проверил все `146` hash-bound local WAV, complete pair structure (`73/73` bona-fide/spoof),
zero sample/audio/text/group overlap c v4 train/dev, оба research-only ledger entries, selected
checkpoint file SHA-256 и pinned CUDA/BF16 runtime (RTX 5060 Ti, `16,616,521,728` bytes). В этом
режиме checkpoint не загружался, calibration logits не вычислялись, temperature не fit-ился и
final inference не выполнялся.

## Текущее состояние

Preflight теперь навсегда открывает единственный calibration execution. Он загрузит только
hash-verified selected checkpoint, score-ит ровно 146 frozen RU assets и fit-ит единственный
temperature scalar; execution lock будет создан до первого logit. Нельзя повторять preflight,
менять assets/contract или запускать final inference.

## One-time calibration execution

Execution lock создан до загрузки checkpoint:
`artifacts/v4/xlsr-sls-model-v4-ru-calibration-v1.execution.json`, SHA-256
`d626816463dae52abc282a917b4eacb5a4f338c975d2c82ef0bd2e1b855db08e`.
Versioned machine report:
[`xlsr_sls_model_v4_ru_calibration_v1.json`](xlsr_sls_model_v4_ru_calibration_v1.json), SHA-256
`3a8bffe16f7aafad2713ace468f0cea23188bcdeb905864179a22505647d73c1`.

Он строго загрузил selected epoch-2 state, вычислил ровно один logit на каждом из `146` frozen
assets и fit-нул scalar `temperature = 0.7253568769` только на этих `73` complete RU pairs.
NLL уменьшился с `0.02308078` до `0.02073807`, ECE (15 bins) — с `0.01674657` до `0.01287263`,
но Brier вырос с `0.00675670` до `0.00736947`. Поэтому результат — смешанный calibration
diagnostic, не безусловное улучшение и не качество final/product модели.

Execution не выбирал threshold, epoch, checkpoint, архитектуру, augmentation или assets. Final
inference и detector feedback остаются `false`; scalar применим только если отдельный immutable
final contract явно hash-bind-ит этот report и не изменяет final set.
