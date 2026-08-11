# XLS-R+SLS Stage A v2

**Статус:** GPU run завершён 12 августа 2026.

Hash-pinned plan `configs/research/xlsr_sls_stage_a_v2.json` имеет SHA-256
`cc5de1383082eda290d07f90337022b95ea95a3c57fe5030255f14f5d0cb3ae7`. Он использует только
RuASD-v2 train (`1 471`) и исторический PyAra dev (`61`), а права закрепляет immutable ledger
snapshot `5ef01f6f648280e1eb6905be15a9921b78fc78d479d293c86a9f102234cc7477`.

Encoder XLS-R-300M был полностью frozen и находился в eval mode; обучался только SLS-head.
RTX 5060 Ti, CUDA/BF16, 3 эпохи, batch 16. Run занял `47.996` с, peak allocated VRAM —
`2 776 100 352` bytes.

| Выбранная epoch | dev loss | accuracy | balanced accuracy |
| ---: | ---: | ---: | ---: |
| 3 | 0.20181 | 0.90164 | 0.89945 |

Checkpoint SHA-256 — `a97673126552ae8b443dadcec1852c88b2e93ad62837ebaddcdb38c3a10e2810`,
selected head state — `bb50295ac34cf45caae57526422b9dccae8e329e31fbdc5d6aa9cebc15d485b1`,
report SHA-256 — `50383418f284228bec52ea74cf590e790b0ad0090767e53ee902e5497b785708`.
Final inference и calibration в Stage A не выполнялись.
