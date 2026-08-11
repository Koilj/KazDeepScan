# XLS-R + SLS

`XlsrSlsClassifier` combines an XLS-R encoder with:

1. softmax-взвешенной смесью всех SSL hidden states;
2. masked attentive statistics pooling (mean + standard deviation);
3. classifier, returning **one raw spoof logit per 4.04-секундному окну**.

Модель не делает `sigmoid`, не агрегирует окна и не калибрует вероятность. Эти операции
должны происходить отдельно на record-level dev set. При padding маска сэмплов преобразуется
в feature-mask согласно `conv_kernel` и `conv_stride` энкодера, поэтому padding не попадает в
pooling.

Веса `facebook/wav2vec2-xls-r-300m` имеют Apache-2.0 и требуют 16 кГц вход. Перед train
необходимо выполнить local forward smoke test после загрузки конкретной ревизии модели и
зафиксировать её commit hash в model card.

```bash
hf download facebook/wav2vec2-xls-r-300m --local-dir models/xlsr-300m
uv run python scripts/smoke_xlsr_sls.py --model-dir models/xlsr-300m --device cuda
uv run python scripts/smoke_xlsr_sls.py --model-dir models/xlsr-300m --device cuda --precision bf16
```

## Stage A

Stage A обучает только SLS-head, удерживает encoder frozen в eval mode и требует CUDA/BF16.
Строгий JSON-plan фиксирует SHA-256 train/dev manifests, license ledger, локальные XLS-R config
и weights, архитектуру head, optimizer parameters и output paths. Runner не принимает final
manifest, отказывается перезаписывать результаты и сохраняет только head state.

```bash
uv run python scripts/train_xlsr_sls_stage_a.py \
  --plan configs/research/xlsr_sls_stage_a_v1.json \
  --audio-root data --validate-only

uv run python scripts/train_xlsr_sls_stage_a.py \
  --plan configs/research/xlsr_sls_stage_a_v1.json \
  --audio-root data --profile-only
```

Stage A v1 уже выполнен и не перезапускается. Его протокол, CUDA profile, train/dev metrics и
artifact hashes записаны в [research_xlsr_sls_stage_a_v1.md](research_xlsr_sls_stage_a_v1.md).
Stage A v2 на исправленном RuASD-v2 также завершён: epoch 3, dev balanced accuracy `0.89945`.
Точный receipt — [research_xlsr_sls_stage_a_v2.md](research_xlsr_sls_stage_a_v2.md).

## Stage B

Stage B инициализирует SLS-head только проверенным Stage A state, оставляет blocks `0`–`15`
frozen и размораживает последние восемь XLS-R blocks `16`–`23`. Для них используется LR
`1e-5`, для head — `1e-4`; gradient checkpointing и BF16 обязательны. Physical batch `4`
накапливается восемь шагов до effective batch `32` с нормализацией по фактическому числу
examples.

Отдельный plan фиксирует Stage A plan/checkpoint/state receipts, свежий PyAra dev manifest,
train/dev SHA-256, CUDA параметры и output paths. Validator требует нулевое пересечение нового
dev со старым Stage A dev и фиксированным RuASD train. Runner не принимает final manifests,
отказывается перезаписывать артефакты и сохраняет только SLS-head и размороженный encoder tail.

```bash
uv run python scripts/train_xlsr_sls_stage_b.py \
  --plan configs/research/xlsr_sls_stage_b_v2.json \
  --audio-root data --validate-only

uv run python scripts/train_xlsr_sls_stage_b.py \
  --plan configs/research/xlsr_sls_stage_b_v2.json \
  --audio-root data --profile-only
```

Stage B v1 уже выполнен и не перезапускается. Выбран epoch 7 с fresh PyAra dev loss `0.15236`,
accuracy `0.9381` и balanced accuracy `0.9393`; в самом Stage-B train/dev run frozen final
inference и calibration не выполнялись. Полный протокол, CUDA profile и artifact hashes записаны
в [research_xlsr_sls_stage_b_v1.md](research_xlsr_sls_stage_b_v1.md).

Stage B v2 обучен на RuASD-v2 train и PyAra dev v3: по заранее зафиксированному minimum dev loss
выбран epoch 5 (`0.17405`), balanced accuracy `0.92287`. Детали и hashes —
[research_xlsr_sls_stage_b_v2.md](research_xlsr_sls_stage_b_v2.md). После отдельного immutable
preflight ровно один раз выполнены temperature scaling на disjoint calibration v3 и confirmatory
RU/KK/mixed evaluation. Результаты не pooled и не product quality; mixed слой ранее раскрыт v1,
а KK не имеет two-review acoustic gate. Полный receipt —
[research_xlsr_sls_stage_b_v2_research_final_v1.md](research_xlsr_sls_stage_b_v2_research_final_v1.md).

Позднее единожды выполнен **не final** exploratory stress-test на 30 input-pinned KSC2 mixed
pairs. Он имеет отдельный immutable plan и execution lock, не меняет Stage-B weights/calibration
и не допускает quality claim. См. [полный pair-level report](research_xlsr_sls_stage_b_ksc2_mixed_exploratory_30.md).
