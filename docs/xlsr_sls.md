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
