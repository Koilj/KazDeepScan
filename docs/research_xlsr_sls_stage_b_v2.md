# XLS-R+SLS Stage B v2

**Статус:** GPU run завершён 12 августа 2026.

Plan SHA-256 — `edec0e98fabf9af4ebb4526005d03e5af2b1fcae3e5854bc3a6e5c6d83f8d8d5`.
Stage B инициализирован head state Stage A v2, оставил XLS-R blocks 0–15 frozen и обучал blocks
16–23 вместе с head. Train — RuASD-v2 `1 471`; новый PyAra dev v3 — `969` (`474/495`).
Один text group из прежнего candidate был исключён из-за overlap с новым RuASD-v2 train.

RTX 5060 Ti, CUDA/BF16, 15 эпох, physical batch 4, gradient accumulation 8. Run занял
`583.013` с; peak allocated VRAM `2 928 268 800` bytes. Контракт заранее выбирал минимум
`dev_loss`, поэтому выбран epoch 5, хотя отдельная более поздняя epoch имела выше accuracy.

| Выбранная epoch | dev loss | accuracy | bona-fide recall | spoof recall | balanced accuracy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0.17405 | 0.92157 | 0.98312 | 0.86263 | 0.92287 |

Checkpoint SHA-256 — `e112c5c93f2a5af0c567b85eccac0a617c37fa79b4d7cc2b29b4b3289f2764cd`;
selected trainable state — `d03adfe2ebfe7b7361b2a0d9b7902ef7251f7faf139d37a30531e2211e2dd738`;
report SHA-256 — `cca5bf61976f07207d15976497b030420c274c6c1c5a194c4d37db2728f4c651`.
Сам Stage-B train/dev run не выполнял calibration или final inference.
