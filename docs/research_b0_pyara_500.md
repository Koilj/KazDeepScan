# B0 research baseline: PyAra 500

Этот документ фиксирует диагностический результат, а не model release. Все manifests,
лицензионные ограничения и отсутствие speaker-disjoint guarantee описаны в
`docs/data_sources_pyara_ru_v7.md`.

## Training

- Checkpoint: `models/b0-pyara-research-500.pt` (не находится в Git).
- Model: `B0LogMelCnn`; GPU: NVIDIA GeForce RTX 5060 Ti.
- Source: `pyara_ru_v7_research_500_ready.csv`, `481` ready WAV.
- 3 epochs, batch size 16, seed `20260809`; best dev loss: `0.5084`.

## Некалиброванные результаты

| Protocol | Rows (bona-fide / spoof) | Loss | Accuracy |
| --- | ---: | ---: | ---: |
| PyAra local test | 44 (22 / 22) | 0.4053 | 0.8864 |
| ML-DF Italian OOD | 192 (94 / 98) | 0.7340 | 0.5208 |
| RuASD Russian fake-only OOD | 100 (0 / 100) | 0.7922 | 0.3800 |

RuASD accuracy — это только доля correctly predicted fake rows при default decision boundary;
в нём нет bona-fide rows, поэтому EER, FPR и калибровку из него выводить нельзя. ML-DF OOD
содержит оба класса, но другой язык/канал и лишь два source speaker pseudo-ID. Не строить
из этой таблицы product threshold, risk probability, public benchmark или claim о
детектировании deepfake в реальной среде.

## Воспроизводимая evaluation

```bash
uv run python scripts/evaluate_b0.py \
  --checkpoint models/b0-pyara-research-500.pt \
  --manifest data/manifests/ml_df_it_v1_ood_200_ready.csv \
  --audio-root data \
  --license-ledger data/licenses/license_ledger.csv \
  --split ood --device cuda
```

CLI проверяет schema, ledger и SHA-256 assets перед тем, как загрузить checkpoint. Он возвращает
только loss/accuracy и явно помечает result как `calibrated=false`.
