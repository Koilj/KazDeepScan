# XLS-R+SLS model v4 — подготовка isolated dev inputs

Статус на 15 августа 2026: write-once contract
[`xlsr_sls_model_v4_dev_inputs_v1.json`](../../configs/research/v4/xlsr_sls_model_v4_dev_inputs_v1.json)
исполнен один раз на CUDA. Machine receipt:
[`xlsr_sls_model_v4_dev_inputs_v1.json`](xlsr_sls_model_v4_dev_inputs_v1.json), SHA-256
`62d946bf3dd069b77c4c519bea9f8c8bf8b513176796f0841470e3a8f170ab0e`.
Это не training contract и он не разрешает training, checkpoint selection, calibration или final
inference.

## Зафиксированный состав

- RU dev — immutable historical PyAra manifest из `969` ready rows: `474` bona-fide и `495`
  spoof. Проверка против frozen v4 train дала ноль общих sample ID, audio SHA-256, text hash и
  parent-group ID.
- KK dev — `600` заранее ранжированных кандидатов только из оригинального KSC SLR102 `dev`
  split. `571` source rows прошли QA; на них создано `571` Silero V4 raw WAV, из которых `535`
  прошли QA. Первые `474` полностью QA-ready KSC/Silero V4 pairs заморожены; `126` пар были
  единственным объявленным reserve. Число `474` ограничено меньшей RU bona-fide cell, поэтому
  оно не выбрано по model output.
- KK spoof — только fixed `kz_M1`, `kz_M2`, `kz_F1`, `kz_F2`, `kz_F3` Silero V4 profiles, с
  text-only KSC input; random profile, `voice_path`, reference audio и cloning запрещены.
- Каждый новый raw asset канонизируется в mono PCM-16 WAV 16 kHz, проходит technical QA/VAD,
  exact/near historical audio screen и within-pool screen. Near hit остаётся не eligible,
  без ручного detector-based выбора.

Контракт hash-pins PyAra dev, combined 20k train, frozen minimal ledger, Silero lock, KSC/Silero
implementation, runner и existing historical fingerprint inventory. Source KSC archive также
привязан к exact size и SHA-256 из ledger. Один source text был отсеян до extraction как
несовместимый с text contract; все `29` source и `36` spoof QA rejects имеют только причину
`insufficient_speech`. Historical/within-pool exact и near-audio collisions отсутствуют. Новые
raw/processed assets и runtime journals остаются Git-ignored.

## Результаты

| Output | Rows | SHA-256 |
| --- | ---: | --- |
| source raw | 600 | `43aa48675517fd4aa611a0fe9d2788162ad3f1a2603e21597ffaaee0067bde92` |
| source ready | 571 | `7c8321c73692455918fd8f17dd28fdd0e0a96fb18620ebce369e0f2918177d87` |
| Silero raw | 571 | `78b04675dafd5542d90f7d476adcb4995bcdc98141e913a9d9a2f6e40e5fb8ea` |
| Silero ready | 535 | `4d4a1f9e11f0412e2f8c663da3bc5468c7936c9d705a50bee52ada0a27db6e20` |
| frozen KK pairs | 948 | `d2cf0c63a8b873f2a9a3b89c637878f7778c68286439881a51929d56fb7a26a3` |
| combined dev | 1,917 | `30cd8808502770b9d996e7f54e28da78f25c9d107822b217ffa8b3a8e29ab554` |

Combined dev содержит immutable PyAra `969` rows и `474 × 2` KK pairs. Проверка против frozen
20k train — ноль общих sample ID, audio SHA-256, text hash и parent-group ID. Это всё ещё не
speaker-independence claim: PyAra и KSC не дают проверенных speaker groups.

## Исполненная команда

```bash
PYTHONPATH=src .venv/bin/python scripts/build_v4_isolated_dev_inputs.py \
  --archive /home/ruslan/Downloads/ISSAI_KSC_335RS_v1.1_flac.tar.gz \
  --project-root . --data-root data --device cuda
```

CUDA preflight подтвердил RTX 5060 Ti, exact archive size, verified Silero bundle и отсутствие
всех output paths. Повторный запуск, resynthesis, external backfill и training не разрешаются.
Следующий безопасный шаг — отдельный full training contract; он должен hash-pin combined train и
этот combined dev manifest до любого training execution.
