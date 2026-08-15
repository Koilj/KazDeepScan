# XLS-R+SLS model v4 — подготовка isolated dev inputs

Статус на 15 августа 2026: подготовлен, но ещё не исполнен write-once contract
[`xlsr_sls_model_v4_dev_inputs_v1.json`](../../configs/research/v4/xlsr_sls_model_v4_dev_inputs_v1.json).
Он не является training contract и не разрешает training, checkpoint selection, calibration или
final inference.

## Зафиксированный состав

- RU dev — immutable historical PyAra manifest из `969` ready rows: `474` bona-fide и `495`
  spoof. Проверка против frozen v4 train дала ноль общих sample ID, audio SHA-256, text hash и
  parent-group ID.
- KK dev — `600` заранее ранжированных кандидатов только из оригинального KSC SLR102 `dev`
  split; первый freeze target — `474` полностью QA-ready KSC/Silero V4 pairs, а `126` пар
  образуют единственный разрешённый reserve. Число `474` ограничено меньшей RU bona-fide cell,
  поэтому оно не выбрано по model output.
- KK spoof — только fixed `kz_M1`, `kz_M2`, `kz_F1`, `kz_F2`, `kz_F3` Silero V4 profiles, с
  text-only KSC input; random profile, `voice_path`, reference audio и cloning запрещены.
- Каждый новый raw asset канонизируется в mono PCM-16 WAV 16 kHz, проходит technical QA/VAD,
  exact/near historical audio screen и within-pool screen. Near hit остаётся не eligible,
  без ручного detector-based выбора.

Контракт hash-pins PyAra dev, combined 20k train, frozen minimal ledger, Silero lock, KSC/Silero
implementation, runner и existing historical fingerprint inventory. Source KSC archive также
привязан к exact size и SHA-256 из ledger. Новые raw/processed assets и runtime journals остаются
Git-ignored.

## Запуск после preflight

```bash
PYTHONPATH=src .venv/bin/python scripts/build_v4_isolated_dev_inputs.py \
  --archive /home/ruslan/Downloads/ISSAI_KSC_335RS_v1.1_flac.tar.gz \
  --project-root . --data-root data --device cuda
```

До запуска нужно убедиться, что CUDA доступна, output paths и slice namespace отсутствуют, а
working tree содержит именно hash-pinned runner. При любом hash mismatch, existing output,
insufficient pair count или audio-leakage hit runner завершается без publish manifests/receipt;
resynthesis, external backfill и training не разрешаются.
