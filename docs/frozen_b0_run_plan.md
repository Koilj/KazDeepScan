# Frozen B0 run plan

## Зачем нужен отдельный plan

`unseen_generator_ood_v1.json` фиксирует состав train/dev/final и проверяет отсутствие
source/text/asset leakage. Этого недостаточно для нового честного запуска: seed, архитектура,
длина окна и training hyperparameters могли бы быть выбраны после просмотра final results.

`scripts/train_b0_unseen_suite.py` поэтому не принимает эти параметры из командной строки. Он
читает строгий versioned run-plan, который заранее закрепляет:

- SHA-256 suite, license ledger и каждого manifest;
- полную конфигурацию B0, seed, число эпох, batch size, окно, optimizer parameters и device;
- заранее выбранные пути checkpoint и JSON report.

Runner проверяет все manifests, права и SHA-256 аудио до fitting. Epoch выбирается только по dev
loss. После этого state dict хешируется, и каждый final test читается ровно один раз; aggregate,
class, generator-family и voice strata собираются в одном проходе. Checkpoint и report сначала
полностью записываются во временные файлы и только затем атомарно публикуются без перезаписи.
Существующий output блокирует повторный запуск.

## Строгая схема v1

Все пути относительны каталогу plan-файла. Ниже шаблон, а не готовое разрешение на запуск:

```json
{
  "schema_version": 1,
  "run_id": "frozen-unseen-generator-suite-v2-b0-seed-20260818",
  "purpose": "research",
  "suite": {
    "path": "unseen_generator_ood_v2.json",
    "sha256": "SUITE_SHA256"
  },
  "license_ledger": {
    "path": "../../data/licenses/license_ledger.csv",
    "sha256": "LICENSE_LEDGER_SHA256"
  },
  "manifests": [
    {"path": "../../data/manifests/train.csv", "sha256": "TRAIN_MANIFEST_SHA256"},
    {"path": "../../data/manifests/dev.csv", "sha256": "DEV_MANIFEST_SHA256"},
    {"path": "../../data/manifests/final_a.csv", "sha256": "FINAL_A_SHA256"},
    {"path": "../../data/manifests/final_b.csv", "sha256": "FINAL_B_SHA256"},
    {"path": "../../data/manifests/final_c.csv", "sha256": "FINAL_C_SHA256"}
  ],
  "model": {
    "name": "b0_logmel_cnn",
    "config": {
      "sample_rate": 16000,
      "n_fft": 512,
      "hop_length": 160,
      "n_mels": 80,
      "dropout": 0.2
    }
  },
  "training": {
    "seed": 20260818,
    "epochs": 5,
    "batch_size": 16,
    "window_samples": 64600,
    "learning_rate": 0.0001,
    "weight_decay": 0.0001,
    "num_workers": 0,
    "device": "cuda"
  },
  "outputs": {
    "checkpoint": "../../models/b0-unseen-generator-suite-v2.pt",
    "report": "../../artifacts/b0-unseen-generator-suite-v2.json"
  }
}
```

Не добавлять в `manifests` посторонние CSV: runner требует точное совпадение с manifests,
на которые ссылается suite. Любое изменение закреплённого файла после создания plan приводит к
SHA-256 error до чтения модели.

## Порядок нового запуска

1. Создать новый suite с новыми, ранее не раскрытыми final assets. Suite v1 уже использован и
   не должен запускаться повторно.
2. До оценки зафиксировать suite, manifests, ledger, run-plan и реализацию в Git.
3. Создать каталоги, заранее указанные в `outputs`, если их ещё нет.
4. Выполнить безопасный preflight без fitting и final inference:

   ```bash
   uv run python scripts/train_b0_unseen_suite.py \
     --plan configs/research/unseen_generator_b0_run_v2.json \
     --audio-root data --validate-only
   ```

5. После проверки plan один раз выполнить ту же команду без `--validate-only`. Не менять plan,
   suite, manifests, threshold, calibration или architecture по результатам final.

JSON report является execution receipt. Его и checkpoint нельзя удалять ради повторного
запуска того же protocol. Результаты остаются personal-research metrics, а не product score.
