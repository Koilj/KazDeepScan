# XLS-R+SLS model v4 — full training contract v1

**Дата:** 15 августа 2026

## Зафиксированный protocol

[Контракт](../../../configs/research/v4/xlsr_sls_model_v4_training_v1.json) определяет один
research-only run `xlsr-sls-model-v4-train-v1`. Он hash-pins combined train (`20 000` строк),
isolated bilingual dev (`1 917` строк), frozen eight-source license ledger, local XLS-R 300M
config/weights, runtime и десять implementation files. Checkpoint пишется только в ignored
`checkpoints/xlsr-sls-model-v4/xlsr-sls-model-v4-train-v1/model.pt`; versioned report —
`docs/artifacts/v4/xlsr_sls_model_v4_training_v1.json`; local execution lock запрещает repeat и
overwrite.

Train содержит ровно по `5 000` строк для каждого `RU/KK × bona-fide/spoof` cell. Dev содержит
RU `474/495` и KK `474/474` bona-fide/spoof строк. На preflight все intersections train/dev по
`sample_id`, asset SHA-256, `text_hash` и `parent_group_id` равны нулю; source lineages также
disjoint. Calibration, final paths, final evaluation, detector feedback, network downloads и
output overwrite запрещены самим контрактом.

Протокол: один head warm-up epoch на frozen XLS-R, затем три tail-unfreeze epochs последних
восьми XLS-R blocks с BF16, batch `4`, accumulation `8`, deterministic seed `20260815` и
label/language-balanced sampler без duplicate padding. Checkpoint выбирается только по
`macro_language_dev_loss_ru_kk`; RU/KK balanced accuracy остаётся diagnostic. Calibration и
final не выполняются.

## No-training preflight

`PYTHONPATH=src .venv/bin/python scripts/train_v4_xlsr_sls.py --plan
configs/research/v4/xlsr_sls_model_v4_training_v1.json --audio-root data --validate-only`
завершился успешно. Он проверил SHA-256 всех `21 917` selected local assets, manifests, ledger,
XLS-R config/weights, implementation hashes, CUDA/BF16 и runtime lock на RTX 5060 Ti
(`16 616 521 728` bytes). Ни forward pass, ни checkpoint, ни calibration, ни final inference не
выполнялись; execution lock и report не созданы.

Следующий безопасный шаг — один `--profile-only` tail-unfreeze batch без публикации artifacts,
а затем единственный write-once training run только по этому неизменяемому контракту.
