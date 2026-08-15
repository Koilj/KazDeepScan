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

## Capacity profile

Один `--profile-only` tail-unfreeze train batch и по одному non-selecting RU/KK dev batch
успешно прошли на восьми из 24 XLS-R blocks. Elapsed time `0.9352` s; peak GPU memory —
`2 910 625 792` allocated и `3 288 334 336` reserved bytes. Profile не публиковал artifacts,
не выбирал checkpoint и не менял заранее зафиксированные hyperparameters; его four-example
diagnostics не являются evaluation result.

После profile был выполнен единственный write-once training run, описанный ниже. Calibration и
final оставались запрещены.

## Write-once training execution

`xlsr-sls-model-v4-train-v1` завершился успешно за `1 869.94` s. Один warm-up epoch и три
tail-unfreeze epochs выполнены exactly as pinned. Macro RU/KK dev-loss выбрал tail-unfreeze epoch
2: `0.0841422787` (`RU=0.1241607735`, `KK=0.0441237839`); epochs 1 и 3 дали `0.1507765775` и
`0.1325452515`. Это selection metric на tuning dev, а не final quality claim.

Selected model state SHA-256: `3cfca24a3731d3f9e3c259dcea905be07aefc4fbf2fbefa98189696df01fbe4a`.
Ignored local checkpoint file SHA-256: `8be73165a4e6f65e966fa6d6a162fbb319d7089d1e8c1597c131e9ccb226852f`.
Versioned machine report:
[xlsr_sls_model_v4_training_v1.json](xlsr_sls_model_v4_training_v1.json), SHA-256
`e4ded7e9ebdac93249991a00511b8b788a3e8ca2fe8aaf9637c71be4773a22f9`.

The report verifies the exact plan, all `21 917` assets, zero train/dev sample/audio/text/group
overlap, CUDA/BF16 environment and matching checkpoint state hash. It records no calibration,
no final evaluation and no final inference. The selected dev balanced accuracies (`RU=0.9620`,
`KK=0.9831`) are diagnostics on a checkpoint-selected development role only and must not be
reported as independent performance.

Metadata-only calibration-input и materialization/audio-isolation gates завершены позднее отдельными
receipts. Write-once no-logit preflight и one calibration execution нового
[RU calibration contract](v4_ru_calibration_contract_2026-08-15.md) завершены. Не запускать
final inference или повтор training; следующий этап требует отдельный immutable final contract.
