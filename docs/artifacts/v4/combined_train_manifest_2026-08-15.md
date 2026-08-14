# XLS-R+SLS model v4 — combined train manifest v1

**Status:** completed exactly once. The assembler validated both immutable inputs and all `20,000`
referenced WAV SHA-256 values, then published the combined manifest. It did not create a model,
checkpoint, training report, calibration or inference result.

Machine contract:
[xlsr_sls_model_v4_train_manifest_v1.json]
(../../configs/research/v4/xlsr_sls_model_v4_train_manifest_v1.json)
(SHA-256 `2acf617a17f9904d3bbb2ce93f6fefaad505ca19d430becf60584eaced8f0230`).

The contract hash-binds the `15,000`-row source frozen manifest and its decode receipt; the
`5,000`-row KK spoof frozen manifest; the completed KK spoof audio-gate receipt and its required
governance reconciliation; the current license ledger; and the exact assembler module and runner.
It accepts only four balanced `language/label` cells of `5,000` rows each.

The source and spoof inputs have no shared sample ID, audio SHA-256, relative path or parent group.
They do share exactly `4,604` text hashes, exclusively between `KK bona-fide` and `KK spoof`
inside the same frozen `train` role. That overlap is pinned as an expected within-train pairing
property; it is neither speaker independence nor cross-role/source leakage. The runner rejects any
other count.

The only permitted operations are deterministic merge, manifest/rights/asset validation and
write-once publication. Detector or logit feedback, new synthesis, audio mutation, output
overwrite, training, checkpoint selection, calibration and final inference are all prohibited.

## Published output

- [combined frozen train manifest](../../../data/manifests/v4/xlsr_sls_model_v4_train_frozen_v1.csv):
  `20,000` rows, SHA-256
  `6c0fdbf38e9509749e2c8c458077095c21c3f6a65ddca17e0325dc6c8d00b7ed`;
- [write-once result](xlsr_sls_model_v4_train_manifest_v1.json): SHA-256
  `b0ceb8756d06ca8551ff616a4a3a91c62aaaffe9381c45cb002cb12c13418671`.

The four `language/label` cells each contain exactly `5,000` rows. Sample ID, audio SHA-256 and
parent group are unique across the combined manifest. The result explicitly records
`actual_training_execution=false` and `training_contract_created=false`.

Preflight command:

```bash
PYTHONPATH=src .venv/bin/python scripts/freeze_v4_combined_train_manifest.py --validate-only
```

The one permitted publication command was:

```bash
PYTHONPATH=src .venv/bin/python scripts/freeze_v4_combined_train_manifest.py --publish
```

The resulting receipt retains `actual_training_execution=false`. The next gate is a separate full
training contract, but only after its dev inputs, runtime, hyperparameters and write-once outputs
are independently hash-pinned; actual training remains forbidden until then.
