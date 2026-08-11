# Dialogs Russian conversations — rejected intake, 2026-08-12

Source: [langswap/dialogs-ru-emotional-conversations](https://huggingface.co/datasets/langswap/dialogs-ru-emotional-conversations), snapshot revision `e25ba617b2b56bd1dbf255d3905c51bd8da3d31f`.

## What was verified

The local Hugging Face snapshot at `/home/ruslan/Downloads/dialogs-ru-emotional-conversations` was checked against the pinned snapshot-tree SHA-256 `0e44fb25264c8b4486d968923cf7fcd9bf08c35d5d354a068a723aa02003eff0`.

- All 10,004 published source files match the tree: 9,997 LFS SHA-256 files and 7 Git blob SHA-1 files, totalling 5,571,006,447 bytes excluding Hugging Face cache files.
- The source supplies 9,996 WAV files, all of which hash-match the published LFS entries.
- The four CSVs have the advertised non-overlapping partition: `train` 11,428 rows, `val` 180 rows, and `test` 188 rows. Their union equals the 11,796 unique `metadata.csv` paths.

## Blocking publication defect

The CSVs refer to 11,796 distinct `wavs/*.wav` paths, but the exact published snapshot contains only 9,996. Thus 1,800 metadata rows have no supplied audio:

- `M`: 898 rows;
- `S`: 534 rows;
- `D`: 368 rows;
- claimed-duration loss: 11,437.023452806 seconds (about 3.18 hours).

This is a source-release inconsistency, not a download error. The cache tree and every supplied file agree with the pinned revision. A locally filtered subset would silently redefine the published splits and must not be treated as a final source.

## Decision

`dialogs_ru_v1` is deliberately absent from `data/licenses/license_ledger.csv`; consequently no manifest may use it. It is rejected for research, calibration, and product/final evaluation until the publisher releases an auditable snapshot where every metadata/split path resolves to a WAV, followed by a new source and rights review.

The source card's stated consent and OpenRAIL license are not enough to bypass this data-integrity gate. Even after an upstream repair, the published three-speaker composition would still be too narrow by itself for a speaker-robust final claim.

## Reproduction

From the repository root:

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_dialogs.py \
  --artifact-root /home/ruslan/Downloads/dialogs-ru-emotional-conversations \
  --require-bonafide-final
```

The command verifies all file hashes and then exits non-zero because the release is incomplete. To render the evidence receipt without asserting eligibility, omit `--require-bonafide-final`.
