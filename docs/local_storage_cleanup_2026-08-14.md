# Local storage cleanup — 14 August 2026

## Scope

Cleanup was explicitly limited to local disposable or RU/KK-v4-ineligible bytes. It did not run
training, synthesis, preflight, calibration, detector inference or any completed evaluation.
No versioned manifest, license ledger/snapshot, research config, checkpoint receipt, execution
lock, report or prediction file was modified or removed.

The protected directory `/home/ruslan/Downloads/269-lockdown/` was not touched. Its post-cleanup
top-level file inventory remained:

| File | Bytes |
| --- | ---: |
| `capture.pcapng` | 3,932,088 |
| `memdump.mem` | 4,831,838,208 |
| `updatenow.exe` | 602,112 |

## Removed local bytes

| Path or set | Pre-delete logical bytes | Reason |
| --- | ---: | --- |
| `data/raw/ruasd/ruasd-000000.tar` | 999,813,120 | Exact byte-for-byte duplicate of the retained `/home/ruslan/Downloads/RuASD/ruasd-000000.tar` |
| `data/raw/ml_df/` | 1,487,227,776 | Italian-only historical OOD source; not eligible for RU/KK v4 train |
| `data/raw/ml_df_it_v1/` | 29,384,718 | Derived Italian OOD slice |
| 192 ML-DF paths referenced by `ml_df_it_v1_ood_200_ready.csv` | 25,523,056 | Historical OOD ready audio; all references were `source_name=ml_df_it_v1`, split empty or `ood`, with no train/dev cross-role reference |
| `.tools/ffmpeg/bin/ffplay` | 148,438,696 | Not referenced by project code or documentation; `ffmpeg` and `ffprobe` retained |
| `.mypy_cache/`, `.pytest_cache/`, `.ruff_cache/` | 119,862,108 | Reproducible tool caches |
| 978 `__pycache__` directories | 142,903,463 | Reproducible Python bytecode caches |
| Dataset-client `.cache` directories under retained RuASD/KSC2/FLEURS downloads | 240,725 | Download metadata cache only; source artifacts retained |

`/home/ruslan/Downloads/chatgpt_amd64.deb` had already disappeared before cleanup execution and
was therefore not deleted or counted. Final filesystem available-byte delta against the pre-delete
measurement was `2,971,725,824` bytes (`2.8 GiB` by IEC-rounded `numfmt`); logical target
accounting was `2,953,393,662` bytes, with the difference explained by allocated filesystem blocks,
directory metadata and small concurrent tool-cache changes.

## Preserved historical evidence

- `data/manifests/ml_df_it_v1_ood_200.csv` remained byte-identical, SHA-256
  `7e7eee52f6add9c69dc5777b160d289ece7155ce3cbe742c2a117bb45b44b5be`.
- `data/manifests/ml_df_it_v1_ood_200_ready.csv` remained byte-identical, SHA-256
  `951582973a8f31aca4585f2136e68e35267a62568e988910d4ad417e2a759da8`.
- `data/licenses/license_ledger.csv` remained byte-identical, SHA-256
  `e0428ea721472b8cd57601a4f0aa59dd943854b66d7229ca9a5e05349ee6b809`.
- `configs/research/source_mixed_v1.json` remained byte-identical, SHA-256
  `f2b8bea79f15b78ee289b1a75e3a9da316d9b01839432865cfe70b8a995151e8`.

The historical ML-DF results remain valid as already recorded research evidence, but their local
audio bindings can no longer pass `validate-assets` unless exact pinned source artifacts are
restored. The completed evaluation is not authorized for rerun.

## Preserved v4 inputs

All local RuASD, PyAra, KSC, KSC2, Common Voice RU, FLEURS RU/KK, VoxForge RU, Denis, ToneSpeak,
RU/KK synthetic-family weights, XLS-R base weights, v1/v2/v3 research checkpoints and the B0
checkpoint required by local research inference were retained. Old write-once research contracts
remain immutable. Any historical evaluation asset later admitted to v4 train must be recorded in
a new v4 manifest and must not be claimed as an independent v4 final asset.

## Verification

- Protected `269-lockdown` top-level file inventory and exact sizes matched the pre-delete check.
- All `192` ML-DF processed targets were absent after deletion; no target had a train/dev or
  non-`ml_df_it_v1` CSV reference before deletion.
- Fresh-cache mypy exposed a stale static-ignore dependency on optional `pyarrow`. The loader now
  uses `importlib.import_module("pyarrow.parquet")`, preserving the same fail-closed `ImportError`
  behavior without a package-presence-dependent type ignore; implementation SHA-256 is
  `8dacc50bfadc5f86b7c670f27cd613c44ee60a0858e7fe081a8eb2148f76183a`.
- Ruff passed; strict mypy passed for 217 source files; `git diff --check` passed.
- The exact hash-pinned temporary Linux `pyarrow==22.0.0` overlay produced `318 passed` with no
  skips. Temporary overlay, temporary mypy caches and the disposable `pyarrow` package cache were
  removed after verification. No project `.venv` dependency or frozen package input changed.
