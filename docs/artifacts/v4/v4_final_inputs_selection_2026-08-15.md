# XLS-R+SLS model v4 — final-input metadata selection v1

**Дата:** 15 августа 2026

## Результат

Ровно один metadata-only run завершён успешно. Он заморозил `1 000` source rows: `500` RU
Common Voice bona-fide candidates и `500` KK FLEURS bona-fide candidates. Pairing, extraction,
synthesis, QA, acoustic review, checkpoint loading, calibration и detector inference не
выполнялись.

| Layer | Source split | Selected source/text groups | Frozen selection-row SHA-256 |
| --- | --- | ---: | --- |
| RU | Common Voice RU v24 `test` | 500 client groups / 500 text groups | `05996545ef68b055134ddf3e2602d18e9d875c74a76b6ea675c2a7e212d1931c` |
| KK | FLEURS `kk_kz` `train` | 500 prompt groups / 500 text groups | `d08360fa59ce8271c25e7a120aa44c0bb43de2d34d8dca10b01a70e179f74c82` |

Versioned [metadata selection](../../../data/manifests/v4/xlsr_sls_model_v4_final_metadata_v1.csv)
has SHA-256 `0eec841c6ec55e9060f92fafbc5f8d648243c23552f8bb130aaaaae6e8636f1b`.
Machine receipt has SHA-256 `18bd4f988a75b0b415c707b945eb1886b00a2cf22d4a78ee8e1bf69c4930431a`.
It binds contract SHA-256 `b9b92b764048b86b6407304c4a2f5c244c6efc4ac2b2f66f279daf2976877f2e`.

## Current-history and capacity gates

- Common Voice strict metadata screen excluded all `713` tainted client groups, leaving `5 878`
  records / `1 362` client groups. All `5 878` surviving literal UTF-8 texts meet Qwen's
  metadata-only input bound; the selected `500` are group- and text-disjoint.
- FLEURS selection uses the never-before-selected local `train` split, not the historical
  Stage-C `test` split. Of `3 200` train records, `2 856` / `1 332` prompt-text groups passed
  current-history and fixed KazakhTTS text-normalization compatibility. The selection is
  deterministic and has no reserve or post-selection backfill.
- FLEURS has no public speaker IDs. `unknown` is a shared placeholder, not an identity; only
  `prompt_id`/text group participates in the FLEURS grouping rule. This result does not support
  a speaker-independent claim.

## Limits and next gate

The frozen metadata ledger permits this selection only. Before any WAV byte or TTS output, a new
materialization contract must explicitly re-authorize the exact routes, bind this selection and
the archive/release identities, then perform one-shot source extraction and text-only synthesis,
technical QA/VAD, full historical exact/near-audio isolation, acoustic/language review and an
immutable complete pair lock. A later final-evaluation contract must separately bind that lock,
the selected v4 checkpoint and the RU-only temperature receipt. KK calibrated probabilities and
all final quality claims remain invalid.
