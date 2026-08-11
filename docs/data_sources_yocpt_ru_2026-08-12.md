# YO-CPT-ru — rejected source review, 2026-08-12

Candidate: [NCSpeech/YO-CPT-ru](https://huggingface.co/datasets/NCSpeech/YO-CPT-ru), current public revision `f2f92c5ecf6b0323cf7e7f9599f0e4852f6c536b` (checked 2026-08-12).

## What it is

This is not a Russian spoof source. It is a claimed bona-fide Russian corpus derived from YouTube/YODAS2, intended for TTS continual pre-training. The publisher states 1.63 million 24-kHz single-speaker chunks and about 6,052 hours. The pinned public revision has 384 files, including 379 Parquet shards, totalling 1,008,879,201,440 bytes (about 1.01 TB). Do not download it for this project.

## Blocking provenance, rights, and privacy issues

1. The audio is sourced from YouTube. The YO-CPT-ru license says its authors add CC-BY-4.0 annotations, but the audio remains the intellectual property of the original creators and is asserted to originate from CC-licensed YODAS2 videos. This is a publisher assertion, not a per-record consent or durable rights record. A removal-request contact does not establish final/product rights.
2. YO-CPT-ru says it recovered the original YouTube IDs to fetch clips. The YODAS2 maintainer states that the real IDs are deliberately withheld and that a decoded mapping may be used only for personal purposes and must not be shared. The public YO-CPT-ru derivative therefore has an unresolved upstream-compliance risk.
3. The dataset provides face-derived `global_spk_id` and inferred appearance, age, nationality, gender, and speaker personas. These are sensitive, model-inferred personal-data fields, not consented participant metadata. Its own card says all annotations are automatic and not human-verified.
4. The YO-CPT-ru license states that `global_spk_id` is produced with LVFace weights that are non-commercial-research-only. At minimum, that field cannot support a commercial-clean provenance or speaker-disjoint product protocol.
5. Its bona-fide label is only the output of the publisher's anti-spoof filtering pipeline. It is not independently audited human/synthetic ground truth. The `local_spk_id` and `global_spk_id` values are automatic clustering results, so they are not verified speaker identities for a final split.

## Decision

Reject `yocpt_ru` for every KazDeepScan protocol: no download, no license-ledger entry, no manifest, no training, calibration, research evaluation, or product/final evaluation. It does not address the independent Russian-only spoof-source gap in any case.

The project may reconsider only after receiving written, per-record rights/provenance evidence from the publisher that resolves the YouTube-ID and consent/privacy issues, and only after excluding all face-derived identity/persona fields and re-auditing a suitably small, immutable release. This is not currently an actionable path.

## Primary sources

- [YO-CPT-ru dataset card](https://huggingface.co/datasets/NCSpeech/YO-CPT-ru)
- [YO-CPT-ru license](https://huggingface.co/datasets/NCSpeech/YO-CPT-ru/blob/main/LICENSE.md)
- [YODAS2 dataset card](https://huggingface.co/datasets/espnet/yodas2)
- [YODAS2 discussion on YouTube IDs](https://huggingface.co/datasets/espnet/yodas2/discussions/2)
