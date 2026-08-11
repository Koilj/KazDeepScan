# Full RuASD v1 — personal-research protocol

> Historical v1 receipt. Актуальная исправленная выборка описана в
> [research_ruasd_full_v2.md](research_ruasd_full_v2.md). V1 использовал непроверенное `model`
> в strata; повторять его нельзя.

## Scope

The local full RuASD release is a Russian binary source for personal research only.  It is
recorded in `data/licenses/license_ledger.csv` as `ruasd_ru_v1_full` under
CC-BY-NC-SA-4.0.  The source is not for deployment, commercial training, re-hosting, or a claim
that a detector has independently generalized to unseen people or synthetic voices.

On 9 August 2026 the immutable catalog in
`data/licenses/ruasd_v1_artifact_catalog.csv` was checked against the local
`/home/ruslan/Downloads/RuASD` collection: all 250 TAR SHA-256 values matched, and the safe
audit found 585,353 exact JSON/WAV pairs.  The ledger stores the catalog digest
`eac0c3f9f2476a37417968547ba29f9ce04ac6dfa46be9146bade19acb0e7517`; it is not a substitute
for the individual archive hashes in that catalog.

## Slice construction

`scripts/ingest_ruasd_research.py` deliberately:

1. reads every pinned TAR and accepts only `raw/real_speech` and `raw/tts` metadata;
2. checks a fresh SHA-256 for every archive by default, then requires exact JSON/WAV pairing;
3. selects equal numbers of bona-fide and spoof records by deterministic SHA-256 ranking;
4. reserves `--min-per-stratum` entries for each real `subset` and fake `subset/model` stratum;
5. extracts only selected WAVs atomically, records per-file SHA-256, and splits connected
   source-record/text components into train/dev/test.

The command never uses augmented RuASD audio.  Source transcript text is not copied into the
manifest: only its SHA-256 is used to keep repeated texts together.  A missing transcript uses
a unique source-content key instead.

```bash
uv run python scripts/ingest_ruasd_research.py \
  --archive-dir /home/ruslan/Downloads/RuASD \
  --output-manifest data/manifests/ruasd_ru_v1_full_research_2000.csv \
  --slice-name research-2000 --limit-per-label 1000 --min-per-stratum 1
```

The first 2,000-WAV baseline intentionally uses `--min-per-stratum 1`: the release has more
than 100 fake `subset/model` strata, so reserving ten of each cannot fit in 1,000 spoof rows.
For a larger run, increase `--limit-per-label` before increasing this coverage minimum.

`--skip-sha256` is permitted only where the same unchanged archive set already has a recorded
full pinned-catalog audit; it still checks the exact archive names, sizes, TAR member safety and
JSON/WAV pairing.  It exists to avoid re-reading about 250 GB when repeating a local experiment.

## Essential limitation

The metadata has no verified bona-fide speaker group for the full raw release, and spoof voice
groups are also unknown.  `speaker_pseudo_id` is therefore an opaque source-record key used to
avoid false leakage assertions, not a speaker identity.  `voice_id=unknown` is provenance, not
a voice group.  The split is text-leakage-safe only and every metric from it must be labelled
**personal-research, non-speaker-disjoint**.

## Executed baseline — 9 August 2026

The first run used `--skip-sha256` only after the full `250/250` SHA-256 audit above.  Its two
new TAR walks still checked the pinned archive set, sizes, safe members and JSON/WAV pairing.

| Stage | Result |
| --- | --- |
| Raw slice | 2,000 WAV: 1,000 bona-fide / 1,000 spoof; train/dev/test = 1,564 / 220 / 216 |
| Ready slice | 1,814 WAV: train/dev/test = 1,417 / 202 / 195; bona-fide / spoof = 816 / 998 |
| QA/VAD report | 152 `insufficient_speech`, 32 `signal_too_quiet`, 2 existing processed-asset collisions; all 186 records remain in `ruasd_ru_v1_full_research_2000_rejections.json` |
| Validation | 1,814/1,814 ready asset SHA-256 values, manifest ledger and `--purpose research` protocol passed |

The QA/VAD exclusions were 184 bona-fide and two spoof rows, so the ready data are intentionally
not described as class-balanced.  B0 reports ordinary and balanced accuracy separately for this
reason.  Existing processed-asset collisions were rejected rather than overwritten; this keeps
the repository's no-overwrite invariant intact.
