# RuASD v1 full local release — integrity and protocol audit

## Pinned release and local discovery

The project has a complete local copy of the `lab260/RuASD` release in
`/home/ruslan/Downloads/RuASD`. It contains the 250 TAR artifacts
`ruasd-000000.tar` through `ruasd-000249.tar` from pinned Hugging Face revision
`fcbc87c57b54ef4f58e1135e2813f6f000c2b739`.

`data/licenses/ruasd_v1_artifact_catalog.csv` is an auditable catalog of the
official filename, exact byte size, and SHA-256 LFS OID for every artifact. It
was generated from the official Hugging Face dataset API on 9 August 2026. The
catalog is deliberately separate from the existing `license_ledger.csv` entry
for shard `000000`: it records a collection of 250 immutable artifacts rather
than falsely representing the full release as one archive.

No archive or audio file was copied, extracted, renamed, or modified during
the audit.

## Full metadata audit

`scripts/audit_ruasd_collection.py` validates the exact archive set and byte
sizes, then safely walks every TAR without extracting audio. It rejects
symlinks, hardlinks, nested member paths, non-JSON/WAV members, malformed JSON,
and any archive that does not have exactly one direct JSON and WAV member per
sample.

The complete integrity pass (`sha256_verified_archives=250`) and metadata pass found
`585,353` paired records. Every local archive's exact byte size and SHA-256 matched the
pinned official catalog:

| Group | bona-fide | spoof |
| --- | ---: | ---: |
| raw | 147,097 | 228,266 |
| augmented | 104,998 | 104,992 |

Raw spoof data cover 37 Russian-capable TTS/voice-cloning subsets. Raw
bona-fide data are from Common Voice, Deep-Speech, GOLOS, M-AILABS, OpenSTT,
RUSLAN, RuLS, and three SOVA subsets. Augmented rows must not be mixed into a
baseline split until a separate channel-robustness protocol is specified.

## Critical protocol result

The full download **does not remove the speaker/voice-disjoint blocker**:

- all `147,097` raw bona-fide rows have `speakers=-1` / unknown;
- `223,516` of `228,266` raw spoof rows also have an unknown speaker/voice
  value;
- only `4,750` raw spoof rows expose a non-unknown value, and no matching
  verifiable bona-fide speaker identifiers are available.

Consequently, RuASD can be used only for a clearly labelled Russian
research/OOD robustness study after its selected archive checksums are verified.
It cannot be presented as a speaker-disjoint target-language train/dev/test
corpus, calibration basis, or product release. Its `CC-BY-NC-SA-4.0` terms also
continue to prohibit commercial use.

## Reproducible audit

Use a new report path each time; the command refuses to overwrite prior audit
results. `--verify-sha256` reads the full local release and should be used before
any future selection or extraction.

```bash
.venv/bin/python scripts/audit_ruasd_collection.py \
  --archive-dir /home/ruslan/Downloads/RuASD \
  --verify-sha256 \
  --progress \
  --output-report /tmp/ruasd_full_audit.json
```

The report contains only aggregate metadata counts. It does not expose audio,
transcripts, or any personal identifiers.
