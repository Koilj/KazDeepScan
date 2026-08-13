# VoxForge Russian / Mozilla Data Collective — source-level intake, 13 августа 2026

**Статус:** completed source-level archive intake. Это ещё не candidate, не extraction и не
разрешение на synthesis, training, calibration или detector inference.

## Exact artifact и лицензия

Проверен только локальный архив, предоставленный пользователем вне репозитория:

| Поле | Значение |
| --- | --- |
| Source ID | `voxforge_ru_mdc_2026_05` |
| Publisher / release | Mozilla Data Collective, VoxForge — Russian, release 2026-05-12 |
| Official artifact name | `voxforge-russian-9a8495d3.tar.gz` |
| Local archive bytes | `3,795,197,539` |
| SHA-256 | `7372c6f8d067b8d1651995ad8306b673acaf2cde705ee51295152b96c93de557` |
| License | `GPL-3.0-or-later` |
| Scope in this project | personal research only |

Mozilla Data Collective declares the release as Russian read speech, `15.5` hours / `6,412`
utterances, GPL-3.0. The exact artifact itself carries `644` submission `LICENSE` notices with
GPL-3.0-or-later wording and `644` `etc/GPL_license.txt` copies of GNU GPL v3. This establishes
the archive's licence evidence more precisely than an unpinned website label, but does not turn
it into product clearance or waive speaker/privacy duties.

The source URL and archive identity are recorded in
[the license ledger](../data/licenses/license_ledger.csv); the immutable source audit is
[the versioned receipt](../data/licenses/voxforge_ru_mdc_2026_05_artifact_audit_receipt.json),
SHA-256 `0e8bd5c7d1e02bedc235adcb3bdb7ed3bc7efdd0ff7637339460e3f43c38272f`.

## Проверенный состав

The audit streamed the TAR without extracting WAV bytes to the project. Gzip CRC and TAR
structure passed; unsafe paths, links, special files, duplicate regular member paths and unknown
regular layout are fail-closed errors. It checked every WAV header and both transcript layers.

| Проверка | Фактический результат |
| --- | --- |
| Submissions | `644` |
| Source-provided contributor groups | `194` |
| WAV / `PROMPTS` / `prompts-original` bindings | `6,412 / 6,412 / 6,412` |
| Unique canonical `PROMPTS` texts | `81` |
| Duplicate prompt rows | `6,331` |
| Exact WAV duration | `55,545.812312` s (`15.4294` h) |
| WAV format | `6,412` mono, 48 kHz, 16-bit PCM |

`README` contributor values are source-provided aliases. They are used only as conservative
whole-group keys; neither the archive nor this audit proves that an alias identifies one unique
human. `anonymous` is therefore one conservative group, rather than inferred speakers. Session
directories are retained as provenance but do not justify a speaker-independence claim.

The `81`-text capacity is a material constraint: any future selection must be one text group per
binary pair and must first exclude every group whose text appears in prior project evidence.

## Pre-extraction project-exposure screen

Completed screen binds the source audit and compared both canonical `PROMPTS` text and
`prompts-original` text hashes, sample identities and privacy-preserving contributor-group keys
with every then-current configured role and full manifest inventory.

| Scope | Configuration / manifests | Rows | Exact overlap: sample / spoken text / original text / group / speaker key |
| --- | ---: | ---: | --- |
| Configured roles | `31` / `18` | `12,397` | `0 / 0 / 0 / 0 / 0` |
| Full manifest inventory | `90` | `40,206` | `0 / 0 / 0 / 0 / 0` |

Strict whole-contributor-group exclusion tainted `0` groups: all `6,412` records, `194`
contributor groups and `81` canonical text groups survive. The screen intentionally has no WAV
SHA-256 comparison because it is pre-extraction; an exact asset audit remains mandatory after a
future frozen selection. Immutable screen receipt SHA-256:
`275367a9738bfcc017315cfb3799078c0c3ab1981a318098b0849eaf7893dffe` —
[versioned receipt](../data/manifests/voxforge_ru_mdc_2026_05_metadata_exposure_screen_v1.json).

## Запреты и следующий gate

This stage created no `data/raw/voxforge*` material, manifest, candidate, selection, synthetic
audio, model logit or metric. Raw archive and WAV stay outside Git.

The next permitted action is a separately frozen, limited bona-fide selection policy. It must
bind a seed, selected text/group rule and a candidate size no greater than the surviving `81`
text groups. It still cannot authorize synthesis or detector inference: a separate exact spoof
route, audio QA, pair lock, full-asset acoustic review and immutable one-run evaluation contract
remain required.
