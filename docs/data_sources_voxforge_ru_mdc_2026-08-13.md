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

## Запреты и следующий gate

This stage created no `data/raw/voxforge*` material, manifest, candidate, selection, synthetic
audio, model logit or metric. Raw archive and WAV stay outside Git.

The next permitted action is a **pre-extraction project-exposure screen** across the full
archive's sample/text/contributor identities and all configured roles plus manifest inventory.
It must taint a whole source-provided contributor group on any exact historical overlap. Only a
passed receipt can permit a separately frozen, limited bona-fide selection; it still cannot
authorize synthesis or detector inference.
