# VoxForge Russian / Mozilla Data Collective — source-level intake, 13 августа 2026

**Статус:** completed source-level archive intake, screen и metadata-only selection. Это ещё не
audio candidate, extraction и не разрешение на synthesis, training, calibration или detector
inference.

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
SHA-256 comparison because it is pre-extraction; an exact asset audit remains mandatory after the
completed frozen selection. Immutable screen receipt SHA-256:
`275367a9738bfcc017315cfb3799078c0c3ab1981a318098b0849eaf7893dffe` —
[versioned receipt](../data/manifests/voxforge_ru_mdc_2026_05_metadata_exposure_screen_v1.json).

## Frozen pre-QA selection

The metadata-only selection is complete: it selected all `81` canonical prompt-text groups and
matched them to `81` distinct conservative contributor groups, with one source record per match.
It binds the source audit and this screen, preserves both text-hash layers, excludes raw aliases
from versioned output and forbids backfill. The immutable details and next gate are in the
[selection receipt](voxforge_ru_mdc_pre_qa_selection_v1.md).

## Принятый exact spoof route

Qwen3-TTS CustomVoice `0.6B` Q8_0 GGUF with its separate 12 Hz codec, CrispASR `v0.8.28` CUDA
runtime and fixed baked `aiden` token прошёл six-artifact SHA-256 lock и health check. Its exact
family/name/version is absent from `18,764` historical spoof rows in `59` manifests; legacy Qwen
CustomVoice identifiers and `aiden` aliases are also absent. The detailed immutable evidence is
in the [route review](voxforge_ru_mdc_qwen3_tts_customvoice_route_review_2026-08-13.md).

This proves only an unseen exact route. The voice token is documented as English, not as a verified
Russian person or group; architecture and speaker independence claims remain forbidden. The route
check generated no audio and did not read any VoxForge WAV.

## Запреты и следующий gate

Archive intake and exposure screen created no `data/raw/voxforge*` material, decoded WAV,
manifest, synthetic audio, model logit or metric; the subsequent selection remained metadata-only.
Raw archive and WAV stay outside Git.

The first separate exact spoof-route review remains rejected because `76` historical spoof rows
use the same unversioned UtrobinTTS model identifier. The accepted Qwen route now authorizes only
selection-bound WAV materialization and technical QA. It still cannot authorize pairing, acoustic
review, detector inference, training, calibration or product use.
