# Silero V5.5 RU / fixed `eugene` — route intake, 13 августа 2026

## Решение

`silero_v5_5_ru_eugene_v1` условно принят только как **новый exact
checkpoint/runtime/fixed-profile route** для будущего personal-research RU evaluation. Это не
принятие нового final-набора и не разрешение на synthesis, detector inference, training или
calibration.

Основания: официальный [Silero README](https://github.com/snakers4/silero-models/blob/d9355348e2781dc8fa25a135d1602c530afae24c/README.md)
описывает RU `v5_5_ru` и встроенный speaker `eugene`; pinned
[models.yml](https://github.com/snakers4/silero-models/blob/d9355348e2781dc8fa25a135d1602c530afae24c/models.yml)
указывает package `https://models.silero.ai/models/tts/ru/v5_5_ru.pt`. Лицензия pinned
repository — [CC-BY-NC-SA-4.0](https://github.com/snakers4/silero-models/blob/d9355348e2781dc8fa25a135d1602c530afae24c/LICENSE): маршрут допустим лишь в текущем personal-research
scope; product/commercial use не авторизован, а требования attribution/share-alike сохраняются.

## Пиннинг и ограниченный wrapper

Полный write-once lock:
[`data/licenses/silero_v5_5_ru_eugene_v1_artifact_lock.json`](../data/licenses/silero_v5_5_ru_eugene_v1_artifact_lock.json).

- `v5_5_ru.pt`: `145420684` bytes,
  SHA-256 `50081637b602126ee06cb3bc8a744d25651d2da149ee8864b9a379bfdd934437`;
- source archive at commit `d9355348e2781dc8fa25a135d1602c530afae24c`: `789187` bytes,
  SHA-256 `cca6d3e6e34e03f9fe30c4e33ee2de8e89aa384f95bc0f3143c51af7a72765aa`;
- model lock:
  [`configs/research/silero_v5_5_ru_eugene_v1_models.json`](../configs/research/silero_v5_5_ru_eugene_v1_models.json),
  SHA-256 `39fc9f4748286593ff39fe51688212215c312b4b1d880dc9539e7f43d9ce8edd`;
- local adapter:
  [`src/kds/data/silero_v5_5.py`](../src/kds/data/silero_v5_5.py),
  SHA-256 `9841a7361d9d8c07e81778cbb2609ccd2195a9e53af9e58d021318c0828560f1`.

Wrapper exposes only literal Russian text (only whitespace collapse) to the built-in `eugene`
speaker at `48000` Hz. It has no parameter for reference audio, cloning, random profile, SSML,
`voice_path`, `symbol_durs`, timestamps, language/type override, or intensity override. Control
markup and unsupported input characters fail before the upstream package call.

Before `torch.package.PackageImporter` is allowed to load the package, the adapter checks the
outer artifact hash through the model lock, ZIP CRC, member count/path/type limits, pinned wrapper
and dispatcher hashes, plus the dispatcher's exact pickle `GLOBAL` allow-list. Static audit found
`584` ZIP members (`145562749` uncompressed bytes); a CPU-only load reported exactly
`aidar`, `baya`, `kseniya`, `eugene`, `xenia`. No synthesis or detector inference was performed.
Raw package/source files remain ignored local artifacts and are not in Git.

## Novelty boundary

The pre-synthesis route audit is
[`data/manifests/silero_v5_5_ru_eugene_exact_route_audit_v1.json`](../data/manifests/silero_v5_5_ru_eugene_exact_route_audit_v1.json),
SHA-256 `5850d8d36bb72191f7e9a9516edec14552bcf1fe83aecea154a9bb90964fb955`.
It scans `56` historical manifest files containing `18 605` spoof rows and sees `0` occurrences of
the exact V5.5 package/runtime/fixed-`eugene` route. It nevertheless finds `1 265` historical
Silero-labelled rows (pinned V4 and generic RuASD labels).

Consequently, the only supported novelty statement is **new exact route**. The project must not
claim a new demonstrated architecture family, independent vendor family, or independent speaker
family. The route has no prior audio assets, synthesis, acoustic review, QA, pair lock, final
plan, or model predictions.

## Excluded alternatives

`joefox/tts_vits_ru_hf` and `frappuccino/vits2_ru_natasha` are excluded because RuASD already
names those routes. Piper, MMS and RHVoice are historical/excluded routes. GPT-SoVITS, XTTS,
F5/VoiceCraft-like and any reference-audio/cloning route are excluded because their input surface
cannot satisfy the fixed text-only/no-reference-audio contract. These exclusions are not a claim
that every possible implementation is unsafe; they are the project boundary for this candidate.

## Bona-fide boundary and next safe action

The historical Stage-D/v3 set is immutable evidence, not an input pool: its `55` Common
Voice/Dialog-RU pairs were already scored in Stage-D v2, and v3 used those same exact assets.
They cannot be used as a new blind test; the related `73`-row selection, `18` QA rejects and
original first-250 source slice are also not an acceptable no-backfill route for a new final
contract.

Pinned FLEURS RU has no usable fresh capacity: its `344` release texts are accounted for by prior
routes or the completed Stage-C selection/rejections. The preferred next source is therefore a
new, previously unmaterialized Common Voice RU v24 `test` slice from the verified full archive,
not the old first-250 slice.

The full archive has now passed the pre-extraction metadata screen in
[`data/manifests/common_voice_ru_v24_full_test_metadata_exposure_screen_v1.json`](../data/manifests/common_voice_ru_v24_full_test_metadata_exposure_screen_v1.json),
SHA-256 `f862ae667195c733c7deb6bf25f304a6287890ca87d4dc0ee7cb5e06aa6f46b3`. Its `10 261`
official `test` records / `2 075` client groups were compared against all `12 313` configured
role rows and `39 850` rows in `85` manifest files by sample ID, text hash, parent/client group
and speaker pseudo-ID. Strict whole-client-group exclusion leaves `6 211` records in `1 443`
groups. This is capacity evidence only: no seed, candidate size or clips were frozen; no audio
was extracted, synthesized, QA-reviewed, acoustically reviewed or scored.

The fixed-wrapper literal-text screen then consumed exactly those `6 211` survivors:
[`data/manifests/common_voice_ru_v24_full_test_silero_v5_5_literal_text_screen_v1.json`](../data/manifests/common_voice_ru_v24_full_test_silero_v5_5_literal_text_screen_v1.json),
SHA-256 `4356c3ecbf3a9b68dd7a5d5f4e2ed9347d9c6f105d63d558bfc03dd1403b23d0`.
It permits no external lexical rewrite and requires wrapper-normalized text to equal the source
sentence exactly. `113` direct incompatible transcripts (only unsupported quotation marks or the
minus glyph) taint `106` complete client groups; thus `5 600` records in `1 337` groups remain.
This is a conservative wrapper-grammar exclusion, not a correction of the source text or an
audio-quality result. At the time of this screen, no clip selection, extraction, synthesis,
QA/review or detector inference had occurred.

## Immutable pre-QA selection

The required selection was completed before extracting any MP3. The new
[pre-QA selection receipt](common_voice_ru_v24_silero_v5_5_eugene_pre_qa_selection_v1.md)
uses seed `2026-08-13-silero-v5-5-eugene-pre-qa-candidate-v1` to freeze `80` records — exactly
one per client group — from only these `5,600` survivors. Its CSV SHA-256 is
`73eaf22706419b275517500ebb25973510e8dcccaa94a54f45b4fe2a787f6b50`; receipt SHA-256 is
`a7f0a1b5c3a152c87692e5e9d7d4ac2e02b2b64d9b6933f7f79f29b4e6b6d7ad`.

The `80` count is a conservative pre-QA buffer over the historical `55` pairs, not an expected
QA yield. Selection did not use audio/duration, detector output, metrics or final errors, and all
selected sample IDs, client groups and text hashes are unique. It does not repurpose the old
`55`, old `73`, their `18` rejected partners or remaining first-250 source-intake records.

## Frozen extraction and technical QA

The exact 80 `clip_name` values have now been extracted and passed through normal decode/QA/VAD;
the [materialization receipt](common_voice_ru_v24_silero_v5_5_eugene_pre_qa_materialization_v1.md)
records `75` ready WAVs and five `insufficient_speech` rejections, with no reuse, replacement or
backfill. Raw manifest SHA-256 is
`5543e1e88b688cb79b4401a7ef68ba525d75321cbe98a4837bf7347867a2a9a5`; ready manifest SHA-256 is
`2b183adbfcac9b1a6022dd35c2f8b6ec8f111c01b4b3364596c53aff8906192a`; receipt SHA-256 is
`da5f4a79cf0444d0f2e905c4f178760967306d3a393753736e72cc4a27a0da3e`.

The [literal-text binding receipt](common_voice_ru_v24_silero_v5_5_eugene_pre_qa_text_binding_v1.md)
now binds only the 75 ready rows to their exact archive sentences without rewrite. The completed
[synthesis receipt](common_voice_ru_v24_silero_v5_5_eugene_pre_qa_synthesis_v1.md) stores exactly
one fixed-profile V5.5/eugene raw WAV per bound text. Its
[technical-QA receipt](common_voice_ru_v24_silero_v5_5_eugene_pre_qa_spoof_technical_qa_v1.md)
retained `42` rows and accounted `33` insufficient-speech rejects. No failed row may be replaced,
reselected or resynthesized; the matching
[42-pair lock](common_voice_ru_v24_silero_v5_5_eugene_pre_qa_pairing_v1.md) passed the subsequent
full technical acoustic gate. A new immutable contract and preflight remain required before any
detector inference.

The existing Common Voice archive guard now validates both size and SHA-256 before metadata read
or extraction; see
[`docs/data_sources_common_voice_ru_v24.md`](data_sources_common_voice_ru_v24.md).
