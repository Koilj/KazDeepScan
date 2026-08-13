# Stage D — Dialogs-RU VITS2 / Masha neutral

## Решение

Маршрут условно принят только как новый **exact checkpoint/runtime route** для personal
research: `frappuccino/dialogs-ru-vits2` at
`af7eabc15d087fce701de7261190ccb747c3bc7a`, checkpoint
`averaged_G_615000.pth` (529,555,187 bytes, SHA-256
`b565b7b5311bd29102f15d2700c69ea71569605bc756fb5bb49d98afdc5e1e43`).

Это не доказательство новой архитектурной family или speaker independence. Исторический
RuASD inventory содержит generic `vits2TTS`; full audit 53 manifest'ов / 18,422 spoof
rows нашёл 0 вхождений точного route и 0 вхождений Masha aliases. Receipt:
`data/manifests/stage_d_ru_dialogs_vits2_exact_route_audit_v1.json`, SHA-256
`f6a4ddd6bc334776d0029cbe3151c5e118815c38f160cb58f0e0ad7725a33d00`.

## Rights and safety lock

`configs/research/dialogs_ru_vits2_masha_neutral_v1_models.json` pins the complete
15-file bundle (529,698,251 bytes) including source, config and checkpoint; SHA-256
`c516526c916781ca332fc3b7cd14055c35b0c59bd1e1d15baef723e4d0bb6142`.
The associated artifact/rights lock is
`data/licenses/dialogs_ru_vits2_masha_neutral_v1_artifact_lock.json`, SHA-256
`3979afb3efc3c96434011cbdf0a374ba875e992b440cc0fba0aa81f146153c6b`.

There is an important licensing limitation: the pinned model repository has no separate
`LICENSE` file. Its model card declares OpenRAIL, and the locked Dialogs dataset license at
revision `e25ba617b2b56bd1dbf255d3905c51bd8da3d31f` explicitly covers models trained on
that dataset and their outputs. The project therefore records this route as suitable only
for the stated personal-research scope and its OpenRAIL restrictions, not as a broad
commercial clearance.

The locked local wrapper (`src/kds/data/dialogs_ru_vits2.py`, SHA-256
`188ed0e0d2a8db916160b75f51462cf59fe24bce60a919e88fedd7fa76846c71`) does not run
upstream `tts.py`: upstream uses pickle-capable `torch.load`. The wrapper verifies every
locked byte, uses `torch.load(weights_only=True)`, checks an all-tensor strict state dict,
and exposes no reference-audio, cloning, speaker or emotion parameter. It fixes
`speaker_id=0` (Masha) and `emotion_id=0` (neutral).

## Frozen text and synthesis

The exact Common Voice archive was bound before generation:

- `/home/ruslan/Downloads/cv-corpus-24.0-2025-12-05-ru.tar.gz`
- 7,008,716,262 bytes; SHA-256
  `9a2ed32a0574f74f505cd7740a599f0b9edc9f52ba1e7d6624b66f258db4c0ea`
- 73-row frozen base manifest SHA-256
  `d9ef5b0e91e960e5bf76a43274587a6483ac10e431d391f1ba119c71a05b330f`
- literal-text binding SHA-256
  `59dbb43393a8b15d969985dce653395af6a3cf558eb899912a2bd29af7ef5d7f`.

Exactly 73 raw synthetic WAVs were created, one per frozen base text, with no text
replacement, reselection or metric/model-driven choice. Four literal texts contain `–`,
`—` or `…`; the locked upstream tokenizer drops those nonlexical punctuation characters.
The source text was not rewritten, and each drop is recorded per row. Synthesis receipt:
`data/manifests/stage_d_ru_dialogs_vits2_masha_neutral_synthesis_v1.json`, SHA-256
`978cb1ad25408b2d5ce7948d5d213368154c519bfa5d95cdb4b99c8255df55d8`.

## Technical QA and current gate state

The project preprocessing pipeline decoded and normalized every raw WAV, enforced duration
and audio-quality limits, ran WebRTC VAD, and computed output SHA-256 values. It published
55 16 kHz mono PCM-16 ready WAVs and rejected 18 raw spoof WAVs solely as
`insufficient_speech`. The rejection report is immutable (SHA-256
`0d539d7c740ca7d0d41420b3738227bbea6d8e3df0c1d4b2df282cc57fef569d`). A later read-only
check verified SHA-256 bindings plus all 55 ready files' 16 kHz/mono/PCM-16 decode and
manifest durations. `original_sr=22050` in their ready manifest retains the original TTS
provenance; it is not a claim about the normalized file's sample rate.

Each VAD reject removed its Common Voice partner too. No resynthesis or backfill occurred:
the paired candidate now has exactly 55 binary pairs / 110 assets. Its write-once receipt is
`data/manifests/stage_d_ru_dialogs_vits2_masha_neutral_pairing_v1.json`, SHA-256
`ae0a214335e3bb7627c13848adcb8cc3ae32ff48cf39a4afd8ccbad6318e326f`.

The review packet and two independent worksheets bind every exact WAV and transcript. Two
distinct reviewers completed all 220 decisions. Every locked asset passed both reviews:
`review_status=pass`, `intelligible=yes`, `russian_audible=yes`,
`lexical_content_preserved=yes`, and `severe_artifacts_absent=yes`. A reviewer may set `pass`
only if all of these are `yes`:
`intelligible`, `russian_audible`, `lexical_content_preserved`, and
`severe_artifacts_absent`.

- packet: `stage_d_ru_dialogs_vits2_masha_neutral_acoustic_gate_packet_v1.csv`, SHA-256
  `478b7b470c430423ce3c8eed51374fc7543c69934e18dfb305146d7fc9a4850b`;
- reviewer 1 worksheet: SHA-256
  `8d23eca5db626048a5ef46cd7e844c757b846feac752c2205fc44989d263cbd4`;
- reviewer 2 worksheet: SHA-256
  `2ed8b15234f4dc4732c715992dd8ac8788622c8a54854a4a15d09debf3389de2`.

The gate report is
`data/manifests/stage_d_ru_dialogs_vits2_masha_neutral_acoustic_gate_report_v1.json`, SHA-256
`ba604ab8487c258a7004540dd243ac61a72810c6bac141a2925323bd3dca9acb`.

## Immutable evaluation and one GPU run

The completed gate authorized one separate RU research contract, not a change to any previous
RU/KK/mixed metric. The plan
`configs/research/xlsr_sls_stage_b_v2_stage_d_dialogs_ru_v1.json` (SHA-256
`3509ec2efd12c134e3eba0abdefca9e3fe891e7751abd78c1abdb2bab11f85ec`) pins:

- the XLS-R+SLS Stage-B v2 checkpoint and its encoder bytes;
- 976 pre-existing PyAra calibration assets solely for temperature scaling;
- the 55 Stage-D pairs / 110 final assets;
- a new minimal three-source frozen license ledger, SHA-256
  `d14cdd6fdd235fe2e511178ac4b3ba6aed5e632eb3656c0560d55e5bfbdb787c`;
- the passing acoustic gate, exact-route audit and a project exposure audit over 23 prior
  research configs, 16 referenced manifests and 12,203 rows. It found zero overlap in
  `sample_id`, audio SHA-256 and `text_hash`.

The write-once preflight validated all 1,086 calibration/final bindings and CUDA BF16 without
computing logits. Its SHA-256 is
`b19d990ec4407241fbff1430687473193d270af69d8d9d54cb814d51b2b1d9f1`. Then exactly one GPU
run was executed. The execution lock was written before the first calibration/final inference
and has SHA-256 `b4a75462ae2084d33d0cca9e44d2de44209fd9c6ee9419cd7087aa736e5295d5`.

The immutable report is
`artifacts/xlsr-sls-stage-b-v2-stage-d-dialogs-ru-v1.report.json`, SHA-256
`956e73d9964cea7b342bdbfab7074e4ddbde87a3a646e110d7c239709320ed8c`:

| RU layer | Result |
| --- | ---: |
| Assets / exact pairs | 110 / 55 |
| Accuracy, Wilson 95% CI | 107/110 = 0.9727 [0.9229, 0.9907] |
| Bona-fide recall | 52/55 = 0.9455 [0.8515, 0.9813] |
| Spoof recall | 55/55 = 1.0000 [0.9347, 1.0000] |
| Balanced accuracy | 0.9727 |
| Pairs with both assets correct | 52/55 = 0.9455 |

All three errors were fixed-set Common Voice bona-fide assets predicted as spoof. The model,
fixed 0.5 boundary, calibration role, source WAVs and 55 pairs were not changed after seeing
them. Inference took 12.49 seconds and used 1,645,827,072 bytes peak allocated VRAM.

This result remains personal research only. It does not establish source-, speaker- or
architecture-family independence, and it must not guide a replacement, reselection,
threshold/model change or v3 choice. v3 may now be prepared only with separate train/dev/
calibration contracts and a final set that remains untouched by model-driven decisions.
