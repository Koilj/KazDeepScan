# XLS-R+SLS model v4 — final-input readiness audit v1

**Дата:** 15 августа 2026

## Решение

Новый v4 final manifest ещё не существует и final inference остаётся запрещённым. Audit
подтверждает, что проект имеет local capacity для нового four-cell final, но ни один ранее
materialized/evaluated candidate нельзя перенести в него. Следующий этап — отдельный immutable
metadata-selection/materialization contract, а не сразу evaluation contract.

## Что навсегда исключено из v4 final

- Exact `79` VoxForge/Qwen CustomVoice `aiden` pairs, manifest SHA-256
  `d1a4e77632cde94ca836202557cfcfc5b6c6b10f4e8b45cc148ee270fca9e91b`, уже прошли один
  Stage-B v2 GPU final run. Completion receipt SHA-256
  `68be1164fc137fd87fbb5abd0902775b2c896be597ab451e29f58b3ad9a1539c` explicitly records
  detector inference; these assets, texts and results are not a v4 reserve or backfill.
- Exact FLEURS/KazakhTTS Stage-C assets тоже already inferred. Stage-C report SHA-256
  `85f47838da866845e9eaa76ad3190d06d03b4d2767e835d99fd67fb8366f4cf6` records one GPU run on
  `120` KK final assets / `60` pairs. They cannot be reused even though the source/model route
  remains useful as a provenance reference.
- All other historical final assets remain excluded by the existing v4 full-history policy. No
  historical error/logit has been read for a selection decision.

## Local candidate capacity, но не final approval

| Role | Fresh reservoir / model evidence | Current status |
| --- | --- | --- |
| RU bona-fide | Common Voice RU v24 full-test screen: `5,600` metadata rows / `1,337` client groups after exposure and literal-text exclusions | Enough pre-extraction capacity; no v4 selection/extraction/QA manifest |
| RU spoof | Local Qwen3 CustomVoice model lock exists, but prior exact Qwen assets were inferred | Needs new text-only output from new frozen RU texts; no cloning and no reuse of 79 prior pairs |
| KK bona-fide | Local FLEURS KK source is CC-BY-4.0, but prior Stage-C exact assets were inferred | Needs a newly selected exact-source slice, not the prior 300-row materialization |
| KK spoof | Local KazakhTTS Tacotron2/PWG lock is approved for personal research, but prior Stage-C exact outputs were inferred | Needs new frozen texts and one new text-only synthesis/QA route; family novelty is not claimable |

The Common Voice metadata receipt is SHA-256
`f862ae667195c733c7deb6bf25f304a6287890ca87d4dc0ee7cb5e06aa6f46b3`; its companion literal
text screen is SHA-256 `4356c3ecbf3a9b68dd7a5d5f4e2ed9347d9c6f105d63d558bfc03dd1403b23d0`.
These are metadata-only screens, not a license decision or authorization to extract, synthesize,
evaluate or re-host audio.

## Required next contract

The final-input contract must, before any WAV extraction or synthesis:

1. freeze a new four-source research-only ledger and bind Common Voice, Qwen, FLEURS and
   KazakhTTS rights/model evidence;
2. re-run current-history source/text/group and exact/near-audio exclusion against all v4 roles,
   including completed calibration assets and the two evaluated historical final layers;
3. select fresh source identities/texts deterministically for the declared final target (currently
   `500` bona-fide/spoof pairs for each of RU and KK), with no outcome-driven reserve/backfill;
4. materialize only the frozen source assets, run one-shot text-only synthesis, decode/QA/VAD,
   acoustic/language review and a complete pair lock before any checkpoint logit;
5. create a later, separate final evaluation contract that hash-binds the selected v4 checkpoint,
   the RU-only temperature report, the locked final manifest and a no-logit preflight.

Until those gates complete, no final asset may be scored, no old pair can be substituted, and no
calibrated v4 final quality or KK calibrated-probability claim is valid.
