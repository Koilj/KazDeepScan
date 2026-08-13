# XLS-R+SLS Stage-B v2 — VoxForge RU / Qwen3-TTS CustomVoice `aiden` contract v1

**Статус:** immutable research evaluation contract prepared and technically validated without
audio inference. No preflight receipt, execution lock, logits or detector result exists yet.

## Fixed final layer

The final layer is exactly the completed `79`-pair / `158`-asset VoxForge RU and fixed Qwen
CustomVoice `aiden` candidate. Its manifest SHA-256 is
`d1a4e77632cde94ca836202557cfcfc5b6c6b10f4e8b45cc148ee270fca9e91b`.

The full acoustic/language gate is pinned by report SHA-256
`bf7a6d84c7ecb71462c128b70f68d8f939ebece916fc11e87c6b9ac8afd26029`: all `158` exact WAV
bytes have two complete pass decisions. Distinct reviewer pseudo-IDs are only a technical minimum;
they do not prove organizational independence.

The preceding project-exposure audit is pinned by SHA-256
`8696bfbea8d9f59451881bcf6ee875ff235c2e281c7b9ddbe5be4ecf74804a72`. It covers all `33`
then-versioned research configs and their `18` referenced manifests (`12,397` rows), with exact
candidate overlap `0/0/0` for sample ID/audio SHA-256/text hash. The route audit is also pinned
(`234e1a49ecc06f4fc7025c4e713af858279d4c35b1d366b4ab052d7a847f5513`): no exact
checkpoint/runtime route, legacy Qwen identifier or `aiden` alias was found in its historical
scope. This supports exact-route absence only, not architecture-family, vendor-family,
Russian-native-voice or speaker independence.

## Frozen model, calibration and rights boundary

The contract reuses the already selected XLS-R+SLS Stage-B v2 checkpoint, not a new model choice:
checkpoint SHA-256 `e112c5c93f2a5af0c567b85eccac0a617c37fa79b4d7cc2b29b4b3289f2764cd`, selected trainable
state SHA-256 `d03adfe2ebfe7b7361b2a0d9b7902ef7251f7faf139d37a30531e2211e2dd738` and pinned XLS-R
revision `1a640f32ac3e39899438a2931f9924c02f080a54`.

Temperature may be fitted only on the existing 976-row PyAra calibration role. The model boundary
is fixed at calibrated spoof probability `0.5`; threshold selection is prohibited. Static contract
validation found zero calibration/final overlap for `sample_id`, audio SHA-256,
`parent_group_id`, `speaker_pseudo_id` and `text_hash`.

The three-source frozen ledger has SHA-256
`f23415c2e57995426e2562059accd24ede5b2da3abad449e99425a3f1c6f2f16`. It contains only PyAra,
VoxForge RU and the fixed Qwen/Aiden route. VoxForge GPL-3.0-or-later and all explicit
personal-research restrictions remain binding; this contract does not authorize training,
calibration on the final route, re-hosting, product use or a commercial claim.

## Write-once contract

The immutable [plan](../configs/research/xlsr_sls_stage_b_v2_voxforge_ru_mdc_qwen3_tts_customvoice_aiden_v1.json)
has SHA-256 `9e36b5d6a35cfa0b796ff24e62f3bfa78667d0b1d9da993f1863a2fe61c421cc`. It pins the new
VoxForge/Qwen wrapper, its V5.5 contract base, the Stage-D numerical engine, all shared
implementation dependencies, evidence, checkpoint, calibration and fresh write-once output paths.
Unlike the earlier V5.5 wrapper, the final report path explicitly records
`detector_inference_performed=true` only after final logits have completed.

The three output paths under `artifacts/` are currently absent. Plan loading and strict input
validation read no model logits and performed no detector inference.

## Следующий безопасный шаг

Commit the contract and frozen ledger first. Then run exactly one `--validate-only` preflight. It
must validate all `1,134` assets (`976` calibration + `158` final), pinned hashes, license scope,
leakage, CUDA/BF16 and write a new preflight receipt without loading final logits. Only a successful
preflight may authorize the single write-once inference run. Any contract/input/output mismatch
must fail closed; do not replace assets, change calibration/boundary or retry around an error.
