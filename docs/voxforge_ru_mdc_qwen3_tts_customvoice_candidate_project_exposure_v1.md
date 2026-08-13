# VoxForge RU / Qwen3-TTS CustomVoice `aiden` — project-exposure audit v1

**Статус:** completed fail-closed audit of the exact `79`-pair candidate against all currently
versioned research configurations and their referenced roles. Detector inference was neither
performed nor authorized.

## Bound evidence

The audit revalidated the immutable `158`-asset candidate, its pairing receipt, the exact-route
audit and the completed two-review acoustic/language gate before scanning configured roles.

- candidate: `158` rows / `79` exact pairs, SHA-256
  `d1a4e77632cde94ca836202557cfcfc5b6c6b10f4e8b45cc148ee270fca9e91b`;
- acoustic gate: `158/158` passed assets, SHA-256
  `bf7a6d84c7ecb71462c128b70f68d8f939ebece916fc11e87c6b9ac8afd26029`;
- exact-route audit: `0` exact Qwen/Aiden route overlap, SHA-256
  `234e1a49ecc06f4fc7025c4e713af858279d4c35b1d366b4ab052d7a847f5513`.

The write-once [exposure receipt](../data/manifests/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_pre_qa_candidate_project_exposure_v1.json)
has SHA-256 `8696bfbea8d9f59451881bcf6ee875ff235c2e281c7b9ddbe5be4ecf74804a72`.

## Current configured-role scope

The receipt pins all `33` JSON files then present under `configs/research`, their hashes and every
manifest reference. Deduplication yielded `18` unique manifests / `12,397` rows. The exact
candidate comparison found:

| Blocking field | Overlap |
| --- | ---: |
| `sample_id` | 0 |
| audio `sha256` | 0 |
| `text_hash` | 0 |

This is configured-role exposure evidence, not a claim that precursor copies are absent from the
manifest inventory: the source/raw/ready/pair receipts intentionally preserve candidate lineage.
The separate historical route audit covers generator-route evidence. Together they establish only
that the exact assets/texts are absent from prior configured model roles and the exact pinned route
was absent from its historical spoof inventory.

## Ограничения и следующий безопасный шаг

The audit does not prove source, speaker, vendor, architecture-family, Russian-native-voice or
organizational reviewer independence. VoxForge GPL-3.0-or-later and the project's personal-research
scope still apply. The audit contains `detector_inference_performed=false` and
`detector_inference_authorized=false`.

The separate immutable
[XLS-R+SLS evaluation contract](research_xlsr_sls_stage_b_v2_voxforge_ru_mdc_qwen3_tts_customvoice_aiden_v1.md)
is now prepared. It pins this receipt, the acoustic gate, exact candidate, selected checkpoint, a
disjoint fixed calibration role, fixed `0.5` boundary, applicable license ledger, implementation
hashes and new write-once output paths. The next safe step is its single no-logit preflight; only a
successful preflight may authorize one inference run.
