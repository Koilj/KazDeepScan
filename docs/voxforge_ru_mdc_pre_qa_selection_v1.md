# VoxForge Russian / Mozilla Data Collective — immutable pre-QA selection v1

**Статус:** completed metadata-only selection. Это не extraction, audio QA, synthesis, pairing,
acoustic review или detector inference.

## Решение о размере и правило

Заморожена вся доступная text capacity: `81` records, ровно по одному на каждый из `81`
canonical `PROMPTS` text groups и на каждый из `81` distinct conservative contributor groups.
Это не означает `81` verified speakers: source `README` aliases остаются только
privacy-preserving group keys.

Selector повторно проверил exact archive и immutable parent receipts:

- source audit SHA-256 `0e8bd5c7d1e02bedc235adcb3bdb7ed3bc7efdd0ff7637339460e3f43c38272f`;
- metadata exposure screen SHA-256
  `275367a9738bfcc017315cfb3799078c0c3ab1981a318098b0849eaf7893dffe`.

Из `6,412` screen survivors / `194` contributor groups / `81` text groups сначала
seed `2026-08-13-voxforge-ru-mdc-pre-qa-candidate-v1` ранжирует canonical text hashes, затем
строит seeded deterministic augmenting-path maximum matching текстов с разными contributor groups
и, наконец, выбирает одну source record в каждой text/group cell. Правило не читает WAV payload,
не использует duration, QA, model/detector output, metrics или historical final errors. Оба слоя
transcript hashes уникальны среди selected records; replacement/backfill запрещён.

## Неизменяемые outputs

- [selection CSV](../data/manifests/voxforge_ru_mdc_2026_05_pre_qa_selection_v1.csv): `81` rows,
  SHA-256 `d181e24b290ee66cfc190a935b8d23132548306a17e3db1814b778d667202fb9`;
- [selection receipt](../data/manifests/voxforge_ru_mdc_2026_05_pre_qa_selection_v1.json):
  SHA-256 `18dc659ce30a6eaec03cdc27b74e709e066d556b142b063fc2a48b7c4fc1224f`.

The outputs retain only hashed submission/contributor identities and prompt IDs; no source
contributor alias or transcript text is committed. A later materializer must resolve the selected
pseudonymous IDs only against the byte-pinned local archive and must reject any changed binding.

## Ограничения и follow-on state

This selection created no `data/raw/voxforge*`, decoded WAV, manifest, synthetic WAV, pair,
model logit or metric. GPL-3.0-or-later and personal-research restrictions still apply, and the
result is neither source-independent nor speaker-independent.

The attempted UtrobinTTS review remains rejected because its unversioned historical model
identifier was already used in `76` spoof rows. Qwen3-TTS CustomVoice Q8_0 / fixed `aiden` then
passed the exact-route audit with `0` overlaps in `18,764` historical spoof rows; see the
[accepted route review](voxforge_ru_mdc_qwen3_tts_customvoice_route_review_2026-08-13.md).

The completed selection-bound [materialization](voxforge_ru_mdc_pre_qa_materialization_v1.md)
retained `79/81` source rows after technical QA; two quiet rejects are accounted without
replacement. Those exact rows subsequently completed literal-text binding, one-shot synthesis,
technical QA, exact pairing and a full `158/158`
[two-review gate](voxforge_ru_mdc_qwen3_tts_customvoice_pre_qa_acoustic_gate_v1.md). The Qwen
route still uses a documented English baked token, so the gate establishes only exact-asset
Russian audibility and content preservation, not a Russian-native voice claim. Detector inference
was subsequently completed exactly once under the immutable
[XLS-R+SLS contract](research_xlsr_sls_stage_b_v2_voxforge_ru_mdc_qwen3_tts_customvoice_aiden_v1.md).
The selection and its two rejects remain unchanged; repeat inference and error-driven tuning are
prohibited.
