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

## Ограничения и следующий безопасный gate

This selection created no `data/raw/voxforge*`, decoded WAV, manifest, synthetic WAV, pair,
model logit or metric. GPL-3.0-or-later and personal-research restrictions still apply, and the
result is neither source-independent nor speaker-independent.

The attempted UtrobinTTS review is rejected because its unversioned historical model identifier
was already used in `76` spoof rows; see the [route review](voxforge_ru_mdc_utrobinmv_vits_route_review_2026-08-13.md).
The next permitted action is to find another exact **spoof route** whose identifier is absent from
historical spoof manifests, then pin a text-only TTS runtime, weights/artifact hashes,
license/rights scope, fixed public voice and generation parameters. Only a passed route receipt
may authorize selection-bound WAV materialization and its full asset/technical QA; it still cannot
authorize detector inference.
