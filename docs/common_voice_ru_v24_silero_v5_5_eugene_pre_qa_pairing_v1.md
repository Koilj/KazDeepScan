# Common Voice RU v24 / Silero V5.5 `eugene` — immutable pairing v1

**Статус:** exact technical-QA-retained pair lock completed. The subsequent full 84-asset
technical acoustic gate and the one governed detector-inference run are complete.

## Pair rule and accounting

The lock binds the original `75` ready bona fide rows, all `75` raw spoof rows, the technical-QA
receipt and its `33` permanent spoof rejects. A rejected spoof excludes its complete pair; no
remaining bona fide-only row, historical Stage-D/v3 pair, resynthesis, replacement, backfill,
metric or detector output participates in selection.

| Layer | Rows |
| --- | ---: |
| Frozen ready bona fide | 75 |
| Raw fixed-eugene spoof | 75 |
| Spoof technical-QA rejects | 33 |
| Retained binary pairs | 42 |
| Immutable candidate assets | 84 |

The candidate contains `42` bona fide and `42` spoof 16 kHz WAVs. Each pair has the same frozen
text ID/text hash, and candidate assets pass license-ledger and SHA-256 asset validation.

## Versioned outputs

- [84-asset candidate manifest](../data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_pairs_v1.csv):
  SHA-256 `9227455d70c30f6f931902951421abb86b62627513feeb1cfc8530cb38bf2d71`;
- [pairing receipt](../data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_pairing_v1.json):
  SHA-256 `579e32ccf4725e27c4024427e6ca3b29fb6ad759efdccb6558cfc5a348b48d01`.

## Следующий безопасный шаг

The [acoustic-gate receipt](common_voice_ru_v24_silero_v5_5_eugene_pre_qa_acoustic_gate_v1.md)
records a full `84/84` technical pass. A project-exposure audit now confirms zero exact
sample/audio/text overlap with all 30 currently configured research roles. This pair lock may
advance only to one separate immutable contract; no inference, resynthesis, replacement, backfill
or error-driven tuning has occurred.
