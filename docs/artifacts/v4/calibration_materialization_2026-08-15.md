# XLS-R+SLS model v4 — RU calibration materialization и audio-isolation v1

**Дата:** 15 августа 2026

## Решение

Контракт завершён со статусом
`ru_calibration_pairs_frozen_checkpoint_scoring_contract_required`. Он материализовал только
заранее замороженный RU input, выполнил one-shot text-only eSpeak route, technical
decode/QA/VAD и exact/near-audio isolation, затем заморозил complete pair lock. Это **не**
calibration run: checkpoint не загружался, temperature не fit-ился, detector/final inference
не выполнялись.

Контракт
[`xlsr_sls_model_v4_calibration_materialization_v1`](../../../configs/research/v4/xlsr_sls_model_v4_calibration_materialization_v1.json)
имеет SHA-256 `1b661b6011dfed77fb80b58aa8fe56add54bf7618775f71fc3ce7a1eff2e2f4a`.
Machine receipt:
[`xlsr_sls_model_v4_calibration_materialization_v1.json`](xlsr_sls_model_v4_calibration_materialization_v1.json),
SHA-256 `e69388ccbcc85901bda0da60d3e65fcab7b8a721b2bc9f57c275442469cd681f`.

## Выполненная изоляция

- Новый frozen two-source ledger закрепляет только local personal-research materialization/QA и
  pair lock: [CSV ledger](../../../data/licenses/frozen/xlsr_sls_model_v4_calibration_materialization_v1.csv),
  SHA-256 `25f5c571eef4bb8c4fd15cd6ae0a1c589cbbd112571eac31d2f078d70c1755e0`.
  Training, fitting и inference в нём запрещены.
- Точный VoxForge archive повторно bound по имени, размеру `3,795,197,539` bytes и SHA-256
  `7372c6f8…93de557`; материализованы ровно `81` fresh frozen WAV. `79` прошли canonical
  decode/QA/VAD; ranks `19` и `46` отклонены только как `insufficient_speech`.
- Для каждого из этих `79` source-ready texts выполнена ровно одна local text-only eSpeak NG
  1.52.0 synthesis attempt (без reference audio или cloning). `73` synthetic WAV прошли
  decode/QA/VAD; ranks `9`, `12`, `24`, `48`, `61` и `78` отклонены только как
  `insufficient_speech`. Resynthesis, replacement и backfill отсутствуют.
- Current-history screen перед materialization охватил `115` manifest files, `140,012` rows и
  `84,605` unique historical audio hashes. `84,213` имеют canonical 256-bit fingerprint;
  оставшиеся `392` intentionally unavailable ML-DF assets сохранены как exact-only references.
  Исторические и within-pool raw/decoded exact и near-audio gates выполнены до pair lock.
- Complete lock содержит `73` exact-text RU pairs (`146` assets), SHA-256
  `a8a367549f566222690ea199955e19b51315182fe329a30dc765e24edc5b5d71`:
  [pair manifest](../../../data/manifests/v4/xlsr_sls_model_v4_calibration_pairs_frozen_v1.csv).
  Speaker independence не доказана и не заявляется; KK coverage/probability claim отсутствуют.

## Versioned outputs

| Output | Rows | SHA-256 |
| --- | ---: | --- |
| source raw manifest | 81 | `9758b26ad9bdbdd32bc1811309ebc0fedfca5d2ec0358c05d644e5cdc579eb28` |
| source ready manifest | 79 | `f301af5962f913e72d81401b67a89c4b1f9beb28faf442d6d346b24541917b33` |
| eSpeak raw manifest | 79 | `2946033bdfed20397dac4d58a5fbb1c56895bdbca83e1b56cbfedd80cf9bb38b` |
| eSpeak ready manifest | 73 | `a4392d48d38f378baca2bb727532f9e02afc74f9d92a0e6ac8710693c65ce266` |
| full audio-gate inventory | 160 | `aacfda22b6b2224648355b0261dab03e6c747d13f510dbb15b665d0c6a326525` |
| complete pair lock | 146 | `a8a367549f566222690ea199955e19b51315182fe329a30dc765e24edc5b5d71` |

Raw and canonical WAV bytes remain only in Git-ignored `data/raw/v4/` and `data/processed/v4/`
namespaces; they are bound by the manifests and are not added to Git or release.

## Следующий разрешённый шаг

Сначала нужен новый explicit rights/ledger decision: текущий frozen ledger намеренно запрещает
temperature fitting. Только если этот decision законно разрешит fitting, возможен отдельный
immutable checkpoint-scoring-and-calibration contract, который hash-bind-ит 73-pair lock,
selected v4 checkpoint, RU-only method и write-once outputs. До этого запрещены checkpoint
loading, temperature fitting, detector/final inference и любая повторная materialization/synthesis.
