# Common Voice RU v24 / Silero V5.5 `eugene` — acoustic gate v1

**Статус:** two completed 84-row response forms have passed the fail-closed technical gate.
The gate authorizes preparation of one new immutable evaluation contract; detector inference has
not been performed.

## Immutable packet

The packet has exactly `84` rows / `42` matched binary pairs from the completed pair lock. Each
row binds label, sample ID, processed relative path, exact audio SHA-256, text hash and literal
Common Voice transcript.

- [acoustic packet](../data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_acoustic_packet_v1.csv):
  84 assets, SHA-256 `3cf55fd48b6bf81ae593661df2a40b4b9e4fb11eb238ca1fce553dea1a3f29f7`;
- [worksheet A](../data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_acoustic_review_reviewer_a_v1.csv):
  84 decisions by `reviewer_1`, SHA-256
  `48a3d21eb5c3f580291fa9ea266c70acf578ea2e4ab67c199b2dfafbff6d628d`;
- [worksheet B](../data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_acoustic_review_reviewer_b_v1.csv):
  84 decisions by `reviewer_2`, SHA-256
  `645a067c81fa26c1a8d4de7a592bdb5686140c49202685f8fe58a2ac3dd271e3`.

The packet retains its generated CRLF bytes. The completed form bytes are now immutable evidence:
do not normalize or edit any of these CSVs because their SHA-256 bindings would change.

## Required real-review procedure

Two genuinely independent reviewers must each listen to all 84 pinned WAVs and create one **new
response copy** from a template. Before review, replace every placeholder reviewer ID in that new
copy consistently with a real distinct pseudonymous reviewer ID; do not alter packet-bound fields.
For every asset each reviewer records:

- `review_status=pass` only if `intelligible`, `russian_audible`,
  `lexical_content_preserved` and `severe_artifacts_absent` are all `yes`;
- otherwise records `fail` or `inconclusive`, with a concise note.

The evaluator accepted exactly two 84-row forms with distinct reviewer IDs and matching packet
SHA-256. All `84/84` assets received two complete `pass/yes/yes/yes/yes` decisions, so the
technical gate passes. The write-once [gate report](../data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_acoustic_gate_report_v1.json)
has SHA-256 `cb9604a6a2c41fa16ce6e0c8c1947e44c0d0d21d626b88ccbf90673c872c3631`.

Distinct pseudonymous IDs are a technical minimum, not proof of organizational independence;
the actual assignment must satisfy the protocol's independence requirement. The report confirms
only the exact-byte acoustic criteria. It does not establish source, speaker, vendor or
architecture-family independence, calibration, product quality, or a detector result.

The completed gate authorizes one new evaluation contract only. A fail, missing decision, altered
binding or incomplete form would fail closed and would not authorize resynthesis, replacement,
backfill, selective omission or tuning.
