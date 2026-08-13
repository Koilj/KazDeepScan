# Common Voice RU v24 / Silero V5.5 `eugene` — acoustic gate v1

**Статус:** packet и две blank worksheets prepared; human acoustic decisions have not been made.
This is a hard governance barrier before any evaluation contract or detector inference.

## Immutable packet

The packet has exactly `84` rows / `42` matched binary pairs from the completed pair lock. Each
row binds label, sample ID, processed relative path, exact audio SHA-256, text hash and literal
Common Voice transcript.

- [acoustic packet](../data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_acoustic_packet_v1.csv):
  84 assets, SHA-256 `3cf55fd48b6bf81ae593661df2a40b4b9e4fb11eb238ca1fce553dea1a3f29f7`;
- [worksheet A](../data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_acoustic_review_reviewer_a_v1.csv):
  SHA-256 `e53bbce49edb4c84a2d417c681e5e84d0a5a311873d130a430c45447fb7ce3ef`;
- [worksheet B](../data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_acoustic_review_reviewer_b_v1.csv):
  SHA-256 `2c96ade8b6726a24b8a2762576fb5683a1ab3f3eb71d165b1478c4ba2fe224bc`.

The worksheets are deliberate immutable templates: every row remains `inconclusive` with
`unknown` answers. `reviewer_a` and `reviewer_b` are placeholders, not people, assignments or
decisions. These three CSVs use their generated CRLF bytes; do not normalize or edit them because
their SHA-256 bindings would change.

## Required real-review procedure

Two genuinely independent reviewers must each listen to all 84 pinned WAVs and create one **new
response copy** from a template. Before review, replace every placeholder reviewer ID in that new
copy consistently with a real distinct pseudonymous reviewer ID; do not alter packet-bound fields.
For every asset each reviewer records:

- `review_status=pass` only if `intelligible`, `russian_audible`,
  `lexical_content_preserved` and `severe_artifacts_absent` are all `yes`;
- otherwise records `fail` or `inconclusive`, with a concise note.

The evaluator accepts exactly two 84-row forms with distinct reviewer IDs and matching packet
SHA-256. It authorizes a new evaluation contract only if every one of the 84 assets gets two
independent complete `pass/yes/yes/yes/yes` decisions. Distinct pseudo-IDs alone do not prove
organizational independence; that must be true in the actual review assignment.

No report is published from these blank forms and no inference is authorized. A fail, missing
decision, altered binding or incomplete form fails closed; it does not authorize resynthesis,
replacement, backfill or selective omission.
