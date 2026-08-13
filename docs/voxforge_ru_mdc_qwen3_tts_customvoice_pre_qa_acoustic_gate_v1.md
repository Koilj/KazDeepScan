# VoxForge RU / Qwen3-TTS CustomVoice `aiden` — acoustic/language gate v1

**Статус:** две completed 158-row формы прошли fail-closed gate. Все `158/158` exact assets
получили по два полных `pass/yes/yes/yes/yes` решения. Gate report remains a pre-inference record;
the subsequent immutable contract completed exactly one detector run.

## Immutable packet

The packet contains exact retained bytes from the immutable `79`-pair lock. Before publication it
revalidated the pair receipt, license ledger, SHA-256 of all `158` audio assets, and the literal
`PROMPTS` transcript from the byte-pinned VoxForge archive. It contains `79` bona-fide and `79`
fixed-Aiden spoof WAVs, grouped only by their exact shared text hash.

- [acoustic packet](../data/manifests/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_pre_qa_acoustic_packet_v1.csv):
  `158` rows, SHA-256 `59e6a83a9866412c12d3daf0d235d990e43548ef07c32eb2dd25baf5d5e3c3dc`;
- [reviewer A form](../data/manifests/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_pre_qa_acoustic_review_reviewer_a_v1.csv):
  `158` решений от `reviewer_1`, SHA-256
  `9f07f7eaf1f5bd19e0345fd08218b3a218cfaa07342cee642e013d7f31bf490e`;
- [reviewer B form](../data/manifests/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_pre_qa_acoustic_review_reviewer_b_v1.csv):
  `158` решений от `reviewer_2`, SHA-256
  `559d60e6a04f21f7b2db7d8f582d92812e9e96a538e737e7537d4b5e102ef116`.

The evaluator accepted exactly two complete forms with distinct technical reviewer pseudo-IDs,
matching packet SHA-256 and unchanged packet-bound fields. This is a technical anti-duplication
requirement, not proof of organizational independence or of the actual review assignment.

The packet retains the original CRLF bytes emitted by the standard CSV writer. Filling the two
response forms through the spreadsheet/editor workflow produced LF completed copies, as in the
earlier Common Voice/Silero gate. The hashes above bind those exact completed bytes; none of the
three versioned evidence files may now be normalized or edited.

## Как заполнить формы

Each reviewer listens to the asset named by `relative_path` under local `data/`, checks that its
SHA-256 remains the packet's `audio_sha256`, and independently changes every row:

- `review_status`: `pass`, `fail` or `inconclusive`;
- `intelligible`, `russian_audible`, `lexical_content_preserved`,
  `severe_artifacts_absent`: `yes`, `no` or `unknown`;
- `notes`: a concise reason for any non-pass result.

The only passing decision is: both reviewers independently set `review_status=pass` and all four
answers to `yes` for the same exact WAV. Any `fail`, `inconclusive`, `no`, `unknown`, missing
decision or disagreement would make that asset ineligible; its full pair would be excluded without
resynthesis, replacement or backfill. In the completed forms no such decision occurred.

The write-once
[gate report](../data/manifests/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_pre_qa_acoustic_gate_report_v1.json)
has SHA-256 `bf7a6d84c7ecb71462c128b70f68d8f939ebece916fc11e87c6b9ac8afd26029`.
It contains `316` review rows, `158` per-asset `pass` results and explicitly records
`detector_inference_performed=false`. The report confirms only intelligibility, Russian
audibility, literal-content preservation and absence of severe artifacts for these exact WAVs; it
does not establish Russian-native speaker identity, source/speaker/vendor/architecture
independence, calibration, product quality or detector performance.

## Follow-on state

The completed gate is immutable evidence. The subsequent
[project-exposure audit](voxforge_ru_mdc_qwen3_tts_customvoice_candidate_project_exposure_v1.md)
covered `33` research configs / `18` referenced manifests / `12,397` prior rows and found
`0/0/0` sample/audio/text overlap. Only a separately reviewed immutable evaluation contract may
bind that audit, this gate, a fixed checkpoint/calibration/boundary, license limits and write-once
output paths. The gate report's `evaluation_contract_authorized=true` permits preparation of that
contract; it does not itself authorize or perform detector inference.
That contract and its preflight subsequently passed, and the single
[write-once result](research_xlsr_sls_stage_b_v2_voxforge_ru_mdc_qwen3_tts_customvoice_aiden_v1.md)
is now complete. Preserve the gate bytes and do not repeat inference or tune on final errors.
