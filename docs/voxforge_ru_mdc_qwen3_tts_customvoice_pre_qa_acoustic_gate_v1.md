# VoxForge RU / Qwen3-TTS CustomVoice `aiden` — acoustic/language gate v1

**Статус:** packet и две формы опубликованы; gate **pending**. Никакого acoustic decision или
detector inference ещё нет.

## Immutable packet

The packet contains exact retained bytes from the immutable `79`-pair lock. Before publication it
revalidated the pair receipt, license ledger, SHA-256 of all `158` audio assets, and the literal
`PROMPTS` transcript from the byte-pinned VoxForge archive. It contains `79` bona-fide and `79`
fixed-Aiden spoof WAVs, grouped only by their exact shared text hash.

- [acoustic packet](../data/manifests/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_pre_qa_acoustic_packet_v1.csv):
  `158` rows, SHA-256 `59e6a83a9866412c12d3daf0d235d990e43548ef07c32eb2dd25baf5d5e3c3dc`;
- [reviewer A form](../data/manifests/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_pre_qa_acoustic_review_reviewer_a_v1.csv):
  `158` pending rows, SHA-256 `fa2679deda16aa8d17676aea3626c8038c176ff96c113b2a54ef192dbd005bfa`;
- [reviewer B form](../data/manifests/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_pre_qa_acoustic_review_reviewer_b_v1.csv):
  `158` pending rows, SHA-256 `70f7e85ddf4c083a71df23cad8443d5981706d18846b68abb7d1bc49d1a44efe`.

The forms use distinct technical reviewer pseudo-IDs. This is a technical anti-duplication
requirement, not proof of organizational independence.

The three immutable CSV files intentionally retain CRLF line endings emitted by the standard CSV
writer. `git diff --check` reports those CRLF bytes as trailing whitespace; they must not be
normalized, because that would change the SHA-256 bindings above. All non-CSV staged paths are
checked separately.

## Как заполнить формы

Each reviewer listens to the asset named by `relative_path` under local `data/`, checks that its
SHA-256 remains the packet's `audio_sha256`, and independently changes every row:

- `review_status`: `pass`, `fail` or `inconclusive`;
- `intelligible`, `russian_audible`, `lexical_content_preserved`,
  `severe_artifacts_absent`: `yes`, `no` or `unknown`;
- `notes`: a concise reason for any non-pass result.

The only passing decision is: both reviewers independently set `review_status=pass` and all four
answers to `yes` for the same exact WAV. Any `fail`, `inconclusive`, `no`, `unknown`, missing
decision or disagreement makes that asset ineligible; its full pair must be excluded without
resynthesis, replacement or backfill.

## Следующий безопасный шаг

Fill both forms independently and return them unchanged in schema and packet binding. Then run the
gate evaluator to create a versioned report. Only if all `158` assets pass may a separate immutable
evaluation contract decide whether one detector-inference run is authorized; that authorization is
not created by this packet or by the reviews alone.
