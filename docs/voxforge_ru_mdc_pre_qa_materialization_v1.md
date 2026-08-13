# VoxForge RU — pre-QA materialization v1, 13 августа 2026

**Статус:** completed frozen bona-fide extraction and technical QA. Subsequent synthesis, pairing,
the `158/158` acoustic/language gate and exactly one governed detector run are complete.

## Immutable inputs and outputs

The materializer revalidated the exact `3,795,197,539`-byte VoxForge archive and the frozen
`81`-row selection before reading any selected member. Each selection pseudonym was rebound to its
same archive submission, prompt, contributor-group and both transcript hashes in memory; only the
81 selected regular WAV members were atomically written outside Git under `data/raw/`.

- [raw manifest](../data/manifests/voxforge_ru_mdc_2026_05_pre_qa_raw_v1.csv): `81` rows,
  SHA-256 `be370c3018825f54daa54a641a387f2037cd699d3037a34da00704f713ca69be`;
- [ready manifest](../data/manifests/voxforge_ru_mdc_2026_05_pre_qa_ready_v1.csv): `79` rows,
  SHA-256 `001eba24021673327a17b7fc94418b334c902007cdee507ddba6a17a06f660e4`;
- [materialization receipt](../data/manifests/voxforge_ru_mdc_2026_05_pre_qa_materialization_v1.json),
  SHA-256 `da4d53054379c3698858f5ae75b5289e40d0f82bd5fe870a2efa4c5b0beeb353`.

`AudioPreparationPipeline` decoded each raw WAV to mono PCM 16 kHz, applied technical quality
checks and WebRTC VAD. `79/81` are ready. Ranks `61` (`ru_0089`) and `70` (`ru_0076`) were
rejected as `signal_too_quiet`; both are explicitly recorded and cannot be replaced, backfilled or
resynthesized. Raw source WAVs and normalized assets remain ignored by Git.

## Boundary

This confirms only source-asset binding and technical readiness. It does not establish the human
speaker identity, acoustic language quality, a synthetic counterpart, or a binary candidate. The
Qwen route remains fixed to its documented English `aiden` token. The subsequent full-asset review
passed, but only establishes Russian audibility and content preservation for the exact reviewed
bytes; it does not establish a Russian-native voice.

## Follow-on state

The separate [literal-text binding](voxforge_ru_mdc_qwen3_tts_customvoice_pre_qa_text_binding_v1.md)
for exactly the `79` ready rows retained both source transcript hashes and the two accounted
rejects. Exactly one Qwen3-TTS CustomVoice / `aiden` WAV per bound text, technical QA, pair lock
and the [two-review gate](voxforge_ru_mdc_qwen3_tts_customvoice_pre_qa_acoustic_gate_v1.md) are
now complete. The subsequent project-exposure audit, immutable evaluation contract and preflight
passed, and exactly one final run completed; see the
[final receipt](research_xlsr_sls_stage_b_v2_voxforge_ru_mdc_qwen3_tts_customvoice_aiden_v1.md).
No replacement/backfill, repeat inference or final-error tuning is allowed.
