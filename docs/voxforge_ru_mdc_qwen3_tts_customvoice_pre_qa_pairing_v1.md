# VoxForge RU / Qwen3-TTS CustomVoice `aiden` — immutable pairing v1

**Статус:** completed exact binary pair lock. The subsequent two-review acoustic/language gate
passed all `158/158` assets; detector inference has not been performed.

## Pairing rule

The gate accepted only the completed `79`-row VoxForge bona-fide ready manifest, Qwen raw/ready
spoof manifests, one-shot synthesis receipt and synthetic technical-QA receipt. It revalidated
every asset hash and ledger entry before producing a candidate. It admits a pair only when both
the `text_hash` and `text_id` match exactly; no duration, waveform score, detector output or other
metric influenced retention.

The synthetic QA retained all rows, so the candidate contains all and only the frozen ready layer:

| Layer | Rows |
| --- | ---: |
| Frozen ready VoxForge bona-fide inputs | 79 |
| Raw fixed-Aiden synthesis | 79 |
| Technical-QA rejected spoof rows | 0 |
| Ready fixed-Aiden spoof rows | 79 |
| Exact retained pairs | 79 |
| Candidate assets | 158 |

Every pair preserves one literal source text. There is no reselection, resynthesis, replacement,
backfill, reuse or metric-based choice. This records a matched research candidate only; it makes
no Russian-native, speaker identity, group or architecture-independence claim.

## Versioned outputs

- [paired candidate manifest](../data/manifests/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_pre_qa_pairs_v1.csv):
  `158` rows, SHA-256 `d1a4e77632cde94ca836202557cfcfc5b6c6b10f4e8b45cc148ee270fca9e91b`;
- [pairing receipt](../data/manifests/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_pre_qa_pairing_v1.json):
  SHA-256 `2c26bb5151e4278bdc633cad44b3f59764f8929e5234ed58ac6080312825c994`.

## Следующий безопасный шаг

The full packet and two independent `158`-asset forms are now completed and bound by the
[acoustic-gate receipt](voxforge_ru_mdc_qwen3_tts_customvoice_pre_qa_acoustic_gate_v1.md). The next
safe step is a current project-exposure audit followed, only if clean, by a separate immutable
evaluation contract. Detector inference remains prohibited until that contract and its no-logit
preflight pass.
