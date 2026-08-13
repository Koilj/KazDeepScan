# VoxForge RU / Qwen3-TTS CustomVoice `aiden` — technical QA v1

**Статус:** completed raw-synthesis technical QA. Binary pairing, acoustic/language review and
detector inference have not been performed.

## Вход и результат

The QA gate accepted only the completed `79`-row raw layer from
[the one-shot synthesis manifest](../data/manifests/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_pre_qa_raw_v1.csv),
SHA-256 `d378466b47dc4e05e575457ebdda5677ad685465df7d70cc6a9c2d7207f0f990`. It revalidated that
the synthesis receipt records exactly one successful synthetic WAV per bound text, zero failed
attempts and no replacement/backfill before processing.

`AudioPreparationPipeline` decoded every raw WAV, produced mono 16 kHz PCM-16 WAV, applied
quality limits and WebRTC VAD. The ready manifest retains `original_sr=24000` as raw-generation
provenance; the actual normalized files were independently checked as `79` 16 kHz mono PCM-16
WAVs (`52,480`–`236,800` frames).

| Outcome | Rows |
| --- | ---: |
| Raw fixed-Aiden synthetic layer | 79 |
| Ready normalized spoof layer | 79 |
| Fully accounted technical rejects | 0 |

There is no reuse, resynthesis, replacement or backfill. Every raw sample is present exactly once
in the ready layer or the immutable empty rejection report; all manifests/assets pass license-ledger
and SHA-256 validation.

## Versioned outputs

- [ready spoof manifest](../data/manifests/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_pre_qa_ready_v1.csv):
  `79` rows, SHA-256 `18ac2f20a90f30c3fe55444eb7d81761712e0be41151cf90f3118eab70f7174c`;
- [generic rejection report](../data/manifests/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_pre_qa_technical_qa_rejections_v1.json):
  SHA-256 `68af32e27db479074786c0d4509459b3113eb07fb9886578c14beda6d99e832f`;
- [technical-QA receipt](../data/manifests/voxforge_ru_mdc_qwen3_tts_customvoice_aiden_pre_qa_technical_qa_v1.json):
  SHA-256 `aa60514236798829d5e61d90f3cd2591fca10ea0b76f2248c4d0a740cfa85a4b`.

## Следующий безопасный шаг

The exact `79`-pair lock is now published in the
[pairing receipt](voxforge_ru_mdc_qwen3_tts_customvoice_pre_qa_pairing_v1.md). It does not
authorize a language claim: the full `158` assets must receive two independent acoustic/language
reviews before any detector evaluation; a rejected review asset reduces the candidate set without
regeneration or replacement.
