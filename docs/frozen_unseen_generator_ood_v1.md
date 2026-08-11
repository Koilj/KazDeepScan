# Frozen unseen-generator suite v1

## Назначение

Это **research-only evaluation contract**, а не новый training corpus, calibration protocol или
product claim. Он фиксирует train/dev до оценки и затем проверяет три независимые Kazakh TTS
family, отсутствующие среди generator family train/dev:

| Роль | Источник | Строк | Generator family |
| --- | --- | ---: | --- |
| Train | RuASD raw research | 1 417 | `tts` |
| Dev | PyAra research | 61 | `unspecified_synthesis` |
| Frozen final 1 | KSC + KazEmoTTS | 718 | `gradtts_hifigan_emotional_tts` |
| Frozen final 2 | KSC + Spark-TTS | 762 | `llm_bicodec_controlled_tts` |
| Frozen final 3 | KSC + eSpeak NG | 716 | `formant_rule_based_tts` |

The contract is [`unseen_generator_ood_v1.json`](../configs/research/unseen_generator_ood_v1.json).
It accepts shared corpus provenance `ksc_slr102` among the final tests, but requires zero overlap
of their `sample_id`, audio SHA-256 and `text_hash`. It also requires one exact bona-fide/spoof
pair per final-test text, unique synthetic `source_name` and a spoof family not present in train
or dev.

```bash
kds validate-unseen-generator-suite configs/research/unseen_generator_ood_v1.json \
  --license-ledger data/licenses/license_ledger.csv
```

The validator does not fit a model and does not read scores. Before any future evaluation, create
one new checkpoint selected **only** by the fixed PyAra dev set, freeze it, and evaluate each
final manifest exactly once. Do not select epoch, threshold, calibration, augmentation or model
architecture from any of the three frozen final tests. Report class accuracy, generator-family
strata and confidence intervals separately; do not combine them into a product "accuracy".

LRLspoof is deliberately absent: it is a 452 GB sequential spoof-only release and cannot provide
the binary, source-auditable Kazakh evaluation required by this contract.
