# B0 frozen unseen-generator suite v1 — один финальный прогон

## Protocol

Контракт [`unseen_generator_ood_v1.json`](../configs/research/unseen_generator_ood_v1.json)
прошёл validation до fitting. Новый checkpoint
`models/b0-unseen-generator-suite-v1.pt` обучен только на RuASD train (`1 417`) с seed
`20260817`; epoch выбирался только по PyAra dev (`61`). Минимальный dev loss `0.5541` был на
epoch 4. Ни один из трёх frozen Kazakh final test не участвовал в выборе epoch, threshold,
calibration, augmentation или architecture.

После выбора checkpoint каждый final manifest оценён ровно один раз на CUDA, без calibration:

| Frozen family | Files | Correct | Balanced accuracy | Bona fide recall | Spoof recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| KazEmoTTS Grad-TTS + HiFi-GAN | 718 | 659 | 0.9178 | 0.8357 (300/359) | 1.0000 (359/359) |
| Spark-TTS LLM + BiCodec | 762 | 711 | 0.9331 | 0.8740 (333/381) | 0.9921 (378/381) |
| eSpeak NG formant | 716 | 673 | 0.9399 | 0.8799 (315/358) | 1.0000 (358/358) |

The 95% Wilson intervals for ordinary accuracy are respectively `0.8954–0.9358`,
`0.9131–0.9487` and `0.9201–0.9551`. These intervals describe only the finite frozen clips;
they do not establish deployment robustness, a risk score, calibration or speaker independence.
The recurring limitation is visible in bona-fide KSC recall, not in a claim that all unseen
generators are solved.

## Strata and recorded limitation

Spark-TTS has three spoof errors: `male:high:low` (30/32) and `male:low:high` (30/31); every
other declared control passed in this one run. eSpeak NG has 358/358 spoof correct across its
twelve deterministic controls. All corresponding per-voice Wilson intervals were emitted by
one invocation of `scripts/evaluate_b0.py`; no final-set decision was made from them.

KazEmoTTS was evaluated by the earlier `train_b0_matrix.py` path, which at that moment emitted
class metrics but not family/voice strata. It is deliberately reported as **class-only** above;
the frozen KazEmoTTS test will not be re-run merely to fill that table. The trainer has now been
corrected so future final evaluations save `final_test_stratified_metrics` in the
checkpoint and JSON output.

This is personal research only. The shared KSC provenance across final tests is audited by the
suite contract: their sample IDs, audio SHA-256 and text hashes are disjoint. It does not make
the tests speaker-independent, nor does it authorize use of a model/API score.
