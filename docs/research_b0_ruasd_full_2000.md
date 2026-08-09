# B0 on full RuASD research-2000 — 9 August 2026

## Protocol

Checkpoint: `models/b0-ruasd-full-research-2000.pt` (local, Git-ignored).  It was trained on
the ready manifest from [research_ruasd_full_v1.md](research_ruasd_full_v1.md), five epochs on
the local RTX 5060 Ti, `seed=20260809`, batch size 16 and `--purpose research`.

The dataset is Russian, raw-only and transcript-leakage-safe, but **not speaker- or spoof-voice-
disjoint**.  The checkpoint is an experimental artifact, not a calibrated detector or a release.

## Training and in-source holdout

Best dev loss was `0.2500` at epoch 5; dev balanced accuracy was `0.9085`.  The held-out RuASD
test split has 195 records (88 bona-fide, 107 spoof): loss `0.3163`, ordinary accuracy `0.8769`,
bona-fide recall `0.9205`, spoof recall `0.8411`, balanced accuracy `0.8808`.

This is a useful pipeline sanity result only.  It does not establish generalization to new
speakers, voice identities, channels or generators.

## Transfer checks, all uncalibrated

| Evaluation set | Rows / labels | Result | Interpretation |
| --- | ---: | --- | --- |
| ML-DF Italian OOD | 192 / 94 bona-fide, 98 spoof | balanced accuracy `0.9262` | Cross-lingual source shift, but a small and potentially easy benchmark; not a Russian/Kazakh deployment claim. |
| PyAra Russian test | 44 / 22 bona-fide, 22 spoof | balanced accuracy `0.7273`; bona-fide `1.0000`, spoof `0.4545` | The model still misses many PyAra spoofs. This is the clearest current warning against claiming robust transfer. |
| KSC Kazakh test | 245 bona-fide only | bona-fide recall `0.9673` | A one-class false-positive check only; balanced accuracy is unavailable. |
| RuASD shard-000000 fake-only | 100 spoof only | spoof recall `0.9100` | No exact sample overlap with this run (0/100), but it is the same RuASD source and generator ecosystem. It is **not** independent OOD and must not be pooled with the rows above. |

All metrics use the model's default logit sign boundary and are not calibrated probabilities,
threshold recommendations, EER, or fraud/risk scores.

## Next research step

Compare the current B0 against a source-mixed training protocol only after defining an explicit
held-out source matrix.  In particular, do not merge the RuASD shard-000000 OOD manifest with a
full-RuASD training run, and do not turn the strong ML-DF number into a generalization claim.
