# XLS-R+SLS Stage-B v2 — Common Voice RU / Silero V5.5 `eugene` contract v1

**Статус:** immutable research contract, its one write-once `--validate-only` preflight and exactly
one CUDA/BF16 inference run are complete. The candidate, reviews, checkpoint, calibration role and
boundary were not changed after preflight; repeat inference and error-driven tuning are forbidden.

## Fixed final layer

The final layer is exactly the completed `42`-pair / `84`-asset Common Voice RU and fixed Silero
V5.5 `eugene` candidate. Its candidate manifest SHA-256 is
`9227455d70c30f6f931902951421abb86b62627513feeb1cfc8530cb38bf2d71`.

The full technical acoustic gate is pinned by its report SHA-256
`cb9604a6a2c41fa16ce6e0c8c1947e44c0d0d21d626b88ccbf90673c872c3631`: all 84 exact WAV bytes
have two complete pass decisions. The gate's distinct pseudonymous reviewer IDs are only a
technical minimum; they do not themselves prove organizational independence.

The preceding project-exposure audit is pinned by SHA-256
`6071deb2f60ca914e475611addf81ef2cf81b485c2b9b86826c5a135c0cca3ff`. It covers all 30
then-versioned research configs and their 17 referenced manifests (`12,313` rows), with exact
candidate overlap `0/0/0` for sample ID/audio SHA-256/text hash. The exact-route audit is also
pinned (`5850d8d36bb72191f7e9a9516edec14552bcf1fe83aecea154a9bb90964fb955`): no exact V5.5/eugene
route row was found among its historical inventory, but legacy Silero evidence forbids
architecture-family, vendor-family and speaker-independence claims.

## Frozen model and rights boundary

The contract reuses the already selected XLS-R+SLS Stage-B v2 checkpoint, not a new model choice:
checkpoint SHA-256 `e112c5c93f2a5af0c567b85eccac0a617c37fa79b4d7cc2b29b4b3289f2764cd`, selected trainable
state SHA-256 `d03adfe2ebfe7b7361b2a0d9b7902ef7251f7faf139d37a30531e2211e2dd738` and pinned XLS-R
revision `1a640f32ac3e39899438a2931f9924c02f080a54`.

Temperature may be fitted only on the existing 976-row PyAra calibration role. The model boundary
is fixed at calibrated spoof probability `0.5`; no threshold selection is allowed. The minimal
three-source ledger has SHA-256
`6de966f1197626e15ac786a38f2abbd211f701a51979c7378b79fb611c9100e8`. Silero's CC-BY-NC-SA route
remains personal-research-only: product use, training and calibration of that route are forbidden.
The contract is the required governance condition for this single detector-evaluation run, not a
product or commercial authorization.

## Write-once execution lock

The immutable [plan](../configs/research/xlsr_sls_stage_b_v2_common_voice_ru_v24_silero_v5_5_eugene_v1.json)
has SHA-256 `cdf3fcbb496006478e575c024963cca497854dae1ce17775e58d95ae4d74cadf`. It pins both the
new V5.5-specific wrapper and its shared Stage-D numerical engine by SHA-256, plus every
implementation dependency, input receipt and output path.

The write-once preflight validated all `1,060` assets (`976` calibration plus `84` final) and
confirmed Python `3.13.15`, Torch `2.11.0+cu128`, CUDA `12.8` and BF16 support on an NVIDIA
GeForce RTX 5060 Ti. It performed no training, threshold selection or detector inference. Its
ignored local receipt is
`artifacts/xlsr-sls-stage-b-v2-common-voice-ru-v24-silero-v5-5-eugene-v1.preflight.json`, SHA-256
`3df30bd5a70bcb471d57db4a85765658396e6a2491ab9937731018e37e4206f3`.

## One-time final result

The one-time run wrote its execution lock before calibration/final logits (SHA-256
`0bad2d746c2036068e1c6ae1d160b294ab22a351b3a56f00867825b1fdb28015`) and completed in 12.26 s.
The immutable ignored local report is
`artifacts/xlsr-sls-stage-b-v2-common-voice-ru-v24-silero-v5-5-eugene-v1.report.json`, SHA-256
`06744ecd71244791efe431c6473b59c8c9962f7b27273ed919495365bbe48991`.

There is one immutable status-metadata defect: the final report's top-level
`detector_inference_performed` remains `false`, inherited from the no-logit preflight mapping,
although its execution lock, `status=ok` / `mode=one_time_gpu_inference` and 84 final sample
results prove that the run occurred. The report was not altered and the run was not repeated. The
versioned [execution reconciliation receipt](../data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_execution_reconciliation_v1.json)
binds the plan, preflight, lock and report and records this defect; SHA-256
`0a301c3e4f14d7a5ea048b0cfda7bfae7eda83b6913b787b34d10fc7c079c5b4`.

Temperature was fitted only on the fixed 976-row calibration role (`1.29954`; NLL
`0.15976 → 0.15424`), without threshold selection. With the pre-pinned calibrated boundary `0.5`,
the exact 84-asset layer produced:

| Metric | Result |
| --- | ---: |
| Accuracy, Wilson 95% CI | 84/84 = 1.0000 [0.9563, 1.0000] |
| Bona-fide recall | 42/42 = 1.0000 [0.9162, 1.0000] |
| Spoof recall | 42/42 = 1.0000 [0.9162, 1.0000] |
| Balanced accuracy | 1.0000 |
| Pairs with both assets correct | 42/42 = 1.0000 [0.9162, 1.0000] |

Peak allocated VRAM was 1,645,827,072 bytes. This is a fixed-set result, not permission to retry
or tune the model. Its apparent perfection is especially not a generalization claim: all bona-fide
assets originate from Common Voice and all spoof assets from this one fixed V5.5/eugene route.
The test is neither source-, speaker-, vendor- nor architecture-family-independent, and remains
personal research rather than product quality.
