# XLS-R+SLS Stage-B v2 — Common Voice RU / Silero V5.5 `eugene` contract v1

**Статус:** immutable research contract is prepared and its static inputs have been validated.
No preflight, execution lock, calibration fit, detector inference, threshold selection or final
metric has been performed.

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

The only safe next action is one `--validate-only` preflight over the 976 calibration plus 84 final
assets. It must create its write-once receipt before any logits. Only if that receipt passes may
the one CUDA/BF16 inference run create its execution lock and report. Candidate bytes, reviews,
calibration role, checkpoint, boundary and code cannot be changed after preflight; a failed
preflight does not authorize replacement, backfill, rerun or tuning.

This remains personal research rather than product quality. It is asset-level blind only: the
candidate's base source, voice provenance and broader Silero family are not independent.
