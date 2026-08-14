# Changelog

Все значимые изменения KazDeepScan фиксируются здесь. Immutable research receipts и более
подробная история находятся в `PROJECT_STATUS.md` и `docs/`.

## [v1.0.0-research] — 2026-08-14

Первый завершённый personal-research release.

- безопасный audio decode/QA/VAD/windowing pipeline;
- строгие manifest, asset, license, consent и leakage validators;
- воспроизводимые B0 и XLS-R+SLS training/evaluation contracts;
- завершённые RU/KK/mixed, Stage-C, Stage-D, V5.5/eugene и VoxForge/Qwen research receipts;
- корректно остановленный Denis/VoxCPM2 route `stop_below_minimum_60` без backfill/inference;
- CLI `kds 0.1.0 (KazDeepScan v1.0 Research)` и FastAPI/OpenAPI `1.0.0-research`;
- exact-hash Linux test overlay с `pyarrow 22.0.0`; optional ToneSpeak Parquet tests больше не
  skipped в release QA;
- fail-closed API без bundled scorer, raw audio или model weights.

Python distribution metadata сохранено как `0.1.0`, чтобы не менять hash-pinned
`pyproject.toml`/`uv.lock` завершённых write-once runs. Это не speaker-independent,
product-calibrated или commercially cleared detector release.
