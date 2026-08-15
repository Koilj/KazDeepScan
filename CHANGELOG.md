# Changelog

Все значимые изменения KazDeepScan фиксируются здесь. Immutable research receipts и более
подробная история находятся в `PROJECT_STATUS.md` и `docs/`.

## Unreleased — documentation reconciliation

- актуальные сводки обновлены по первичному v4 final-evaluation receipt от 16 августа 2026 года:
  final GPU run завершён на `792` reviewed pairs (`332` RU, `460` KK);
- устранены устаревшие сводные утверждения, что v4 final ещё не запускался; historical receipts
  не изменялись и по-прежнему описывают состояние только на момент своего write-once этапа;
- v4 final явно закрыт от повторного inference, threshold/calibration work, retraining, backfill
  и resynthesis. Код, данные, manifests, hashes, model weights и versioned receipts не менялись.
- добавлен post-v4 local-only roadmap без новых источников: он отделяет locked v4 от уже
  реализованной B0 проверки внешнего пользовательского аудио и не авторизует новые ML-runs.

## Post-v1.0 — local storage cleanup — 2026-08-14

- удалены только local caches, exact duplicate первого RuASD TAR, unused `ffplay`
  и Italian ML-DF OOD media bytes, непригодные для RU/KK v4 train;
- protected `/home/ruslan/Downloads/269-lockdown/`, основные RU/KK sources, TTS-family
  weights и old write-once evidence сохранены;
- optional `pyarrow` import переведён на `importlib`, чтобы fresh-cache mypy не
  зависел от наличия test overlay; runtime fail-closed behavior не изменилось;
- exact temporary `pyarrow==22.0.0` overlay подтвердил `318 passed`; временный
  overlay и его package cache после проверки удалены.

## Post-v1.0 — local user-audio research inference v1 — 2026-08-14

- добавлен strict `research_user_audio_only` contract с file/state SHA-256 checkpoint gate;
- добавлены `kds validate-research-inference` и opt-in `kds research-infer`;
- добавлен отдельный `/v1/research/analyze` ASGI factory с обязательным acknowledgment;
- output явно uncalibrated, не содержит `risk_score` и запрещает probability/fraud/product claims;
- frozen evaluation runners, manifests, execution locks, reports и tagged v1.0 release не
  изменены и не перезапущены.

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
