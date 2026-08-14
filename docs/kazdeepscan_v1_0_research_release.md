# KazDeepScan v1.0 Research — release contract

**Release ID:** `kazdeepscan-v1.0-research`

**Source release version:** `1.0.0-research`

**Python distribution metadata:** `0.1.0` (historical, intentionally preserved)

**Git tag:** `v1.0.0-research`

**Scope:** completed personal-research toolkit

**Status:** completed

Machine-readable QA receipt:
[`data/releases/kazdeepscan_v1_0_research_release_receipt.json`](../data/releases/kazdeepscan_v1_0_research_release_receipt.json).

## Как проект должен быть закончен

Release считается завершённым только при одновременном выполнении всех условий:

1. Git tag, `kds.__release__`, CLI и OpenAPI однозначно публикуют identity v1.0 Research.
2. Historical Python package inputs остаются byte-identical: `pyproject.toml` SHA-256
   `debc12027b79900ad127dbc125efa0d931d8f7181509056cbdd9f49554467e39`, `uv.lock` SHA-256
   `61314bb9c647eab9f53c898143757ed17c321a9812e56e05fc9928f3ea5183a1`; locked sync проходит.
3. Все `307` pytest items проходят без failed/skipped; ToneSpeak выполняется с exact-hash Linux
   test overlay `pyarrow==22.0.0`, не изменяющим historical lockfile.
4. Ruff и strict mypy проходят на `src`, `tests`, `scripts`, `services`.
5. Wheel и sdist собираются; wheel устанавливается в пустое temporary environment, где
   `kds --version` и import smoke подтверждают source release identity и distribution `0.1.0`.
6. CLI manifest/license validation проходит на versioned research manifest.
7. API tests подтверждают `healthz=200`, а readiness/analyze без scorer — fail closed `503` без
   risk score.
8. README, `PROJECT_STATUS.md`, `План реализации.md`, API docs и changelog согласованы.
9. Git содержит только source/config/manifests/receipts: raw audio, weights, caches и build
   artifacts отсутствуют.
10. Один atomic release commit помечается annotated tag `v1.0.0-research`; branch и tag
    публикуются в configured GitHub `origin`.

## Что можно использовать

- `kds inspect-audio` для безопасной локальной подготовки audio;
- manifest/assets/license/training/source-matrix validators;
- deterministic group-aware split assignment;
- frozen personal-research scripts, configs и receipts как воспроизводимый reference;
- `services.api.main:app` как fail-closed local upload/preprocessing boundary;
- проект как основу для отдельной последующей v1.x/v2 development.

Установка и минимальная проверка:

```bash
uv sync --all-extras --locked
uv pip install --require-hashes \
  -r requirements/kazdeepscan-v1.0-research-test-linux.txt
.venv/bin/kds --version
.venv/bin/pytest
.venv/bin/uvicorn services.api.main:app --host 127.0.0.1 --port 8000
```

Overlay рассчитан на CPython 3.11–3.13, Linux x86_64. Он существует отдельно именно потому,
что обычное добавление dependency изменило бы SHA-256 inputs уже завершённых write-once runs.

## Что release намеренно не предоставляет

- model weights или automatically selected checkpoint;
- готовый `RiskScorer` и public/product risk score;
- raw/processed audio datasets;
- speaker identity, fraud proof или speaker-independent claim;
- commercial/data-rights clearance для third-party research datasets/models;
- разрешение повторять write-once inference либо настраиваться по final errors.

API без будущего отдельно авторизованного scorer обязан оставаться `model_unavailable`. Это
делает v1.0 пригодным для безопасной research-разработки, но не выдаёт экспериментальный
checkpoint за продукт.

## Зафиксированное состояние evidence

Research results и их ограничения перечислены в
[`PROJECT_STATUS.md`](../PROJECT_STATUS.md). Последний Denis × VoxCPM2 candidate корректно
завершён `stop_below_minimum_60`: `64/64` raw generated, `53` ready, `11`
`insufficient_speech`, без retry/replacement/backfill и без detector inference.

Новые datasets, model routes, product scorer или additional inference после v1.0 должны
получить новые versioned contracts и не изменять уже опубликованные receipts.
