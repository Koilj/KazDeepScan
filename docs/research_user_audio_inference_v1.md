# Local user-audio research inference v1

**Статус:** completed post-v1.0 development

**Contract:** `configs/inference/b0_user_audio_local_research_v1.json`

**Purpose:** `research_user_audio_only`

## Граница нового этапа

Этот inference предназначен только для внешнего пользовательского файла. Он не является новым
evaluation run и не меняет ни один завершённый Stage-C/Stage-D/v3/V5.5/VoxForge/Denis protocol.
Новый код:

- не вызывает training/evaluation scripts;
- не обходит write-once output reservations;
- не открывает final manifests и не обходит их candidate gates;
- не создаёт execution lock, prediction CSV, report или новый checkpoint;
- не fit-ит temperature, threshold, architecture или augmentation;
- не использует пользовательские outcomes для tuning.

Checkpoint содержит прежнее aggregate final metadata как часть уже созданного payload, но loader
не использует эти значения для выбора модели, boundary или transform. Он проверяет только
закреплённые identity/config/state constraints и загружает state read-only.

## Pinned model и preprocessing

- model: local Git-ignored `models/b0-unseen-generator-suite-v1.pt`;
- checkpoint SHA-256:
  `7b620af0c7e20788550b432c1d428b4e29e0a9c57cedc2fa549687c46b200539`;
- state-dict SHA-256:
  `a586c3234c01fd04ecf70a1b14be38b2a9611b12f5ebbfa951433f3a7f9024de`;
- safe load: `torch.load(weights_only=True)`, CPU map, exact top-level metadata и strict state;
- architecture: `b0_logmel_cnn`, 16 kHz mono;
- media/size/duration checks, FFmpeg normalization, QA и WebRTC VAD выполняются до scoring;
- minimum speech: `2.5 s`;
- window: `64 600` samples, hop `32 000`; short speech window повторяется до model size;
- per-window logits агрегируются duration-weighted mean в raw-logit space;
- fixed training boundary: raw logit `0.0`; post-hoc threshold selection отсутствует;
- sigmoid публикуется только как `uncalibrated_spoof_score`, не как probability.

CLI запрещает source paths ниже project `data/`, `models/`, `artifacts/`, `checkpoints/`. API
принимает upload только во временный private directory после explicit
`acknowledge_research_only=true` и отдельного `confirm_external_user_audio=true`, сохраняет
streaming limit `50 MiB`, duration limit `10 min` и удаляет temporary bytes после запроса. Ни
CLI, ни API не сохраняют normalized WAV или model outputs автоматически.

## Output semantics

Каждый successful response содержит:

- `research_only=true`;
- exact contract/model/checkpoint identity;
- SHA-256 входного файла;
- model-independent QA/VAD diagnostics;
- aggregate и per-window raw logits;
- `uncalibrated_spoof_score` и только fixed-zero-boundary `bonafide_like`/`spoof_like`;
- `calibrated=false`, `probability_claim=false`, `fraud_claim=false`, `product_grade=false`;
- обязательное предупреждение и полный limitation list.

`risk_score`, fraud verdict, identity claim и product band отсутствуют. Training-data overlap
пользовательского файла не проверен. B0 evidence не является speaker-independent и не даёт
commercial/data-rights clearance.

## Versioned implementation

- contract SHA-256:
  `4cc12f8ee44855970297207e30b8691d07938456ad6e776d3fbd35ae0484cd77`;
- `src/kds/inference/__init__.py`:
  `889b997e89383607ea915abc381c11238089471cff87febb4c486e920b526fc6`;
- `src/kds/inference/research.py`:
  `704897198543287db43a2020400d2e5ddf54fc2e01d67dab6a4049fcb7691172`;
- `src/kds/serving/research_api.py`:
  `ed55181d33ecf799c5d32f7add59373d946ea19b5c43fc824b7db634ca2f6f78`;
- `src/kds/cli.py`:
  `66690464c17d22c2efa6ab06de601b3400c8ac06ffca3e9af1cef8d9659b168e`;
- `tests/test_research_inference.py`:
  `53c9a80f0a6c4510eeab07ab3a724545b2d97908584b8d3e97c84ab853d546e8`;
- `tests/test_research_inference_api.py`:
  `335c2c6dd1917b3f4fda5fdedc2193c9d4ec6348316bc9f61e36e0e8a3735c5a`.

Historical package inputs не изменены:

- `pyproject.toml`:
  `debc12027b79900ad127dbc125efa0d931d8f7181509056cbdd9f49554467e39`;
- `uv.lock`:
  `61314bb9c647eab9f53c898143757ed17c321a9812e56e05fc9928f3ea5183a1`.

## QA receipt

- одиннадцать новых contract/checkpoint/engine/path-guard/CLI/API tests проходят;
- legacy API/release tests подтверждают, что `/v1/analyze` по-прежнему fail closed;
- полный suite: `318 passed`, `0 failed`, `0 skipped`;
- Ruff: passed;
- strict mypy: passed;
- `git diff --check`: passed;
- exact local checkpoint validation: passed;
- отдельный ASGI factory публикует только `/healthz`, `/readyz`,
  `/v1/research/analyze` и стандартные OpenAPI docs;
- external technical smoke использовал только новый `/tmp/kds-external-user-smoke-voiced.wav`,
  SHA-256 `047ab6944c4c9a7c669f94186bfc5f77f473aaf85564966adf32cf5ac2daaf4a`: `5.0 s`,
  VAD speech `4.97 s`, два windows, CLI status `ok`;
- smoke является искусственным harmonic input, не evaluation sample; его score не интерпретировался,
  не сравнивался с label и не использовался для изменения contract/code.

## Обязательное предупреждение

> Результат является некалиброванным исследовательским сигналом сходства. Он не является
> вероятностью, идентификацией говорящего, доказательством мошенничества или product-grade
> оценкой и не должен использоваться для автоматических решений.

Следующий model, calibration или product route требует нового versioned contract. Existing
write-once runs и их final errors для этого использовать запрещено.
