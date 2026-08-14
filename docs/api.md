# KazDeepScan v1.0 Research API-каркас

ASGI entrypoint: `services.api.main:app`.

OpenAPI metadata публикует source-release version `1.0.0-research` и title
`KazDeepScan Research API`. Это usable preprocessing/upload boundary для local research
integration, но не model release. Python distribution metadata остаётся историческим `0.1.0`.

```bash
KDS_FFMPEG_BINARY="$PWD/.tools/ffmpeg/bin/ffmpeg" \
KDS_FFPROBE_BINARY="$PWD/.tools/ffmpeg/bin/ffprobe" \
uv run uvicorn services.api.main:app --host 127.0.0.1 --port 8000
```

- `GET /healthz` подтверждает, что процесс жив.
- `GET /readyz` возвращает 503, пока не загружен обученный и калиброванный scorer.
- `POST /v1/analyze` принимает multipart `audio`, сохраняет его в UUID-временный файл с
  жёстким лимитом 50 MiB и удаляет по окончании запроса.

До появления model release endpoint **не отдаёт** случайный B0 logit. Готовая запись получает
`503 model_unavailable`; короткая или некачественная запись получает честный ответ с
`insufficient_speech` либо `rejected_quality`, без risk score.

Проверка release boundary:

```bash
uv run kds --version
curl -fsS http://127.0.0.1:8000/healthz
curl -i http://127.0.0.1:8000/readyz
```

Ожидается `kds 0.1.0 (KazDeepScan v1.0 Research)`, затем `200` для health и
`503 model_unavailable` для readiness. Перевод `/readyz` в `200` допускается только будущим
отдельным model/product contract.

## Отдельный local research inference API

Новый пользовательский inference не подключён к `/v1/analyze` и не меняет его fail-closed
поведение. Он запускается как другой ASGI factory на другом порту:

```bash
export KDS_RESEARCH_INFERENCE_CONTRACT="$PWD/configs/inference/b0_user_audio_local_research_v1.json"
export KDS_FFMPEG_BINARY="$PWD/.tools/ffmpeg/bin/ffmpeg"
export KDS_FFPROBE_BINARY="$PWD/.tools/ffmpeg/bin/ffprobe"

.venv/bin/uvicorn \
  kds.serving.research_api:create_research_app_from_environment \
  --factory --host 127.0.0.1 --port 8001
```

- `GET /healthz` — процесс жив;
- `GET /readyz` — exact contract/checkpoint загружены;
- `POST /v1/research/analyze` — только внешний пользовательский audio upload;
- multipart `acknowledge_research_only=true` обязателен;
- multipart `confirm_external_user_audio=true` отдельно подтверждает, что это не frozen asset;
- без acknowledgment возвращается `400 research_acknowledgement_required`;
- без external-input confirmation возвращается
  `400 external_user_audio_confirmation_required`;
- без contract/checkpoint приложение не стартует либо readiness остаётся fail closed;
- uploads ограничены `50 MiB / 10 min` и удаляются вместе с temporary directory;
- response не содержит `risk_score`, fraud verdict или product band.

Пример:

```bash
curl -F "audio=@/absolute/path/to/user-audio.wav;type=audio/wav" \
  -F "acknowledge_research_only=true" \
  -F "confirm_external_user_audio=true" \
  http://127.0.0.1:8001/v1/research/analyze
```

`uncalibrated_spoof_score` не является вероятностью. Результат не доказывает мошенничество,
не идентифицирует говорящего, не является speaker-independent или product-grade оценкой и не
может использоваться для автоматических решений. Contract:
[`configs/inference/b0_user_audio_local_research_v1.json`](../configs/inference/b0_user_audio_local_research_v1.json).
