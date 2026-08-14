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
