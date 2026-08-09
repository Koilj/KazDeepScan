# API-каркас

ASGI entrypoint: `services.api.main:app`.

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
