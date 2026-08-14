# Versioned research artifacts

В этой директории хранятся новые подробные receipts, audit summaries и завершённые результаты,
которые должны быть versioned в Git. Каждый этап получает отдельный подкаталог и новые имена
файлов; существующие receipts не перезаписываются.

Raw audio, model weights, checkpoints и generated run outputs сюда не добавляются. Они остаются
в Git-ignored `data/raw/`, `data/processed/`, `models/`, `checkpoints/` и корневом `artifacts/`.

Основные `README.md`, `PROJECT_STATUS.md` и `План реализации.md` содержат только актуальный
статус и ссылки на канонические подробности.
