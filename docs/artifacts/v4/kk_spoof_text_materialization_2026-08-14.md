# XLS-R+SLS model v4 — KK spoof text materialization

**Статус:** `7 200` frozen KSC2 transcripts проверены; synthesis ещё не выполнялся, training
запрещён.

Полный KSC2 multipart archive повторно прочитан с проверкой размеров 10 частей, combined gzip
SHA-256 `43d1ee6725d737a438125a13997a0abde4159de84ef17d1706fe7921e8632cbe`, gzip CRC и безопасного
TAR layout. Извлечена только exact allow-list `7 200` UTF-8 Train transcripts. Каждый текст
нормализован только по whitespace и совпал со своим frozen SHA-256.

Распределение неизменно: по `1 800` строк на Piper, MMS, KazEmoTTS и Spark-TTS; внутри каждой
route `1 500` target + `300` заранее упорядоченного reserve. Все `7 200` transcript members и
canonical text hashes уникальны. Локальные text bytes занимают `29 MiB` в Git-ignored v4 path.

Versioned outputs:

- inventory: `data/manifests/v4/xlsr_sls_model_v4_kk_spoof_text_inventory_v1.csv`, `7 200`
  rows, SHA-256 `f98f33d13eecc483ef3a05676e53636eb0f3e987afc10f5870041ce1889f5752`;
- [machine receipt](xlsr_sls_model_v4_kk_spoof_text_materialization_v1.json), SHA-256
  `54c9b7a24435e6c9c183d90fcbcb528c7ef041bcc9e38c29808c5699ae4123b8`.

Следующий gate — до первого WAV заморозить exact four-route synthesis plan: model/runtime
hashes, profiles, seeds/retries, все `1 800` attempts на route, write-once raw paths и полный
success/failure accounting. Synthesis остаётся text-only: reference/prompt audio, cloning,
network access и отбор по detector output запрещены.
