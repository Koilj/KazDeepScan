# ToneSpeak RU — source-level intake audit

**Статус:** `ToneSpeak` принят только как независимый **personal-research Russian
spoof-only source**. Это не binary final benchmark, не product source и не разрешение
на training, calibration или new model inference.

## Источник и зафиксированная версия

Источник — [Vikhrmodels/ToneSpeak](https://huggingface.co/datasets/Vikhrmodels/ToneSpeak/tree/d40f94cd5c7dcf756a8c59a1c465b834220bec56),
commit `d40f94cd5c7dcf756a8c59a1c465b834220bec56`, опубликованная dataset-card license
`Apache-2.0`. Card прямо заявляет Russian content, создание текстов и prosody prompts с
`GPT-4.1 mini`, а audio — с `GPT-4o mini TTS` в ten named voices. Эти имена совпадают с
официальными built-in TTS voices `alloy`, `ash`, `ballad`, `coral`, `echo`, `fable`,
`nova`, `onyx`, `sage`, `shimmer`. Это подтверждает смысл field `voice_name` в public API
contract, но не доказывает самостоятельно, как автор сгенерировал каждую строку. См.
[ToneSpeak card](https://huggingface.co/datasets/Vikhrmodels/ToneSpeak) и
[official OpenAI Audio API reference](https://platform.openai.com/docs/api-reference/audio/voice-consent-list).

Никакой per-row generation log, OpenAI model snapshot, API request или независимое
доказательство отсутствия reference audio не опубликованы. Поэтому именно source card,
а не локальный вывод, остаётся доказательством generator provenance и `voice_name` имеет
ledger status `source_provided`, а не `verified`.

## Локальный integrity audit

Скачаны только exact public artifacts этой revision: `README.md`, `.gitattributes` и
пять Parquet LFS payloads. Их source lock —
`data/licenses/tone_speak_ru_v1_artifact_lock.json`; receipt streaming-аудита —
`data/licenses/tone_speak_ru_v1_artifact_audit_receipt.json` (SHA-256
`c14d3f0fd38e6ee8675a78b08b627aa43ca618bde52be9c1f90cec8d71996908`).

Аудит fail-closed проверяет tree, размеры и LFS SHA-256 каждого artifact, schema
`audio/text/text_description/voice_name`, Hugging Face sampling-rate metadata, все
embedded MP3 paths/voice names и header каждого MP3. Никакой MP3 не извлекается.

| Проверка | Результат |
| --- | ---: |
| Artifacts (включая card) | 7, `1,686,390,665` bytes |
| Rows | `6,998` = `6,298` train + `700` validation |
| Audio | `6,998` unique paths, `0` identical payloads |
| Format | all MP3, `24,000` Hz |
| Local duration | `105,244.048` s (29.23 h) |
| Cyrillic marker in transcript | `6,998/6,998` |
| Train/validation normalized-text overlap | `0` |

Все десять card-declared voices присутствуют в обоих source splits. Найден ровно один
duplicate normalized transcript, оба instances лежат внутри `train`; это не audio duplicate
и не cross-split leakage. Любой будущий selection обязан всё равно группировать строки по
normalized text hash.

Для повторного audit нужен `pyarrow>=18,<23` в active environment. Команда audit:

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_tone_speak.py \
  --artifact-root data/raw/tone_speak_ru_v1/artifacts \
  --output-receipt /new/empty/path/tone_speak_receipt.json
```

## Независимость от frozen checkpoint

Read-only scan `1,814` rows frozen Stage-A manifest нашёл `0` rows с `OpenAI` generator
name/version и `0` rows с этими ten voice IDs. Scan также не нашёл совпадающих значений среди
`6,997` canonical ToneSpeak text hashes и `60` valid local manifests. Последняя проверка
ограничена общим whitespace-normalized SHA-256 representation: она не является доказательством
отсутствия semantic overlap или источником speaker independence. При публикации конкретного
candidate manifest надо повторить exclusions против всех уже frozen assets.

## Locked research-only OOD candidate v1

Создан один write-once candidate из source `validation`, а не из ToneSpeak `train`: seed
`20260811` выбирает `10` text-group-unique clips каждого из ten voices. Перед выбором были
загружены все `60` valid manifests из `data/manifests/` как exclusion set; восемь CSV с иной
схемой (receipts/rejections) явно записаны как skipped и не выдаются за manifests. Result —
`100` Russian spoof rows, `100` unique text hashes, source train/validation text overlap
already audited as zero.

| Layer | Rows | SHA-256 |
| --- | ---: | --- |
| raw MP3 manifest | 100 | `3957dcc8daa1ea59f994c6221607a521248ff815065ab7b41b3902061c384191` |
| selection receipt | 100 / 10 voices × 10 | `504edfa5fa16be52efac6e70da726fb70ec237457903f63525907634b0990bc8` |
| ready 16-kHz mono WAV manifest | 100 | `45527f97ddbb5be65ea12ff8a8ed7b723e9d1135aff0cde71addd95366dc84a0` |
| preprocessing rejection report | 0 rejections | `67cf220069c9f0bf35d2960a2cd371c0934faafb9cf8b0bbfdeea387c9777250` |
| ready receipt | 100 raw → 100 ready | `9a0f21a8613f0fa752bf6553aa7a300d442226929c7fdddd050c67a55e192ea7` |

Новый candidate закреплён только как `split=ood`, `label=spoof`,
`generator_name=openai_gpt_4o_mini_tts`, `generator_version=source_card_unpinned` и exact
per-row `voice_id`. Preprocessing создал 16-kHz PCM WAV и QA/VAD принял `100/100` assets;
это подтверждает decode/readiness, но не язык или lexical content. Candidate не добавлен к
train/dev/test, не применялся к checkpoint и не является binary final layer.

## Acoustic language-preservation gate — completed

Опубликован immutable `100`-asset listening packet
`tone_speak_ru_v1_ood_acoustic_gate_packet.csv`, SHA-256
`f69fe1a3d082c9d8bfac3ba1ce1f10011fa844b07e9608f840d8c4b59fbfed41`.
Он повторно сверяет source revision, ready receipt, ledger, asset SHA-256 и source validation
transcripts; поле transcript используется только как reference для человека, не как acoustic
decision. Два заполненных 100-row CSV имеют разные formal pseudonymous IDs (`reviewer_1` и
`reviewer_2`); их SHA-256 соответственно
`f432de9aa051aba5afc9e19a011b9611ac8f5164aad170979195e39d03ab3277` и
`ced102a481eb68d22ef04d4777251498a3c2f6bdb2e360f96092f456381d7263`.

Для каждого WAV reviewer ставит `pass` лишь когда он слышит русский и содержание соответствует
pinned transcript; любое `fail`, `unknown`, пропуск, duplicate review ID или disagreement даёт
этому asset `not_eligible`. Strict evaluator принял `200` review rows: `100/100` locked WAV
получили ровно два `pass/yes/yes` decisions. Immutable receipt —
`tone_speak_ru_v1_ood_acoustic_gate_report_v2.json`, SHA-256
`8525df3980210c8e2b4dd827859e0d0c7b1ecb74f512ab6a5eaa628cbeb55df6`; он содержит hashes
обоих review CSV. Формально разные IDs подтверждены validator'ом, но организационную
независимость людей CSV криптографически не доказывает. Даже `100/100 pass` оставляет
`final_or_product_eligible=false`: gate не доказывает human voice-group provenance, generator
API logs, bona-fide balance или final quality.

## Жёсткие ограничения

- Набор состоит только из synthetic/spoof audio. Для binary final layer нужен отдельно
  vetted Russian bona-fide counterpart; ToneSpeak сам его не создаёт.
- Ledger намеренно разрешает только personal research (`train_dev_test_use` и
  `ood_evaluation_use` равны `research_only`), а spoof voice group остаётся
  `source_provided`. Product validator закономерно его отклонит.
- `text` и Cyrillic marker сами по себе не подтверждают акустически русский язык. Этот locked
  100-asset sample прошёл two-review gate; любые иные final-level assets всё ещё потребуют
  собственного locked sample, preprocessing receipt и acoustic review по самим output audio.
- В project есть ровно один locked 100-asset **OOD candidate**. Его acoustic gate завершён, а
  единственный hash-pinned frozen-checkpoint OOD run выполнен: spoof recall на fixed raw boundary
  равен `88/100` и не является binary/final quality. No training, calibration или threshold
  fitting не выполнялись. Plan, execution lock и все logits —
  `docs/research_xlsr_sls_stage_b_tone_speak_ru_ood_100.md`.
