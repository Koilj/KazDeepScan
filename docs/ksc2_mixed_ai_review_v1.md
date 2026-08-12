# KSC2 narrow mixed evidence review v1

**Статус:** опубликован 11 августа 2026 года. Это узкий, воспроизводимый слой
положительных примеров code-switching, а не полная разметка KSC2 и не готовый binary dataset.

## Что именно опубликовано

Исходный immutable candidate packet содержит `2 632` paired audio/transcript rows из
`Test/podcasts` (`1 547`), `Test/radio` (`702`) и `Test/talkshow` (`383`). Все строки packet
сохраняют исходные поля `language=unknown`, `code_switch=unknown`.

В `src/kds/data/ksc2_mixed_review.py` сохранён список из `32` explicit decisions одного AI
semantic review. Для каждой строки publisher переносит transcript и source/audio hashes,
ставит `language=mixed`, `code_switch=true` и записывает 1-based позиции/текстовые формы
русских и казахских evidence tokens. Распределение positives: `22` podcasts, `4` radio,
`6` talkshow. Все не внесённые в этот список `2 600` candidates остаются неизвестными: скрипт
не создаёт для них отрицательную разметку и не пытается предсказать язык.

| Artifact | SHA-256 | Назначение |
| --- | --- | --- |
| `data/manifests/ksc2_test_mixed_annotation_v1.csv` | `5b80ee0b6d9d80907ed77a9c1e21821a6324e9621f3cb96ad1daab17982acc20` | исходные pending candidates |
| `data/licenses/ksc2_test_mixed_annotation_v1_receipt.json` | `64925d25231ecde944ffb934a0d1e7f0fecdac37c8d1923e38d6bd7ec71b802d` | rule и component counts исходного packet |
| `data/licenses/ksc2_test_mixed_annotation_v1_packet_lock.json` | `c4d731de6b8b772f7c0c1f5e70be3edde668f1e065bc730472cc4e3a8491fe25` | packet → receipt → KSC2 source-lock |
| `data/manifests/ksc2_test_mixed_ai_review_v1.csv` | `63257415ad744bf3095e28f6dede7e6e53e9608587258170cdd924651f5075e2` | 32 positive evidence rows |
| `data/licenses/ksc2_test_mixed_ai_review_v1_receipt.json` | `e078a73322250992a3f2aa5322f731f884a237d3bf6a13256ea8f2ff6f201bd5` | provenance и hash published CSV |

## Метод и границы

Решения — это direct semantic reading транскриптов одним AI reviewer, зафиксированная в
source-control positive list. Каждое решение требует одновременно явной русской фразы/слова и
явного казахского token evidence; валидатор дополнительно проверяет позиции, их непересечение и
наличие казахского специфического символа в сохранённом казахском token. Эта последняя проверка
контролирует целостность уже сохранённого evidence, но не является языковым классификатором.

Не используются component path, доля code-switching в статье, подсчёт кириллических букв,
external LID, ASR или runtime heuristic. Поэтому 32 строки можно честно называть только
single-AI transcript-review evidence. Они не подтверждены аудио, не имеют независимой проверки
и не измеряют coverage/accuracy относительно всего packet.

Этот слой нельзя использовать как:

- отрицательную разметку для остальных строк;
- complete mixed corpus annotation;
- KDS `ManifestRow`, binary training/final-test input, calibration или основание для model/API
  score.

До такого использования нужен отдельный неизменяемый binary protocol с QA/VAD, audit overlap,
независимой spoof-половиной и заранее закреплённым назначением split.

## Почему не был использован готовый автоматический обход

Статья KSC2 подтверждает code-switching лишь на уровне component aggregate rates; per-row labels
в release не опубликованы. [KSC2 Interspeech paper](https://www.isca-archive.org/interspeech_2022/mussakhojayeva22_interspeech.pdf)
не даёт готового сопоставления записи с языковой разметкой.

- [KazRusCSW-mBERT](https://huggingface.co/liminovna/KazRusCSW-mbert) предназначен для token
  LID, но access gated, а model card указывает ограниченное тестирование и не публикует лицензию.
- [kaznlp](https://github.com/makazhan/kaznlp) публикует heuristic word-level LID и сам
  документирует примерно `83%` token accuracy, что недостаточно, чтобы подменить ей факт
  конкретной строки.
- [fastText language identification](https://fasttext.cc/docs/en/language-identification.html)
  поддерживает `kk` и `ru`, но является document-level LID, а не token-level code-switching
  annotation.
- [KRASR Kazakh–Russian Whisper](https://huggingface.co/KRASR/kazakh-russian-asr-whisper-small-lora)
  — ASR model, не LID; её card сообщает нестабильное automatic language detection на коротких
  смешанных utterances и смешанный WER `0.5626`.

Следовательно, ни один вариант не был внесён в проект как генератор labels. Он мог бы дать
candidate ranking, но не основание автоматически менять `language` для 2 600 непросмотренных
строк.

## Воспроизведение

Publisher отказывается перезаписывать артефакты. Для проверки v1 нужно сохранить существующие
hashes; для нового review version укажите новые output paths и новый versioned receipt.

```bash
uv run python scripts/publish_ksc2_ai_mixed_review.py \
  --packet data/manifests/ksc2_test_mixed_annotation_v1.csv \
  --packet-receipt data/licenses/ksc2_test_mixed_annotation_v1_receipt.json \
  --packet-lock data/licenses/ksc2_test_mixed_annotation_v1_packet_lock.json \
  --reviewed-at 2026-08-11T00:00:00Z \
  --output-csv data/manifests/ksc2_test_mixed_ai_review_v2.csv \
  --output-receipt data/licenses/ksc2_test_mixed_ai_review_v2_receipt.json
```

Команда воспроизводит набор fixed decisions, не запускает модель и не скачивает данные. Новая
версия должна пройти те же integrity checks и иметь свой отдельный receipt.

Первый QA/VAD-ready bona-fide candidate, технический Silero smoke-test и границы дальнейшей
synthesis описаны в [KSC2 mixed bona-fide candidate v1](ksc2_mixed_candidate_v1.md).

Для нового Stage-C suite исторический v1 не изменялся. Отдельный disjoint review и его
QA/rejection accounting описаны в
[KSC2 mixed semantic evidence v2 delta](ksc2_mixed_ai_review_v2_delta.md).
