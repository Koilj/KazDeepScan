# LRLspoof: исключённый вариант

## Что проверено

10 августа 2026 выполнен только read-only audit публичного Hugging Face release
[`lab260/LRLspoof`](https://huggingface.co/datasets/lab260/LRLspoof), revision
`793f667a579756602193bb5b783ba16e80bcb7e6`.

- Dataset card объявляет лицензию MIT и `1 304 455` spoof-only TTS utterances в 66 языках.
- Публичный `data/labels.parquet` (6.1 MiB metadata) содержит `36 000` путей `kazakh/` и
  четыре записанных варианта TTS directory: `espeak`, `pipertts_part1`,
  `pipertts_part2`, `turkic_tts`.
- Это **не binary corpus**: у release отсутствует bona-fide class. Его нельзя вписывать в
  source-mixed matrix как самостоятельный test source и нельзя молча склеивать с KSC:
  происхождение текста, записи, канал и split были бы разными.

## Почему intake пока не выполняется

Аудио опубликовано как единый последовательный `lrl_spoof.tar.gz`, разбитый на 69 частей.
Уже первая часть имеет `10 737 418 240` bytes; release указывает общий размер около 452 GB.
Из-за gzip/tar layout получить только каталог `kazakh/` без предшествующих частей невозможно.

Операционное правило проекта запрещает начинать download набора больше 2 GB без локально
предоставленного пользователем archive. Поэтому не скачивались ни audio parts, ни сам
dataset; в `license_ledger.csv` его не добавляли и в manifest не использовали.

## Решение

**Не использовать LRLspoof в этом проекте.** Причины окончательные для текущего scope:

- release требует sequential download около `452 GB`, а казахский каталог нельзя получить
  выборочно;
- corpus spoof-only, поэтому из него нельзя получить самостоятельный binary test без
  неаудируемого смешивания bona-fide data другого происхождения;
- проектный intake limit — 2 GiB.

Не скачивать archive, не добавлять его в ledger, не запрашивать shards и не смешивать вручную
с KSC. Audit оставлен только как воспроизводимое объяснение исключения.
