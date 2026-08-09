# LRLspoof: проверенный внешний кандидат, не локальный intake

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

LRLspoof остаётся сильным кандидатом для отдельного **spoof-only external stress report**
после того, как пользователь предоставит проверяемую локальную копию. Даже тогда потребуется
отдельный intake с hashes и собственный протокол: число 1-SRR из dataset card использует
внешний threshold, поэтому его нельзя подменять текущей некалиброванной balanced accuracy.

Для ближайшего казахского binary research layer выбран другой, полностью воспроизводимый
путь: bona-fide KSC плюс две локальные TTS-family с зафиксированными revision, лицензиями и
text provenance. Он будет явно помечен как derived research stress source, а не как замена
LRLspoof или независимый speaker benchmark.
