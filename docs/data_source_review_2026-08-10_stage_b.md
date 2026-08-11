# Проверка дополнительных данных для XLS-R Stage B — 10 августа 2026

## Что искали

Проверены готовые русские и казахские binary anti-spoofing источники, которые могли бы
расширить train/dev без собственной записи голосов. Запись участников, reference audio,
voice cloning и имитация конкретного человека исключены из scope. Локальные полные RuASD и
PyAra v7 в `/home/ruslan/Downloads` считаются основными доступными research-источниками.

## XMAD-Bench

Официальный pinned dataset card:
<https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/XMAD/blob/0649579d161d55c1f6e74cc81a2fbbbe930bf036/README.md>.
Связанная статья: <https://arxiv.org/abs/2506.00462>.

XMAD содержит `368 085` строк, семь языков, включая русский, бинарную метку и дополнительные
поля attack/source. Лицензия release — `CC-BY-NC-SA-4.0`. Это наиболее содержательный
найденный готовый multilingual кандидат, но он не включён в Stage B v1:

- русская часть использует Common Voice и M-AILABS;
- эти bona-fide source уже входят в полный RuASD, используемый для train;
- у текущего RuASD slice нет проверяемых speaker IDs, поэтому speaker/source leakage между
  RuASD train и русским XMAD нельзя надёжно исключить;
- полный release разбит на 200 крупных Parquet shards и требует отдельного intake с
  закреплённой ревизией, лицензией, selective extraction и проверкой фактических групп.

Поэтому XMAD остаётся кандидатом для отдельного OOD/source-audit, а не скрытой заменой dev в
уже начатом Stage B.

## Казахские кандидаты

- Common Voice Scripted Speech Kazakh 24.0:
  <https://dev.mozilladatacollective.com/datasets/cmj8l8c9n00dfnlovvjrd3ksd>. Официальная
  карточка указывает только bona-fide речь: `2 750` clips, `2.39` validated hours и `193`
  speakers. Это полезный будущий bona-fide слой, но не готовый binary anti-spoof benchmark.
- LRLspoof: <https://huggingface.co/datasets/lab260/LRLspoof>. В проекте уже проведён
  pinned audit. Казахская часть spoof-only, а release опубликован единым последовательным
  архивом примерно `452` GB без selective Kazakh download. Источник остаётся исключённым.

Готового казахского binary anti-spoof release с пригодным selective download, понятными
правами и проверяемыми независимыми группами в этом поиске не найдено. Поэтому недостающий
казахский spoof coverage продолжает строиться только text-only TTS family с закреплёнными
весами и встроенными публичными voices; bona-fide кандидаты проходят отдельный source intake.

## Принятое решение

Для Stage B v1 расширен локальный PyAra dev, потому что его archive уже полностью проверен и
разрешён для personal research. Новый slice исключает все record/text groups старого PyAra
dev и затем дополнительно удаляет любой text overlap с RuASD train. Это улучшает объём dev,
но не превращает его в speaker-disjoint benchmark: PyAra не публикует speaker IDs.

