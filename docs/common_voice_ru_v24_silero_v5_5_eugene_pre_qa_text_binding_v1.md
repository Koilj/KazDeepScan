# Common Voice RU v24 / Silero V5.5 `eugene` — literal-text binding v1

**Статус:** завершён write-once binding текста для candidate; это ещё не synthesis, acoustic
review, binary pairing или detector inference.

## Неизменяемые входы

Binding принял только `75` технически готовых Common Voice RU `test` bona fide rows из
[ready manifest](../data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_ready_v1.csv)
(SHA-256 `2b183adbfcac9b1a6022dd35c2f8b6ec8f111c01b4b3364596c53aff8906192a`). Его родительский
materialization receipt, frozen 80-row selection CSV/receipt, exact-route audit и model lock
проверены по закреплённым SHA-256 до чтения текстов.

Локальный archive `/home/ruslan/Downloads/cv-corpus-24.0-2025-12-05-ru.tar.gz` был повторно
проверен до metadata read: `7,008,716,262` bytes, SHA-256
`9a2ed32a0574f74f505cd7740a599f0b9edc9f52ba1e7d6624b66f258db4c0ea`.

## Результат

Все `75/75` ready rows совпали с source `test.tsv` по sample ID, sentence ID и SHA-256 literal
sentence. Wrapper normalizer `silero_v5_5_ru_literal_whitespace_only_v1` оставил каждую строку
byte-equivalent после штатной archive canonicalization: нет lexical rewrite, external normalizer,
stress model, text replacement или reselection. Для binding не использовались audio, duration,
detector либо metric.

Receipt:
[text binding JSON](../data/manifests/common_voice_ru_v24_silero_v5_5_eugene_pre_qa_text_binding_v1.json),
SHA-256 `8418fb0f8725e1408cc7b60886f0e981c90fc255715f147e93c76175ab3978d0`; deterministic
row-binding digest `1f202d97df4c1f4cf94a2460fbbef99a54d338fddcb93efaeaaf89ee7db7088d`.

## Граница следующего шага

Разрешён только один fixed built-in `eugene` 48 kHz synthetic WAV для каждого из этих 75 bound
texts. Любая synthesis/technical-QA ошибка permanently уменьшает candidate: replacement,
regeneration и backfill запрещены. До двух независимых full-asset acoustic reviews и нового
immutable paired plan detector inference и настройка по candidate запрещены.
