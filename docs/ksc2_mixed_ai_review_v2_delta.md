# KSC2 mixed semantic evidence v2 delta — Stage C

**Статус:** опубликован 12 августа 2026 года как disjoint single-AI transcript-review delta.
Это не полная разметка KSC2, не acoustic certificate и не разрешение выполнять detector
inference.

## Решение

Размер `1` fresh mixed group из inventory v1 недостаточен для осмысленной отдельной метрики.
Поэтому до фиксации Stage-C selection выполнен второй узкий semantic pass по уже
hash-pinned `2 632` KSC2 candidates. Ranking по консервативным русским фразам применялся только
для сокращения объёма чтения; он не создавал labels. В source control опубликованы только `59`
явных положительных решений, каждое с позициями и формами русского и отдельного казахского
token evidence.

Четыре неоднозначных ranking hits, основанные только на `проблема`, `так` или слабом `там`,
оставлены `unknown`. Все остальные не просмотренные строки также остаются `unknown`.

| Артефакт | Rows | SHA-256 |
| --- | ---: | --- |
| `ksc2_test_mixed_ai_review_v2_delta.csv` | 59 | `80464931a73dbb366d4140db7409698ad47ccf67b43bfc50201af66e08fe2554` |
| `ksc2_test_mixed_ai_review_v2_delta_receipt.json` | receipt | `9935c12c7c4f19bb67cb33ac50eca610787a978b6b2bf406aa900718de9edeeb` |
| `ksc2_test_mixed_ai_review_v2_delta_raw.csv` | 59 | `03f72d3b72814d85a31bbc58b86b51fa926cd043c67b79eb8b3236cbfeb99338` |
| `ksc2_test_mixed_ai_review_v2_delta_ready.csv` | 57 | `2675c051bc8a0cf559159d927f6cddaaef47344172a5ce722b149b0e9144a3da` |
| `ksc2_test_mixed_ai_review_v2_delta_rejections.json` | 2 | `0a754616aa94d540ebc2393d018b98904cf6bbd86a8f14577182c7cc4972b855` |
| `fresh_research_suite_source_inventory_v2.json` | inventory | `915294f0c608435d4775217f686cd6d2933eabfb4efd65e0a7a1d38d9d0db39a` |

Распределение semantic positives: `40` podcasts, `14` radio, `5` talkshow. QA/VAD отклонил
две записи как `insufficient_speech`; причины и exact source paths сохранены в rejection
receipt. Пересечений annotation ID с 32-row v1 review нет.

## Итоговая доступная ёмкость

С учётом 31 QA-ready v1 rows и 57 QA-ready v2-delta rows имеется `88` ready mixed rows. Из них
30 уже раскрыты через прежний Silero candidate, поэтому Stage C доступны `58` fresh groups:
одна ранее известная fresh v1 row и 57 новых v2-delta rows.

Inventory v2 повторно проверяет оба полных pinned FLEURS locale releases, обе KSC2 review/ready
версии и прежний exposed manifest. Текущие ёмкости до Stage-C selection: `55` RU release groups,
`60` QA-ready KK groups и `58` QA-ready mixed groups.

## Ограничения

- semantic evidence получено одним AI reviewer и не является независимой human annotation;
- наличие RU/KK tokens в transcript не доказывает сохранение языка будущим TTS;
- speaker IDs KSC2 не верифицированы;
- полный asset-level two-listener gate обязателен после synthesis и до inference;
- оставшиеся `2 541` candidates не являются отрицательными примерами и остаются неизвестными.
