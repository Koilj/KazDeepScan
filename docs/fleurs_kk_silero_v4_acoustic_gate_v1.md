# FLEURS KK / Silero V4 — post-inference acoustic gate v1

**Статус:** завершён 12 августа 2026; две заполненные формы строго проверены, write-once gate
receipt опубликован.

Этот gate относится ровно к `304` WAV (`152` FLEURS KK / Silero V4 пары), уже использованным в
confirmatory run. Он проверяет только audibility, соответствие казахскому тексту и отсутствие
явных акустических дефектов. Результат детектора уже известен, поэтому даже полностью успешный
gate останется post-inference evidence и не превратит метрику `1.0000` в blind final или product
quality.

## Зафиксированные входы

| Артефакт | Строк | SHA-256 |
| --- | ---: | --- |
| `fleurs_kk_v1_silero_v4_test_152.csv` | 304 | `e23f1ce80dbc866eafd4fe1f488f2f55c5705bef1bd9b12e33ce80632328dac2` |
| `fleurs_kk_v1_silero_v4_acoustic_gate_packet.csv` | 304 | `1ff70e56f516b09be1dba7b3c3e5c6d890538be076122cf08ae716fada401267` |
| `fleurs_kk_v1_silero_v4_acoustic_review_reviewer_1.csv` | 304 | `8cb82ef7737586537e43b4e1ae653147ade75abb73b90d51adfd40d97628c0a7` |
| `fleurs_kk_v1_silero_v4_acoustic_review_reviewer_2.csv` | 304 | `6286695353379a9e6ae76b79e5a386a865d7176e1f76c7049f7e6a243da62096` |
| `fleurs_kk_v1_silero_v4_acoustic_gate_report_v1.json` | 304 results | `4147cc8df2658af46c42530ea74e1d9550d5645ab3c176f0462fc93789f47810` |

При публикации packet повторно проверены manifest schema, license ledger, SHA-256 всех 304
audio assets и полный pinned FLEURS `kk_kz` release. Packet содержит exact asset paths/hashes и
source transcripts, но не содержит model predictions, logits или detector errors.

## Результат gate

- review rows: `608` (`304` на reviewer ID);
- reviewer IDs: `reviewer_1`, `reviewer_2`;
- решения: `304 pass`, `0 not_eligible`;
- `all_assets_acoustically_verified: true`;
- `evidence_timing: post_inference`;
- `metric_status_changed: false`;
- `blind_final_eligible: false`;
- `final_or_product_eligible: false`.

В предоставленных формах оба reviewer ID поставили `pass/yes/yes/yes` всем 304 assets. Строгая
проверка подтвердила полноту, разные IDs, отсутствие дублей, точное совпадение packet SHA-256,
sample IDs, asset SHA-256, путей и transcripts. Техническая проверка IDs не может сама доказать
организационную независимость людей; это остаётся условием проведения review.

## Контракт заполненных форм

Рецензенты работают независимо и не смотрят predictions. В каждой строке WAV находится по пути
`data/<relative_path>`, а ожидаемый текст указан в `input_transcript`. Не изменяйте поля от
`protocol_id` до `reviewer_pseudo_id` включительно.

Заполняются только:

- `review_status`: `pass`, `fail` или `inconclusive`;
- `audio_audible`: `yes`, `no` или `unknown`;
- `kazakh_text_matches`: `yes`, `no` или `unknown`;
- `no_obvious_defects`: `yes`, `no` или `unknown`;
- `notes`: обязательная краткая причина для `fail`/`inconclusive`, без персональных данных.

Статусы логически строгие:

- `pass` допустим только при трёх `yes`;
- `fail` требует хотя бы одного `no` и непустой `notes`;
- `inconclusive` требует хотя бы одного `unknown`, не допускает `no` и требует непустой
  `notes`.

Начальные `inconclusive/unknown` с пустой `notes` намеренно не проходили evaluator: так receipt
нельзя было случайно опубликовать до фактического прослушивания.

Нельзя удалять строки, дублировать решения или передавать обе формы одному рецензенту. Разные
`reviewer_pseudo_id` технически проверяются, но фактическая независимость остаётся
организационной обязанностью рецензентов.

## Историческая команда публикации

Receipt уже существует и write-once guard запрещает повтор. Выполненная команда:

```bash
.venv/bin/python scripts/prepare_fleurs_kk_silero_v4_acoustic_gate.py evaluate \
  --packet data/manifests/fleurs_kk_v1_silero_v4_acoustic_gate_packet.csv \
  --reviews data/manifests/fleurs_kk_v1_silero_v4_acoustic_review_reviewer_1.csv \
  --reviews data/manifests/fleurs_kk_v1_silero_v4_acoustic_review_reviewer_2.csv \
  --output-report data/manifests/fleurs_kk_v1_silero_v4_acoustic_gate_report_v1.json
```

Команда потребовала ровно `608` решений: два разных полных reviewer-набора по всем `304`
assets. Любое изменение packet binding, противоречивый enum, пропуск или повтор приводит к
отказу. Существующий receipt не перезаписывается.
