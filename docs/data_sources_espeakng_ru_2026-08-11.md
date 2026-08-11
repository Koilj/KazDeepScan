# eSpeak NG Russian — независимая RU spoof generator family v1

**Статус:** 11 августа 2026 года создан personal-research RU candidate. Это controlled local
text-to-speech source, а не найденный внешний Russian-only benchmark и не final/product result.

## Источник и pin

Использован уже локально проверенный eSpeak NG `1.52.0` source/runtime bundle: source archive,
binary и четыре Debian runtime packages имеют суммарно `29 248 111` bytes, exact size/SHA-256
закреплены в [`espeakng_ru_v1_models.json`](../configs/research/espeakng_ru_v1_models.json),
SHA-256 `6e84fc467f65bda1a581a0bff3779f5b04337289325e63bda40667e1ff629825`.

Lock использует тот же already-verified local package directory, что и прежний Kazakh eSpeak run,
но не скачивает и не дублирует artifacts. До synthesis bundle повторно прошёл exact verification.
Проверка package layout подтверждает bundled `lang/zle/ru` и `ru_dict`; runtime разрешает только
закреплённые `ru`/`kk` voices, а этот protocol принимает строго `ru`.

eSpeak NG — rule-based formant synthesizer, не neural voice. Adapter принимает только UTF-8 text
через stdin, извлекает Debian payload во временный каталог с path/type/size checks и не имеет
reference-audio, voice-path, voice-cloning или imitation route. Двенадцать profiles —
детерминированные speed/pitch/amplitude controls, не человеческие speaker identities.

## Text-disjoint FLEURS RU base

Полный ready FLEURS RU test base имеет `289` bona-fide WAV. Из него исключены все `214` text groups,
уже опубликованные в frozen FLEURS/Silero candidate. Новый base содержит оставшиеся `75` rows;
пересечение с удержанными groups по text hash равно нулю.

| Артефакт | Rows | SHA-256 |
| --- | ---: | --- |
| `fleurs_ru_v1_espeakng_base_75.csv` | 75 bona-fide | `2a04cddedf39164108a8c7c76ed51b2d9e07bd4882ec79740e0b5ab98eb5ec9e` |
| selection receipt | 75 selected, 214 held | `a027cb7613e44fd4faf4ee49c589cfe6c943008ccb934de1ae7e502068ddc05c` |

Builder повторно сверяет assets и ledger, а receipt хранит exact hashes исходного full base и
existing Silero candidate. Это предотвращает silent reuse уже раскрытых text groups.

## Synthesis, QA и paired candidate

Перед synthesis full FLEURS release повторно проверяется и каждый source transcript сверяется с
`sample_id`, `text_id` и `text_hash`; текст не нормализуется, не переводится и не фильтруется.
Русский eSpeak создал `75/75` raw WAV. Общий QA/VAD preprocessing сохранил `75/75`; пустой
audio-rejection receipt всё равно опубликован.

| Артефакт | Rows | SHA-256 |
| --- | ---: | --- |
| raw spoof manifest | 75 | `504d986ce9534fff927b327e104cb0e897eeed54937288572beb815876b20838` |
| ready spoof manifest | 75 | `8bcab40578a5dac1f3f96dba6265be19f0f62300c5b6ae2c65560bcccdbf1db8` |
| text-accounting receipt | 75, no text rejection | `79a5bb0055e45a314d5fd2d8c7147f5fa3434b2d378da466674c9d9fafa4f12e` |
| audio-rejection receipt | 0 rejection | `038c65b506f7fd6659af52839b9e491507ea7cb607bbe01581495381a18bd83e` |
| paired candidate `fleurs_ru_v1_espeakng_test_75.csv` | 75 pairs / 150 assets | `65b12c43df18fcf7d1ea6b38d2e951cc505963241f5d821092b035b56c9a3af8` |

Paired builder подтвердил exact pair text identity и отсутствие sample ID, asset SHA-256 и text-hash
пересечений с FLEURS/Silero, RuASD train, historical/fresh PyAra dev, calibration dev, четырьмя
KSC final layers и frozen KSC2/Silero stress candidate. `kds validate-manifest` и
`kds validate-assets` проверили все `150/150` published assets.

## Acoustic language-preservation gate — завершён

Чтобы не выдавать input transcript provenance за содержание WAV, опубликован write-once listening
packet `fleurs_ru_v1_espeakng_acoustic_gate_packet.csv`: `75` exact pairs / `150` assets,
SHA-256 `7aecbebbd91444fe75bf261c15715862a7bb194bb501e8862d01b3f02059ff40`.
Перед publication повторно сверены candidate manifest, ledger, все `150` asset SHA-256 и полный
FLEURS RU release; packet связывает FLEURS source transcript/text hash с обеими asset hash каждой
пары.

Gate fail-closed требует для каждого WAV две записи от разных reviewer pseudo-ID со значениями
`review_status=pass`, `russian_audible=yes` и `lexical_content_preserved=yes`. Две 150-row forms
прошли строгую evaluation: `300` review rows, `150/150` asset decisions `pass`, `75/75` pairs.
Write-once receipt `fleurs_ru_v1_espeakng_acoustic_gate_report_v1.json` имеет SHA-256
`ec9f6129391687ef12799a8a6b06ad9a6dfd308a3defba1c9cda09a81e2380ea`.

Проверка кода может установить разные pseudo-ID и полноту CSV, но не может криптографически
доказать организационную независимость людей. Gate устанавливает только audibility русского языка
и lexical preservation зафиксированных assets; он не разрешает calibration, inference, final или
product claim.

Реализация: `src/kds/eval/fleurs_ru_acoustic_gate.py` и
`scripts/prepare_fleurs_ru_espeakng_acoustic_gate.py`. Она не запускает ASR или detector и не
делает language decision по текстовой эвристике.

## Границы применения

- Это новая **generator family** относительно current RuASD/PyAra train/dev и Silero V4, но не
  внешний independent Russian spoof release. Поиск такого external release остаётся отрицательным.
- FLEURS не даёт public speaker IDs; deterministic controls не являются spoof-voice groups.
- Current result подтверждает input transcript provenance, asset integrity, QA/VAD и narrow
  two-review acoustic language/lexical preservation для locked WAV. Это нельзя расширять на
  другие assets или подменять этим результатом source independence, calibration либо final quality.
- Source и все derivatives остаются personal research; final quality, threshold selection,
  calibration, checkpoint inference и product claim не выполнялись.

## Реализация

- `src/kds/data/espeakng.py` — pinned runtime и text-only synthesis.
- `src/kds/data/fleurs_espeakng.py` — text-disjoint selection и row provenance.
- `scripts/build_fleurs_ru_espeakng_base.py` — selection receipt.
- `scripts/synthesize_fleurs_ru_espeakng.py` — write-once raw synthesis.
- `scripts/publish_fleurs_ru_espeakng_text_accounting.py` — explicit zero-rejection accounting.
- `scripts/build_fleurs_silero_v4_final.py --spoof-source fleurs_ru_v1_espeakng` — generic paired
  candidate proof; filename historical, logic больше не привязан только к Silero.
- `scripts/prepare_fleurs_ru_espeakng_acoustic_gate.py` — write-once packet, reviewer templates и
  fail-closed evaluator.
