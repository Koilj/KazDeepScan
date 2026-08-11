# RuASD full research slice v2

**Статус:** выполнен 12 августа 2026; personal research, non-speaker-disjoint.

V2 исправляет provenance-дефект первой выборки. Поле `model` в source JSON оказалось
неоднородным: среди значений встречались числа и другие данные, которые нельзя выдавать за имя
генератора. Теперь identifier принимается только в ограниченном ASCII-формате и обязан содержать
букву; missing/invalid значения становятся `unspecified_by_source`. Исходное недостоверное
значение не сохраняется: receipt содержит только его SHA-256. Strata строятся по `label/subset`,
а не по непроверенному model field.

## Зафиксированные артефакты

| Артефакт | Результат | SHA-256 |
| --- | --- | --- |
| raw manifest v2 | 2 000: 1 000 / 1 000 | `c9162c554e92548b0fa6896e89cdec06297d2b800ecc87ab76cdb7f2750ea13d` |
| selection receipt | 250 TAR, 37 spoof subsets | `f7f0f2541ae8f70c45ec573fbf9ce81cc0b5e86b36a97c18e362b32ae5a1cf7a` |
| ready manifest v2 | 1 815: 817 / 998 | `82518a2e4e40ef39fa3bcc0c197f717ef83e5f7310cfee53bf546a55fe3ba4cc` |
| QA/VAD rejection report | 185 rows | `2c9b744ed6eeafd4c313a3c5155e23fbde24a6d69e821b22c36dee77ad28064e` |

Ready split: train `1 471`, dev `185`, test `159`. Из `185` exclusions: `161`
insufficient speech, `22` too quiet и `2` excessive clipping. Preprocessing безопасно
переиспользовал только `7` уже существовавших normalized WAV, когда одновременно совпали
`sample_id` и SHA-256 raw audio; остальные `1 808` assets обработаны заново. Все `1 815`
published asset hashes проверены.

Collection SHA-256 повторно не читались только потому, что exact 250-artifact collection уже
прошёл полный `250/250` audit. Intake всё равно проверил имена, размеры, безопасный TAR layout и
JSON/WAV pairing в двух независимых проходах. Raw archives остаются нужны для полного
воспроизведения выборки.

## Ограничения

RuASD не публикует verified speaker IDs для bona-fide и надёжные voice groups для большинства
spoof. Поэтому split защищает доступные record/text keys, но не доказывает speaker/voice
independence. `CC-BY-NC-SA-4.0` сохраняет только personal-research scope.
