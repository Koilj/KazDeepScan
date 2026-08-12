# Stage C — source/rights review и fresh-asset inventory

**Дата проверки:** 12 августа 2026 года  
**Scope:** personal research; без записи голосов, reference audio, voice cloning и model
inference.  
**Решение:** fresh-source inventory зафиксирован, но ни один проверенный TTS-кандидат не прошёл
все обязательные gates. Artifact lock, synthesis и detector inference не запускаются.

## 1. Воспроизводимый inventory

Полный локальный FLEURS release повторно проверен на pinned revision
`4683b04af03d2d9549064c7d72060a9a94bb6046`: exact sizes/hashes всех RU/KK artifacts,
TAR CRC, безопасный layout и совпадение TAR/TSV membership. Затем exact text groups были
сопоставлены с уже оценёнными project manifests. KSC2 candidates, semantic review, QA-ready
manifest и ранее оценённый 30-pair manifest связаны по annotation/sample ID и transcript hash.

Write-once receipt:
`data/manifests/fresh_research_suite_source_inventory_v1.json`, SHA-256
`434a3b12d11e654e4de4438d60a42448cb4badd21a6a0fb700a02a1ce0507b3f`.

| Роль | Полный доступный источник | Уже оценено | Fresh сейчас | Что ещё требуется |
| --- | ---: | ---: | ---: | --- |
| RU FLEURS | 344 unique test texts | 289 | 55 release groups | новая extraction и QA/VAD |
| KK FLEURS | 349 unique test texts | 152 | 60 уже QA-ready + 137 release groups | для 137 — extraction и QA/VAD |
| KSC2 mixed | 2 632 candidates | 30 | 1 уже QA-ready | semantic review для 2 600 pending rows |

Важно: `55` RU и суммарные `197` KK — только release-level capacity до новой selection,
extraction и QA. Они ещё не являются approved final assets. У KSC2 только один свежий row уже
имеет semantic evidence и QA-ready WAV: `ksc2_v1:Test/podcasts/09_03_368`.

Inventory не доказывает source- или speaker-independence: FLEURS и KSC2 уже являются известными
project sources и не публикуют достаточные verified speaker groups. Возможен только честно
обозначенный **asset-level-blind research suite**.

## 2. Проверка новой TTS family

### IMS Toucan — отклонён как новая architecture family

Проверены official source repository, model repository и paper:

- source revision `3cc2094d9c7123336eda7e299ac0bc90319ca9ff`, code license `Apache-2.0`;
- model revision `e0afe0ef703d2178dd7dc74ec298693ddb10e720`, repository license
  `Apache-2.0`;
- official `supervised_languages.json` содержит `rus` и `kaz`;
- `ToucanTTS.pt`: `200 782 528` bytes, SHA-256
  `b36d5d79669ef2b36b1edbf6196132ba95c9e6b03c799d679191e259fe561a59`;
- `Vocoder.pt`: `124 772 377` bytes, SHA-256
  `3f4fa1ea04b2f723cdf4b7fed3ccc73b07fd8dd84723f1e8bc7dee80094ffdbf`;
- interface может использовать `checkpoint["default_emb"]` без reference audio, но одновременно
  включает optional ECAPA reference-audio route. Любой возможный adapter обязан был бы удалить
  этот route и запретить скрытые downloads.

Несмотря на техническую поддержку языков, candidate не проходит главный gate. Official paper
описывает acoustic model как `FastSpeech-2-like` с `FastPitch-style` conditioning и `HiFi-GAN`.
В проекте уже использована family `fastpitch_hifigan_torchscript_tts` (Silero V4). Отличающийся
checkpoint, multilingual frontend и model name не превращают близкую architecture route в новую
независимую family.

Есть и дополнительные ограничения: краткая model card не закрепляет exact training snapshot и
происхождение встроенного `default_emb`; последний checkpoint commit описан как «new default
speaker», но не публикует проверяемый voice ID/rights record. Для текущего research scope это
требовало бы отдельного ограничения, однако architecture overlap уже достаточен для stop.

Official evidence:

- [IMS Toucan source](https://github.com/DigitalPhonetics/IMS-Toucan/tree/3cc2094d9c7123336eda7e299ac0bc90319ca9ff);
- [ToucanTTS model revision](https://huggingface.co/Flux9665/ToucanTTS/tree/e0afe0ef703d2178dd7dc74ec298693ddb10e720);
- [supervised language list](https://huggingface.co/Flux9665/ToucanTTS/blob/e0afe0ef703d2178dd7dc74ec298693ddb10e720/supervised_languages.json);
- [architecture and training-data paper](https://arxiv.org/abs/2406.06403).

### Остальные проверенные routes

| Candidate | Решение | Причина |
| --- | --- | --- |
| RHVoice | reject | official supported-language list содержит Russian, но не Kazakh |
| Meta MMS TTS | reject | поддерживает `rus`/`kaz`, но VITS family уже использована проектом |
| Qwen3-TTS / AIT-Syn | reject | official Qwen3-TTS не заявляет Kazakh; AIT-Syn Kazakh route требует reference voice cloning |
| Coqui XTTS v2 | reject | Kazakh отсутствует в declared language set; основной route использует cloning |
| SeamlessM4T v2 | reject | UnitY2 был бы новой family, но Kazakh не поддерживается как speech-output target |

Переход к разным генераторам для RU/KK/mixed сейчас также не принят: он усложнит сравнение,
а для mixed всё равно не найден text-only, no-reference route с однозначной RU/KK code-switch
поддержкой и полной artifact provenance.

## 3. Stop decision

Этап source/rights review выполнен корректно именно как **отрицательный gate result**:

- TTS weights и новые крупные datasets не скачивались;
- model lock и license-ledger row для отклонённых candidates не создавались;
- fresh assets не извлекались и не выбирались после просмотра detector outputs;
- synthesis, acoustic review и XLS-R inference не выполнялись;
- старые final plans и результаты не изменялись.

Следующий допустимый шаг — найти либо создать действительно новую TTS architecture family,
которая одновременно поддерживает RU, KK и mixed text-only synthesis со встроенным фиксированным
voice profile и проверяемыми правами. Только после успешного review разрешены artifact lock и
frozen selection policy. Ослабление требования «новая architecture family» является изменением
научного контракта и требует отдельного решения владельца проекта.
