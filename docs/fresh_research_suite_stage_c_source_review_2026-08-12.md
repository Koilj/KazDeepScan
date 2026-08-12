# Stage C — source/rights review, route decision и fresh-asset inventory

**Дата проверки:** 12 августа 2026 года

**Scope:** personal research; без записи голосов, reference audio, voice cloning и detector
inference.

**Решение:** абсолютная новизна TTS-архитектуры заменена на проверяемую новизну точного
маршрута `checkpoint + runtime`. Запреты на cloning/reference audio и переиспользование exact
assets не ослаблены. ISSAI KazakhTTS2 Male2 Tacotron2 + ParallelWaveGAN принят как Stage-C
candidate и прошёл artifact/config/technical-smoke gates. Два независимых listening review
одобрили `kk`, `ru` и `mixed` для подготовки evaluation candidate. Detector inference остаётся
запрещён до полного asset-level gate и immutable plan.

## 1. Почему изменён критерий

RuASD train/dev manifests содержат в основном названия генераторов и общий family `tts`, но не
надёжные architecture IDs или checkpoint provenance. Поэтому утверждение «эта архитектура
отсутствовала в обучении» по имеющимся данным нельзя ни доказать, ни опровергнуть. Сохранять
такой gate означало бы выдавать предположение за проверенный факт.

Новый fail-closed контракт доказывает только доступные факты:

- точная тройка `generator_family + generator_name + generator_version`, где version включает
  SHA-256 checkpoints, не встречалась ни в одном сохранённом manifest;
- reference audio и voice cloning запрещены самим runtime contract;
- family/component и fixed-speaker alias overlap публикуются отдельно;
- результат нельзя называть architecture- или speaker-independent;
- разные fixed text-only routes для RU/KK/mixed допустимы, если один route не проходит
  pre-inference language gate. Это не меняет раздельное вычисление метрик.

Иными словами, ослаблено только недоказуемое название novelty, а не защита от leakage.

## 2. Воспроизводимый fresh inventory

Полный локальный FLEURS release повторно проверен на pinned revision
`4683b04af03d2d9549064c7d72060a9a94bb6046`: exact sizes/hashes RU/KK artifacts, TAR CRC,
безопасный layout и совпадение TAR/TSV membership. KSC2 candidates связаны с semantic review,
QA-ready manifest и ранее оценёнными assets.

Write-once receipt:
`data/manifests/fresh_research_suite_source_inventory_v1.json`, SHA-256
`434a3b12d11e654e4de4438d60a42448cb4badd21a6a0fb700a02a1ce0507b3f`.

| Роль | Полный доступный источник | Уже оценено | Fresh сейчас | Что ещё требуется |
| --- | ---: | ---: | ---: | --- |
| RU FLEURS | 344 unique test texts | 289 | 55 release groups | extraction и QA/VAD |
| KK FLEURS | 349 unique test texts | 152 | 60 QA-ready + 137 release groups | для 137 — extraction и QA/VAD |
| KSC2 mixed | 2 632 candidates | 30 | 1 QA-ready | semantic review для 2 600 rows |

Эти числа являются capacity до selection, а не approved final assets. Без нового corpus и
verified speaker IDs возможен только честно обозначенный **asset-level-blind research suite**.

## 3. Принятый generator route

Используется официальный репозиторий [IS2AI/Kazakh_TTS](https://github.com/IS2AI/Kazakh_TTS/tree/fc906048ff5914a3528d1ae4ed6f7ccd94d71383),
а не ранее отклонённый `IS2AI/TurkicTTS` без объявленной repository license. Официальный
Kazakh_TTS repository публикует CC-BY-4.0 license, пять Tacotron2 checkpoints и соответствующие
ParallelWaveGAN vocoders. Для проекта разрешён только один фиксированный профиль Male2.

Model lock: `configs/research/kazakhtts_tacotron2_pwg_v1_models.json`, SHA-256
`1ee5150c1c9c5f69b33cc4a4a67148904079651830b0a144f8647af10c8cf68e`.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| source revision archive | 17 551 | `889cd6d323d09a36bb1e479dcccf4850382e9f4d8b752496958836467db7d7ad` |
| Male2 Tacotron2 archive | 107 280 461 | `ea80b8227c94e12c87f447270c17d93d5eebe89b7df3f7c76f7b30523a5cf2c2` |
| Male2 ParallelWaveGAN archive | 15 610 294 | `3efb427b99a1a51b53b3cf08841a64f040b41b70c791e4b3a8f4edc7dca71337` |

Проверены CRC обоих ZIP, allowlisted extraction и inner SHA-256 для metadata, acoustic config,
normalization statistics, Tacotron2 checkpoint, vocoder config и vocoder checkpoint. Config
подтвердил `char + Tacotron2`, 80 mel bands и 22 050 Hz; vocoder — ParallelWaveGAN 0.4.8,
80-channel conditioning и 22 050 Hz. Runtime зафиксирован как ESPnet 0.10.6 +
ParallelWaveGAN 0.6.1; для удалённого из нового SciPy alias `signal.kaiser` используется узкий
локальный shim на `signal.windows.kaiser`.

Route exposure receipt:
`data/manifests/fresh_suite_stage_c_generator_route_gate_v1.json`, SHA-256
`54da4a704b59f2a6ff24395a90dee013b3a1fd1a48e65355bab4008426358f57`.

- проверено 46 manifest-файлов и 17 657 сохранённых spoof-строк;
- найдено 313 точных исторических routes;
- exact candidate route overlap: `0`;
- generator-family overlap: `0`;
- alias `ISSAI_KazakhTTS2_M2`: 312 строк через ранее оценённый Piper route.

Последний пункт принципиален: новый checkpoint/runtime route не делает голос новым. Этот слой
может проверять устойчивость к другому способу синтеза того же опубликованного профиля, но не
устойчивость к новому speaker identity.

## 4. Технический pre-detector smoke

Frozen smoke plan SHA-256:
`92f695ad964c27b4e749a728919083c51f4a9ead37ec3f0e9ccea67db3a3f40b`.

CUDA/PyTorch 2.11 успешно загрузил старый ESPnet checkpoint в safe weights-only default mode.
Все три входа входят в checkpoint token list; созданы конечные mono PCM WAV 22 050 Hz:

| Язык | Официальный статус | Duration | Peak | Решение сейчас |
| --- | --- | ---: | ---: | --- |
| KK | supported | 3.286 s | 0.361 | listening pass, 2 reviewers |
| RU | не заявлен upstream | 4.470 s | 0.355 | listening pass, 2 reviewers |
| mixed | не заявлен upstream | 2.705 s | 0.419 | listening pass, 2 reviewers |

Smoke report SHA-256:
`fc10d5660eca06a44bfc7433838ac7043ee5ee93171b277d4034f10356c4377b`.
Detector checkpoint, logits и predictions не открывались.

Listening packet содержит три exact WAV и не содержит model predictions. Packet SHA-256:
`12dd6caa8bff9332708e1b365002545ddec9c7ea15b92ad82cb89de82ac37dea`.
Две формы заполнены разными reviewer IDs и строго проверены. Для каждого языка оба решения
`pass/yes/yes/yes/no`: intelligible speech, сохранённый текст и язык без тяжёлых артефактов.
Gate receipt SHA-256:
`946c3a3a59fdd437553c2fe8e93d4ade157e718cf67505abb1216c02cbc82a73`;
`approved_input_languages=["kk", "mixed", "ru"]`, но `detector_inference_authorized=false`.
Подробный receipt: `docs/fresh_suite_stage_c_kazakhtts_acoustic_gate_v1.md`.

## 5. Отклонённые routes и дальнейшее ветвление

IMS Toucan остаётся отклонённым именно как заявленная новая architecture family: его
FastSpeech-2-like/FastPitch-style + HiFi-GAN components близки уже использованному Silero V4.
После смены критерия его можно было бы повторно рассматривать как exact route, но KazakhTTS
предпочтён из-за более простого fixed-voice/no-reference interface, официальных artifact links
и успешно проверенной локальной совместимости.

RHVoice не имеет официальной Kazakh поддержки; MMS/VITS уже представлен; Qwen3-TTS не заявляет
Kazakh; Coqui XTTS/AIT-Syn требуют cloning/reference voice; SeamlessM4T v2 не поддерживает
Kazakh как speech-output target.

После reviews решение принято по языкам:

1. KK pass разрешает перейти к frozen selection 60 уже QA-ready fresh KK groups.
2. RU/mixed pass разрешает только сбор отдельного fresh candidate и новый full-asset acoustic
   gate до detector inference.
3. Все три роли прошли; альтернативный exact route сейчас не требуется.
4. При трёх pass сначала фиксируются selection policy, rejection accounting, ledger
   snapshot и immutable run plan. Detector inference остаётся последним одноразовым действием.

Следующий корректный автоматический этап — заморозить selection policy и exact bona-fide groups
до массового synthesis. Следующий человеческий gate потребуется уже для всех созданных assets.
