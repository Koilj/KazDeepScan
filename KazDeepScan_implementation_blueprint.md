# KazDeepScan — инженерный план реализации

**Версия:** 1.0  
**Проверено:** 8 августа 2026  
**Цель:** сервис, который по записи речи на русском, казахском или смешанной русско-казахской речи выдаёт *калиброванную оценку риска синтезированности*, а не обещание «абсолютной истины».

## 1. Итоговое решение

Стартовая архитектура проекта:

| Слой | Выбранное решение | Почему |
|---|---|---|
| Вход | ffmpeg, WAV/FLAC/MP3/OGG/M4A, максимум 10 минут в MVP | безопасное декодирование и одинаковый тракт для файлов и голосовых |
| Сегментация | WebRTC VAD + проверка качества речи | убирает тишину и не даёт модели принимать решение по пустому аудио |
| Модель-кандидат №1 | XLS-R-300M + SLS head | мультиязычный SSL-фронтенд, Apache-2.0, подходит для ru/kk, реалистичен на 16 ГБ VRAM |
| Обязательный конкурент | XLS-R-300M + AASIST head | проверяет, даёт ли графовая голова выигрыш именно на казахском и телефонном тестах |
| Продуктовая модель | победитель фиксированного bake-off по худшей OOD-метрике, не по общей accuracy | нельзя честно объявить одну модель «лучшей» до независимого казахского и телефонного теста |
| Вероятность | логиты окон → агрегация записи → temperature scaling на отдельном validation set | только так вероятность относится к целой записи |
| API | FastAPI, один worker на GPU, Redis/RQ для длинных файлов | удобно начать локально и масштабировать |
| Хранение | исходник кратковременно, метаданные в PostgreSQL, MinIO/S3-совместимое хранилище при развёртывании | голос — чувствительные персональные данные |
| Эксперименты | MLflow локально + DVC + Git + Hydra-конфиги | не отправляет голосовые записи в чужой облачный трекер по умолчанию |

**Не надо делать в первой версии:** определение личности говорящего, распознавание мошенничества по смыслу разговора, перехват обычных GSM-звонков, «детектор любого ИИ на 100%», ансамбль из десятка моделей и экспорт в ONNX до измерения PyTorch-инференса.

## 2. Что исправить в приложенном черновике

| В черновике | Решение в плане |
|---|---|
| Жёстко зафиксированы будущие версии пакетов, включая PyTorch 2.12 | Не пинить непроверенную версию. На день установки брать команду с [официального селектора PyTorch](https://pytorch.org/get-started/locally/), Python 3.11 и сборку CUDA 12.8+. Для RTX 5060 Ti (sm_120) это критично. Драйвер с твоего nvidia-smi 595.84 и заявленной CUDA 13.2 не требует отдельной установки CUDA Toolkit для обычных PyTorch wheels. |
| ASVspoof 5 назван основным train-набором продукта | Это обязательный **исследовательский benchmark**, но перед любым коммерческим обучением нужно прочитать его LICENSE.txt. Сам Zenodo прямо просит это сделать. |
| Ru/kk-фейки предлагается брать из моделей без единого реестра прав | Нужны два контура: research и commercial-clean. Вес модели, её код, TTS-выход и исходный голос имеют разные права. |
| Вердикт записи — среднее калиброванных вероятностей окон | Неверно: среднее вероятностей не является калиброванной вероятностью записи. Сначала агрегируются **сырые логиты**, затем один calibrator обучается на record-level validation set. |
| XLS-R + AASIST выбран навсегда до измерений | Это хороший кандидат, но «навсегда» — риск. На свежем русском RuASD TCM-ADD, SLS/XLS-R и SSL-системы меняют места в зависимости от канала; чистая accuracy не предсказывает телефонную устойчивость. |
| ONNX Runtime CPU задан как единственный продуктовый инференс | Сначала PyTorch GPU на твоей RTX и измерение p50/p95. CPU ONNX добавляется только после проверки численного parity и реальной латентности. |
| W&B по умолчанию | Для голосовых лучше локальный MLflow; в облако можно отправлять только числа, конфиги и хеши после отдельного решения. |
| F5-TTS и XTTS выглядят допустимыми для продукта | Их официальные pretrained weights — non-commercial. Оставить только в research/OOD-контуре. |

## 3. Ограничение задачи и честный UX

Модель распознаёт **акустические признаки** TTS, voice conversion, клонирования голоса и частично перепроигранного синтезированного звука. Она не доказывает, кто именно говорил, не проверяет смысл истории «я попал в беду», не выявляет мошенника как человека и не гарантирует детекцию нового генератора.

В интерфейсе вместо фразы «это дипфейк» выводить:

| Риск | Условие после калибровки | Текст пользователю |
|---|---|---|
| Низкий | p_fake < T_low | «По имеющимся акустическим признакам синтез не обнаружен. Это не подтверждает личность собеседника.» |
| Неопределённо | T_low ≤ p_fake < T_high или низкое качество | «Запись недостаточно однозначна. Проверьте источник другим способом.» |
| Высокий | p_fake ≥ T_high | «Есть сильные признаки синтезированной или изменённой речи. Не сообщайте коды и перезвоните человеку по известному номеру.» |
| Не анализируется | меньше 2.5 с речи, перегрузка, слишком сильный шум | «Недостаточно пригодной речи для надёжной оценки.» |

Пороги выбираются не «0.5», а по product validation set. Для антифрод-режима важнее ограничить ложные обвинения: отдельно публикуются FPR при выбранном T_high и FNR при T_low.

## 4. Схема сервиса

~~~mermaid
flowchart TD
    A["Файл или аудиопоток"] --> B["Decode + проверки"]
    B --> C["VAD и окна речи"]
    C --> D["XLS-R + classifier"]
    D --> E["Агрегация логитов и калибровка"]
    E --> F["Риск, сегменты, причина отказа"]
~~~

### Два режима

1. **Файл/голосовое.** POST /v1/analyze принимает файл, отвечает JSON. До 60 секунд можно обрабатывать синхронно; длиннее — job_id и очередь.
2. **Звонок/стрим.** Только если сервис получает аудио законно: WebSocket от мобильного приложения, SIP/WebRTC-клиента или собственного Asterisk/FreeSWITCH-контура. Обычный Android/iPhone не даёт приложению универсально перехватывать любой сотовый звонок; это не надо обещать в MVP.

Для стрима буферизировать речь в 4.04-секундные окна с шагом 1.0–2.0 секунды. Первый корректный результат появляется после накопления минимум 2.5–4 секунд речи.

## 5. Данные: главное место, где выиграет или проиграет проект

### 5.1. Юридически разные контуры

| Контур | Назначение | Что можно включить |
|---|---|---|
| Research | диплом, эксперименты, статьи, сравнение моделей | ASVspoof 5, RuASD, MLAAD, F5-TTS, XTTS и другие NC-ресурсы по их условиям |
| Commercial-clean | будущий сервис с коммерческим использованием | собственные записи с информированным согласием; данные и TTS-выходы с проверенным разрешением на коммерческое ML-обучение |
| Hold-out/OOD | финальная честная проверка, **никогда не train и не выбор гиперпараметров** | новые TTS/API, новые телефоны, реальные consented voice clones, независимые дикторы и тексты |

Не переносить ни один файл между контурами «на глаз». У каждой строки манифеста должно быть поле rights_basis и ссылка на зафиксированную версию лицензии/согласия.

### 5.2. Реальная речь

| Источник | Язык | Практическая ценность | Режим |
|---|---|---|---|
| [KSC2](https://issai.nu.edu.kz/kz-speech-corpus/) | kk и code-switch | около 1.2 тыс. часов, 600 тыс.+ записей, CC BY 4.0 и прямо допускается индустрией при атрибуции | основной источник казахской bona fide речи |
| [KSC OpenSLR-102](https://www.openslr.org/102/) | kk | около 332 часов, CC BY 4.0 | дополнение, device/speaker diversity |
| [KSD OpenSLR-140](https://www.openslr.org/140/) | kk | 554 часа с мобильных iOS/Android — очень полезно против bias «студийная реальная речь» | research и OOD; для закрытого продукта CC BY-SA 3.0 требует отдельной юридической оценки |
| [Mozilla Common Voice](https://mozilladatacollective.com/organization/cmfh0j9o10006ns07jq45h7xk) | ru, kk | короткие записи, CC0 | коммерчески чистое дополнение и внешний test |
| [Golos](https://github.com/sberdevices/golos) | ru | около 1 240 часов crowd и far-field речи | пока research/кандидат: прочитать приложенный PDF-лицензионный договор до использования в clean-контуре |
| [OpenSTT](https://learn.microsoft.com/en-us/azure/open-datasets/dataset-open-speech-text) | ru | есть телефонный домен, но страница указывает исходные условия и NC-ограничение | только research/OOD |
| Собственная коллекция | ru, kk, mixed | единственный источник, который реально отражает казахстанские телефоны, акценты и согласие | ключевой продуктовый набор |

### 5.3. Готовые fake-наборы

| Набор | Содержание | Как использовать |
|---|---|---|
| [ASVspoof 5](https://zenodo.org/records/14498691) | около 2 000 дикторов, более 20 атак и 7 adversarial-атак, 142.3 ГБ | benchmark и pretraining research; сначала проверить LICENSE.txt |
| [ASVspoof 2021 DF](https://www.asvspoof.org/index2021.html) | TTS/VC с кодеками и распространением | обязательный test на codec/channel shift |
| [RuASD](https://huggingface.co/datasets/lab260/RuASD) | русский: 37 современных TTS/VC, 691.68 часа fake и 234.07 часа real | лучший русскоязычный research benchmark; лицензия CC BY-NC-SA 4.0, значит не в коммерческом train |
| [MLAAD](https://huggingface.co/datasets/mueller91/MLAAD) | многоязычные фейки, включая русский | research/OOD; CC BY-NC 4.0 |
| [LRLspoof](https://huggingface.co/datasets/lab260/LRLspoof) | 2 732 часа spoof-only в 66 языках, включая казахский; 3 казахских синтезатора в диагностике | особенно ценен как **казахский fake-recall OOD test**; не склеивать напрямую с KSC2, потому что он spoof-only и модель легко выучит происхождение набора |

Последнее ограничение важно: сами авторы LRLspoof отдельно предупреждают, что к spoof-only набору нельзя просто добавить real из другого домена — модель начнёт отличать домены, а не deepfake.

### 5.4. Собственный набор KazDeepScan

Минимальная реалистичная цель для первой обучаемой версии:

| Часть | RU | KK | Содержание |
|---|---:|---:|---|
| Bona fide clean | 40 ч | 40 ч | смартфон, ноутбук, гарнитура, тихая комната |
| Bona fide channel | 30 ч | 30 ч | реальный VoIP/SIP, speakerphone, 8 кГц, Bluetooth, шум |
| TTS | 75 ч | 75 ч | 3–5 разных семейств, равный баланс по семействам и голосам |
| Consented voice-clone | 20 ч | 20 ч | только голоса участников, давших отдельное согласие на имитацию |
| Partial fake | 10 ч | 10 ч | заменённые фразы и склейки с известными границами |
| Независимый test | 20 ч | 20 ч | новые люди, тексты, устройства, генераторы и каналы |

Это примерно 340 часов до on-the-fly аугментаций. Для MVP можно начать с половины, но **не уменьшать независимый test**. После подтверждения дизайна нарастить до 300+ ч bona fide и 500+ ч fake на язык.

#### Как собирать

- 100+ дикторов на язык в сумме, с возрастом, разными регионами, мужчиной/женщиной, русско-казахским code-switching.
- Записывать только свои нейтральные сценарии: бытовой диалог, банковские предупреждения, просьба перезвонить, числа, имена, спонтанная речь. Не использовать реальные мошеннические разговоры.
- Получить два согласия: на запись для detector training и, отдельно, на генерацию ограниченного клона. Участник может отозвать согласие; его аудио и производные fake тогда удаляются из следующей версии набора.
- Хранить исходник в исходной частоте (WAV/FLAC 24/48 кГц); для модели создавать копию 16 кГц mono FLAC. Никогда не терять исходный файл и его SHA-256.
- Для телефонного поднабора записать **реальные** тракты: SIP/WebRTC, handset, громкая связь, Bluetooth; синтетическую речь воспроизвести через динамик и заново захватить этими траками. Это отдельный класс «playback/channel», а не подмена TTS.

### 5.5. Манифест вместо папок с непонятными файлами

Одна строка Parquet/CSV на каждый audio asset:

~~~text
sample_id, sha256, split, label, language, code_switch,
parent_group_id, source_name, source_license, rights_basis,
speaker_pseudo_id, text_id, text_hash, duration_s,
generator_family, generator_name, generator_version, voice_id,
clone_consent_id, device, capture_route, original_sr,
codec, augmentation_chain, augmentation_seed, created_at
~~~

parent_group_id связывает оригинал, синтез по тому же тексту, replay и все производные. Группа целиком попадает ровно в один split.

## 6. Создание собственных фейков

Цель не в том, чтобы сгенерировать миллион одинаковых голосов. Нужно покрыть **разные механизмы синтеза** и сохранить происхождение каждого файла.

| Источник | RU | KK | Роль | Правило |
|---|---|---|---|---|
| [Piper voices](https://huggingface.co/rhasspy/piper-voices) | да | да | доступные локальные baseline TTS | сам repo заявлен MIT, но обязательно проверить model card конкретного голоса |
| [KazakhTTS2 recipe](https://github.com/IS2AI/Kazakh_TTS) | — | да | важнейший локальный казахский источник; 5 голосов, ESPnet, CC BY 4.0 | добавить атрибуцию и сохранить версию весов/вокодера |
| Silero base CIS | проверить | проверить | дополнительное семейство в clean-контуре только для поддерживаемых языков | base cis-tts объявлен MIT, но сначала автоматический language/качество тест |
| Коммерческие TTS API | да | зависит от провайдера | только OOD-test, ближе к реальной угрозе | не включать в train без письменной проверки ToS и прав на ML-обучение выходов |
| F5-TTS | да, после quality-test | zero-shot, после quality-test | research OOD | код MIT, но официальные pretrained weights CC BY-NC 4.0 |
| XTTS-v2 | да | после quality-test | research voice-clone OOD | Coqui Public Model License: только non-commercial |
| Собственный clone на consenting voices | да | да | самый приближённый к угрозе и юридически управляемый источник | отдельное согласие, закрытое хранение reference-аудио, запрет чужих голосов |

Не считать Kokoro покрытием русского или казахского, пока конкретная модель не проходит language acceptance test. Не считать «модель умеет zero-shot» доказательством качественного казахского фейка: нужны native-speaker прослушивание, ASR-CER и ручной аудит.

### Quality gate синтезированного файла

Файл допускается в data pool, если:

1. декодируется без ошибки, один канал, нет цифровых обрывов;
2. длится 2.5–20 секунд и содержит ≥2 секунд речи;
3. clipping меньше установленного порога, нет DC offset/пустого сигнала;
4. transcript сохраняется, а ASR/CER служит только фильтром явной неразборчивости;
5. 2% на генератор вручную прослушаны носителем языка;
6. нет утечки: голос, текст, generator_name и parent_group не пересекают splits.

## 7. Аугментация: реалистично и симметрично

Любая аугментация применяется с одинаковой вероятностью к real и fake. Иначе классификатор узнает «MP3 значит fake», а не артефакты синтеза.

| Тип | Параметры для старта | Вероятность |
|---|---|---:|
| RawBoost | официальная waveform-аугментация: линейное/нелинейное искажение, импульсный шум | 0.30 |
| Шум MUSAN | речь/музыка/шум, SNR 5–30 dB; отдельный hard-set 0–5 dB | 0.35 |
| RIR | реальные/симулированные RIR, RT60 примерно 0.2–1.2 с | 0.25 |
| Телефония | G.711 μ-law/A-law, G.722, GSM, AMR-NB, Opus 8/12/16/24 kbps; 16→8→16 кГц | 0.45 |
| Платформы | AAC/M4A, MP3, OGG/Opus, Telegram-подобный Opus | 0.25 |
| Microphone/device | EQ, low/high-pass, gain ±6 dB, mild saturation | 0.20 |
| Packet loss/jitter | 0–5% frame loss и PLC-симуляция | 0.15 |
| Replay | воспроизведение через динамик и повторный захват или отдельный RIR+EQ режим | отдельный task/group |

[MUSAN](https://www.openslr.org/17/) имеет CC BY 4.0, [RIRS_NOISES](https://www.openslr.org/28/) — Apache-2.0. [RawBoost](https://github.com/TakHemlata/RawBoost-antispoofing) особенно полезен, потому что моделирует искажения waveform без внешнего аудио.

Правила:

- Не применять одновременно пять эффектов к каждой записи. Для train выбирать 0–2 эффекта, а для отдельного stress-test — заранее названные цепочки.
- Сохранять seed и цепочку в manifest; не материализовывать все варианты на диск — применять on-the-fly.
- Codec family и generator family балансировать в каждом split.
- Не использовать denoise, dereverb или агрессивную loudness-normalization до классификации: они могут удалить именно тот артефакт, который нужен детектору. Разрешено удалить DC, привести канал к mono и безопасно ограничить пиковый уровень.

## 8. Признаки и модель

### 8.1. Что подаётся в модель

Главный признак — **raw waveform 16 кГц**, нормированный в диапазон [-1, 1]. XLS-R извлекает SSL-представления; это не ASR и не требует транскрипта во время инференса.

- log-Mel и спектрограмма нужны для QA/UI и лёгкого baseline;
- LFCC + GMM/LCNN — обязательный sanity baseline, но не основная модель;
- не добавлять F0, breath или speaker embeddings в MVP: это повышает сложность и риск demographic/speaker bias, не решая основной OOD-проблемы.

### 8.2. Bake-off, который определит продуктовую модель

| Кандидат | Реализация | Зачем |
|---|---|---|
| B0 | LFCC + LCNN или ResNet18 | быстрый контроль: pipeline, labels, data leakage |
| B1 | XLS-R-300M + SLS | стартовый продуктовый кандидат, сравнительно простая голова |
| B2 | XLS-R-300M + AASIST | проверка graph-attention на тех же окнах и сплитах |
| B3 | TCM-ADD | исследовательский соперник, если его код/лицензия/VRAM проходят аудит |

**Выбор:** модель выигрывает, если имеет наименьший worst-group EER среди kk-OOD, ru-OOD, real phone, synthetic phone и partial-fake, при приемлемом p95 latency. Не выбирать по ASVspoof-only score. Это подтверждает RuASD: на clean Russian set TCM-ADD получил EER 0.143, SLS/XLS-R 0.221, AASIST3 0.231, а при шуме/реверберации порядок меняется.

### 8.3. Рекомендованный первый trainable вариант

**XLS-R-300M + SLS binary head.**

1. [XLS-R-300M](https://huggingface.co/facebook/wav2vec2-xls-r-300m) (Apache-2.0) принимает 16 кГц и предобучен на 128 языках.
2. Запросить hidden states transformer-слоёв.
3. SLS обучает softmax-веса для слоёв; это даёт взвешенную свёртку SSL-признаков.
4. LayerNorm → attentive statistics pooling (mean+std) → Dropout 0.2 → MLP 2048→256→1.
5. Выход — один **logit fake**, а не сразу softmax probability.

Референс SLS показывает конкурентоспособность на ASVspoof 2021 DF и In-the-Wild, но его старый fairseq-код нельзя копировать как среду проекта. Реализовать head поверх Transformers AutoModel — это совместимо с современным PyTorch и Blackwell.

### 8.4. Окна и агрегация

- Базовое окно: 64 600 samples ≈ 4.04 с, как у многих ASVspoof baseline.
- Train: случайный speech-aware crop; короткий файл циклически дополняется только после VAD.
- File inference: окна 4.04 с, шаг 2.0 с; для partial fake параллельно хранить window scores.
- Full-fake score: duration-weighted average **logits** окна.
- Partial-fake score: top-k mean logits, где k=max(2, ceil(20% окон)); его threshold калибруется отдельно.
- Затем record-level logits проходят temperature scaling. Нельзя калибровать окна, усреднить p и назвать это вероятностью записи.
- Если речи меньше 2.5 с — отказаться от вероятности.

## 9. Обучение на RTX 5060 Ti 16 ГБ

Твоего железа достаточно для данного проекта, но не для бессмысленного обучения 1B/2B моделей с нуля.

| Этап | Что обучаем | Настройка |
|---|---|---|
| Smoke test | B0 на 1–2 ч | подтверждает загрузку, labels, GPU, метрики и API |
| Stage A | только SLS head | encoder frozen, batch 16–32, 3–5 эпох |
| Stage B | head + последние 8 блоков XLS-R | bf16, batch 4, gradient accumulation 8, gradient checkpointing, 15–25 эпох |
| Stage C | full fine-tune только если Stage B не проходит OOD | batch 1–2, accumulation 16+, по профилировщику VRAM |

Стартовые гиперпараметры:

~~~yaml
seed: 20260808
sample_rate: 16000
window_samples: 64600
optimizer: AdamW
lr_head: 1.0e-4
lr_encoder: 1.0e-5
weight_decay: 1.0e-4
warmup_ratio: 0.10
scheduler: cosine
loss: BCEWithLogitsLoss
sampler: class_balanced + language_balanced + generator_balanced
precision: bf16
gradient_clip_norm: 1.0
early_stopping: worst_group_eer
~~~

Если batch уже сбалансирован, BCEWithLogitsLoss лучше, чем бездумно добавлять сильные class weights. Focal loss — только отдельная ablation, не базовая настройка.

Проверка окружения после установки:

~~~bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_arch_list())"
~~~

В выводе должен быть доступен GPU и поддержка sm_120. Не собирать PyTorch из исходников, пока это не доказанно необходимо.

### Диск и память

1 ТБ NVMe подходит для MVP и выбранных подмножеств, но не для удобного одновременного хранения всех benchmark-ов. Один только LRLspoof занимает около 452 ГБ, а ASVspoof 5 — 142.3 ГБ.

| Резерв на SSD | Что хранить |
|---:|---|
| 150 ГБ | Ubuntu, пакеты, Hugging Face cache, Docker |
| 250–350 ГБ | активный рабочий data slice и собственный clean-набор |
| 80–120 ГБ | checkpoints, MLflow artifacts, отчёты |
| 150–200 ГБ | DVC cache и запас на временные decode-файлы |

Не хранить одновременно LRLspoof, ASVspoof 5, полный KSC2/KSD и materialized-аугментации. LRLspoof использовать через целевой язык/генератор либо на внешнем SSD. Для research-роста лучше добавить внешний SSD 2 ТБ. DVC настроить на hardlink/reflink, иначе cache может незаметно удвоить объём данных.

## 10. Splits, тесты и метрики

### 10.1. Непересекающиеся группы

| Раздел | Нельзя пересекать между train/dev/test |
|---|---|
| Диктор | один реальный диктор и все его clone-версии |
| Текст | текст и его TTS-производные |
| generator | минимум один generator family полностью hold-out |
| Устройство/канал | минимум один тип телефона/кодека hold-out |
| Источник датасета | хотя бы один corpus полностью hold-out |

Обычный random split аудиофайлов здесь даёт искусственно высокую оценку из-за одинаковых дикторов, текстов и TTS-артефактов по обе стороны.

### 10.2. Набор отчётов

| Метрика | Для чего |
|---|---|
| ROC-AUC и PR-AUC | общая дискриминация; PR-AUC важен при реальной редкости фейков |
| EER | стандарт anti-spoofing, сравнение с ASVspoof/RuASD |
| FPR@TPR и FNR@FPR | реальные рабочие пороги |
| ECE, Brier score, reliability diagram | показывает, можно ли назвать p_fake вероятностью |
| Worst-group EER | защита от хорошего среднего за счёт плохого казахского/телефонного сегмента |
| p50/p95 latency, RTF, VRAM | пригодность сервиса |
| abstention rate | сколько честно неанализируемых записей |

t-DCF считать только там, где есть нужный ASV-сценарий и протокол. Для обычного API без speaker verification это не главная метрика.

### 10.3. Обязательные независимые тесты

1. clean ru, clean kk и code-switch;
2. unseen generator, включая облачный TTS только после законного получения тестовых аудио;
3. unseen codec: Opus, AMR-NB, GSM, G.711;
4. реальная запись через телефон и speakerphone;
5. шум, RIR, шум+codec, RIR+codec;
6. replay TTS;
7. частично заменённая фраза с известной границей;
8. короткая речь 2.5–4 с и длинная речь 30–180 с;
9. срез по языку, полу/возрастной группе (если законно собраны и агрегированы), устройству и уровню шума.

Нельзя публиковать «95% accuracy» без таблицы этих групп.

## 11. API и безопасная обработка аудио

### API-контракт

~~~text
POST /v1/analyze
  multipart: audio, mode = file|call, language_hint = ru|kk|mixed|auto

200:
{
  "analysis_id": "uuid",
  "status": "ok|insufficient_speech|unsupported|queued",
  "risk_score": 0.82,
  "risk_band": "high",
  "model_version": "kds-xlsr-sls-0.1.0",
  "speech_seconds": 18.7,
  "segments": [{"start_s": 0.0, "end_s": 4.04, "risk_score": 0.79}],
  "calibration_domain": "ru_kk_telephony_v1",
  "disclaimer": "..."
}

GET /v1/jobs/{analysis_id}
GET /healthz
GET /readyz
WS /v1/stream
~~~

### Безопасность

- Limit: file size 50 MiB и длительность 10 минут в MVP; проверять MIME **и** фактическое декодирование.
- Decode через отдельный непривилегированный процесс/контейнер; не доверять расширению файла.
- UUID вместо имени файла, SHA-256 для дедупликации.
- TLS, API key/OAuth, rate limit, audit log без сырого аудио.
- По умолчанию удалить оригинал после ответа или максимум через 24 часа; хранить только score, модель, timestamp и технические метаданные.
- Аудио, consent forms и voice embeddings шифровать at rest, разделять ключи и доступы.
- Не писать содержимое аудио, текст из ASR или token в application logs.
- Не использовать загруженное пользователем аудио для retraining без отдельного opt-in.

Голосовые записи могут быть персональными/биометрическими данными. Закон Казахстана о [персональных данных](https://adilet.zan.kz/eng/docs/Z1300000094) требует целевого, достаточного по объёму сбора и согласия в обычном сценарии. Для продукта нужны privacy notice, срок хранения, удаление по запросу и отдельная юридическая проверка локализации/передачи данных. Это инженерное требование, не юридическая консультация.

## 12. Стек и структура репозитория

### Стек

| Область | Технология |
|---|---|
| Python | 3.11, venv или uv |
| ML | PyTorch/Torchaudio с CUDA 12.8+, Transformers, Accelerate, Safetensors |
| Аудио | ffmpeg, soundfile, librosa только для анализа, audiomentations, WebRTC VAD |
| Конфиги | Hydra + YAML |
| Данные | Parquet manifests, DVC remote на внешнем SSD/MinIO |
| Эксперименты | MLflow local, TensorBoard по желанию |
| Backend | FastAPI, Pydantic, Uvicorn, Redis + RQ |
| Хранилище | PostgreSQL; MinIO только когда нужны сохранённые аудио/job artifacts |
| UI | Gradio для internal demo, затем отдельный web client |
| Качество | Ruff, Pytest, mypy, pre-commit |
| Деплой | Docker Compose для API/Redis/Postgres; train на хосте |

### Дерево

~~~text
kazdeepscan/
  pyproject.toml
  README.md
  configs/
    data/
    model/
    train/
    eval/
  data/
    manifests/              # DVC tracked, без сырого audio в Git
    licenses/
    consent/
  src/kds/
    audio/                  # decode, VAD, windows, validation
    data/                   # manifest, splitter, sampler, QA
    augment/
    models/                 # xlsr_sls.py, xlsr_aasist.py, baseline.py
    train/
    eval/                   # metrics, calibration, reports
    serving/
  scripts/
    ingest.py
    build_manifest.py
    synthesize_authorized.py
    train.py
    evaluate.py
    calibrate.py
    benchmark_latency.py
  services/api/
  tests/
  model_cards/
~~~

Никаких raw audio, consent PDF, .env, API keys, checkpoint weights и DVC cache в Git.

## 13. Порядок разработки

| Неделя | Результат |
|---|---|
| 1 | репозиторий, lockfile, GPU smoke test, ffmpeg/VAD, manifest schema, B0 на мини-наборе |
| 2 | ingestion KSC2/Common Voice/разрешённого ru, data QA, групповой split, первый data card/license ledger |
| 3 | генератор authorised fake, Piper/KazakhTTS2, provenance manifests, RawBoost и codec augmentations |
| 4 | XLS-R + SLS Stage A/B, MLflow, record-level evaluator и temperature scaler |
| 5 | XLS-R + AASIST competitor, bake-off и error analysis на ru/kk/channel |
| 6 | FastAPI + Gradio, file mode, latency profiling, security limits |
| 7 | SIP/WebSocket streaming prototype и реальная телефонная тестовая коллекция |
| 8 | model card, data card, reproducibility report, демонстрация и финальный independent test |

## 14. Критерии готовности MVP

MVP можно показывать, только если выполнены все пункты:

- есть отдельные ru, kk, mixed и telephony test manifests;
- нет parent_group/speaker/text/generator leakage;
- есть baseline B0 и минимум B1/B2 comparison;
- опубликованы EER, PR-AUC, ECE, FPR/FNR и worst-group таблица;
- calibration fit делался только на dev, а не test;
- API отказывается от короткой/некачественной речи;
- исходные записи не остаются на сервере по умолчанию;
- понятны права каждого train-источника;
- есть model card с known failure modes;
- ручная проверка содержит минимум 50 реальных и 50 fake записей на язык, включая телефон.

## 15. Главные риски и защита

| Риск | Защита |
|---|---|
| Model учит dataset fingerprint | group-aware split, matched own data, source hold-out |
| Новый TTS обходится | unseen-generator test, ежеквартальный data refresh, abstain при OOD |
| Телефонный codec рушит качество | codec-first training и real telephony hold-out |
| Казахский хуже русского | не объединять языки только ради средней метрики; отдельные thresholds/reports |
| Ложное обвинение человека | high threshold, три состояния, предупреждение «проверьте другим каналом» |
| Лицензионный риск | research/commercial-clean separation, ledger и audit до релиза |
| Утечка голосов | согласия, минимизация хранения, шифрование, удаление, нет retraining без opt-in |
| Мало VRAM | 300M, bf16, partial unfreeze, grad accumulation, 4-секундные окна |

## 16. Источники, на которых основаны решения

- [PyTorch: актуальная матрица установки](https://pytorch.org/get-started/locally/)
- [XLS-R-300M model card и лицензия](https://huggingface.co/facebook/wav2vec2-xls-r-300m)
- [SSL anti-spoofing reference](https://github.com/TakHemlata/SSL_Anti-spoofing)
- [SLS + XLS-R reference](https://github.com/QiShanZhang/SLSforASVspoof-2021-DF)
- [AASIST official implementation](https://github.com/clovaai/aasist)
- [ASVspoof 5 on Zenodo](https://zenodo.org/records/14498691)
- [RuASD paper](https://arxiv.org/abs/2604.02374) и [dataset card](https://huggingface.co/datasets/lab260/RuASD)
- [LRLspoof paper](https://arxiv.org/abs/2603.02364) и [dataset card](https://huggingface.co/datasets/lab260/LRLspoof)
- [KSC2](https://issai.nu.edu.kz/kz-speech-corpus/), [KSC](https://www.openslr.org/102/), [KSD](https://www.openslr.org/140/)
- [KazakhTTS2/ESPnet recipe](https://github.com/IS2AI/Kazakh_TTS) и [Piper voices](https://huggingface.co/rhasspy/piper-voices)
- [RawBoost](https://github.com/TakHemlata/RawBoost-antispoofing), [MUSAN](https://www.openslr.org/17/), [RIRS_NOISES](https://www.openslr.org/28/)
- [F5-TTS licensing notice](https://github.com/SWivid/F5-TTS) и [XTTS-v2 license](https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt)
- [Закон РК «О персональных данных и их защите»](https://adilet.zan.kz/eng/docs/Z1300000094)

---

### Первая практическая задача

Не скачивать сразу сотни гигабайт. Начать с маленького, но корректного data slice: 5 ч ru + 5 ч kk bona fide, 5 ч ru + 5 ч kk authorised/Piper/KazakhTTS2 fake, 4 фиксированных канала, групповой split и B0. Если этот срез не проходит проверку на утечки и API, большой датасет только масштабирует ошибку.
