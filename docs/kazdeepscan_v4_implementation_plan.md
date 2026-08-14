# KazDeepScan XLS-R+SLS v4 — план реализации расширенного train

**Статус:** design-only, реализация не начата.

**Дата локального аудита:** 14 августа 2026 года.

**Scope:** только personal research; не product model, не fraud detector и не калиброванная
вероятность.

Этот документ является единственным подробным планом v4. Он не разрешает запуск intake,
synthesis, training или final inference и не меняет v1/v2/v3, их manifests, checkpoints,
execution locks, reports или write-once результаты. Перед началом каждого этапа создаются новые
versioned contracts и новые output paths.

Название `model v4` не связано с уже существующим историческим contract
`source-mixed-research-v4-sparktts-final-test`. Чтобы не столкнуться с ним по `run_id` или путям,
новая ветка везде использует префикс `xlsr-sls-model-v4`.

## 1. Цель и критерий готовности

Цель — обучить отдельный XLS-R+SLS v4 checkpoint на `20 000–30 000` прошедших QA исходных
train-записях. Предпочтительный frozen target — `24 000` строк:

| Язык | bona-fide | spoof | Всего |
| --- | ---: | ---: | ---: |
| RU | 6 000 | 6 000 | 12 000 |
| KK | 6 000 | 6 000 | 12 000 |
| **Всего** | **12 000** | **12 000** | **24 000** |

В target считаются только уникальные base recordings после decode, QA и VAD. Augmentation,
повторные окна одного файла и regenerated variants не увеличивают размер набора. Допустимый
нижний предел — `20 000`, но только при сохранении равных четырёх `language × label` cells.
Если после исчерпания локальных eligible данных хотя бы одна cell не достигает `5 000`, этап
останавливается: размер не дополняется дубликатами, историческими final assets или ослаблением
leakage policy.

Готовый этап v4 должен иметь:

- новые и неизменяемые train/dev/calibration/final manifests и receipts;
- нулевой detectable overlap по audio, text, group/speaker и source lineage между ролями;
- несколько train-only TTS-family и минимум по одной полностью held-out RU и KK family для
  нового final;
- checkpoint, выбранный только по заранее объявленной dev-метрике;
- calibration, не видевшую dev или final;
- ровно один final inference после lock-before-logits;
- отдельный model card с честным описанием unknown speaker provenance и project history.

## 2. Что подтверждено текущим локальным аудитом

Ниже разделены три разных понятия: проверенная локальная ёмкость, потенциальная роль в v4 и
уже сертифицированное количество. Большой source archive не равен готовому train manifest.

| Источник | Подтверждённая локальная ёмкость | Решение для v4 |
| --- | --- | --- |
| RuASD full | `585 353` paired rows: raw `147 097` bona-fide + `228 266` spoof, augmented `104 998` + `104 992`; 250 TAR совпали с pinned catalog | Использовать только raw и только в train. Исключить augmented, всю `bonafide/CommonVoice` strata (`31 456` rows), все исторические v1/v2/v3 keys и любые неразрешимые collisions. До нового QA остаётся верхняя candidate-оценка `115 641` bona-fide + `228 266` spoof, а не готовый объём. |
| Common Voice RU v24 | exact archive `7 008 716 262` bytes / `201 326` MP3. Исторический full-test screen оставлял `6 211` rows / `1 443` client groups; literal V5.5 gate — `5 600` / `1 337` | Новый RU final-only source. Старый screen предшествует последующим 80-row selection и 42-pair evaluation, поэтому его числа нельзя считать текущим clean count. До selection нужен новый whole-client-group exposure screen по полному project history. |
| KSC2 | exact multipart archive `80 809 122 212` bytes / `645 860` paired FLAC; один лишний transcript исключён. В пяти новых `Test` components подтверждены `6 023` pairs. `crowdsourced` и `tts` запрещены | Основной KK bona-fide train reservoir только после нового component-level audit. Использовать nonlegacy components; legacy KSC/KazakhTTS2 lineage, historical reviewed/evaluated rows и все whole groups с overlap исключить. |
| KSC2 mixed layer | `2 632` extracted candidates; `91` semantic decisions, `88` QA-ready; `2 541` rows остаются unknown | Не считать pending rows ни `mixed`, ни чистыми `kk`. Historical 30-pair и Stage-C rows не включать в v4 final. Mixed не входит в обязательные 24 000 RU/KK train rows и может стать только отдельным будущим layer. |
| Explicit KK synthetic | `921` Piper/MMS + `359` KazEmoTTS + `381` Spark-TTS + `358` eSpeak NG accepted WAV, всего `2 019` по четырём family | Exact bytes уже были frozen final tests и не входят в v4 train/final по умолчанию. Проверенные model bundles можно применить к новым train-only KSC2 texts по отдельным v4 contracts. |
| Другие synthetic RU/KK | local ToneSpeak release содержит `6 998` embedded RU MP3; отдельно сохранены accepted Silero, Dialogs-RU, Qwen, KazakhTTS, eSpeak и VoxCPM2 assets | Каждый route проходит v4 eligibility gate. Historical final/failed routes, запрет training/calibration в receipts и family exposure сохраняются. VoxCPM2 Denis route остаётся `stop_below_minimum_60` и не открывается повторно. |
| Другие локальные bona-fide | PyAra, KSC SLR102, FLEURS RU/KK, VoxForge RU и Denis присутствуют локально и имеют versioned audits | Разрешены только как source-exclusive dev/calibration/final candidates после проверки current ledger и lineage. Новые datasets не ищутся, пока этот inventory не исчерпан. |

Итог аудита: целевые `24 000` train rows реалистичны по raw capacity. Однако точное число
**безопасно eligible** строк сейчас не сертифицировано, потому что ещё нет общего v4 exposure
graph, fresh KSC2 train-component inventory и QA результатов. До завершения Gate A нельзя писать,
что v4 располагает `24 000 ready` или что speaker-disjointness доказана.

Требование использовать локальные synthetic выполняется без подмены ролей: unexposed RuASD и
ToneSpeak assets могут стать train candidates, а hash-pinned KK TTS bundles — создать fresh
train-only outputs. Уже оценённые exact synthetic assets допускаются только через отдельное
all-or-none migration decision, принятое без использования их per-row logits/errors; после этого
они навсегда исключаются из v4 dev/calibration/final и не служат сравнением с v4. Если
historical receipt запрещает training для конкретного route, migration невозможна.

### Непреодолённое ограничение speaker provenance

RuASD не публикует verified speaker ID для bona-fide и надёжный voice group для большинства
spoof. KSC2 и FLEURS также не дают достаточных публичных per-row speaker identifiers. Поэтому
абсолютное отсутствие скрытого совпадения человека между независимыми corpora доказать нельзя.
Практический fail-closed contract должен:

1. никогда не делить строки одного unknown-speaker source lineage между ролями;
2. применять все доступные source group/client/program/session IDs транзитивно;
3. выполнять cross-role speaker-embedding near-match screen и ручной review пограничных
   совпадений как дополнительную защиту, не выдавая его за identity proof;
4. маркировать итог как `not_verified_speaker_independent`.

Если требование трактуется как математически доказанная speaker independence, текущих metadata
недостаточно и обучение должно остаться blocked. Нельзя заменить неизвестный speaker выдуманным
ID или объявить разные datasets разными людьми без evidence.

## 3. Предварительное распределение источников по ролям

Распределение замораживается до synthesis. Один upstream source lineage не может переходить в
другую роль даже при разных filenames или source-provided split.

### 3.1. Train: preferred target 24 000

- `RU bona-fide = 6 000`: raw RuASD, но без внутренней Common Voice strata, augmented rows и
  project-history closure.
- `RU spoof = 6 000`: raw RuASD с заранее ограниченной квотой на subset и, только если новый
  ledger snapshot разрешит, fresh unevaluated ToneSpeak rows. Нельзя считать 37 RuASD subset
  names 37 доказанными architecture family; family claims делаются только после provenance map.
- `KK bona-fide = 6 000`: KSC2 nonlegacy train components после исключения `crowdsourced`,
  `tts`, legacy KSC/KazakhTTS2 и всех historical groups/texts.
- `KK spoof = 6 000`: по одному fresh output на frozen KK train text. Квота распределяется
  примерно поровну между train-only Piper/MMS, KazEmoTTS, Spark-TTS и eSpeak NG; пятая family
  допускается только если она не зарезервирована для final и проходит rights/route gate.

Для каждой cell сначала создаётся oversubscribed pre-QA pool с заранее объявленным reserve
order. QA может убрать плохие assets; оно не может выбирать строки по detector score. После
достижения target публикуются полный rejection accounting и frozen train manifest. Backfill после
просмотра dev/final запрещён.

### 3.2. Dev и calibration

Preferred dev — отдельный bilingual binary layer из локальных sources, отсутствующих в train и
final. Возможный RU source — PyAra; возможный KK source — KSC SLR102 только если lineage audit
докажет, что KSC2 train полностью исключил включённые в KSC/KazakhTTS2 компоненты и exact/near
overlap. Synthetic dev family не должна присутствовать в train или final.

Calibration получает третий source set и не участвует в выборе epoch, architecture или
augmentation. Локальный кандидат для RU — новые VoxForge rows и отдельный calibration-only
synthetic source. Для KK сейчас нет автоматически одобренного четвёртого независимого
bona-fide source. Поэтому Gate A должен выбрать один из двух честных вариантов:

- найти достаточный local-only KK source lineage и сделать bilingual calibration; или
- оставить calibration RU-only, не калибровать KK score и не называть общий v4 output
  probability.

Нельзя использовать одну FLEURS/KSC/Common Voice lineage сразу в dev, calibration и final ради
симметричной таблицы. Если после полного локального аудита строгий bilingual four-role contract
невозможен, это versioned blocker; только после него можно отдельно решить вопрос о поиске нового
KK source.

### 3.3. Новый независимый final test

Предпочтительный local-only final содержит равные RU/KK и bona-fide/spoof cells. Базовый target —
`2 000` assets: по `500` exact bona-fide/spoof pairs для RU и KK.

- RU bona-fide: fresh whole-client-group-disjoint Common Voice RU rows.
- KK bona-fide: fresh Google FLEURS KK rows из source split, полностью зарезервированного для
  final и не использованного в train/dev/calibration.
- RU spoof: новая exact-text synthesis family, полностью отсутствующая в v4 train/dev/calibration.
- KK spoof: другая held-out family, также отсутствующая во всех v4 tuning roles.

FLEURS уже находится локально и прошёл source audit, поэтому это не поиск нового dataset.
Использовать KSC2 одновременно в train и final нельзя: это нарушит требование source-disjointness.
Использовать Common Voice внутри RuASD train также нельзя, поэтому соответствующая RuASD strata
исключается целиком.

Final должен быть независим от **v4** по exact assets, texts, source lineages и generator family.
Некоторые локальные TTS-family уже встречались в старых экспериментах; такой history раскрывается
в contract и не позволяет называть final project-level unseen. Если ни одна local family не
проходит held-out-family audit, final остаётся blocked, а не понижается до same-family test.

## 4. Gate A — обязательный аудит до любых новых assets

Gate A ничего не обучает и не синтезирует. Его outputs — только versioned inventories и решения.

1. Зафиксировать новый `v4_project_history_inventory` со всеми configs, manifests, rejection
   reports, pairing receipts, execution locks и final selections v1/v2/v3.
2. Повторно сверить exact archive size/SHA-256 и license status для RuASD, Common Voice, KSC2 и
   выбранных TTS bundles. Старые locks не перезаписывать.
3. Построить canonical `source_lineage_id`: dataset, upstream subset/component, collection,
   program/session/client group и synthetic text source. Простого `source_name` недостаточно.
4. Для всех candidates посчитать raw/decoded SHA-256, canonical audio fingerprint, exact и
   normalized text hashes, sample ID, parent group, speaker/group ID и generator route.
5. Построить транзитивные connected components leakage graph. Если любой node касается двух
   ролей, весь component исключается из менее приоритетной роли.
6. Проверить exact audio duplicate, near-audio duplicate, exact/canonical/near-text overlap,
   group/speaker overlap, source-lineage overlap, TTS-family/voice overlap и collision с каждым
   historical final.
7. Опубликовать capacity table `candidate → excluded by reason → pre-QA eligible` для каждой
   `role × language × label × source × family` cell.
8. Принять fail-closed решение: `proceed_24k`, `proceed_20k_to_29999` или
   `stop_local_capacity_exhausted`.

Gate A запрещено упрощать из-за стоимости полного streaming audit. Нельзя считать старый
metadata screen актуальным после появления новых manifests.

## 5. Подготовка v4 data

### 5.1. Selection до synthesis

- seed, target, source/family quotas, reserve order и stop rules фиксируются в JSON contract;
- bona-fide groups и texts распределяются по ролям до генерации spoof;
- один text ID принадлежит только одной роли, включая canonical/normalized variants;
- final selection не читает checkpoint, logits, errors или prior final per-row outcomes;
- historical final exact assets не используются как v4 train shortcuts.

### 5.2. Synthesis contract

Каждая family имеет отдельные model lock, runtime lock, source revision, checksum, voice/control
ID, seed и output namespace. Разрешён только local text-to-speech без reference audio, voice
cloning, prompt audio, external normalizer, retry по detector outcome или сети во время run.
Неудачный output учитывается как rejection; повторная генерация разрешена только если её правило
зафиксировано до первой попытки и не зависит от detector/model score.

### 5.3. Audio QA

Все bona-fide и spoof проходят один и тот же decode/16 kHz mono PCM/VAD/quality pipeline.
Публикуются raw manifest, ready manifest и полный rejection report. Автоматические gates как
минимум проверяют decode completeness, duration, speech duration, clipping, silence/quietness,
NaN/Inf, channel/rate contract и asset SHA-256.

Для train выполняется стратифицированный listening audit по language/source/family/voice и всем
редким rejection modes. Для final каждый exact asset проходит две независимые blinded review
формы до detector inference. Predictions и logits никогда не попадают в review packet.

### 5.4. Augmentation

Augmentation применяется только после split и только к train, симметрично для обоих labels и
языков. Policy label-agnostic, deterministic по sample seed и не создаёт новые manifest rows,
участвующие в заявленном размере. Dev, calibration и final остаются без augmentation.

## 6. Model и training protocol

Первый v4 experiment сохраняет XLS-R+SLS architecture, чтобы измерять эффект расширенных данных,
а не смешивать его с новой моделью. Любое изменение backbone/head требует отдельного pre-final
ablation contract и не может быть принято по final result.

Предлагаемый порядок:

1. новый head warm-up на frozen XLS-R;
2. staged unfreeze последних XLS-R blocks с BF16, gradient accumulation и deterministic seed;
3. class/language-balanced sampler без duplicate padding;
4. primary checkpoint criterion — заранее зафиксированный macro dev loss по доступным RU/KK
   cells; balanced accuracy/EER остаются diagnostics;
5. seed, epoch, learning rate, unfreeze depth и augmentation выбираются только по dev;
6. после выбора checkpoint training code hash, environment lock, manifest hashes и state-dict
   hash замораживаются;
7. temperature/threshold допускаются только по calibration contract. При отсутствии clean KK
   calibration KK остаётся uncalibrated с заранее фиксированной boundary;
8. после final lock запрещены retraining, backfill, threshold change и повтор final inference.

Checkpoint сохраняется отдельно, например
`checkpoints/xlsr-sls-model-v4/<run_id>/model.pt`; этот каталог остаётся Git-ignored. В Git входят
только state-dict/file SHA-256, config, environment receipt, summary и model card.

## 7. Final evaluation и отчётность

До final создаётся no-logit preflight, который проверяет:

- exact hashes всех manifests, assets, reviews, config и checkpoint;
- нулевые pairwise leakage counts по всем ключам;
- запрет final paths в training/dev/calibration loaders;
- CUDA/BF16/runtime readiness без forward pass по final;
- отсутствие старого execution lock/report по новому `run_id`.

После lock выполняется один final run. Отчёт содержит раздельно RU/KK, bona-fide/spoof,
source, TTS-family и voice/control strata, pair accuracy, balanced accuracy, EER и доверительные
интервалы там, где они статистически определены. Общая pooled RU+KK accuracy не используется как
headline. Ошибки final сохраняются только в immutable receipt и не становятся входом следующего
v4 решения.

## 8. Новая структура артефактов

Новые v4 files не смешиваются с историческими flat receipts:

```text
configs/research/v4/          новые contracts и training configs
data/manifests/v4/            versioned raw/ready/frozen manifests и machine receipts
data/licenses/frozen/         отдельные immutable v4 ledger snapshots
docs/v4/                      human-readable audit, QA, training и final receipts
data/raw/v4/                  локальные raw assets, Git-ignored
data/processed/v4/            локальные normalized assets, Git-ignored
checkpoints/xlsr-sls-model-v4/ checkpoints, Git-ignored
artifacts/v4/                 local execution outputs, Git-ignored
```

Существующие historical receipts не перемещаются, не переименовываются и не форматируются:
пути и bytes могут быть частью hash-pinned contracts. Новые scripts обязаны писать через новые
paths и отказываться от overwrite.

## 9. Правило документации

Документация имеет одного владельца факта:

- `README.md` — назначение, safety boundary, короткий quick start и ссылки;
- `PROJECT_STATUS.md` — только текущий завершённый этап, действующие ограничения и следующий gate;
- `План реализации.md` — короткая текущая roadmap и ссылка на этот v4 plan;
- `KazDeepScan_implementation_blueprint.md` — стабильная архитектура;
- `docs/v4/` — подробные versioned audits, QA, training/final receipts;
- `data/manifests/v4/` и `data/licenses/frozen/` — machine-readable truth и exact hashes.

Исторические числа не копируются в README/status/roadmap. В них остаётся одна ссылка на
canonical receipt. Существующие immutable historical receipts не удаляются. Документационный
cleanup выполняется отдельным docs-only этапом до v4 data implementation, со сверкой ссылок и
фактов; он не меняет ни один historical byte-locked CSV/JSON receipt.

## 10. Этапы и stop conditions

| Этап | Versioned результат | Условие перехода |
| --- | --- | --- |
| A. Local capacity/exposure audit | inventories, lineage map, leakage graph, capacity receipt | Все target cells имеют подтверждённый pre-QA reserve; права разрешают роль |
| B. Frozen selection | role contracts и candidate manifests | Нулевой cross-role overlap до synthesis |
| C. Materialization/synthesis/QA | raw/ready/rejection receipts | `20k–30k` balanced unique train; no outcome-driven backfill |
| D. Documentation cleanup | короткие main docs, ссылки на canonical receipts | Нет broken links/duplicated current facts |
| E. Train/dev selection | new checkpoint, report, hashes | Checkpoint выбран только по dev |
| F. Calibration | calibration receipt | Final не читался; claims соответствуют RU/KK coverage |
| G. Final preflight/reviews | immutable plan и no-logit lock | Все exact assets/reviews/leakage gates прошли |
| H. One-shot final | execution lock, report, model card | Один run; никаких post-final изменений |

Немедленный stop обязателен при license conflict, недостаточном local capacity, невозможности
разделить source lineage, неполном rejection accounting, неизвестном overwrite, попадании final
data в tuning, несоответствии hash или попытке повторить write-once run.

## 11. Что не сделано этим документом

- не созданы v4 manifests/configs/checkpoint/directories;
- не извлечён и не изменён ни один audio asset;
- не запускались synthesis, QA, training, calibration или inference;
- не изменены v1/v2/v3 и существующие immutable receipts;
- не искались и не скачивались новые datasets/models;
- `24 000 ready` не заявлены: подтверждена только достаточная raw candidate capacity.

Первый безопасный практический шаг после отдельного разрешения на реализацию — только Gate A:
новый read-only local capacity/exposure audit с versioned receipt. До его завершения нельзя
создавать v4 split или synthetic WAV.
