# KazDeepScan XLS-R+SLS v4 — план реализации расширенного train

**Статус:** capacity/integrity, frozen metadata selection v2 и source decode/QA/audio-leakage
gate завершены. Source train заморожен на `15 000` строк (`3 × 5 000`); принято
`proceed_20k_balanced`. Все `7 200` KK spoof texts exact-проверены, синтезированы и прошли
общий QA/VAD/audio-leakage screen: `6 200` eligible и ровно `5 000` frozen (`4 × 1 250`).
Combined balanced `20 000` train manifest уже write-once заморожен (`4 × 5 000`). Full training
contract, его no-training preflight и one-batch tail-unfreeze capacity profile завершены; actual
training завершён с hash-bound checkpoint selection; RU calibration и один final inference
после two-review pair lock завершены.

Four-route hash-pinned offline synthesis contract и resumable runner завершили все four routes:
`7 200/7 200` raw WAV (`4 × 1 800`) без runtime reject. Общий hash-pinned audio QA/leakage gate
frozen before processing.

**Дата локального аудита:** 15 августа 2026 года.

**Scope:** только personal research; не product model, не fraud detector и не калиброванная
вероятность.

Этот документ является единственным подробным планом v4. Завершённые capacity audit,
metadata-only selection, source decode/QA и KK spoof audio gate привели к combined balanced
`20 000` train manifest, isolated bilingual dev и отдельному full training contract. Его
no-training preflight прошёл до actual training и final inference. V1/v2/v3, их
manifests, checkpoints, execution locks, reports и write-once результаты не меняются. Перед
началом каждого следующего этапа создаются новые versioned contracts и новые output paths.

Название `model v4` не связано с уже существующим историческим contract
`source-mixed-research-v4-sparktts-final-test`. Чтобы не столкнуться с ним по `run_id` или путям,
новая ветка везде использует префикс `xlsr-sls-model-v4`.

## 1. Цель и критерий готовности

Цель — обучить отдельный XLS-R+SLS v4 checkpoint на `20 000–30 000` прошедших QA исходных
train-записях. Первоначальный preferred target `24 000` не прошёл source QA: RU bona-fide
оставил `5 706` eligible. Без outcome-driven backfill текущий frozen target — `20 000` строк:

| Язык | bona-fide | spoof | Всего |
| --- | ---: | ---: | ---: |
| RU | 5 000 | 5 000 | 10 000 |
| KK | 5 000 | 5 000 | 10 000 |
| **Всего** | **10 000** | **10 000** | **20 000** |

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
| RuASD full | Fresh Gate A подтвердил `585 353` paired rows и SHA-256 всех 250 TAR: raw `147 097` bona-fide + `228 266` spoof, augmented `104 998` + `104 992` | Только raw/train. После исключения Common Voice strata и консервативного historical accounting pre-QA верхние оценки равны `113 649` bona-fide и `226 166` spoof; это не ready rows. |
| Common Voice RU v24 | Fresh whole-history screen exact архива оставил `5 882` rows / `1 363` client groups | RU final-only reservoir; selection, extraction и QA ещё не выполнялись. |
| KSC2 | Fresh Gate A подтвердил 10 exact parts, `645 860` paired FLAC и один лишний transcript. Пять разрешённых Train components содержат `202 961` пар до консервативного historical accounting | Основной KK bona-fide train reservoir. `crowdsourced`, `tts`, legacy lineage и historical collisions запрещены; текущая pre-QA верхняя оценка `202 870`, не ready rows. |
| KSC2 mixed layer | `2 632` extracted candidates; `91` semantic decisions, `88` QA-ready; `2 541` rows остаются unknown | Не считать pending rows ни `mixed`, ни чистыми `kk`. Historical 30-pair и Stage-C rows не включать в v4 final. Mixed не входит в обязательные 24 000 RU/KK train rows и может стать только отдельным будущим layer. |
| Explicit KK synthetic | `921` Piper/MMS + `359` KazEmoTTS + `381` Spark-TTS + `358` eSpeak NG accepted WAV, всего `2 019` по четырём family | Exact bytes уже были frozen final tests и не входят в v4 train/final по умолчанию. Проверенные model bundles можно применить к новым train-only KSC2 texts по отдельным v4 contracts. |
| Другие synthetic RU/KK | local ToneSpeak release содержит `6 998` embedded RU MP3; отдельно сохранены accepted Silero, Dialogs-RU, Qwen, KazakhTTS, eSpeak и VoxCPM2 assets | Каждый route проходит v4 eligibility gate. Historical final/failed routes, запрет training/calibration в receipts и family exposure сохраняются. VoxCPM2 Denis route остаётся `stop_below_minimum_60` и не открывается повторно. |
| Другие локальные bona-fide | PyAra, KSC SLR102, FLEURS RU/KK, VoxForge RU и Denis присутствуют локально и имеют versioned audits | Разрешены только как source-exclusive dev/calibration/final candidates после проверки current ledger и lineage. Новые datasets не ищутся, пока этот inventory не исчерпан. |

Итог capacity/integrity части Gate A: решение `proceed_24k` подтверждает достаточную локальную
pre-QA candidate capacity. Проверены 429 versioned project-history files, 99 manifest files /
40 942 rows с versioned дубликатами и пять exact локальных KK TTS-family. Точное число
**безопасно eligible** ready-строк ещё не сертифицировано: frozen v2 selection доказал
sample/text exclusion и role-root isolation, а raw materialization — provenance exact bytes и
отсутствие raw SHA-256 collisions у `21 598` допущенных source rows. Decoded/near-audio leakage
и QA/VAD ещё не выполнены. Нельзя писать, что v4 уже
располагает `24 000 ready` или что speaker-disjointness доказана. Канонический подробный
результат: [v4 Gate A capacity](artifacts/v4/gate_a_2026-08-14.md) и
[source raw materialization](artifacts/v4/source_raw_materialization_2026-08-14.md).

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

- `RU bona-fide = 5 000`: raw RuASD, но без внутренней Common Voice strata, augmented rows и
  project-history closure.
- `RU spoof = 5 000`: raw RuASD с заранее ограниченной квотой на subset. ToneSpeak не входит в
  canonical v2 train role. Нельзя считать RuASD subset names доказанными architecture family;
  family claims делаются только после provenance map.
- `KK bona-fide = 5 000`: KSC2 nonlegacy train components после исключения `crowdsourced`,
  `tts`, legacy KSC/KazakhTTS2 и всех historical groups/texts.
- `KK spoof = 5 000`: по одному fresh output на frozen KK train text. V2 pre-QA pool распределён
  по `1 800` строк между четырьмя train-only family: Piper, MMS, KazEmoTTS и Spark-TTS; frozen
  target — `1 250` ready на family. eSpeak
  исключён из train и зарезервирован только для RU calibration, чтобы family roots не пересекали
  роли.

Для каждой cell сначала создаётся oversubscribed pre-QA pool с заранее объявленным reserve
order. QA может убрать плохие assets; оно не может выбирать строки по detector score. После
достижения target публикуются полный rejection accounting и frozen train manifest. Backfill после
просмотра dev/final запрещён.

### 3.2. Dev и calibration

V2 резервирует dev как отдельный bilingual binary layer: RU PyAra и KK KSC SLR102/Silero V4.
KSC2 train использует только пять явно разрешённых nonlegacy Train components; materialization
gate всё равно обязан доказать отсутствие exact/near overlap с KSC SLR102. Synthetic dev family
не присутствует в train, calibration или final.

Calibration получает третий source set и не участвует в выборе epoch, architecture или
augmentation. V2 резервирует fresh exact VoxForge RU rows и eSpeak RU как calibration candidates;
их права, unused text/groups и audio leakage должны пройти отдельный gate до создания calibration
manifest. Для KK нет автоматически одобренного четвёртого независимого bona-fide source, поэтому
зафиксирован RU-only вариант: KK score не калибруется и общий v4 output не называется
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

Metadata portion завершена каноническим v2 packet: `28 800` rows, `4 × 7 200`, нулевые
historical sample/text intersections и disjoint role roots. V1 отклонён до materialization из-за
FLEURS corpus-family и eSpeak family crossing. Точные hashes и reconciliation:
[frozen train candidates](artifacts/v4/train_candidate_selection_2026-08-14.md).

Source materialization завершила exact raw-audio часть для `21 600` RuASD/KSC2 candidates.
`21 598` unique raw assets допущены к decode/QA; две TeraTTS записи отклонены из-за exact
collision со старым RuASD OOD-100 manifest, без replacement/backfill. Decoded hashes,
near-audio fingerprints и QA/VAD остаются обязательным следующим gate:
[source raw materialization](artifacts/v4/source_raw_materialization_2026-08-14.md).

Source decode/QA gate обработал все `21 598` rows, оставил `18 930` eligible, исключил один
historical near-hit и заморозил `15 000` source train rows (`3 × 5 000`). Exact decoded history
и within-pool collisions равны нулю. Решение `proceed_20k_balanced`, полный accounting и hashes:
[source decode/QA](artifacts/v4/source_decode_qa_2026-08-14.md).

KK spoof text materialization повторно проверила полный KSC2 multipart archive и exact извлекла
`7 200` unique transcripts (`4 × 1 800`, в каждой route `1 500` target + `300` reserve).
Все `7 200` synthetic raw WAV теперь созданы без runtime reject:
[KK spoof texts](artifacts/v4/kk_spoof_text_materialization_2026-08-14.md).
Отдельный synthesis plan hash-bind'ит inputs, runner и четыре model/adapter route, запрещает
network/reference audio/cloning/detector feedback и публикует только полный route accounting:
[KK spoof synthesis](artifacts/v4/kk_spoof_synthesis_plan_2026-08-14.md).
Все four routes опубликовали по `1 800` raw rows (`1 500` target + `300` reserve) без runtime
rejection; это ещё не готовые train rows до common gate:
[MMS synthesis](artifacts/v4/xlsr_sls_model_v4_kk_spoof_kk_mms_kaz_v1_synthesis_v1.json) и
[KazEmoTTS synthesis](artifacts/v4/xlsr_sls_model_v4_kk_spoof_kk_kazemotts_v1_synthesis_v1.json),
[Piper synthesis](artifacts/v4/xlsr_sls_model_v4_kk_spoof_kk_piper_issai_high_v1_synthesis_v1.json) и
[SparkTTS synthesis](artifacts/v4/xlsr_sls_model_v4_kk_spoof_kk_sparktts_v1_synthesis_v1.json).
Общий gate reuse'ит canonical v4 decode/QA/VAD и screens all current/historical audio before
fixed `4 × 1,250` freeze. Он завершён: `6 200/7 200` rows eligible, `1 000` rejected only for
`insufficient_speech`, frozen rows exactly `5 000`, and historical/within-pool exact and near
audio intersections are zero: [KK spoof audio gate](artifacts/v4/kk_spoof_audio_gate_2026-08-15.md).
The output receipt's overly broad training claim is corrected without changing its bytes by the
[governance reconciliation](artifacts/v4/xlsr_sls_model_v4_kk_spoof_audio_gate_governance_v1.json):
only the combined-manifest and separate training-contract preflight are now authorized.
The combined manifest is now frozen with exactly four `5,000`-row cells and `4,604` allowed
within-train KK text overlaps: [combined train manifest](artifacts/v4/combined_train_manifest_2026-08-15.md).

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
docs/artifacts/v4/            human-readable и machine audit/QA/training/final receipts
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
- `docs/artifacts/v4/` — подробные versioned audits, QA, training/final receipts;
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
| F0. Calibration metadata inputs | fresh source identities, isolation receipt | 81 fresh VoxForge identities/groups frozen; prior historical text overlap disclosed |
| F0.5. Calibration materialization/isolation | raw/ready manifests, audio gate, 73-pair lock | Exact source/eSpeak bindings, QA/VAD and full-history exact/near gate passed; separate RU calibration contract now permits only one write-once scoring/temperature run |
| F. Calibration | RU temperature receipt | One write-once 73-pair RU-only fit completed; no final read, no KK probability claim |
| G. Final preflight/reviews | immutable plan и no-logit lock | Все exact assets/reviews/leakage gates прошли |
| H. One-shot final | execution lock, report, model card | Один run; никаких post-final изменений |

Немедленный stop обязателен при license conflict, недостаточном local capacity, невозможности
разделить source lineage, неполном rejection accounting, неизвестном overwrite, попадании final
data в tuning, несоответствии hash или попытке повторить write-once run.

## 11. Итоговый scope v4

- capacity gate, canonical metadata selection v2 и source raw/decode/QA завершены: `18 930`
  source rows eligible, `15 000` frozen в train source manifest; combined balanced train и
  isolated dev созданы, selected ignored checkpoint создан, а RU calibration pair manifest
  создан отдельно; final audio manifest, two-review pair lock и final report завершены;
- materialized raw audio находится только в новом Git-ignored v4 namespace; historical audio
  assets не изменялись;
- raw synthesis и common QA/VAD/audio-leakage завершены: `6 200` KK spoof rows eligible and
  `5 000` (`4 × 1 250`) frozen; combined train manifest содержит `20 000` rows (`4 × 5 000`);
  full training contract, no-training preflight и write-once training завершены; checkpoint
  выбран только по macro RU/KK dev loss, RU-only calibration и one-time final inference также
  завершены;
- isolated dev-input contract выполнен на CUDA: он reuse'ит `969` PyAra dev rows, source QA
  оставил `571/600` KSC rows, Silero QA — `535/571`, и `474` KSC SLR102/Silero V4 KK pairs
  заморожены без detector feedback или backfill; combined dev содержит `1 917` rows;
- calibration-input gate завершён metadata-only: `81` fresh VoxForge exact source identities и
  `81` distinct new contributor groups frozen after current-history isolation audit; historical
  VoxForge texts disclosed, while v4 train/dev sample/text/group intersections are zero.
  Follow-on materialization/audio-isolation gate повторно bound archive, materialized `81` WAV,
  retained `79` source-ready, generated exactly `79` one-shot eSpeak WAV and froze `73` exact RU
  pairs after QA/VAD/full-current-history exact/near screen. A separate RU calibration contract
  now binds this pair lock, a narrow research-only fitting ledger and the selected checkpoint;
  its write-once no-logit preflight and one calibration run completed: temperature is
  `0.72535688`, NLL/ECE decreased and Brier increased; final inference did not occur;
- не изменены v1/v2/v3 и существующие immutable receipts;
- не искались и не скачивались новые datasets/models;
- `24 000 ready` не заявлены: подтверждена только достаточная pre-QA candidate capacity.

Explicit rights/ledger decision теперь versioned отдельно от materialization ledger и разрешает
только research-only RU temperature fitting. Required no-logit preflight and exactly one execution
нового checkpoint-scoring-and-calibration contract завершены на `73` frozen pairs. Read-only
[final-readiness audit](artifacts/v4/v4_final_readiness_2026-08-15.md) уже исключил прежние
inferred Qwen/VoxForge и FLEURS/KazakhTTS assets. Первый
[metadata-only final-input contract](artifacts/v4/v4_final_inputs_contract_2026-08-15.md)
завершил selection `500` RU + `500` KK source text groups без audio/model operations. Следующий
[final materialization/review contract](artifacts/v4/final_materialization_contract_2026-08-15.md)
подготовлен, а его read-only
[preflight](artifacts/v4/final_materialization_preflight_2026-08-15.md) прошёл без outputs. Он
разрешал только one-shot extraction/synthesis этих exact rows, QA/VAD, current-history
isolation, две независимые review forms и pair lock. Первая Qwen output write failure остановила
attempt после source extraction; [failure receipt](artifacts/v4/final_materialization_attempt_failure_2026-08-15.md)
запрещает retry/resynthesis и требует нового recovery contract. Reconciliation pair lock затем
прошёл отдельный [one-time final evaluation](artifacts/v4/final_reconciliation_evaluation_2026-08-16.md);
repeat inference, threshold selection и any backfill запрещены.
