# B0 source-mixed research v2 — controlled Kazakh test

**Дата:** 10 августа 2026
**Статус:** воспроизводимый personal-research experiment; не model release, не calibration и
не API score.

## Вопрос эксперимента

Проверить, переносится ли B0, обученный на одном русском source и выбранный на другом, на
изолированный контролируемый казахский binary test. Это не проверка «deepfake detection в
целом»: final test специально построен из KSC bona-fide речи и её синтеза двумя TTS family.

Contract: [`configs/research/source_mixed_v2_kk.json`](../configs/research/source_mixed_v2_kk.json).
Он прошёл `kds validate-source-matrix`: между ролями нет совпадающего `source_name`, а
обычные sample/SHA-256/group/text leakage checks и ledger policy применены до обучения.

| Роль | Источник | Строк | Назначение |
| --- | --- | ---: | --- |
| train | RuASD Russian ready | 1 417 | B0 fit |
| dev | PyAra Russian ready | 61 | выбор epoch только по dev loss |
| final test | KSC-derived Kazakh v1 | 1 842 | единственная final evaluation после выбора epoch |

Технический и лицензионный protocol теста описан в
[`data_sources_ksc_derived_kk_v1.md`](data_sources_ksc_derived_kk_v1.md). Вкратце: в нём
`921` KSC bona-fide clips и `921` paired synthetic clips с тем же `text_id`; synthetic часть
состоит из `475` MMS/VITS и `446` Piper clips. Пины моделей, локальные хэши и QA/rejection
reports зафиксированы. Исходные сведения: [KSC / OpenSLR 102](https://www.openslr.org/102),
[Piper Kazakh voice](https://huggingface.co/rhasspy/piper-voices/tree/main/kk/kk_KZ/issai/high)
и [MMS Kazakh TTS](https://huggingface.co/facebook/mms-tts-kaz).

## Запуск и результаты

Три запуска используют заранее закреплённые seeds `20260810`, `20260811`, `20260812`, пять
эпох каждый. Checkpoint выбирался по минимальному PyAra dev loss; final KSC test не участвовал
ни в выборе epoch, ни в подборе порога. Все числа ниже получены заново командой
`scripts/evaluate_b0.py` на checkpoint соответствующего запуска. Порог logit = 0, калибровки
нет.

| Seed | Best epoch | Dev loss | Correct / 1 842 | Accuracy = balanced accuracy (95% Wilson) | KSC bona-fide recall (95% Wilson) | Synthetic recall (95% Wilson) |
| --- | ---: | ---: | ---: | --- | --- | --- |
| 20260810 | 5 | 0.5461 | 1 512 | 82.08% (80.27–83.77%) | 64.17% (61.02–67.20%) | 100.00% (99.58–100.00%) |
| 20260811 | 5 | 0.5892 | 1 649 | 89.52% (88.04–90.84%) | 79.04% (76.30–81.55%) | 100.00% (99.58–100.00%) |
| 20260812 | 2 | 0.5409 | 1 681 | 91.26% (89.88–92.46%) | 82.52% (79.93–84.84%) | 100.00% (99.58–100.00%) |

Поскольку test уравновешен, accuracy и balanced accuracy здесь совпадают. Это не означает,
что они будут совпадать при иной prevalence spoof.

Стратифицированная повторная оценка каждого из трёх checkpoint дала одинаковый результат
для обеих synthetic family:

| Страта | Correct / N | Recall | 95% Wilson |
| --- | ---: | ---: | --- |
| MMS/VITS | 475 / 475 | 100.00% | 99.20–100.00% |
| Piper neural TTS | 446 / 446 | 100.00% | 99.15–100.00% |
| Piper individual voice | 70–80 / 70–80 | 100.00% | нижняя граница 94.80–95.42% |

## Корректная интерпретация

Получился полезный stress signal, но не высокий универсальный показатель качества. Модель
без ошибок распознаёт синтез именно двух закреплённых family на парных текстах, одновременно
ошибочно помечая `161`–`330` из `921` настоящих KSC clips как spoof. Разброс bona-fide recall
между seed (`64.17%`–`82.52%`) достаточно велик, чтобы прямо запретить любой risk score,
порог или product claim.

Ограничения эксперимента существенны:

- это только две generator family; шесть Piper voice ID не превращаются в шесть независимых
  генераторов;
- paired text контролирует содержание, но не проверяет генерализацию на новых текстах,
  каналах, устройствах или людях;
- KSC не даёт пригодных verified speaker groups, поэтому это не speaker-independent benchmark;
- MMS model имеет CC-BY-NC-4.0, а обе synthetic ветви и данный набор остаются только для
  personal research;
- доверительные интервалы характеризуют конечные `921`/`475`/`446` clips, но не неопределённость
  по будущим TTS, доменам или способам записи.

Следующий содержательный шаг — не добавлять ещё голосов Piper, а получить независимый
казахский spoof source или создать 3–5 новых generator family с отдельным OOD. Аудит
[LRLspoof](data_sources_lrlspoof_2026-08-10.md) показывает потенциальные 36 000 Kazakh spoof
paths, но текущая официальная публикация — единый примерно 452 GB gzip/tar archive. Его нельзя
надёжно скачать выборочно по папке Kazakh: gzip требует последовательного чтения. Нужны либо
пер-языковые random-access shards/manifest от авторов, либо локально предоставленный полный
archive с отдельным streaming extraction protocol. До этого LRLspoof не входит в ledger и не
смешивается с KSC.

## Достаточность размера

Для этого узко определённого controlled test `921` примера на класс — достаточный minimum:
при доле около 50% 95% Wilson interval имеет полу-ширину примерно 3.2 п.п. Это **не**
достаточный размер для вывода «модель работает на казахском» и тем более для эксплуатации.

Практическая следующая цель для исследовательской оценки: 3–5 независимых TTS/VC family,
по 300–500 accepted clips на family, то есть 1 500–2 500 spoof clips и столько же
сопоставимых bona-fide clips. Для будущего обучения более разумная цель — от 5 000 clips
каждого класса с разнообразием устройств и условий, а финальный test и unseen-generator OOD
должны быть заморожены до обучения. Это engineering target проекта, а не отраслевой норматив.

## Как создать собственный набор

Собственный dataset реален и предпочтительнее полного 452 GB чужого архива, если нужны права
и независимое покрытие. Процесс должен быть таким:

1. Получить документированное согласие каждого человека на запись, обучение, synthetic
   derivatives и предполагаемый режим использования. Сами согласия и PII хранятся вне Git;
   в проект попадает только pseudonymous registry с проверяемыми scope.
2. До записи зафиксировать квоты по языку/диалекту, устройству, помещению, шуму и текстам.
   Разнести людей, тексты и при возможности устройства между train/dev/test до генерации.
3. Собрать bona-fide записи и synthetic материалы минимум от 3–5 независимых TTS/VC family.
   На каждую строку записывать model revision, file SHA-256, voice ID, runtime, параметры и
   seed. Голоса одного checkpoint не считать независимыми family.
4. Применить один и тот же безопасный pipeline: decode, 16 kHz mono, VAD/QA, SHA-256 assets,
   manifests, rejection reports, checks лицензий и sample/text/group leakage. Не удалять
   неудачные записи молча.
5. Заморозить final test и отдельный unseen-generator OOD до обучения. Отчитываться отдельно
   по class, family, voice, условию и confidence interval; не подбирать threshold по test.

Пока этих гарантий нет, результат остаётся исследовательским сравнением, а не доказательством
мошенничества и не основой автоматического решения о человеке.
