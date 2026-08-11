# XLS-R + SLS Stage B v1

## Назначение и границы

Stage B частично дообучает проверенный XLS-R+SLS Stage A checkpoint для personal research.
Разморожены только последние восемь transformer blocks XLS-R; SLS-head продолжает обучение.
Runner не принимает final manifest и ни разу не открывал frozen KazEmoTTS, Spark-TTS или
eSpeak NG final tests. Threshold, record-level calibration и API score не подбирались.

Собственные голоса не записывались. Данные взяты из готовых локальных RuASD и PyAra; полный
поиск дополнительных источников и решение по XMAD/Kazakh candidates записаны в
[data_source_review_2026-08-10_stage_b.md](data_source_review_2026-08-10_stage_b.md).

## Новый независимый PyAra dev

Полный локальный PyAra v7 archive был повторно проверен по закреплённым размеру, SHA-256,
ZIP layout и точному TSV/WAV membership. Детерминированно выбраны `500` bona-fide и по `100`
spoof для каждого из пяти algorithms. При выборе исключены все `500` record IDs и `499`
уникальных text hashes исходного PyAra research slice.

- raw manifest: `data/manifests/pyara_ru_v7_fresh_dev_1000.csv`, `1 000` rows;
- после decode/QA/VAD: `973` rows и `27` явных rejections;
- строгий Stage B filter обнаружил ещё три text hashes, встречавшиеся в RuASD train, и удалил
  всю соответствующую dev-группу;
- итог: `data/manifests/pyara_ru_v7_fresh_dev_1000_ready_v2.csv`, `970` rows — `474`
  bona-fide и `496` spoof;
- manifest SHA-256:
  `7ae57e0714b5a245079902e06adaca7a02c423418e2e37bfded6d55c6db0d2f0`;
- exclusion report SHA-256:
  `446cd616f77ec66b7d9a2c7bcc72705cb5a4b66cb5c31817bf943fca6ca82154`.

У нового dev нулевое пересечение с RuASD train и старым Stage A dev по доступным sample ID,
asset SHA-256, text hash и parent group. Все `2 387/2 387` train/dev assets повторно
проверены. Ограничение остаётся существенным: PyAra не раскрывает speaker IDs, поэтому этот
dev text/source-disjoint относительно выбранных строк, но не доказан как speaker-disjoint.

## Зафиксированный run plan

- plan: `configs/research/xlsr_sls_stage_b_v1.json`;
- plan SHA-256: `264c10cfdee74e0ee2e75990bbbeeb5725f0aef0e254b8cdf90029015c40e731`;
- base Stage A plan SHA-256:
  `4cdb63f6490238f7c453afeab59f3c32bf0bdf1bbc5598e6d95e7847832f6d7e`;
- initial head checkpoint SHA-256:
  `46e8187551bb0d9defcb45eaae78f3e084fbff57667f593c3e696eea75a88a12`;
- initial canonical head-state SHA-256:
  `1370b9b81e0c61f0ced94d29fdcc15e28ba28f5240ef724683b8bfd0cdb490e6`;
- train: RuASD `train`, `1 417` rows (`637` bona-fide, `780` spoof);
- dev: fresh PyAra `dev`, `970` rows (`474` bona-fide, `496` spoof);
- seed `20260820`, 15 эпох, physical batch `4`, accumulation `8`, effective batch `32`;
- BF16, AdamW, encoder LR `1e-5`, head LR `1e-4`, weight decay `1e-4`, gradient clip `1.0`;
- последние blocks `16`–`23` из `24`, gradient checkpointing включён;
- checkpoint заранее выбирается только по минимальному `dev_loss`.

Gradient accumulation нормализует градиент по фактическому числу examples, в том числе для
последней неполной accumulation group. Это проверено тестом эквивалентности накопленного и
полного batch update.

## CUDA preflight и profile

Запуск выполнен на NVIDIA GeForce RTX 5060 Ti (`16 616 521 728` bytes VRAM), PyTorch
`2.11.0+cu128`, CUDA runtime `12.8`, Transformers `5.14.1`; BF16 поддерживается. Preflight
проверил plan receipts, initial head state и все assets до создания модели.

Одно train- и одно dev-окно подтвердили конфигурацию: `100 769 792` trainable encoder
parameters, `660 251` head parameters, blocks `16`–`23`, gradient checkpointing. Profile
занял `0.527` с; peak VRAM — `2 902 534 656` allocated и `3 288 334 336` reserved bytes.
Profile артефактов не публиковал.

## Результат

Заранее закреплённый `dev_loss` выбрал epoch 7:

| Metric | Train | Fresh PyAra dev |
|---|---:|---:|
| Loss | `0.02551` | **`0.15236`** |
| Accuracy | `0.9915` | `0.9381` (`910/970`) |
| Balanced accuracy | `0.9917` | `0.9393` |
| Bona-fide recall | `0.9937` | `0.9895` (`469/474`) |
| Spoof recall | `0.9897` | `0.8891` (`441/496`) |

Epoch 14 имел немного более высокую dev accuracy (`0.9423`) и balanced accuracy (`0.9434`),
но худший loss (`0.20345`). Критерий после просмотра результатов не менялся: epoch 7 является
реальным argmin dev loss среди всех 15 эпох. Поздние колебания dev при почти идеальном train
подтверждают переобучение/высокую дисперсию и не позволяют считать результат устойчивой
оценкой новых людей, corpus или каналов.

Полное обучение заняло `532.32` с. Пиковая память — `2 928 268 800` allocated и
`3 326 083 072` reserved bytes. Опубликованы:

- `models/xlsr-sls-stage-b-v1.pt`: `405 780 969` bytes, SHA-256
  `18c967a8881404140ccda04fc6234079ac4b2802425e4111f3fef59bef505c32`;
- `models/xlsr-sls-stage-b-v1-report.json`: `35 514` bytes, SHA-256
  `281ca58256db5d229ce3e86c6cf8a9ed60735907dbb18d52f3346806b39579b7`;
- canonical hash 139 tensors выбранного trainable state:
  `59ad0812e14d33abec00ba5225876de4c208efa9c8f8f9061e253e60df9d1089`.

Read-only audit повторно загрузил checkpoint с `weights_only=True` и безопасным allowlist
для `TorchVersion`, пересчитал state hash, подтвердил epoch 7 как argmin и проверил, что state
содержит только SLS-head и blocks `16`–`23`. Receipt содержит
`frozen_final_evaluation_performed=false` и `calibrated=false`.

## Что разрешено дальше

Stage B v1 нельзя повторно запускать или перезаписывать. Это выбранный train/dev checkpoint,
но ещё не model release. Следующий шаг должен сначала зарегистрировать отдельный final-run
contract с ранее не использованными данными; только после его заморозки допустим один final
inference. Нельзя использовать уже раскрытые B0 final результаты для настройки Stage B,
постфактум менять selection metric, подбирать threshold или включать checkpoint в API.

