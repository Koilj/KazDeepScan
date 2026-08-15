# XLS-R+SLS model v4 — final materialization attempt failure v1

**Дата:** 15 августа 2026, `2026-08-15T22:30:00+06:00`

## Статус

One-shot materialization attempt **не завершён** и не может быть повторён по этому contract.
Перед началом exact read-only preflight успешно прошёл; затем были atomically extracted source
slices, но первый RU Qwen route остановился fail-closed до записи первого synthetic WAV. Нет
final raw/ready manifests, audio QA/VAD inventory, review packet/forms, pair lock или
materialization receipt. Detector checkpoint, calibration, detector inference и final inference
не выполнялись.

Использованный contract:
[`xlsr_sls_model_v4_final_materialization_v1`](../../../configs/research/v4/xlsr_sls_model_v4_final_materialization_v1.json),
SHA-256 `9d471f8e961a530a209fa2652344a0c12d4dfc28c03b4c28f5e14bd4bac2088a`.

## Фактическая остановка

Locked CrispASR/Qwen process завершился с exit code `16`: ему был передан относительный target
`data/raw/v4/xlsr_sls_model_v4_final_materialization_v1/ru_qwen/ru_qwen_001_3c6fd2b8ea62.wav`,
тогда как runtime запускается из собственного `cwd`; этот relative parent там не существует.
Runtime не записал WAV. Append-only one-shot journal содержит ровно один `planned` event для
rank `1`, SHA-256 `7ad051fff0d79b084b26274521c34aa6636c21d64d021cd38482b41961e91315`; generated event
отсутствует.

Это считается осуществлённой synthesis attempt, даже хотя audio byte не был сохранён. Поэтому
resynthesis этого row, replacement/backfill и простой повтор старого runner запрещены.

## Сохранённые локальные следы

Только Git-ignored raw source directories были опубликованы до ошибки; они ещё не являются
versioned manifests и не допускаются к QA/review/inference:

| Local slice | Files | Bytes | Ordered `(path,size,SHA-256)` aggregate SHA-256 |
| --- | ---: | ---: | --- |
| RU Common Voice source | 500 | 17,877,957 | `4e93ccecab98d0275692cb602927324db8cf584ea04c7993edffc9efbe1c2ee0` |
| KK FLEURS source | 500 | 421,864,520 | `aab8cdcbe4207603986cdf0e2a0fde887a10c80815a884650a722b6a8a2936c4` |
| RU Qwen synthetic | 0 | 0 | not applicable |
| KK KazakhTTS synthetic | 0 | 0 | not started |

## Required decision

The original one-shot authorization is exhausted/incomplete. Любой recovery должен быть
оформлен новым immutable contract, который явно решит, допустимо ли (а) сохранить rank `1` как
irrecoverable reject и продолжить только ранее не-attempted rows без backfill, либо (б) навсегда
abandon this 1,000-row selection and сделать новую metadata-only selection. До такого решения
запрещены любые synthesis/resynthesis, pair lock и final inference.
