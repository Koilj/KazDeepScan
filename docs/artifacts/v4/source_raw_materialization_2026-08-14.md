# XLS-R+SLS model v4 — materialization исходного train audio

**Статус:** исходные RuASD/KSC2 bytes материализованы и прошли exact raw-audio gate;
decoded-аудит, near-audio fingerprints и QA/VAD ещё не выполнены. Synthesis и training не
разрешены.

Из канонического v2 metadata packet извлечены все `21 600` source candidates: `14 400` RuASD
WAV и `7 200` KSC2 FLAC. RuASD повторно привязан к metadata всех `250/250` pinned TAR, после
чего извлечены только выбранные записи из `127` затронутых архивов. KSC2 прочитан полным
multipart stream: повторно проверены размеры 10 частей, SHA-256 объединённого gzip, gzip CRC,
безопасность TAR layout и точная allow-list из `7 200` файлов.

Raw bytes занимают локально `6.9 GiB` в Git-ignored namespace
`data/raw/v4/xlsr_sls_model_v4_source_raw_v1/`; они не добавляются в Git или release.

## Результат exact gate

| Cell | Извлечено | Допущено к decode/QA | Отклонено |
| --- | ---: | ---: | ---: |
| KK bona-fide | 7 200 | 7 200 | 0 |
| RU bona-fide | 7 200 | 7 200 | 0 |
| RU spoof | 7 200 | 7 198 | 2 |
| **Всего** | **21 600** | **21 598** | **2** |

Повторная полная проверка manifest-to-disk подтвердила `21 598` уникальных путей и `21 598`
уникальных raw SHA-256. Within-pool exact duplicates отсутствуют. Две TeraTTS записи отклонены
как historical exact raw-audio collisions с неизменённым
`data/manifests/ruasd_ru_v1_shard000000_ood_100.csv`:

- `raw_fake_TeraTTS_15d373b7afaf07793684c1ba279178df`, SHA-256
  `e5905cd28bbf9202fcf84d755f57fb3a456fb4bcababc0ce98b9a1a17141d154`;
- `raw_fake_TeraTTS_131bb8e32e2f4ffed14a81aac50aba6f`, SHA-256
  `69562d543d55071a59e5a5011f4bea36a20231ace04acd6c9d72b5aade82e2e7`.

Отклонённые строки остаются в полном inventory с причиной, но отсутствуют в eligible raw
manifest. Замена или outcome-driven backfill не выполнялись.

## Versioned outputs

- полный inventory, `21 600` строк:
  `data/manifests/v4/xlsr_sls_model_v4_source_raw_inventory_v1.csv`, SHA-256
  `f01ee992320c3692a9d9c79e1293b3c8e48f7e118683ea18b6f23f996e08895b`;
- eligible raw manifest, `21 598` строк:
  `data/manifests/v4/xlsr_sls_model_v4_source_raw_eligible_v1.csv`, SHA-256
  `d0d3e7610a00bcbb3929f2f48ed2c5c5febc6285b8b1bfa32cdd0e3680745903`;
- [machine receipt](xlsr_sls_model_v4_source_raw_materialization_v1.json), SHA-256
  `246009d1d8e16bcdf354736efc7ebff297a4b560f3dced15eedd95f8cff6cfc7`.

Machine receipt hash-bind-ит canonical selection/config, source audits, license ledger и
актуальный historical exposure inventory из `99` manifest-файлов. Старые v1/v2/v3 manifests,
raw assets, checkpoints, execution locks и receipts не изменены.

## Граница утверждений и следующий gate

Этот этап доказывает только provenance извлечения, целостность raw bytes и отсутствие
обнаруживаемого exact raw-audio overlap. Он не доказывает:

- корректность полного decode в canonical mono PCM 16 kHz;
- отсутствие exact duplicate после декодирования или near-audio overlap;
- прохождение technical QA и WebRTC VAD;
- speaker independence;
- наличие `24 000 ready` train rows.

Следующий обязательный этап — независимо декодировать `21 598` eligible source rows, вычислить
decoded SHA-256 и устойчивые near-audio fingerprints, выполнить QA/VAD и сверить результаты со
всей project history. Только после versioned rejection accounting можно заморозить source часть
train и перейти к заранее определённому KK spoof synthesis.
