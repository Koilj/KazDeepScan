# XLS-R+SLS model v4 — source decode, QA/VAD и audio leakage gate

**Статус:** source часть train заморожена на `15 000` строк; принято решение
`proceed_20k_balanced`. Разрешён только следующий KK spoof synthesis gate. Training не разрешён.

Все `21 598` допущенных raw RuASD/KSC2 assets декодированы FFmpeg в новый Git-ignored
content-addressed namespace: mono PCM S16LE WAV, 16 кГц. Для каждого файла повторно вычислены
decoded SHA-256, RMS, clipping, DC offset, WebRTC VAD speech duration и 256-bit
speech-spectral perceptual fingerprint. Локальный processed слой занимает `4.8 GiB` и не входит
в Git/release.

## QA/VAD и eligibility

| Cell | Raw gate | QA ready | Near-hit исключён | Eligible |
| --- | ---: | ---: | ---: | ---: |
| KK bona-fide | 7 200 | 6 027 | 0 | 6 027 |
| RU bona-fide | 7 200 | 5 707 | 1 | 5 706 |
| RU spoof | 7 198 | 7 197 | 0 | 7 197 |
| **Всего** | **21 598** | **18 931** | **1** | **18 930** |

Technical rejection accounting:

- `2 659` — `insufficient_speech` (`1 173` KK bona-fide, `1 486` RU bona-fide);
- `4` — `signal_too_quiet` (`3` RU bona-fide, `1` RU spoof);
- `3` — `excessive_clipping` (RU bona-fide);
- decode errors отсутствуют.

Thresholds не подбирались по model score: использованы project defaults — RMS не ниже
`-55 dBFS`, clipped fraction не выше `0.02`, минимум `2.5 s` WebRTC VAD speech.

## Exact и near-audio screen

Exact decoded-audio collisions с project history: `0`. Within-pool exact decoded duplicates:
`0`. Within-pool near hits: `0`.

Historical inventory содержит `28 400` уникальных manifest SHA-256 из 99 ранее закреплённых
manifests. Все `28 008` локально доступных assets hash-verified и fingerprinted. Остальные `392`
относятся только к двум сохранённым Italian ML-DF OOD manifests, media bytes которых были ранее
удалены как непригодные для RU/KK v4; их exact hashes и metadata остаются в screen.

Одна RU bona-fide запись получила два эквивалентных near references на raw/processed версии
одного старого PyAra asset: Hamming distance `12/256`, speech-duration delta `0.0556`. Запись
`ruasd_ru_v1_full:ruasd-000153:raw_real_Deep-Speech_In_6kJDNFQI_45` консервативно исключена из
ready/frozen manifests без попытки доказать или опровергнуть тождество по модели. Fingerprint —
только duplicate-candidate screen, не speaker identity evidence.

## Balanced train decision

Первоначальный preferred target `24 000` требовал по `6 000` строк на cell. После полного
заранее замороженного target+reserve прохода RU bona-fide оставил `5 706` eligible, поэтому
`24 000` не заявляются. Outcome-driven backfill, расширение pool или поиск нового dataset не
выполнялись.

В пределах заранее разрешённой нижней границы принят `proceed_20k_balanced`:

- frozen RU bona-fide: `5 000`;
- frozen RU spoof: `5 000`;
- frozen KK bona-fide: `5 000`;
- pending KK spoof target: `5 000`.

Frozen source manifest содержит `15 000/15 000` уникальных sample IDs, decoded SHA-256,
canonical text hashes и conservative parent groups. Выбор выполнен только по frozen rank после
QA/leakage exclusions. Replacement/backfill не было.

## Versioned outputs

- full decode/QA inventory, `21 598` rows:
  `data/manifests/v4/xlsr_sls_model_v4_source_decode_qa_inventory_v1.csv`, SHA-256
  `8815000a16d778a83c3d22474c24bcbba41a74f8964177d26b992b83a1af262d`;
- historical fingerprint inventory, `28 400` rows:
  `data/manifests/v4/xlsr_sls_model_v4_historical_audio_fingerprints_v1.csv`, SHA-256
  `ff48f80b0729a8b2c40d02cf0b5f830c11d27d87380fedae6ba57264a0708e08`;
- eligible ready manifest, `18 930` rows:
  `data/manifests/v4/xlsr_sls_model_v4_source_ready_v1.csv`, SHA-256
  `30ff4902842556f6fa7f50ad2dfb9152bb89b0fa383f8a4dd93bd32f39af2183`;
- frozen source train manifest, `15 000` rows:
  `data/manifests/v4/xlsr_sls_model_v4_source_train_frozen_v1.csv`, SHA-256
  `f174e60e4cc13e58779ad3c1556a3aa20d737538dc696636f11ec440a8d11025`;
- [machine receipt](xlsr_sls_model_v4_source_decode_qa_v1.json), SHA-256
  `634416a0bd7b450ece58a0bad6b55f202ff78d9acd269648d5bfddb83864fe0d`.

## Следующий gate

Разрешено синтезировать только frozen v2 KK spoof candidates через четыре train-only routes:
Piper, MMS, KazEmoTTS и Spark-TTS, в новых write-once v4 namespaces. Для balanced target каждой
family требуется `1 250` ready rows; existing `1 800` candidates/family дают заранее
замороженный QA reserve. Synthetic outputs проходят тот же decode/QA/VAD, decoded exact/near
audio screen и полный rejection accounting. До frozen `5 000` KK spoof manifest training,
checkpoint selection, calibration и final inference запрещены.
