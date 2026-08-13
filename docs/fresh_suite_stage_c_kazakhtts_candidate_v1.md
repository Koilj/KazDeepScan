# Stage C KazakhTTS normalized synthesis and candidate v1

**Статус:** 12 августа 2026 года опубликован exact balanced candidate; full-asset acoustic
review ожидает заполнения двух независимых форм. Detector inference не выполнялся и не разрешён.

## Почему потребовалась нормализация

Первая массовая попытка передала модели исходные surface forms. Character-inventory gate
принял 96 строк (`58 mixed / 38 RU`) и явно отклонил 72 (`60 KK / 12 RU`) из-за цифр,
латиницы и неподдерживаемой пунктуации. Эта попытка сохранена как compatibility-failure receipt,
но не используется в candidate и не скрывается.

До повторного synthesis был write-once заморожен deterministic normalizer. Он изменил все 60
KK и 12 RU текстов, не изменил 58 mixed текстов и сохранил исходные `text_id/text_hash`.
Нормализованный текст и его SHA-256 записаны отдельно в provenance каждой spoof-строки.
Нормализация служит только совместимости с token list: detector outputs, метрики и backfill не
использовались.

## Synthesis, QA и pairing

Нормализованный CUDA run создал `168/168` WAV, текстовых отказов не было. Обычный project
audio pipeline принял 167 файлов и отклонил одну mixed-строку как `insufficient_speech`.
Перегенерации и замены не выполнялись. Итоговый candidate содержит 167 точных текстовых пар:

| Роль | Bona fide | Spoof | Пар |
| --- | ---: | ---: | ---: |
| RU | 50 | 50 | 50 |
| KK | 60 | 60 | 60 |
| mixed | 57 | 57 | 57 |
| Всего | 167 | 167 | 167 |

Метрики должны рассчитываться отдельно по языкам. Claims остаются
`source_independent=false` и `speaker_independent=false`; 312 исторических строк используют тот
же Male2 voice alias через другой Piper route.

## Project exposure

Candidate сравнивался по `sample_id`, exact audio SHA-256 и `text_hash` с 15 уникальными
manifest-файлами, на которые ссылается 21 существующий research config (`11 869` prior rows).
Все три overlap count равны нулю. Отдельный ранее замороженный route audit также имеет exact
route overlap `0`. Это доказывает asset/text/route freshness только в указанном project scope,
но не source- или speaker-independence.

## Full-asset gate

Packet содержит все 167 QA-ready synthetic assets, exact audio SHA, исходный текст,
нормализованный synthesis text и их bindings. Обе формы созданы с fail-closed значениями
`inconclusive/unknown/unknown/unknown/unknown`. Для pass каждая строка у двух разных reviewers
должна стать `pass/yes/yes/yes/no`. Обе независимые формы заполнены: все 334 решения строгого
контракта прошли, `167/167` exact synthetic assets получили итог `pass`. Write-once receipt
`fresh_suite_stage_c_kazakhtts_full_acoustic_gate_report_v1.json` имеет SHA-256
`9a12f235072ce5ae4c3bd6bb0616a804a710abea74f3c3b387cebe12baf8153c` и разрешил создание
immutable inference plan. После exact validate-only preflight один GPU inference run завершён;
повтор блокирован execution lock. Раздельные результаты и ограничения описаны в
`docs/research_xlsr_sls_stage_b_v2_fresh_suite_stage_c_v1.md`.

## Ключевые артефакты

| Артефакт | SHA-256 |
| --- | --- |
| `fresh_suite_stage_c_kazakhtts_synthesis_v1_report.json` | `c5154a317b12de51e4dc117aa3d82d26712f2d9459e7012f2aaf93190c5c8b1e` |
| `fresh_suite_stage_c_kazakhtts_normalization_v1.json` | `d1f77e41349f6c787cf2cdbabfc09a1bfd3b153fd006ca4c2ec7d8975cfa9bf0` |
| `fresh_suite_stage_c_kazakhtts_raw_v2.csv` | `bae3d8bf2d92ac280715468c3326adf04bc187d87598c519c0c34f047cc12ee4` |
| `fresh_suite_stage_c_kazakhtts_normalized_synthesis_v2_report.json` | `644f6bac4278cf4d95ce60eafddab1bfc001b828d5438b6424418c37faac920d` |
| `fresh_suite_stage_c_kazakhtts_ready_v2.csv` | `3f91a91ecc7627964fe15fe57ba0e443a976a692202122d64107ea81add27b78` |
| `fresh_suite_stage_c_kazakhtts_audio_rejections_v2.json` | `e6510490a83c9069ab5977156d972c6e9dd3414fec210300191e93ec2d0cd702` |
| `fresh_suite_stage_c_kazakhtts_pairing_v1.json` | `9012dcad2065d90072a41c5dd66eac08326156229b6ba08b87f4563998c2026a` |
| `fresh_suite_stage_c_candidate_v1.csv` | `64d2f54d59f05eeb9211db7fba3280036798fc94c91b30440122001caca215c9` |
| `fresh_suite_stage_c_candidate_project_exposure_v1.json` | `f9d2980e5577d1955df6eb098bf380fe4be8d89fcc29bb618d50f37ee72037cc` |
| `fresh_suite_stage_c_kazakhtts_full_acoustic_gate_packet_v1.csv` | `6b4ada9ffd10b3c1cb1c30f6b1794b483e4ec31017333b7401cff31d414f5d14` |
| `fresh_suite_stage_c_kazakhtts_full_acoustic_review_reviewer_1.csv` | `0e797abd78170de4b453ece05cd47e4ebdbd9bdd457f490b1ad1c5163aac9697` |
| `fresh_suite_stage_c_kazakhtts_full_acoustic_review_reviewer_2.csv` | `03616fec42b78c6b3d29caad57d665e52a4c202cf9dd4bd5f8d90231273e3421` |
| `fresh_suite_stage_c_kazakhtts_full_acoustic_gate_report_v1.json` | `9a12f235072ce5ae4c3bd6bb0616a804a710abea74f3c3b387cebe12baf8153c` |

Raw/processed WAV и model weights остаются локальными и исключены из Git; manifests и receipts
фиксируют их exact bytes.
