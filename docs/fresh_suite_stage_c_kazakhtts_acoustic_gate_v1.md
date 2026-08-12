# Stage C KazakhTTS — pre-inference language gate v1

**Статус:** завершён 12 августа 2026 года; два независимых набора решений прошли строгую
проверку, `kk`, `ru` и `mixed` допущены к подготовке нового evaluation candidate.

Gate относится только к трём техническим smoke WAV фиксированного маршрута ISSAI KazakhTTS2
Male2 Tacotron2 + ParallelWaveGAN. Он проверяет разборчивость, сохранение текста и языка и
отсутствие тяжёлых акустических артефактов. Детектор, его predictions и logits не использовались.

## Зафиксированные входы

| Артефакт | Строк | SHA-256 |
| --- | ---: | --- |
| `fresh_suite_stage_c_kazakhtts_smoke_v1_report.json` | 3 WAV | `fc10d5660eca06a44bfc7433838ac7043ee5ee93171b277d4034f10356c4377b` |
| `fresh_suite_stage_c_kazakhtts_acoustic_gate_packet_v1.csv` | 3 | `12dd6caa8bff9332708e1b365002545ddec9c7ea15b92ad82cb89de82ac37dea` |
| `fresh_suite_stage_c_kazakhtts_acoustic_review_reviewer_1.csv` | 3 | `ee9d5139f84bb4a7ad63aac4c412171eea32eb71c70c2baf2a172953f8fc7a29` |
| `fresh_suite_stage_c_kazakhtts_acoustic_review_reviewer_2.csv` | 3 | `0e305a42cfef1678030170e7d157a2be370d1e4ccae35eef3c8b3cee6300093a` |
| `fresh_suite_stage_c_kazakhtts_acoustic_gate_report_v1.json` | 3 results | `946c3a3a59fdd437553c2fe8e93d4ade157e718cf67505abb1216c02cbc82a73` |

Обе формы точно связаны с packet SHA-256 и хешами трёх WAV. Reviewer IDs различаются; каждая
форма содержит полный набор `kk`, `mixed`, `ru` без дублей и пропусков. Техническая проверка
разных псевдонимов не может сама доказать организационную независимость людей — это условие
проведённого прослушивания.

## Результат

- review rows: `6`, по три на каждого reviewer ID;
- `kk`: `pass` двумя слушателями;
- `ru`: `pass` двумя слушателями;
- `mixed`: `pass` двумя слушателями;
- `approved_input_languages`: `["kk", "mixed", "ru"]`;
- `detector_inference_authorized`: `false`.

Каждое решение имеет строгую комбинацию `pass/yes/yes/yes/no`: речь разборчива, текст и язык
сохранены, тяжёлые артефакты не отмечены. Это снимает языковое ветвление для выбранного exact
route, но не утверждает качество всех будущих WAV и не делает результат source- или
speaker-independent. Известный overlap с Male2 speaker alias остаётся раскрытым.

## Что разрешено дальше

Разрешена только подготовка нового pre-inference candidate для трёх языковых ролей:

1. заранее зафиксировать selection policy и точные bona-fide groups;
2. синтезировать связанные spoof assets и опубликовать полный rejection accounting;
3. проверить project exposure/leakage;
4. получить два full-asset acoustic review набора;
5. только после этого создать immutable inference plan, выполнить preflight и один GPU run.

Текущий smoke gate сам по себе не разрешает detector inference. Существующий JSON receipt
write-once и не должен пересоздаваться или изменяться.

## Историческая команда

```bash
.venv/bin/python scripts/kazakhtts_stage_c_acoustic_gate.py evaluate \
  --packet data/manifests/fresh_suite_stage_c_kazakhtts_acoustic_gate_packet_v1.csv \
  --reviewer-1 data/manifests/fresh_suite_stage_c_kazakhtts_acoustic_review_reviewer_1.csv \
  --reviewer-2 data/manifests/fresh_suite_stage_c_kazakhtts_acoustic_review_reviewer_2.csv \
  --evaluated-at 2026-08-12T20:40:34+05:00 \
  --output-report data/manifests/fresh_suite_stage_c_kazakhtts_acoustic_gate_report_v1.json
```
