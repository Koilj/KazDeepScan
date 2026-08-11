# KSC2/Silero mixed — acoustic language-preservation gate v1

**Статус:** завершён 11 августа 2026 года для уже замороженных 30 KSC2/Silero pairs.
Это fail-closed quality gate, а не новая разметка, не detector evaluation и не основание назвать
checkpoint final-quality.

## Зачем нужен отдельный gate

У исходных KSC2 rows есть только narrow single-AI transcript evidence, а у Silero WAV — только
provenance intended input text. Ни transcript, ни успешный QA/VAD не доказывают, что синтезатор
действительно произнёс и русский, и казахский фрагмент. Автоматический ASR может быть вспомогательным
инструментом слушающего, но не является самодостаточным certificate: он может ошибиться именно на
коротком code-switching fragment. Поэтому gate требует независимой акустической проверки.

## Неизменяемый packet

`data/manifests/ksc2_mixed_v1_silero_v4_acoustic_gate_packet.csv` содержит ровно `60` rows:
один bona-fide и один spoof WAV для каждой из `30` frozen pairs. Его SHA-256:

`225f5cfe70eb422ef4c5cf131c81537eefea3a4bc9eabd24ace4df87af620421`.

Каждая строка связывает фактический asset SHA-256, path, text hash, полный исходный transcript,
Russian evidence tokens и Kazakh evidence tokens с pair-lock. Перед публикацией script повторно
проверил SHA-256 всех 60 local WAV. Packet не меняет candidate manifest и не запускает XLS-R.

## Результат v1

Два раздельных review CSV содержат по `60` решений с различными pseudonymous IDs `reviewer_1`
и `reviewer_2`. Validator подтвердил для каждого locked WAV две записи `pass` с
`ru_evidence_audible=yes`, `kk_evidence_audible=yes` и `lexical_content_preserved=yes`.
Следовательно, `60/60` assets и `30/30` pairs прошли именно этот узкий acoustic-preservation
contract.

Итоговый write-once receipt:
`data/manifests/ksc2_mixed_v1_silero_v4_acoustic_gate_report_v1.json`, SHA-256
`3585cda150e09a40a57bee50f3209e02e836b86738027c224902ae98d98eed01`.

Контракт технически проверяет, что IDs различаются и что reviewer не дублирует свой asset.
Он не может криптографически доказать реальную независимость людей; это остаётся
организационным требованием review process. Результат не изменяет frozen exploratory XLS-R
run и не отменяет остальные final/product blockers.

## Правило допуска

Для **каждого** из 60 locked audio assets нужны два отдельных review CSV от разных
`reviewer_pseudo_id`. Каждый reviewer прослушивает именно asset из packet и фиксирует:

- `review_status=pass`;
- `ru_evidence_audible=yes`;
- `kk_evidence_audible=yes`;
- `lexical_content_preserved=yes`.

Если хотя бы один asset имеет меньше двух review, reviewer повторён, есть `fail`/`inconclusive` или
хотя бы одно `no`/`unknown`, этот asset получает `not_eligible`; смешанная пара не может войти в
final protocol. Никаких score thresholds, confidence cutoffs, ASR-derived labels или
post-hoc исключений не предусмотрено. Reviewer IDs должны быть pseudonymous; не записывайте в CSV
имя, контактные данные или другую PII.

Даже стопроцентный pass этого gate подтверждает только аудиальную сохранность locked RU/KK lexical
evidence. Он **не** доказывает speaker independence, независимость spoof source, право product use,
calibration или final quality. Поэтому machine-readable report намеренно оставляет
`final_or_product_eligible=false`.

## Воспроизведение и выполнение review

Packet уже опубликован write-once. Для каждого независимого reviewer нужно создать новый template
с другим pseudonymous ID; начальные значения `inconclusive` и `unknown` нельзя считать решением.

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_ksc2_mixed_acoustic_gate.py prepare-review \
  --packet data/manifests/ksc2_mixed_v1_silero_v4_acoustic_gate_packet.csv \
  --reviewer-pseudo-id reviewer_a \
  --output-review data/manifests/ksc2_mixed_v1_silero_v4_acoustic_review_a.csv
```

Второй reviewer создаёт отдельный file с `reviewer_b`, не копируя решение первого. После полного
прослушивания всех audio можно создать отдельный write-once receipt:

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_ksc2_mixed_acoustic_gate.py evaluate \
  --packet data/manifests/ksc2_mixed_v1_silero_v4_acoustic_gate_packet.csv \
  --reviews data/manifests/ksc2_mixed_v1_silero_v4_acoustic_review_a.csv \
  --reviews data/manifests/ksc2_mixed_v1_silero_v4_acoustic_review_b.csv \
  --output-report data/manifests/ksc2_mixed_v1_silero_v4_acoustic_gate_report_v1.json
```

Runner проверяет exact packet hash, every asset binding, enum values и отсутствие повторного решения
того же asset тем же reviewer. Он публикует решение по всем 60 assets, включая непрошедшие; output
нельзя перезаписать. Этот v1 уже выполнен и повторять его на тех же review files не нужно.

## Реализация и проверки

- Contract/validator: `src/kds/eval/mixed_acoustic_gate.py`.
- CLI: `scripts/prepare_ksc2_mixed_acoustic_gate.py`.
- Unit test: `tests/test_mixed_acoustic_gate.py`.

Профильные Ruff, mypy и pytest проверки прошли. Gate намеренно изолирован от
`evaluate_xlsr_exploratory_mixed.py`: exploratory XLS-R run остаётся единственным и не
повторяется для этого процесса.
