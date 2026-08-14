# XLS-R+SLS model v4 — frozen train candidate metadata

**Статус:** канонический packet v2 заморожен; разрешена только materialization исходного аудио.
Synthesis и training не разрешены.

V2 выбрал `28 800` metadata candidates: по `7 200` на каждую cell `RU/KK × bona-fide/spoof`.
В каждой cell первые `6 000` имеют состояние `target`, следующие `1 200` — заранее
упорядоченный QA reserve. RU берётся только из raw RuASD без Common Voice; KK bona-fide — по
`1 440` текстов из пяти nonlegacy KSC2 Train components. KK spoof запланирован по `1 800`
текстов для четырёх train-only families: Piper, MMS, KazEmoTTS и Spark-TTS.

Текущая project-history сверка дала `0` пересечений по selected sample ID и exact/canonical text
hash. Source-lineage roots и TTS-family roots в v2 попарно разделены между
train/dev/calibration/final. Speaker independence не доказана. Raw/decoded audio hashes,
near-audio fingerprints и QA ещё отсутствуют, поэтому `24 000 ready` не заявлены.

Канонические bindings:

- config v2 SHA-256: `f04324f995d67fb7e21266eb1dfe019a0bd0744563e82c0553f2371cda0cb11b`;
- candidate CSV v2 SHA-256: `9a88442e32593147b07d55a13ee8ba4e97656d4deec0d6837e225c3b05436e43`;
- selection receipt v2 SHA-256: `79c19b7a69588d8baf1a4eb5d8e9cc4b1ab6752b2c6352b19b0912fd5d23f2ae`;
- [machine governance and reconciliation](xlsr_sls_model_v4_train_candidate_selection_governance_v1.json).

V1 сохранён неизменным, но отклонён до materialization: он пересекал FLEURS corpus family между
calibration/final и eSpeak family между train/calibration. Его downstream use запрещён.
Кроме того, v2 selection receipt содержит слишком широкое поле
`historical_exact_assets_selected=false`: до извлечения оно означает только отсутствие
sample/text collision, а не проверку exact audio. Governance receipt фиксирует это без
перезаписи исходного write-once файла.

Следующий gate: извлечь только v2 RuASD/KSC2 source candidates в новые Git-ignored v4 paths,
посчитать raw/decoded SHA-256 и near-audio fingerprints, выполнить общий QA/VAD и закрыть
historical/cross-role audio leakage. До успешного receipt нельзя синтезировать KK spoof.
