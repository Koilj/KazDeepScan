# XLS-R+SLS model v4 — final-input metadata contract v1

**Дата:** 15 августа 2026

## Статус

Контракт завершён одним metadata-selection run: заморожены `500` RU + `500` KK source text
groups. [Selection receipt](v4_final_inputs_selection_2026-08-15.md) фиксирует exact rows и
current-history screen. До следующего отдельного contract нет final audio manifest, WAV,
synthetic output, QA, pair lock, checkpoint loading, calibration или detector inference.

## Изоляция и маршруты

- RU — только previously unselected Common Voice RU v24 `test` metadata, один record на
  client group и text group; Qwen CustomVoice `aiden` проверяется только как fixed literal-text
  route. Ранее inferred `79` VoxForge/Qwen pairs исключены текущим full-history screen.
- KK — только FLEURS `kk_kz` `train` split. Это отдельный от historical Stage-C `test` split;
  для FLEURS единственный source-group key — `prompt_id`/text group. Общий placeholder
  `speaker_pseudo_id=unknown` не является speaker identity и не может ни исключать все rows,
  ни поддерживать speaker-independent claim.
- KazakhTTS проверяется только на explicit text-normalization compatibility. Ни модель, ни TTS
  runtime не загружаются на этом этапе.

Frozen ledger запрещает materialization/evaluation для всех четырёх routes. Следующий отдельный
contract сможет разрешить extraction и one-shot text-only synthesis только для exact selected
metadata rows, с полным QA/VAD/audio-isolation/pair-lock gate. Final evaluation contract остаётся
третьим, самостоятельным шагом.
