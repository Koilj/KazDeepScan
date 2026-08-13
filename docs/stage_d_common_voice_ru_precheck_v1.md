# Этап D — Common Voice RU bona-fide precheck

**Статус:** подготовлен только bona-fide кандидат; это не binary suite, не evaluation plan и не
разрешение на detector inference.

## Что проверено

Исходный `common_voice_ru_v24_first_250_ready.csv` содержит `84` `test` bona-fide RU строки.
Перед выбором не запускались detector, ASR, LID, model metric или анализ Stage-C ошибок.

Первая write-once попытка selection (`…candidate_v1.csv`, SHA-256
`32ffa7772426277d330748d7a6218debf3a914d4ead022398202e55809ea958c`) исключила пять строк с
text overlap в configured roles, но оставила четыре строки из тех же pseudo-speaker/parent groups.
Она заблокирована и не может быть частью будущей оценки; inference не выполнялся.

Исправленный v2 policy исключает всю candidate group, если хотя бы одна её строка пересекается с
configured role по `sample_id`, audio SHA-256, `text_hash`, `parent_group_id` или
`speaker_pseudo_id`:

- вход: `84` Common Voice RU `test` строки;
- прямые configured-role text overlaps: `5`;
- group-tainted exclusions: `11` строк;
- результат: `73` RU bona-fide `test` строки, manifest SHA-256
  `d9ef5b0e91e960e5bf76a43274587a6483ac10e431d391f1ba119c71a05b330f`;
- selection receipt SHA-256
  `7dce6e5fc87608daaef1fea0714030783ec4b02725c38bfa686ba8f93c4a1db1`.

Project exposure v2 прошёл против всех `22` research configs, `16` их manifest bindings и
`12 203` configured rows: overlap по всем пяти comparison fields равен нулю. В дополнительном
inventory scan проверены `39 320` non-candidate rows: нет duplicate sample/audio/text; два
parent/speaker overlaps раскрыты как неиспользованные source-lineage rows, не как model role.
Receipt SHA-256: `1b36548ba2d7f533aa64ff4497aa69aae886c0d0e5e2100d36f530f4dd19435f`.

Source-group independence не заявляется: Common Voice `client_id` — только source-provided
pseudonymous ID. В v2 audit явно связаны raw MP3, ready WAV и заблокированный v1 selection; ни
один из этих технических predecessor manifests не указан в prior research config.

## Проверка новых RU TTS маршрутов

Ниже перечислены только результаты provenance screen, без скачивания модели, synthesis или
inference.

- Russian Piper отвергнут: RuASD historical manifests уже содержат `ru_RU` Piper voices
  `dmitri`, `ruslan`, `denis` и `irina` в model train/dev roles.
- Meta `facebook/mms-tts-rus` имеет официальный text-to-waveform VITS checkpoint (CC-BY-NC-4.0,
  revision `a6f0f76c028c49175f42074ee79bd3e17ee1dd47`, один speaker и
  `speaker_embedding_size=0`). Но RuASD v2 train уже содержит `19` spoof rows с
  `generator_name=mms_tts_rus` / `generator_version=mms-tts-rus`; historical source не pin-ит
  revision. Нельзя честно доказать новый exact route, поэтому MMS Russian отклонён.
- RHVoice уже отмечен как seen generator family в
  [external RU source search](russian_spoof_source_search_2026-08-11.md) и не рассматривается.

Следствие: в проекте пока нет допустимого нового RU spoof route для pairing с этими 73
bona-fide rows. Новый model v3, synthetic generation, acoustic review и final inference не
начинаются.

## Воспроизводимость

- `scripts/select_unexposed_bonafide_candidate.py` фиксирует selection без model output.
- `scripts/audit_candidate_project_exposure.py` fail-closed сравнивает candidate с configured
  roles и inventory; exact candidate rows исключаются только из явно связанных source manifests.
- Unit tests: `tests/test_bonafide_candidate.py`, `tests/test_candidate_exposure.py`.

Следующий разрешённый шаг — найти и полностью проверить новый Russian text-only TTS route с
pinned model bytes, fixed non-cloning profile, явными правами и отсутствием historical route
overlap. Только затем можно создавать paired synthetic assets и две independent acoustic review
формы.
