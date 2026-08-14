# Evidence tiers and VoxCPM2 holdout candidates — 14 августа 2026

**Статус:** policy, Denis 1.0 source intake/current exposure/frozen selection/bona-fide QA/VAD и
official VoxCPM2 artifact/source/project-history/runtime/CUDA-smoke gates завершены. Denis QA
оставил minimum layer `64/79` без backfill. Candidate synthesis и detector inference не
выполнялись. Denis archive и VoxCPM2 model/source bytes проверены вне Git; MCSKL и
VoxCPM2-KZ-Darwin локально не загружались.

**Scope:** personal research. Ранее выполненные write-once runs, их manifests, reports, hashes и
запреты на повтор остаются неизменными.

## 1. Новая иерархия силы доказательств

Проект больше не требует «идеального» готового binary dataset и не требует от поставщика TTS
исчерпывающего перечня каждой обучающей реплики как предварительного условия для любого
исследования. Вместо бинарного `independent / rejected` используются три уровня.

| Уровень | Обязательные условия | Допустимый claim | Недопустимый claim |
| --- | --- | --- | --- |
| **Основной независимый слой** | новый human-корпус; новое generator family; text-only run без reference/prompt audio; права, exact artifacts и project exposure проверены; данные TTS достаточны для каждого заявленного измерения независимости | только те измерения независимости, которые фактически подтверждены | training-data, speaker или organizational independence без отдельного доказательства |
| **External source/family holdout** | новая human/TTS пара; новый human source и generator family относительно проекта; reference/prompt audio отсутствуют; TTS training sources раскрыты неполно | `external source- and generator-family-disjoint holdout; TTS training-data overlap unverified` | подтверждение отсутствия TTS training-data overlap или полностью независимый benchmark |
| **Same-family sensitivity test** | новый checkpoint, adapter, voice или language specialization внутри уже использованной family | sensitivity к checkpoint/языку/voice/config | новый независимый holdout или дополнительная generator-family evidence layer |

Полный список обучающих реплик TTS не является формальным обязательным артефактом. Но если
поставщик раскрывает лишь объём или общую схему training data, проект не превращает отсутствие
сведений в доказательство отсутствия overlap: такой результат автоматически остаётся external
holdout. Для stronger claim достаточно не конкретного формата списка, а проверяемого evidence,
которое действительно исключает заявленный overlap.

## 2. Неизменяемые запреты и облегчённый размер

Остаются fail-closed запреты на:

- human sources, уже использованные проектом, их repack/derivative copies и источники с
  неизвестными правами на человеческую речь;
- любой `reference_wav_path`, `prompt_wav_path`, reference/prompt audio, voice cloning или
  попытку воспроизвести идентичность человека;
- выбор, замену, backfill, regeneration, retry, reselection или настройку checkpoint, seed,
  параметров, threshold/calibration/augmentation после просмотра detector result;
- повтор выполненного write-once run или использование его final errors для следующего recipe.

Минимальный **готовый после technical QA** размер нового binary layer снижен до `60` exact pairs;
цель — `79`. Pre-QA buffer может быть больше, но selection policy, порядок и отсутствие backfill
фиксируются до synthesis. Для multi-speaker corpus сначала максимизируется число разных
source-provided speaker groups, затем строки добавляются round-robin с заранее установленным
per-speaker cap. Универсальное число cap не выдумывается до знания фактической metadata.

Single-speaker corpus может пройти только как явно ограниченный external holdout. `79` строк
одного человека не становятся speaker-robust evidence только из-за размера.

Если поставщик не публикует duration table, exact duration разрешено получить из byte-pinned
audio полным проходом audited decoder. Такой derived duration должен быть отдельно маркирован и
не считается VAD/acoustic QA. Это не ослабляет требования к archive identity, text/audio binding
или waveform readiness.

## 3. Contract для cloning-capable TTS в text-only режиме

Наличие cloning capability больше не отклоняет модель само по себе. Конкретный pinned route
допустим только если wrapper и receipt обеспечивают одновременно:

1. Полные commit/revision, source/runtime revision, license, размеры и SHA-256 всех required
   files; seed и generation parameters фиксируются до первого WAV.
2. Runtime работает из локального allow-listed snapshot: `local_files_only=True`, denoiser не
   загружается, сетевые обращения технически запрещены и проверены.
3. `reference_wav_path=None`, `prompt_wav_path=None`, `prompt_text=None`; wrapper не принимает
   эти значения от CLI/config. Voice-design/control prefix также запрещён для claim
   `default voice`.
4. `normalize=False`, `denoise=False`, `retry_badcase=False`; ровно одна попытка на frozen text,
   без скрытого retry или отбора «лучшего» WAV. Если exact upstream всё равно применяет
   детерминированный whitespace collapse, он разрешён только как заранее объявленный transform
   с literal/canonical hash binding; semantic TN/rewrite остаётся запрещён.
5. Output маркируется `text-only default voice`; не заявляются fixed real identity, совпадение с
   bona-fide speaker, cloning или speaker independence.
6. Smoke test выполняется на отдельном non-candidate тексте и без detector inference. Candidate
   synthesis разрешается только отдельным последующим contract после intake и exposure gate.

## 4. Denis 1.0: подтверждённые факты и новое ограничение

Первичная карточка [Mozilla Data Collective](https://mozilladatacollective.com/datasets/cmiup9seu01flnv076fexaqp9)
указывает:

- dataset ID `cmiup9seu01flnv076fexaqp9`, steward Open Home Foundation;
- latest archive name `denis-1-0-3f60c388.tar.gz`, отображаемый размер `104.52 MB`;
- `CC0-1.0`, примерно два часа scripted Russian speech, один speaker, WEBM;
- запись через Piper Recording Studio и отсутствие post-processing/validation;
- запрет re-identification и наличие отдельной pretrained Piper voice.

MDC [Data Provider Terms](https://mozilladatacollective.com/terms/providers) требуют от
поставщика подтвердить необходимые права, permissions и consents. Это полезная contractual
warranty, но сами terms одновременно говорят, что платформа не гарантирует законность, точность
или качество dataset. Поэтому rights evidence сильнее, чем у неизвестного web corpus, но не
равно независимому consent audit. Consumer terms, dataset restrictions и запрет re-identification
остаются обязательными; raw archive нельзя добавлять в Git или rehost.

Карточка не публикует exact byte size, SHA-256, число utterances или таблицу durations. Теперь
завершён отдельный [локальный source intake](data_sources_denis_1_0_mdc_2026-08-14.md): browser
download `1764973737766-ru_RU-denis.tar.gz` имеет `109,594,943` bytes и SHA-256
`75e2c63c5082df7623c6a98c529718b22015dfbd2d38a1ea328635f4dd4ccf9b`. Он содержит `1,150`
exact unique UTF-8 text/audio pairs; все `1,150` payloads полностью декодируются как 48-kHz
stereo Ogg/Opus, хотя members имеют `.webm` suffix. Decoded duration — `6,719.465` s; `1,143`
rows имеют duration `>=2.5` s до VAD/acoustic QA. На source-gate этапе это подтверждало только
feasibility minimum `60` и target `79`; последующий frozen QA/VAD оставил `64` ready rows.

### Историческая speaker-lineage экспозиция

В полном current project scope есть `12` уникальных historical RuASD spoof sample IDs с
`generator_name=piperTTS`, `generator_version=ru_RU-denis-medium`: `11` в когда-либо configured
train и `1` в dev. Ранее названные `7` относятся только к v2; v1 добавляет ещё `5`. Raw/ready
copies дают `24` manifest rows, но не `24` уникальных samples. Официальная [Piper model card](https://huggingface.co/rhasspy/piper-voices/blob/main/ru/ru_RU/denis/medium/MODEL_CARD)
связывает `ru_RU-denis-medium` с CC0 `OHF-Voice/voice-datasets`, а Denis 1.0 у MDC ссылается на
доступную pretrained Piper voice. Это сильное указание, что historical synthetic voice обучена
на речи этого же человека, хотя текущие public cards не дают cryptographic archive-to-checkpoint
binding.

Проект применяет fail-closed трактовку: пока обратное не доказано, Denis 1.0 **не является
speaker-disjoint** относительно detector training. Human archive остаётся новым direct source,
но route нельзя называть speaker-independent или speaker-robust.

## 5. Official OpenBMB VoxCPM2: подтверждённые факты

Проверены официальный [model card](https://huggingface.co/openbmb/VoxCPM2),
[repository](https://github.com/OpenBMB/VoxCPM),
[API reference](https://voxcpm.readthedocs.io/en/latest/reference/api.html) и
[technical report](https://arxiv.org/html/2606.06928v1).

- Exact HF revision закреплён как `bffb3df5a29440629464e5e839f4d214c8714c3d`, официальный
  source commit — `ee8161e9e1b7b082cb5721a3a9980da4204401e6`; mutable `main` не используется.
- Модель имеет 2B parameters, Apache-2.0 и официальную поддержку Russian; basic TTS запускается
  без reference audio и выдаёт 48 kHz output.
- Это новая для project manifests tokenizer-free diffusion-autoregressive family
  (`LocEnc -> TSLM -> RALM -> LocDiT`), а не Piper/MMS/VITS/RHVoice/eSpeak/Qwen3-TTS.
- Exact `9` files / `4,960,731,703` bytes включают `model.safetensors` и `audiovae.pth`.
  Safetensors header/`577` BF16 tensors и contiguous payload проверены; AudioVAE ZIP CRC,
  pickle `GLOBAL` allow-list и `312`-tensor CPU `torch.load(weights_only=True)` state прошли.
  Tokenizer Python и весь official source tar также проверены до model load.
- Exact current source требует Python `>=3.10`, а не прежнее наблюдение `<3.13`. Для smoke всё
  равно выбран отдельный Python 3.12 environment; основной project venv не меняется.
- Upstream безусловно схлопывает whitespace даже при `normalize=False`. Project wrapper заранее
  выполняет только `" ".join(text.split())`, связывает literal/canonical hashes и запрещает
  semantic normalizer.
- Model card рекомендует 1–3 generation attempts из-за вариативности. Project contract это
  сознательно запрещает: один frozen seed, одна попытка, любой reject остаётся reject.

Technical report раскрывает более `2` млн часов multilingual training speech: Chinese/English
составляют большинство, остальные 28 языков имеют примерно `1K–50K` часов каждый. Названия и
полный состав корпусов, включая Russian, не опубликованы. Поэтому нельзя проверить, входил ли
Denis/OHF material в training data; Apache-2.0 weights/code license сама по себе этого не
доказывает.

## 6. Решение по Denis 1.0 × official VoxCPM2

**Source, VoxCPM2 artifact/source/project-history/runtime/smoke и Denis selection/QA gates
пройдены для minimum external-holdout layer.** Exact source sample/audio/text overlap равен нулю
по current pre-selection scope `35` configs и `95` manifest files; history screen проверил
`40,682` rows / `19,001` spoof rows и нашёл `0` VoxCPM rows. Generator family новый, но
historical speaker-lineage ограничение остаётся. Если runtime и последующие gates пройдут,
допустимая маркировка результата:

> external human-source- and generator-family-disjoint RU holdout; TTS training-data overlap
> unverified; likely historical speaker-lineage exposure through RuASD Piper Denis; single
> speaker; not speaker-robust or speaker-independent; personal research only.

Это полезнее ещё одного exact-route test на Common Voice/VoxForge и historical TTS family, но не
закрывает основной independent/speaker-robust evidence gap. Последующие smoke, Denis selection
и bona-fide QA/VAD выполнены отдельными write-once gates. Synthetic generation и detector
inference по-прежнему не разрешены.

## 7. MCSKL × VoxCPM2-KZ-Darwin

Human side существенно сильнее по diversity, но intake пока заблокирован:

- [data paper](https://openhumanitiesdata.metajnl.com/articles/10.5334/johd.529) сообщает 33
  recordings, около 12 часов, естественные разговоры и time-aligned transcripts; abstract пишет
  `78` participants, а metadata section — `73`;
- тот же paper показывает `CC BY 4.0` в abstract/page metadata, но dataset-description license
  называет `Attribution-NonCommercial-ShareAlike 4.0`;
- официальный [SPEAK DMP](https://site.unibo.it/msca_speak/en/pubblicazioni/data-management-plan/@@download/file/MSCA-SPEAK-DMP.pdf)
  прямо относит MCSKL audio и transcripts к `CC BY-NC-SA 4.0`.

До проверки license file и terms **в фактическом OSF archive** действует более ограничительная
трактовка `CC BY-NC-SA 4.0`; ни `CC BY`, ни точное число speaker groups не считаются
закреплёнными.

[VoxCPM2-KZ-Darwin model card](https://huggingface.co/AMAImedia/VoxCPM2-KZ-Darwin-NOESIS-BF16)
указывает Apache-2.0, BF16 safetensors, Kazakh primary, Russian secondary и baked-in
`voxcpm_kaz_lora`, но:

- observed short revision `c0aa555` ещё не заменён полным pinned commit;
- base указан как `sozkz/VoxCPM2`, а не официальный `openbmb/VoxCPM2`;
- `voxcpm_kaz_lora` не имеет в card проверяемого source revision, training-corpus description и
  rights chain;
- snapshot listing не показывает отдельный AudioVAE artifact, поэтому offline completeness
  должна быть доказана до load.

После выбора official VoxCPM2 для RU этот KZ checkpoint относится к той же VoxCPM2 family. Даже
при успешных rights/artifact gates MCSKL × KZ-Darwin сможет быть только **same-family KK
sensitivity/source-diversity test**, а не вторым новым generator-family holdout. Альтернатива —
отказаться от RU VoxCPM2 и первой использовать KZ family, но текущие license/provenance blockers
делают это худшим следующим шагом.

## 8. Исторические исключения не открываются повторно

- `RUSLAN`, `SOVA`, `RuLS` и `M-AILABS` уже входят в RuASD upstream; также не возвращаются
  перечисленные там GOLOS/OpenSTT routes.
- Utrobin VITS остаётся rejected по immutable review: `76` historical rows с тем же model name
  и неизвестной historical revision запрещают использовать его как backfill.
- Piper, MMS, RHVoice, eSpeak NG, Silero, Dialogs VITS2, Qwen3-TTS и другие уже зафиксированные
  families/routes остаются historical. Новый checkpoint внутри них — sensitivity/exact-route
  evidence, а не новая family.

## 9. Следующий безопасный этап

1. **Завершено:** владелец предоставил browser-downloaded Denis archive вне Git; exact local
   bytes/SHA-256 закреплены несмотря на отличие browser/source-card filenames.
2. **Завершено:** read-only intake проверил gzip/TAR safety, `1,150` text/audio bindings, полный
   decode, derived duration, rights limitations и feasibility `60/79`.
3. **Завершено:** current source-wide sample/audio/three-text-hash screen v2 дал zero direct
   overlap по `35` configs / `95` manifests; historical lineage осталась `12` unique Piper Denis
   samples. Frozen metadata target содержит ровно `79` rows с category balance `27/26/26`.
4. **Завершено:** official VoxCPM2 model revision и source commit, `9` model files, source TAR,
   safetensors, AudioVAE weights-only load, tokenizer/source code, narrow wrapper и historical
   generator-family exposure закреплены.
5. **Завершено:** official frozen lock установлен в isolated Python 3.12; первый interface call
   versioned как pre-generation failure без WAV, после correction ровно один actual non-candidate
   CUDA smoke дал `48 kHz` mono output при `0` network attempts и null/false forbidden controls.
   Smoke не повторять.
6. **Завершено:** exact extraction и normal decode/QA/VAD оставили `64/79` ready rows; все `15`
   rejects — `insufficient_speech`, reuse/replacement/backfill `0/false/false`. Target 79 не
   достигнут, но заранее установленный minimum 60 пройден.
7. Следующим отдельным contract закрепить literal/canonical binding и one-shot VoxCPM2 synthesis
   только для 64 ready texts. Detector inference требует последующих synthetic QA, exact pair
   lock, двух independent reviews, current exposure audit и write-once evaluation contract.
