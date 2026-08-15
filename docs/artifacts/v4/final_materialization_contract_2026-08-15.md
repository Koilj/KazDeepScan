# XLS-R+SLS model v4 — final materialization/review contract v1

**Дата:** 15 августа 2026

## Статус

Контракт подготовлен и hash-pinned, но **не выполнен**. Он является единственным разрешением на
следующий one-shot этап и не изменяет metadata-only selection: ровно `500` RU Common Voice
`test` rows и `500` KK FLEURS `train` rows остаются единственными source identities. До успешного
run нет новых final WAV, ready manifests, QA/VAD results, review packet, review decisions или
pair lock. Detector checkpoint, calibration, detector и final inference не загружаются и не
разрешены; hash-pinned local TTS models разрешены только для указанного synthesis route.

Контракт
[`xlsr_sls_model_v4_final_materialization_v1`](../../../configs/research/v4/xlsr_sls_model_v4_final_materialization_v1.json)
имеет SHA-256 `fabd4c28b1f70815d57747007cfb3504424407ee7ce2e23cdb157dac3dd71b1d`.
Новый narrow ledger:
[`xlsr_sls_model_v4_final_materialization_v1.csv`](../../../data/licenses/frozen/xlsr_sls_model_v4_final_materialization_v1.csv),
SHA-256 `87aedfd77beb51891abe81477289d8061db50f7488db5b1d802d01e20481a09a`.

## Разрешённая последовательность

1. Read-only preflight повторно проверяет metadata selection, Common Voice archive, полный
   FLEURS release, frozen ledger, full project-history fingerprint evidence и оба hash-pinned
   local TTS route. Он не создаёт WAV или final receipt.
2. Только после preflight один write-once run извлекает exact `500` RU и `500` KK source assets,
   затем делает ровно по одной text-only synthesis attempt для тех же `500` RU через fixed
   Qwen CustomVoice `aiden` и `500` KK через fixed KazakhTTS Tacotron2/PWG. Reference audio,
   cloning, network downloads, replacement, backfill и resynthesis запрещены.
3. Все четыре raw cells проходят canonical decode, technical QA/VAD и full current-history
   exact/near-audio isolation. Только ready assets публикуются в versioned review packet;
   rejected/pending assets не заменяются.
4. Два независимых reviewer'а заполняют две exact-byte-bound versioned CSV формы. Для asset
   нужен один полный verdict от каждого reviewer'а; pair допускается только если оба verdict'а
   `pass/yes/yes/yes/no` для status, intelligibility, lexical content, language и severe
   artifacts соответственно.
5. Отдельная команда создаёт immutable pair lock только после обеих complete review forms. Она
   не может принять произвольные CSV или общую reviewer identity. Даже после lock final
   inference остаётся запрещённым: для него понадобится самостоятельный no-logit/evaluation
   contract.

Raw/processed audio и local model bundles остаются Git-ignored. Versioned будут только
manifests, gate inventory, review packet/forms и machine receipts с SHA-256.

## Безопасный следующий шаг

Запустить только `preflight` этого контракта. Если он проходит, последующий one-shot
`materialize` run не должен менять checkpoint, calibration или выполнять detector/final
inference. После materialization необходима реальная независимая acoustic/language review;
автоматически или задним числом одобрять формы нельзя.
