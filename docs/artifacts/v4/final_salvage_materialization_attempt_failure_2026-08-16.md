# XLS-R+SLS model v4 — final salvage materialization attempt failure v1

**Дата:** 16 августа 2026, `2026-08-16T00:20:00+06:00`

## Статус

Salvage one-shot attempt исчерпан и не может быть повторён этим contract. Это не TTS runtime
failure: один разрешённый KazakhTTS pass создал `227/227` new WAV и технический decode/QA/VAD
прошёл для всех `1 994` raw assets. Publication намеренно fail-closed остановился до создания
versioned raw/ready manifests, inventory, review packet/forms, pair lock и materialization
receipt: текущий packet builder требует ready-сторону для каждого ready asset, тогда как QA
может оставить только одну сторону source/spoof пары.

Ни detector checkpoint, calibration, detector inference, final inference, replacement/backfill
или resynthesis не выполнялись.

## Hash-bound traces

Plan:
[`xlsr_sls_model_v4_final_salvage_materialization_v1.json`](../../../configs/research/v4/xlsr_sls_model_v4_final_salvage_materialization_v1.json),
SHA-256 `b886674e97dfe9261d6d79c88e5a9a5ba03b5f7421b2fe0bc08e47276d782f11`.

Новый append-only KazakhTTS journal: `454` events (`227` planned + `227` generated), SHA-256
`20559fc14fc3f42ff7b3b6b422a51b64a24d8dfca0305961b8505e227120eb41`. Новый Git-ignored raw
namespace содержит `227` WAV, `85,726,980` bytes; ordered
`(relative-path,SHA-256,size)` aggregate SHA-256
`386d8e24ccbce5069d57c527051143627322568647a91eee238f4ae621ec2022`.

Decode journals (all complete) закреплены следующими SHA-256:

- RU source `499`: `72d271e96697b15dc2bcd3dddf117d569746fb11891717f9943e1f2ce93f8ad4`;
- KK source `498`: `aec13e8d91720317a300516e271c67667887a657141fa9ddd0e9d7564d747d83`;
- RU Qwen spoof `499`: `ed1588bd6a3d4a6466ce32509db6fee6e8cbbbdd82eb82b262db8fe27abc5961`;
- KK KazakhTTS spoof `498`: `4170d72c8bb0517973e23154a875be93d5745445388800554316f3e7ad06f275`.

## QA facts and required next boundary

Decode QA results are source `ready/rejected`: RU `387/112` (`insufficient_speech`), KK
`460/38` (`signal_too_quiet`); spoof: RU `351/148` (`insufficient_speech`), KK `498/0`.
The failed publication did not authorize any response to those rejects.

A new immutable reconciliation contract must consume only the exact existing raw assets and these
complete decode journals, apply full-history isolation, and publish review inputs only for
complete eligible source/spoof pairs. It must not extract, synthesize, resynthesize, replace,
backfill or infer.
