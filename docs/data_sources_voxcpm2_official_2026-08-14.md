# Official OpenBMB VoxCPM2 — artifact/source/runtime gate, 14 августа 2026

**Статус:** exact model/source artifacts, project-history novelty, isolated runtime и один
non-candidate CUDA text-only smoke завершены. Denis frozen metadata selection и bona-fide QA/VAD
также завершены (`64/79` ready, no backfill); subsequent immutable 64-row text-binding и
one-attempt contract были закреплены до candidate WAV. Candidate synthesis и synthetic QA/VAD
завершены один раз: `64/64` raw, `53` ready, `11` final rejects, `stop_below_minimum_60`.
Detector inference не выполнялся и для этого route запрещён; smoke/synthesis не повторять.

**Допустимая роль artifacts:** stopped personal-research external source/generator-family holdout
candidate для Denis 1.0. Это не подтверждает отсутствие Denis/аналогичных записей в training
data модели, идентичность default voice или speaker independence и не образует evaluation layer.

## Exact artifacts

Закреплены официальный model snapshot `openbmb/VoxCPM2` revision
`bffb3df5a29440629464e5e839f4d214c8714c3d` и официальный source commit
`ee8161e9e1b7b082cb5721a3a9980da4204401e6`. Локальные model/source bytes находятся в ignored
`models/`; в Git входят только проверки, lock/config и aggregate receipts.

| Проверка | Результат |
| --- | ---: |
| Model files / total bytes | `9 / 4,960,731,703` |
| Main `model.safetensors` | `4,580,080,592` bytes, SHA-256 `f7f964…6891d` |
| Safetensors header / tensors | `71,496` bytes / `577` BF16 tensors |
| Safetensors payload | `4,580,009,088` contiguous bytes, no gaps/overlap |
| `audiovae.pth` | `376,951,122` bytes, SHA-256 `94b5d5…bf1` |
| AudioVAE ZIP | `316` members, full CRC pass |
| AudioVAE weights-only state | `312` CPU tensors, `94,213,957` elements |
| Source tar.gz | `4,107,908` bytes, SHA-256 `5af8b4…0674` |
| Source TAR | `98` members: `75` files / `23` dirs, safe paths/types |
| Source `torch.load` calls | `11/11` pass explicit `weights_only=True` |
| Declared code/model license | `Apache-2.0` |
| Exact upstream Python requirement | `>=3.10` |
| Model output sample rate | `48,000 Hz` |

Main safetensors был проверен без deserialization: JSON header, tensor metadata и полный
contiguous payload. AudioVAE сначала прошёл ZIP/path/CRC и pickle `GLOBAL` allow-list, затем был
загружен только как CPU state через `torch.load(weights_only=True, mmap=True)`. VoxCPM class не
импортировался и модель не создавалась.

Versioned receipt:
[`data/licenses/voxcpm2_official_v1_artifact_source_receipt.json`](../data/licenses/voxcpm2_official_v1_artifact_source_receipt.json),
SHA-256 `84d5572840837d5fb6ef92202c53be4162d8116cdced0abfbcf7e0414890fcb8`.

## Новый generator family, но не training-data independence

Полный current history screen связал hashes всех `95` manifest CSV и проверил `40,682` rows,
включая `19,001` spoof rows / `6,690` unique spoof sample IDs. Поиск `voxcpm` в
`generator_family`, `generator_name`, `generator_version` и `voice_id` дал `0` rows. В history
есть `14` generator families, но VoxCPM среди них не было.

Отсюда разрешён точный claim: **official OpenBMB VoxCPM2 — новый generator family для
сохранённой истории проекта**. Absolute architecture novelty не заявляется, потому что
historical metadata не универсальны. Также не разрешён training-data claim: model card
раскрывает лишь aggregate scale, а не полный список источников. Поэтому Denis × VoxCPM2
остаётся `external source holdout, TTS training-data overlap unverified`.

Immutable history screen:
[`data/manifests/voxcpm2_official_project_history_screen_v1.json`](../data/manifests/voxcpm2_official_project_history_screen_v1.json),
SHA-256 `0a86e2677f0d8ec64dd2206720d7d95a6b9ec01e2aec38f4669a9544399558d7`.

## Облегчённый, но fail-closed text contract

Upstream API умеет cloning, continuation, normalizer, denoiser и retry. Это не запрещает саму
модель, потому что проект допускает её только через собственный narrow wrapper:

- local exact snapshot и `local_files_only=True`;
- внешний network block плюс offline environment variables;
- `reference_wav_path=None`, `prompt_wav_path=None`, `prompt_text=None`;
- no LoRA, no denoiser, `normalize=False`, `denoise=False`, `retry_badcase=False`;
- один attempt, fixed seed `20260814`, `cfg=2.0`, `10` steps, `min_len=2`, `max_len=4096`;
- output маркируется `text-only default voice`; voice identity не заявляется.

Проверка официального `core.py` выявила, что upstream безусловно схлопывает whitespace даже при
`normalize=False`. Поэтому правило уточнено: разрешена только заранее объявленная
`collapse_whitespace = " ".join(text.split())`. До selection связываются SHA-256 literal и
collapse-whitespace вариантов, а в модель передаётся второй. Semantic normalization, числа,
ударения, переписывание или language-dependent TN запрещены. Это покрывает `46` Denis строк с
NBSP и `18` с trailing whitespace без молчаливого post-selection rewrite.

Wrapper:
[`src/kds/data/voxcpm2_text_only.py`](../src/kds/data/voxcpm2_text_only.py), SHA-256
`3dcc290594a6af2670203b1dfd9ff500b96dbaf425b5ebe21011abfe57f12cbd`.
Model contract:
[`configs/research/voxcpm2_official_text_only_v1_models.json`](../configs/research/voxcpm2_official_text_only_v1_models.json),
SHA-256 `544a2ad4df100c5e39b76ca92dcd4aafe9150de2139e7bca608435e32b0a9168`.

## Isolated runtime и единственный smoke

Официальный `uv.lock` SHA-256
`fc066d21d09656c5060892baad096c53af6774c0947fad5bf6c676ea73c47c9b` установлен из exact Git
checkout того же source commit в отдельный Python `3.12.13`. Current `uv 0.12.3` требует для
этого legacy lock режим `--frozen`: `--locked` хотел бы обновить metadata из-за malformed
specifier одной upstream dependency, а менять официальный lock запрещено. Установлены `160`
unique distributions; fingerprint
`60158bb4e2dd9dbef6a0defdf517b98b3c5df21811af13e0b0a48c25de1e5779`.

Первый launcher загрузил модель, но остановился до входа в upstream `_generate`: wrapper передал
`streaming=False`, а public `VoxCPM.generate` добавляет этот keyword сам. Python выдал `TypeError`
при binding аргументов; stochastic generation/WAV не было. Ошибка не скрыта:
[`data/licenses/voxcpm2_official_v1_cuda_smoke_pre_inference_failure_v1.json`](../data/licenses/voxcpm2_official_v1_cuda_smoke_pre_inference_failure_v1.json),
SHA-256 `d803afad53782aeb38be2b29ea6182c44652cea5692cd715dd094b3efc98ff41`.
Исправление только удалило дублирующий API-owned keyword; seed/model/text/parameters не менялись.

После correction выполнен первый и единственный фактический generation call:

| Проверка | Результат |
| --- | ---: |
| Runtime | `torch 2.10.0+cu128`, CUDA `12.8`, RTX 5060 Ti, BF16 |
| Network | `bwrap --unshare-net` + offline env + socket guard; `0` attempts |
| Non-candidate screen | `0` literal/canonical matches среди `1,150` Denis texts |
| Model load | local-only, no denoiser/LoRA; `14.060360` s |
| Generation calls | `1`, fixed seed `20260814`; `3.383638` s |
| Reference/prompt audio/text | `None / None / None` |
| Normalizer / denoiser / retry | `false / false / false` |
| Output | mono PCM-16 WAV, `48,000 Hz`, `161,280` frames / `3.360000` s |
| WAV bytes / SHA-256 | `322,604` / `4678b294…a25c` |
| Peak / RMS | `0.641680241 / 0.126388862` |
| Peak CUDA allocated/reserved | `5,606,945,792 / 5,758,779,392` bytes |

Smoke receipt:
[`data/licenses/voxcpm2_official_v1_cuda_smoke_v1.json`](../data/licenses/voxcpm2_official_v1_cuda_smoke_v1.json),
SHA-256 `0c24a8325d5c1159b2ac2885ebb46d8e38386c984d6908822170a442ea3d6982`.
WAV остаётся в ignored `models/` и не входит в Git. Это technical format/load evidence, не
listening, Russian-intelligibility, acoustic-quality или identity evidence.

## Решение и следующий gate

Artifact/source/history/runtime/smoke gate **пройден**. Старое предположение `Python <3.13`
удалено: exact source требует `>=3.10`, а фактический runtime закреплён на `3.12.13`.

Denis metadata selection уже заморожен: target `79`, category balance `27/26/26`, ranking без
duration/audio-quality/model signals. Bona-fide decode/QA/VAD оставил `64` ready rows и `15`
`insufficient_speech` rejects без backfill. Отдельный
[immutable 64-row literal/canonical/NFKC binding](denis_1_0_mdc_voxcpm2_pre_qa_text_binding_v1.md)
и one-attempt synthesis contract уже завершён; receipt SHA-256
`943a9595968996f29da1a13f213e28419fc2c7b5215df790e4d4c440528f2b7b`. Единственный frozen
offline synthesis run затем дал `64/64` raw WAV при одном model load, `0` network attempts и без
retry/resynthesis/backfill. Один normal synthetic decode/QA/VAD pass сохранил `53` rows и
отклонил `11` `insufficient_speech`; minimum `60` не достигнут. Technical receipt закрепил
`stop_below_minimum_60`, поэтому pair/review/exposure/evaluation/inference gates для этого route
не открываются. Подробности:
[synthesis/technical-QA receipt](denis_1_0_mdc_voxcpm2_pre_qa_synthesis_and_technical_qa_v1.md).
