# Official OpenBMB VoxCPM2 — artifact/source gate, 14 августа 2026

**Статус:** exact model/source artifacts и project-history novelty gate завершены. Runtime
environment, CUDA model load и text-only smoke ещё не выполнены. Никаких candidate texts,
synthesis или detector inference в этом этапе не было.

**Допустимая будущая роль:** personal-research external source/generator-family holdout для
Denis 1.0. Это не подтверждает отсутствие Denis/аналогичных записей в training data модели,
идентичность default voice или speaker independence.

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
`4f9ad38da0ccbda4b97d263ee0c382d91b11edfceb9fea87d0a4229a79c32cde`.
Model contract:
[`configs/research/voxcpm2_official_text_only_v1_models.json`](../configs/research/voxcpm2_official_text_only_v1_models.json),
SHA-256 `23e9fd73c0b65bfe3ab650c617d4ca4e2385e24b56152968c97ebbc0435e0472`.

## Решение и следующий gate

Artifact/source/history gate **пройден**. Старое локальное предположение `Python <3.13` удалено:
точный current source требует `>=3.10`. Для воспроизводимости всё равно предпочтителен отдельный
CPython 3.12; доступен system-managed Python `3.12.13`, но environment/dependency lock ещё не
создан и не считается готовым.

Следующий безопасный этап — отдельно создать isolated runtime из exact source commit, закрепить
dependency lock и выполнить один **non-candidate** Russian text-only CUDA smoke при физически
заблокированной сети. Smoke обязан доказать local load, `48 kHz` output и фактические null/false
parameters; detector inference запрещён. Только после успешного smoke допустим отдельный frozen
Denis metadata selection contract.
