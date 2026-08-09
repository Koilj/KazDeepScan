# Проверка следующих источников данных — 9 августа 2026

Это начальный read-only review до скачивания. После него Common Voice Russian v24, ML-DF
Italian и один допустимый RuASD shard были локально верифицированы и внесены в
`data/licenses/license_ledger.csv` с их фактическими scope. Ни один OOD source из этого файла
не должен использоваться в B0 train/dev/test.

## Russian bona-fide candidate: Common Voice Scripted Speech 24.0 — Russian

- Официальная datasheet: <https://dev.mozilladatacollective.com/datasets/cmj8l8ct700o5nlovbdnv58yr>.
- Указаны CC0-1.0, MP3 archive `mcv-scripted-ru-v24.0.tar.gz` размером 6.53 GB и выпуск
  5 декабря 2025 года.
- Datasheet заявляет 201 326 клипов, 290.23 часа записи (250.48 часа validated), 3 636
  speakers и официальные modeling split: train 26 721, dev 10 253, test 10 253.
- Ограничения страницы: нельзя пытаться установить личность диктора и нельзя перехостить или
  повторно распространять dataset. Указанная intended use — ASR, CALL и языковая поддержка.

**Статус: `owner_authorized_personal_research`, локально верифицирован.** Владелец проекта
разрешил этот источник только для личного обучения без отдельного письменного scope
confirmation. Archive размером `7 008 716 262` байта прошёл gzip CRC; его SHA-256 —
`9a2ed32a0574f74f505cd7740a599f0b9edc9f52ba1e7d6624b66f258db4c0ea`.
Подробности intake — в `docs/data_sources_common_voice_ru_v24.md`. Это внутреннее решение не
расширяет CC0 или условия datasheet: сохраняются запреты на идентификацию дикторов и
re-hosting/re-sharing.

## Spoof candidate: ASVspoof 2021 Logical Access evaluation

- Официальная запись: <https://zenodo.org/records/4837263>.
- Она описывает пары bona-fide/spoof, созданные TTS/VC и переданные с telephony/VoIP effects;
  доступный archive evaluation set имеет размер 7.8 GB.
- В разделе Rights официальной записи поле License пусто.

**Статус: `rejected_missing_license`.** Открытая ссылка на файл не является правом на
коммерческое обучение. Не скачивать и не заносить в ledger, пока правообладатель не даст
явную применимую лицензию.

## Spoof candidate: ASVspoof 5

- Официальная запись: <https://zenodo.org/records/14498691>.
- Dataset предназначен для spoof/deepfake detection, но полный объём указан как 142.3 GB и
  язык в metadata — English.
- Его `LICENSE.txt` — ODC Attribution License:
  <https://zenodo.org/records/14498691/files/LICENSE.txt?download=1>. Она прямо говорит, что
  лицензирует database rights, но не самостоятельные права на individual contents и советует
  отдельно очистить такие права.

**Статус: `rejected_content_rights_unclear`.** Для коммерческого проекта такой уровень
правовой определённости недостаточен. Нужны отдельная документация о правах на каждый audio
content либо письменное разрешение правообладателя. Даже после него ASVspoof 5 остаётся
англоязычным OOD/cross-lingual ресурсом, а не заменой Russian/Kazakh spoof data.

## Russian anti-spoof candidate: RuASD

- Официальный dataset card: <https://huggingface.co/datasets/lab260/RuASD>.
- Card указывает `CC-BY-NC-SA-4.0`, русский язык, bona-fide и TTS/voice-cloning rows от
  37 Russian-capable systems; полный repository — `250 GB`.
- Каждый проверенный file имеет собственный published SHA-256. Shard `ruasd-000000.tar`
  размером `999 813 120` байт прошёл local size/SHA-256/TAR/JSON-WAV-pair verification.

**Статус: current shard `verified`, full release audited but not protocol-eligible.** Shard
`000000` contains `985` raw fake TTS rows (ElevenLabs and TeraTTS) and no bona-fide rows. It
remains a Russian fake-only OOD stress set:
`data/manifests/ruasd_ru_v1_shard000000_ood_100.csv`; non-commercial / share-alike terms
remain in force. Details are in `docs/data_sources_ruasd_shard000000.md`.

The owner has now placed the full 250-artifact release in `~/Downloads/RuASD`. The pinned
official size/SHA-256 catalog and safe metadata audit are in
`docs/data_sources_ruasd_full_v1.md`. All 250 archive SHA-256 values matched the official
catalog, and the audit found `585,353` JSON/WAV pairs. Every raw bona-fide row has unknown
`speakers`, as do almost all raw fake rows. Thus the full download is not a verifiable
speaker/voice-disjoint binary protocol and does not authorize B0 train/dev/test or calibration.
It may only extend a labelled research/OOD study.

## Russian binary candidate: PyAra

- Официальная API metadata: <https://www.kaggle.com/datasets/alep079/pyara>.
- Указаны русский bona-fide/spoof WAV, `CC BY-NC-SA 4.0`, label и algorithm columns.
- Published total size: `38 661 781 089` байт.

**Статус: `owner_authorized_personal_research`, локально проверен.** Владелец проекта вручную
разместил `archive.zip` (`28 092 611 663` байта) в `/home/ruslan/Downloads`. ZIP CRC и SHA-256
`dadf5b795adbd6d635e74f4f9662c3e9a425c88bd76f26731f9e6adbad278b91` проверены, source
внесён в ledger. Созданы raw/ready personal-research manifests и B0 smoke baseline; подробности
в `docs/data_sources_pyara_ru_v7.md`.

PyAra не предоставляет speaker IDs. Поэтому local train/dev/test предотвращает text leakage,
но не доказывает speaker independence. Он не может служить product benchmark, calibration basis
или основанием для external claims; некоммерческие условия `CC-BY-NC-SA-4.0` сохраняются.

## Следующее безопасное действие

Нужен target-language binary protocol с bona-fide и spoof rows, совпадающими по source/channel
либо с явно спроектированной независимой evaluation схемой. До его появления API не выдаёт
score, а ML-DF и RuASD остаются только OOD stress data.
