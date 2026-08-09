# RuASD Russian Anti-Spoofing Dataset — shard `000000`

## Проверенный локальный artifact

- Официальный dataset card: <https://huggingface.co/datasets/lab260/RuASD>.
- Зафиксированный revision: `fcbc87c57b54ef4f58e1135e2813f6f000c2b739`.
- Artifact: `ruasd-000000.tar`.
- Размер: `999 813 120` байт.
- SHA-256: `956efb0e1281ada0dcee6f2ed9498c454552be88b3e9784e52e70c3ef4dfcd67`.
- Лицензия dataset card: `CC-BY-NC-SA-4.0`; scope проекта — только personal research.

Во время публикации этого small OOD slice в проекте был только shard `000000`: загрузка
второго shard тогда была остановлена консервативным лимитом 2 GB. Позднее владелец вручную
разместил полный 250-artifact release в `~/Downloads/RuASD`; все его SHA-256 сверены с
pinned official catalog. Это не меняет назначение данного manifest-а: full metadata audit
показал, что bona-fide speaker IDs неизвестны, поэтому полная коллекция всё равно не может
быть заявлена как speaker/voice-disjoint binary protocol. См.
`docs/data_sources_ruasd_full_v1.md`.

## Состав shard и применение

Проверенный shard содержит ровно `985` пар прямых JSON/WAV members; symlink, hardlink,
вложенные пути, дубликаты и непарные files intake отклоняет. Все `985` rows имеют
`label=fake`, `group=raw`, `source_type=tts`: `306` ElevenLabs и `679` TeraTTS.

Следовательно, shard не является binary corpus и **не может** участвовать в B0
train/dev/test, calibration или product scoring. Он полезен только как Russian fake-only
OOD stress set. Отдельный reproducible slice:

- `data/manifests/ruasd_ru_v1_shard000000_ood_100.csv`;
- `100` WAV: по `50` детерминированных examples ElevenLabs и TeraTTS;
- audio в `data/raw/ruasd/slices/ood-100/`;
- все rows имеют `split=ood`, `label=spoof`, `language=ru`; family и source generator
  сохранены, а нераскрытые source version/voice явно записаны как unknown/unspecified.

`original_sr` остаётся фактической частотой source asset (проверенный example: mono
PCM S16LE, `44 100` Гц). Это raw OOD manifest; преобразование в 16 kHz, если оно станет
нужно в отдельном evaluation protocol, выполняется только через существующий проверяемый
preprocessing workflow и новый output manifest.

## Воспроизведение

Команда не перезаписывает существующие outputs, поэтому здесь приведены новые пути:

```bash
uv run python scripts/ingest_ruasd_ood.py \
  --archive data/raw/ruasd/ruasd-000000.tar \
  --output-manifest data/manifests/ruasd_ru_v1_shard000000_ood_100_repro.csv \
  --slice-name ood-100-repro

kds validate-manifest data/manifests/ruasd_ru_v1_shard000000_ood_100_repro.csv \
  --license-ledger data/licenses/license_ledger.csv \
  --require-ood-generator
kds validate-assets data/manifests/ruasd_ru_v1_shard000000_ood_100_repro.csv --audio-root data
```

Archive и extracted audio не попадают в Git.
