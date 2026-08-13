# Denis 1.0 / Mozilla Data Collective — source intake, 14 августа 2026

**Статус:** completed read-only source-level archive intake и source-wide project-exposure
screen. Это ещё не frozen selection, raw/ready manifest, VoxCPM2 artifact lock, synthesis,
acoustic review или detector inference.

**Допустимая роль:** personal-research external holdout only. Источник single-speaker и имеет
вероятную historical speaker-lineage экспозицию; speaker-disjoint, speaker-independent и
speaker-robust claims запрещены.

## Exact artifact и права

Проверен локальный архив, вручную полученный владельцем вне репозитория:

| Поле | Значение |
| --- | --- |
| Source ID | `denis_1_0_mdc` |
| Publisher / steward | Mozilla Data Collective / Open Home Foundation |
| Browser download name | `1764973737766-ru_RU-denis.tar.gz` |
| Source-card archive name | `denis-1-0-3f60c388.tar.gz` |
| Exact bytes | `109,594,943` |
| SHA-256 | `75e2c63c5082df7623c6a98c529718b22015dfbd2d38a1ea328635f4dd4ccf9b` |
| License on current source card | `CC0-1.0` |
| Embedded license file | нет |
| Project scope | personal research only; no re-identification/re-hosting |

Различие browser-assigned имени и имени в карточке не считается подменой: идентичность этого
intake задают exact bytes и SHA-256. Но archive сам не содержит license file, поэтому CC0 и
rights basis опираются на текущую MDC dataset card и provider terms, а не на embedded notice.
Provider terms требуют от поставщика необходимых прав, permissions и consents; это contractual
warranty, но не независимый consent audit.

Raw archive, аудио и тексты не добавлены в Git. Versioned aggregate receipt:
[`data/licenses/denis_1_0_mdc_artifact_audit_receipt.json`](../data/licenses/denis_1_0_mdc_artifact_audit_receipt.json),
SHA-256 `bcc4e0852b0ebeab0e0ba33f0e3ee5b11903f399a62c5442614686f5b47b489b`.

## Безопасный состав и фактический формат

Audit проверяет expected size/hash до TAR parsing, полностью проходит gzip/TAR stream, отклоняет
traversal, links, special files, duplicate и case-fold duplicate paths, затем связывает каждый
текст с одноимённым audio member. Ни один member не извлекается на диск.

| Проверка | Фактический результат |
| --- | ---: |
| TAR members / regular / directories | `2,304 / 2,300 / 4` |
| Gzip CRC / uncompressed TAR bytes | `pass / 111,523,328` |
| Exact `.txt` / `.webm` pairs | `1,150 / 1,150` |
| Orphans / unsafe / duplicate paths | `0 / 0 / 0` |
| General / Chat / CustomerService records | `550 / 300 / 300` |
| Literal / whitespace-canonical / NFKC-canonical unique texts | `1,150 / 1,150 / 1,150` |
| Texts with NBSP / trailing whitespace | `46 / 18` |
| Fully decoded audio / failures | `1,150 / 0` |
| Filename suffix | `.webm` у `1,150` members |
| Actual decoded container / codec | `OGG / OPUS` у `1,150` members |
| Sample rate / channels | `48,000 Hz / 2` у `1,150` members |

Источник и filenames называют формат WebM, но payload начинается как Ogg и декодируется как
Ogg/Opus. Это единообразная source-format mislabel, не повреждение. Будущий materializer обязан
сохранить suffix/container disclosure и не переименовывать provenance задним числом.

## Длительности и облегчённое правило

Поставщик не публикует отдельную duration table. Проект не требует такую таблицу, если duration
можно воспроизводимо получить из byte-pinned audio: intake полностью декодирует каждый payload и
считает `decoded PCM frames / 48,000` после codec pre-skip. Это единственное дополнительное
облегчение правил, которое понадобилось фактически; права, source reuse, reference audio,
retry/backfill и post-result tuning не ослаблены.

| Метрика | Результат |
| --- | ---: |
| Total decoded frames | `322,534,320` |
| Total decoded duration | `6,719.465000` s (`1.8665` h) |
| Min / median / max | `2.333500 / 5.233500 / 19.313500` s |
| Rows `>=2.5` s / `<2.5` s | `1,143 / 7` |

`1,143` технически декодируемых строк выше прежнего duration floor дают большой запас над
minimum `60` и target `79`. Это pre-QA feasibility, не `ready`: VAD, waveform QA, text
compatibility и acoustic review ещё не выполнялись. Снижать минимум ниже `60` не нужно.

NBSP и trailing whitespace нельзя молча нормализовать после selection. До freeze нужен отдельный
metadata-only text-compatibility rule: либо literal text проходит pinned VoxCPM2 wrapper при
`normalize=False`, либо строка заранее исключается. С `1,150` уникальными текстами это не требует
backfill по результатам detector/TTS.

## Project exposure и speaker lineage

Source-wide screen сравнил все `1,150` candidate identities с `34` research configs, `19`
уникальными configured manifests / `12,555` rows и полным inventory из `95` manifests /
`40,682` rows. Сравнение выполнено по будущему `sample_id`, exact audio SHA-256 и трём вариантам
text SHA-256: literal, whitespace-canonical и NFKC+whitespace-canonical.

| Scope | sample ID | audio SHA-256 | literal text | whitespace text | NFKC text |
| --- | ---: | ---: | ---: | ---: | ---: |
| Configured roles | `0` | `0` | `0` | `0` | `0` |
| Full manifest inventory | `0` | `0` | `0` | `0` | `0` |

По direct artifacts Denis является новым human source, и все `1,150` записей переживают strict
single-speaker group exclusion. Immutable screen:
[`data/manifests/denis_1_0_mdc_source_exposure_screen_v1.json`](../data/manifests/denis_1_0_mdc_source_exposure_screen_v1.json),
SHA-256 `05ad13c4ed9a01825b0a8b9226e262afa91471df48440ad0258b0bc672306c24`.

Предыдущее web-only review учитывало только `7` v2 rows. Полный current manifest screen нашёл
`12` уникальных historical `piperTTS / ru_RU-denis-medium` sample ID: `11` в когда-либо
configured train и `1` в dev. Inventory содержит `24` raw/ready manifest rows, но это те же `12`
sample ID: по `5` raw/ready v1 и по `7` raw/ready v2.

Source card связывает Denis с доступным Piper voice, а официальный Piper card связывает
`ru_RU-denis-medium` с OHF voice data. Cryptographic archive-to-checkpoint binding нет, поэтому
это маркируется `likely exposed fail-closed`, а не доказанное byte-level совпадение. Этого
достаточно, чтобы запретить speaker independence; direct source novelty при этом остаётся
подтверждённой.

## Решение и следующий gate

Denis source gate **пройден** для ограниченного маршрута:

> new direct human source; single speaker; likely historical speaker-lineage exposure; not
> speaker-disjoint, speaker-independent or speaker-robust; personal research only.

Не нужно искать «идеальный» готовый набор или уменьшать target: archive даёт достаточный
лексический/domain запас, хотя не speaker diversity. Будущий selection может распределять строки
между General/Chat/CustomerService, но это не заменяет diversity людей и не меняет claim.

Следующий отдельный этап — только artifact/runtime intake official OpenBMB VoxCPM2: повторно
разрешить full revision, скачать snapshot вне Git, закрепить все sizes/SHA-256, Python 3.12
environment, проверить `audiovae.pth` safe load, tokenizer code и offline wrapper. Пока запрещены
frozen text selection, raw materialization, smoke, candidate synthesis и detector inference.
