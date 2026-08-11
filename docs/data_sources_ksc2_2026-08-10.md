# KSC2 — архивная проверка и ограничения intake

**Статус:** source audit завершён 10 августа 2026 года. KSC2 допустим только как
проверенный personal-research **bona-fide candidate**. Ни manifest, ни TTS-derivative,
ни final evaluation из него пока не созданы.

## Права и provenance

Официальная страница ISSAI прямо указывает Creative Commons Attribution 4.0 и
требование атрибуции для коммерческого применения. Она также подтверждает, что KSC2
включает прежние KSC и KazakhTTS2, а новые компоненты происходят из TV, radio,
senate и podcasts. Hugging Face metadata в pinned revision
`cececbec1049f93f34a7421552500da01971ead8` по-прежнему показывает `mit`; это
противоречие сохранено в audit. Для project ledger использована лицензия первичного
издателя ISSAI — `CC-BY-4.0`, но текущий scope остаётся personal research и не
делает источник product-ready.

Исходный archive не перемещался из `/home/ruslan/Downloads/KSC2/`. Полный
машиночитаемый receipt находится в
`data/licenses/ksc2_v1_artifact_lock.json`.

## Read-only integrity audit

Все десять последовательных частей были проверены одним streaming-проходом без
конкатенации или extraction:

- compressed size: `80 809 122 212` bytes;
- SHA-256 последовательности частей:
  `43d1ee6725d737a438125a13997a0abde4159de84ef17d1706fe7921e8632cbe`;
- gzip CRC и TAR layout прошли; archive содержит только directories и regular files,
  без links или иных опасных member types;
- `645 860` FLAC и `645 861` transcript `.txt` обнаружены;
- у всех аудио есть paired transcript. Единственный лишний transcript —
  `Train/radio/01_69_226.txt`; он не может быть выбран будущим intake;
- `5 044` TV-news пар используют необычные, но согласованные имена
  `*.flac.flac` / `*.txt.txt`. Audit нормализует повторный suffix до logical ID,
  поэтому валидные пары не отбрасываются ложно.

## Компоненты и conservative exclusion

В исходном `Test` есть как минимум следующие новые, явно отделённые component paths:

| Component | Paired bona-fide candidates |
| --- | ---: |
| `parliament` | 773 |
| `podcasts` | 1 547 |
| `radio` | 702 |
| `talkshow` | 383 |
| `tv_news` | 2 618 |

Это `6 023` paired candidates до проверки text overlap и QA/VAD.

`crowdsourced` исключён консервативно из каждого source split: KSC2 subsumes
original KSC, а archive не содержит отдельного component-provenance manifest, который
мог бы доказать отсутствие overlap. Весь `tts` component также исключён. Будущий
intake обязан работать только с перечисленными new components, требовать exact
FLAC/text pair и исключать every existing KSC-derived text/sample/asset key до
публикации output.

## Повторная проверка code-switching

Проверен первоисточник — [статья KSC2 в Interspeech
2022](https://www.isca-archive.org/interspeech_2022/mussakhojayeva22_interspeech.pdf),
а также [официальный training recipe](https://github.com/IS2AI/Kazakh_ASR). Поправка
важна: авторы действительно прямо пишут, что наибольшая доля code-switching найдена
в podcasts, television programs и radio programs. В Table 1 приведены следующие
**агрегированные** rates для test split:

| Название в статье | Path в локальном archive | Test CS rate в статье | Роль для mixed intake |
| --- | --- | ---: | --- |
| Podcasts | `Test/podcasts` | 8,6 % | первый приоритет |
| Television programs | `Test/talkshow` | 6,3 % | первый приоритет |
| Radio programs | `Test/radio` | 3,1 % | первый приоритет |
| Television news | `Test/tv_news` | 1,3 % | не относится к группе с наибольшими rates |
| Parliament | `Test/parliament` | 2,5 % | не относится к группе с наибольшими rates |

То есть `talkshow`, а не `tv_news`, является архивным соответствием television
programs из статьи. Авторы определяли rate как долю предложений со словами обоих
языков; они отдельно оговорили, что казахское слово с русским суффиксом считали
русским, а русское с казахским суффиксом — казахским. Следовательно, paper даёт
обоснованный порядок **кандидатных component paths**, а не лишь общее утверждение
о существовании code-switching.

Однако release не содержит per-row результата этого подсчёта: в TAR есть только
paired audio/transcript files, а официальный GitHub recipe содержит ASR recipe и
инструкцию скачивания, но не CSV/TSV с языковой разметкой или опубликованный
word-level classifier. Поэтому нельзя воспроизвести именно авторский результат для
конкретной строки без дополнительной, явно зафиксированной процедуры. В частности,
нельзя ставить `language=mixed` или `code_switch=true` только по component path,
кириллице либо самодельной word heuristic.

### Выбранный путь к узкому mixed evidence layer

Packet остаётся неизменяемым источником paired `Test/podcasts`, `Test/talkshow` и
`Test/radio` candidates с archive lock, transcript и audio SHA-256. Вместо широкого
автоматического LID выполнен узкий **single-AI semantic transcript review**: в source control
внесён только явно проверенный positive list, а для каждой опубликованной строки сохранены
русские и казахские token positions и соответствующие surface tokens.

Это не character/script heuristic, component-path label, ASR transcription или external LID
score. Все не вошедшие строки, включая неоднозначные русские заимствования и гибридные формы,
по-прежнему `unknown`. Таким образом, `language=mixed`, `code_switch=true` получают только
строки с явными evidence обоих языков, но этот слой имеет single-AI provenance и не выдаётся за
complete corpus annotation.

Для подготовки packet добавлен `scripts/prepare_ksc2_mixed_annotation.py`. Он повторно
stream-читает все parts, сверяет общий archive SHA-256 с source lock, извлекает только
эти три paths и публикует отдельный CSV/receipt только после полной проверки. В его
output намеренно нет режима, который мог бы автоматически присвоить `mixed`:
каждая строка имеет `annotation_state=pending`, `language=unknown`,
`code_switch=unknown`. `tv_news` и `parliament` script не принимает как входной
component selection.

### Опубликованный packet v1

Первый run завершён на локальном неизменённом archive. Он повторно подтвердил global
archive SHA-256 `43d1ee6725d737a438125a13997a0abde4159de84ef17d1706fe7921e8632cbe`
и опубликовал `2 632` FLAC assets (`283 MB`): `1 547` podcasts, `383` talkshow и
`702` radio. Ни одна строка не получила язык или code-switch label.

| Artifact | SHA-256 | Смысл |
| --- | --- | --- |
| `data/manifests/ksc2_test_mixed_annotation_v1.csv` | `5b80ee0b6d9d80907ed77a9c1e21821a6324e9621f3cb96ad1daab17982acc20` | immutable pending candidates, transcripts и audio hashes |
| `data/licenses/ksc2_test_mixed_annotation_v1_receipt.json` | `64925d25231ecde944ffb934a0d1e7f0fecdac37c8d1923e38d6bd7ec71b802d` | archive lock, component counts и rule for still-unlabelled candidates |
| `data/licenses/ksc2_test_mixed_annotation_v1_packet_lock.json` | `c4d731de6b8b772f7c0c1f5e70be3edde668f1e065bc730472cc4e3a8491fe25` | machine-readable packet → receipt → source-lock chain |

После publication независимо сверены CSV/receipt fields, uniqueness `annotation_id`,
наличие и SHA-256 каждого из `2 632` extracted FLAC. Их immutable original не редактируется.
Machine-readable lock закрепляет packet, receipt и source lock. Отдельный publisher
`scripts/publish_ksc2_ai_mixed_review.py` воспроизводит лишь explicit review list и никогда не
применяет language inference к непросмотренным rows. Полученный evidence не является training
manifest и не закрывает binary mixed final protocol без QA/VAD, overlap audit и независимой
spoof половины. Полная процедура и source review — `docs/ksc2_mixed_ai_review_v1.md`.
