# Denis 1.0 × official VoxCPM2 — one-shot synthesis и technical QA v1

**Статус:** write-once synthesis и единственный normal decode/technical-QA/VAD pass завершены.
Все `64` bound texts получили raw WAV, но только `53` прошли synthetic technical QA. Frozen
minimum `60` не достигнут; окончательный outcome — `stop_below_minimum_60`. Pair lock, acoustic
review, exposure audit и detector inference не выполнялись и этим receipt не разрешены.

## Frozen execution

Run использовал immutable 64-row binding SHA-256
`943a9595968996f29da1a13f213e28419fc2c7b5215df790e4d4c440528f2b7b`, official model
revision `bffb3df5a29440629464e5e839f4d214c8714c3d`, source revision
`ee8161e9e1b7b082cb5721a3a9980da4204401e6` и fixed seed `20260814`.

Фактические инварианты:

- один model load и ровно `64` generation attempts;
- `64` successes, `0` failures, `0` retry/resynthesis/replacement/backfill;
- `0` observed upstream network attempts внутри `bwrap --unshare-net` и socket guard;
- no reference/prompt audio or text, LoRA, voice cloning, denoiser или semantic normalizer;
- default synthetic voice identity остаётся `unknown_not_claimed`;
- raw assets: `64` mono PCM16 WAV `48 kHz`, `276.00` s total, range `1.76–11.52` s.

[Raw manifest](../data/manifests/denis_1_0_mdc_voxcpm2_official_pre_qa_raw_v1.csv) имеет `64`
rows и SHA-256 `45c8d5c9fb4d9f9bd9b5745add9b6e738111928b2b5c42a8779e030377195362`.
[Synthesis receipt](../data/manifests/denis_1_0_mdc_voxcpm2_official_pre_qa_synthesis_v1.json)
имеет SHA-256 `b827ba8208d4d44fdaeefaabeaa841355ed580aa253b261dead766a3a16ee83b`.
Raw WAV остаются ignored artifacts и не входят в Git.

## Единственный synthetic QA/VAD pass

Standard `AudioPreparationPipeline` выполнил decode, mono PCM16 WAV `16 kHz`, RMS/clipping/DC
checks и WebRTC VAD ровно один раз с `reused_rows=0`. Результат:

- ready: `53`, `251.52` s total, duration range `2.56–11.52` s;
- rejected: `11`;
- reason counts: `insufficient_speech=11`;
- reuse/retry/resynthesis/replacement/backfill: `0/false/false/false/false`.

[Ready manifest](../data/manifests/denis_1_0_mdc_voxcpm2_official_pre_qa_ready_v1.csv) имеет
SHA-256 `f90a634b80364a3a70046cf66354dbc7c11459f15a375b1e3a61c1f440e3028a`.
[Rejection report](../data/manifests/denis_1_0_mdc_voxcpm2_official_pre_qa_technical_qa_rejections_v1.json)
имеет SHA-256 `38c4da79e2bd0a50168fabb1817f866c6dacbbcd657c8ee18e6846a45e058ecb`.
[Technical-QA receipt](../data/manifests/denis_1_0_mdc_voxcpm2_official_pre_qa_technical_qa_v1.json)
имеет SHA-256 `ca46362313f50f79043dd559f8d739185b51d8cb0dc9dcc0f5dc659e5b02951c`.
Каждая из `64` raw rows учтена ровно один раз как ready или rejected.

## Решение и ограничения

Заранее зафиксированы minimum `60` и target `64` synthetic ready rows. Фактические `53` нельзя
превратить в допустимый layer снижением порога после outcome. Поэтому этот exact route завершён
защитной остановкой: rejected rows не пересинтезировать, selection не менять, backfill не делать,
pair lock и detector inference не создавать.

Маркировка artifacts остаётся узкой:

> external human-source / generator-family holdout candidate; TTS training-data overlap
> unverified; likely historical Denis speaker-lineage exposure; single bona-fide speaker;
> not speaker-independent or speaker-robust; default synthetic voice identity unknown;
> personal research only.

Следующий безопасный research step, если нужен новый evaluation layer, — отдельный genuinely new
human source и generator route с новым intake/exposure/selection contract до просмотра outcome.
Текущие `53` rows не являются разрешением на урезанный post-hoc evaluation.
