# Denis 1.0 × official VoxCPM2 — immutable text binding v1

**Статус:** completed pre-synthesis write-once gate. Ровно `64` Denis ready texts связаны с
exact model/runtime и будущими synthesis/QA paths до первого candidate WAV. Synthetic audio,
acoustic review, pairing и detector inference не выполнялись.

## Frozen inputs

Gate повторно проверил:

- exact Denis archive: `109,594,943` bytes, SHA-256
  `75e2c63c5082df7623c6a98c529718b22015dfbd2d38a1ea328635f4dd4ccf9b`;
- immutable `79`-row selection и `64`-row bona-fide ready manifest SHA-256
  `ae71cca7dcc2854cccca67565f81a0696acc665a6648b9889f5d9abd267891d8`;
- materialization receipt SHA-256
  `c36fc8bcc60c16d5d2493c4bf8b77719f32ca3d9da9ba15d51054b9ee16d5386`:
  target `79`, actual ready `64`, `15` rejects, no reuse/replacement/backfill;
- official VoxCPM2 model lock SHA-256
  `544a2ad4df100c5e39b76ca92dcd4aafe9150de2139e7bca608435e32b0a9168`,
  artifact/source receipt, project-history receipt и единственный completed CUDA smoke;
- frozen wrapper SHA-256
  `3dcc290594a6af2670203b1dfd9ff500b96dbaf425b5ebe21011abfe57f12cbd`.

Для каждой ready row transcript повторно прочитан только из exact TAR member. Receipt хранит
selection rank, source member identity, `text_id`, literal/collapse-whitespace/NFKC hashes,
UTF-8 byte lengths и ready-audio hash, но не plaintext. Все `64` literal/canonical bindings
совпали; canonical inputs имеют `57–353` UTF-8 bytes и укладываются в frozen `max_len=4096`.
Category counts неизбежно следуют bona-fide QA: General `23`, Chat `17`, CustomerService `24`.

## One-attempt contract

До synthesis закреплены следующие инварианты:

- один model load и ровно одна generation call на каждый из `64` bound texts;
- fixed seed `20260814`, `cfg_value=2.0`, `inference_timesteps=10`, `min_len=2`,
  `max_len=4096`;
- передаётся только заранее hash-bound collapse-whitespace text;
- `prompt_wav_path`, `prompt_text`, `reference_wav_path`, LoRA и denoiser — `null`/disabled;
- `normalize=false`, `denoise=false`, `retry_badcase=false`; resynthesis запрещён;
- replacement, reselection и backfill запрещены независимо от generation/QA outcome;
- isolated CPython 3.12, `bwrap --unshare-net`, offline environment и Python socket guard;
- detector/metric access во время generation запрещён;
- ожидаемый raw output — mono PCM-16 WAV `48 kHz`; default voice identity остаётся unknown.

Exact frozen program hashes:

| Program | SHA-256 |
| --- | --- |
| binding runner | `ab50c3d973a05db7d6f35ac2d0e962756843b5a645c5b4022e4418c51133d45b` |
| synthesis runner | `fdf33a572dc7f4b69fe4bf0fdfa64e924b2844e2c175f64cf6873c2f48db2346` |
| normal preprocess runner | `14862f32debe836408d095f4fa7db3d1e54beabee6673f77213fa0330a8470e0` |
| technical-QA publisher | `596d619482227fe5ddba12bfdcbb513f95c8f62921408c6ca7fb2cb5bf49b1fd` |

[Binding receipt](../data/manifests/denis_1_0_mdc_voxcpm2_official_pre_qa_text_binding_v1.json)
имеет SHA-256 `943a9595968996f29da1a13f213e28419fc2c7b5215df790e4d4c440528f2b7b`;
row fingerprint — `b28d1ff99bc50b5dc6879b75a7dee018cef3a0767508cfde2fc660f9156204c0`.

## Claims и следующий безопасный этап

Маркировка не расширяется:

> external human-source / generator-family holdout candidate; TTS training-data overlap
> unverified; likely historical Denis speaker-lineage exposure; single bona-fide speaker;
> not speaker-independent or speaker-robust; default synthetic voice identity unknown;
> personal research only.

Следующий безопасный этап — выполнить один committed runner в frozen offline runtime, завершить
все `64` attempts без retry/backfill, затем ровно один normal decode/QA/VAD pass над полученными
raw WAV и опубликовать technical-QA receipt. Detector inference остаётся запрещён до exact pair
lock, двух независимых full-asset acoustic/language reviews, current project-exposure audit и
отдельного write-once evaluation contract.
