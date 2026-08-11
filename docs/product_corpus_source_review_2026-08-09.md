# Product corpus source review — 9 August 2026

> Архивный review: с 9 августа 2026 текущий scope KazDeepScan изменён на personal research.
> Этот документ не блокирует research use источников с соблюдением их лицензий; он остаётся
> полезным только если scope когда-либо снова станет коммерческим/product.
> Владелец также исключил запись голосов людей: раздел acquisition route ниже исторический и
> не является текущим поручением или рекомендацией.

## Decision

No publicly checked release can yet be entered in the product ledger as a complete Russian or
Kazakh binary corpus. The only positive external lead is **KazakhTTS2**, but it is a
**conditional bona-fide component**, not a complete product protocol. The machine-readable
review is [product_corpus_candidates_2026-08-09.csv](../data/licenses/product_corpus_candidates_2026-08-09.csv).

This is a deliberate stop condition: a public CC license or a column called `speaker_id` is not
enough evidence of product-safe voice rights or of a speaker/voice-disjoint binary protocol.

## Candidate results

### KazakhTTS2 — conditional bona-fide component only

The official [ISSAI repository](https://github.com/IS2AI/Kazakh_TTS/tree/fc906048ff5914a3528d1ae4ed6f7ccd94d71383)
ships a CC BY 4.0 license. The accompanying
[paper](https://aclanthology.org/2022.lrec-1.578.pdf) describes five professional speakers,
271.7 hours, their consent process, and a separate audio/transcript directory for each speaker.
The official Hugging Face snapshot is pinned to
[`821c7d583331fd2ae50283a00147772ef0734fd2`](https://huggingface.co/datasets/issai/KazakhTTS/tree/821c7d583331fd2ae50283a00147772ef0734fd2).

It is not a binary anti-spoofing release: there are no fake labels or verified spoof voice
groups, and five speakers are insufficient evidence for a robust speaker-independent product
benchmark. The release is also larger than the project's 2 GB auto-download limit: six official
parts total 35,730,236,270 bytes. Do not download it automatically.

Before a bona-fide-only intake, obtain written clarification from the rightsholder that covers
the intended commercial detector-training use and the planned handling of voice/personality
rights. A public CC BY text cannot supply a warranty that every third-party right is cleared.
No external message has been sent by this project.

### Excluded candidates

- **MCSKL Module 1:** a publication page describes CC BY, while the project's data-management
  plan identifies the audio/transcripts as CC BY-NC-SA. The primary data-rights position is
  therefore unresolved and must be treated as non-commercial until the rightsholder confirms
  otherwise.
- **YO-CPT-ru:** its data card says that cross-video speaker identifiers are produced by face and
  voice clustering from YouTube-derived data. They are useful technical metadata, but not
  verified consent groups for a product voice corpus.
- **SpeechFake:** its public documentation establishes a multilingual fake-data project, but not
  a released Russian/Kazakh subset with verified per-voice groups and an audited target-language
  rights chain.

Existing PyAra and RuASD remain research-only for the reasons documented in their source notes.

## Required acquisition route

The viable path is a new, consented Russian/Kazakh binary corpus, optionally supplemented by a
separately cleared source such as KazakhTTS2 only after the written clarification above. Before
recording or generating any audio:

1. Keep signed agreements and identifying information in an access-controlled system outside
   this repository.
2. Maintain a local pseudonymous CSV in `data/consents/` and verify it with
   `kds validate-consent-registry`. The directory is Git-ignored by design.
3. Consent for a speaker whose voice can be used to make spoof samples must expressly cover
   product model training, synthetic derivatives, commercial deployment, withdrawal handling,
   and the exact collection version. A `speaker_pseudo_id` alone is not consent.
4. Lock speaker and verified spoof-voice groups into train/dev/test/OOD *before* model training;
   run `kds assign-splits --include-voice-id` only where `voice_id` is truly verified.
5. Register every actual source, archive hash, use policy, and group provenance in
   `data/licenses/license_ledger.csv`, then require
   `kds validate-training-protocol --purpose product` before B0/XLS-R training.

This is an engineering and provenance contract, not legal advice. A qualified reviewer must
approve the actual agreements and source terms before the product policy is marked allowed.
