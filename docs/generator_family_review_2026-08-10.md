# Kazakh generator family review — 10 августа 2026

## Критерий

Family засчитывается только если это отдельная generation chain, а не дополнительный voice,
emotion, seed или sampler одного checkpoint. Для local intake нужны: казахская text-to-speech
поддержка, технически воспроизводимый local runtime, явные права, pinned bytes/revision и
суммарный download не более 2 GiB. Все варианты ниже остаются personal research, пока не
будет отдельного legal/product review.

| Family | Состояние | Проверенное основание | Решение |
| --- | --- | --- | --- |
| Piper neural TTS | уже в KSC-derived v1 | ONNX/VITS Piper voice, 6 profiles | первая family; новые Piper profiles не считаются новыми family |
| Meta MMS/VITS | уже в KSC-derived v1 | отдельный MMS Kazakh VITS checkpoint | вторая family |
| **KazEmoTTS Grad-TTS + HiFi-GAN** | **реализована** | official ISSAI model/source, CC BY 4.0 statement, 248 MB pinned archives, local GPU smoke | третья family; frozen v2 содержит 359 KSC/KazEmoTTS пар, no cloning |
| **Spark-TTS Kazakh LLM + BiCodec** | **реализована** | [model card](https://huggingface.co/ErnarBahat/Spark-TTS-Kazakh) declares Kazakh fine-tune and CC-BY-NC-SA-4.0; official [Spark-TTS](https://github.com/SparkAudio/Spark-TTS) is LLM + BiCodec, not VITS | четвёртая family; 1.861 GB pinned LLM+BiCodec/source bytes verify, controlled GPU runtime cannot load reference audio/wav2vec2, frozen v3 содержит 381 KSC/Spark-TTS pair |
| **eSpeak NG Kazakh formant** | **реализована** | official [language table](https://github.com/espeak-ng/espeak-ng/blob/4870adfa25b1a32b4361592f1be8a40337c58d6c/docs/languages.md) lists `kk`; [upstream](https://github.com/espeak-ng/espeak-ng/tree/4870adfa25b1a32b4361592f1be8a40337c58d6c) declares formant synthesis and GPL-3.0-or-later | пятая family; 29.2 MB source/runtime lock, temporary safe package extraction, no reference audio; frozen v4 содержит 358 KSC/eSpeak NG pair |
| AIT-Syn / AIT-Syn-4L Qwen3-TTS | rejected | cards claim Kazakh support and Qwen3-TTS lineage | repositories require accepting contact-sharing access conditions; 4L is based on Qwen3-TTS-12Hz-1.7B, so its essential weights also cannot be assumed below the 2 GiB project limit. Do not download, pin or use without an explicit accessible license/artifact audit |
| TurkicTTS Tacotron2 + ParallelWaveGAN | rejected | official repo exposes small Kazakh weights | independent architecture, but repository/API provide no declared license for the weights; do not download/use |
| KazakhTTS-OmniVoice | rejected | model card describes a 56% KazakhTTS2 + 44% own synthetic training mixture | no sufficient independent rights/provenance audit for synthetic training component; do not use as a family merely because it has many controls |
| LRLspoof | **исключён** | 36 000 Kazakh spoof paths, но один ~452 GB sequential gzip/tar release и нет bona-fide class | не скачивать и не использовать: нельзя выбрать только Kazakh и нельзя сформировать аудируемый binary test; [audit](data_sources_lrlspoof_2026-08-10.md) сохранён как обоснование |

## Следующий порядок

1. Fresh, text/sample-disjoint KSC/KazEmoTTS source создан и frozen до model evaluation:
   359 QA-accepted pairs в `ksc_derived_kk_v2_kazemotts_test_359.csv`.
2. Spark-TTS runtime audit и fresh KSC paired intake завершены: 381 QA-accepted pair в
   `ksc_derived_kk_v3_sparktts_test_381.csv`; не использовать его для выбора epoch, threshold
   или calibration.
3. Пятая family eSpeak NG завершена как 358 QA-accepted pair в
   `ksc_derived_kk_v4_espeakng_test_358.csv`. TurkicTTS и KazakhTTS-OmniVoice остаются
   отклонёнными; LRLspoof исключён и больше не является вариантом intake. Теперь можно
   проектировать frozen unseen-generator OOD; это не разрешает calibration, API score или
   product claim.
