# YO-CPT-kk — rejected source review, 2026-08-12

Проверен public revision `a8c2d1a23de45c271d1ce5d458416f6bf11dd913`. Это human/bona-fide
Kazakh YouTube corpus для continual TTS pre-training, а не spoof dataset: `156 903` utterances,
около `600` часов. Public tree содержит 40 Parquet shards и около `99.95 GB`, поэтому он не нужен
для KazDeepScan и не скачивался.

Ключевой blocker — не только размер. Собственная LICENSE автора оставляет copyright source
recordings их исходным владельцам YouTube, а release добавляет автоматически полученные face,
persona и cross-video identity fields. `global_spk_id` создан LVFace weights с
non-commercial-research-only terms; automatic anti-spoof/language filters не заменяют consent,
per-record rights и проверенную binary ground truth. Dataset также сохраняет natural RU/KK
code-switching, но не публикует независимый spoof class.

Решение: не добавлять в ledger, не использовать для train/dev/calibration/final/product и не
скачивать. Это тот же класс provenance/privacy риска, что и YO-CPT-ru, при объёме около 100 GB.

Primary source: <https://huggingface.co/datasets/NCSpeech/YO-CPT-kk>.
