# Dusha — source review, 2026-08-12

Проверен `salute-developers/golos` master commit
`5c5c5f87044803fcccdf7e149ef5384c95cff107`. Dusha — Russian human speech emotion corpus, а
не anti-spoof source: `303 963` recordings, `346.6` hours, crowd (`28 GB`) и podcast metadata /
features. Repository прямо не распространяет podcast audio из-за licensing issue.

Custom public license разрешает reproduction/adaptation с attribution/share-alike-like
conditions, но отдельно говорит, что privacy/publicity/portrayal rights в общем случае не
предоставляются. Это требует отдельного legal/privacy решения перед product use. Для текущего
personal research возможен будущий read-only intake только crowd component после exact artifact,
speaker metadata, consent/provenance и overlap audit.

Решение сейчас: **не скачивать**. Dusha не закрывает spoof gap, не является готовым binary final
benchmark и добавит 28 GB human audio, которое текущий frozen RU layer уже не требует. Podcast
component исключён. Источник можно пересмотреть лишь для отдельного bona-fide robustness study,
не смешивая его с claim о spoof detection quality.

Primary source: <https://github.com/salute-developers/golos/tree/master/dusha>.
