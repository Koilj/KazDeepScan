# Локальные raw datasets в `/home/ruslan/Downloads`

Проверено 12 августа 2026. Ничего автоматически не удалялось.

> Это historical snapshot на 12 августа. Текущая cleanup-запись от 14 августа и новые
> ограничения v4 записаны в
> [`local_storage_cleanup_2026-08-14.md`](local_storage_cleanup_2026-08-14.md). Для v4 KSC и Common Voice
> больше не считаются кандидатами на удаление до нового role/capacity audit.

| Путь | Размер | Роль | Решение |
| --- | ---: | --- | --- |
| `RuASD/` | 234 GiB | источник текущего RuASD-v2 train; 250 pinned TAR | оставить для полной воспроизводимости |
| `archive.zip` | 27 GiB | PyAra v7: Stage A/B dev и calibration | оставить до резервной копии; нужен для rebuild |
| `KSC2/` | 76 GiB | KSC2 mixed source и будущий новый mixed audit | оставить сейчас |
| `FLEURS/` | 4.6 GiB | RU/KK final source и возможный acoustic review rebuild | оставить сейчас |
| `ISSAI_KSC_335RS_v1.1_flac.tar.gz` | 18 GiB | исторические KSC-derived Kazakh suites | можно архивировать после backup; текущий v2 final его не читает |
| `cv-corpus-24.0-2025-12-05-ru.tar.gz` | 6.6 GiB | исторический Common Voice RU slice | можно архивировать после backup; текущий Stage A/B/final его не читает |

`ISSAI_KSC...` и Common Voice можно удалить только если владелец сознательно принимает потерю
возможности заново построить их raw/ready slices. Git хранит manifests и receipts, но не raw или
processed audio; они не заменяют архив. Остальные четыре источника пока удалять не рекомендуется.
Перед любым удалением сначала сохранить off-machine backup и проверить его checksum.
