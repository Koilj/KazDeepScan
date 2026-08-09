# Record-level aggregation и калибровка

Модель выдаёт один **raw logit** на каждое speech-window. Чтобы получить оценку для записи:

1. Для full-fake режима окна агрегируются duration-weighted средним logit-ов.
2. Для partial-fake режима используется top-k mean logit, где `k=max(2, ceil(20% окон))`;
   для записи с одним окном `k=1`.
3. Только после этого `TemperatureScaler` обучается на независимом dev set.
4. Вероятность равна `sigmoid(record_logit / temperature)`.

Нельзя калибровать окна, усреднять их `sigmoid` и называть итог калиброванной вероятностью
записи. В отчёт должны войти NLL, Brier score и ECE до/после калибровки.
