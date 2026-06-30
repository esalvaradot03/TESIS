# Modelos — dataset_v1

> Anti-p-hacking: configuraciones fijas, sin tuning, todas reportadas.

**Criterio de señal real (definido a priori):** AUC agregado > 0.55 Y ≥70% folds > 0.53 Y feature de sentimiento |ρ|>0.05 p<0.001.

- Señal de sentimiento (Fase 8): news_count: ρ=-0.0103 p=6.58e-07

## Resumen de modelos

| modelo | folds | AUC medio | acc | prec | recall | folds>0.53 | ¿señal? |
|---|---|---|---|---|---|---|---|
| M1_tecnico | 6 | 0.4987 | 0.500 | 0.501 | 0.504 | 0/6 | no |
| M2_stocktwits | 6 | 0.4963 | 0.496 | 0.494 | 0.407 | 0/6 | no |
| M3_wsb | 6 | 0.4994 | 0.501 | 0.491 | 0.203 | 0/6 | no |
| M4_fnspid | 6 | 0.5017 | 0.501 | 0.502 | 0.743 | 0/6 | no |
| M5_todas | 6 | 0.5021 | 0.504 | 0.504 | 0.575 | 0/6 | no |
| M6_sin_stocktwits | 6 | 0.4993 | 0.496 | 0.496 | 0.436 | 0/6 | no |

### M1_tecnico — AUC por fold

| test year | n | AUC | acc | prec | recall |
|---|---|---|---|---|---|
| 2017 | 26241 | 0.504 | 0.496 | 0.512 | 0.260 |
| 2018 | 56280 | 0.500 | 0.503 | 0.508 | 0.672 |
| 2019 | 23398 | 0.498 | 0.506 | 0.512 | 0.574 |
| 2020 | 29772 | 0.502 | 0.503 | 0.493 | 0.468 |
| 2021 | 28631 | 0.496 | 0.498 | 0.481 | 0.479 |
| 2022 | 22868 | 0.492 | 0.495 | 0.502 | 0.569 |

Top features (importance): volatility_20d(0.181), macd_hist(0.180), bb_width(0.169), bb_pct(0.167), rsi(0.154), volume_ratio(0.149)

### M2_stocktwits — AUC por fold

| test year | n | AUC | acc | prec | recall |
|---|---|---|---|---|---|
| 2017 | 26241 | 0.483 | 0.487 | 0.497 | 0.489 |
| 2018 | 56280 | 0.493 | 0.493 | 0.501 | 0.122 |
| 2019 | 23398 | 0.501 | 0.505 | 0.508 | 0.742 |
| 2020 | 29772 | 0.492 | 0.498 | 0.475 | 0.231 |
| 2021 | 28631 | 0.502 | 0.500 | 0.487 | 0.621 |
| 2022 | 22868 | 0.508 | 0.492 | 0.500 | 0.238 |

Top features (importance): total_count(0.217), unique_users(0.207), bullish_count(0.202), bullish_ratio(0.200), bearish_count(0.175)

### M3_wsb — AUC por fold

| test year | n | AUC | acc | prec | recall |
|---|---|---|---|---|---|
| 2017 | 26241 | 0.498 | 0.506 | 0.509 | 0.892 |
| 2018 | 56280 | 0.498 | 0.494 | 0.514 | 0.040 |
| 2019 | 23398 | 0.501 | 0.495 | 0.514 | 0.090 |
| 2020 | 29772 | 0.502 | 0.509 | 0.493 | 0.101 |
| 2021 | 28631 | 0.499 | 0.514 | 0.485 | 0.073 |
| 2022 | 22868 | 0.498 | 0.489 | 0.429 | 0.022 |

Top features (importance): avg_score(0.375), total_comments(0.346), mention_count(0.279)

### M4_fnspid — AUC por fold

| test year | n | AUC | acc | prec | recall |
|---|---|---|---|---|---|
| 2017 | 26241 | 0.500 | 0.492 | 0.504 | 0.315 |
| 2018 | 56280 | 0.503 | 0.506 | 0.508 | 0.825 |
| 2019 | 23398 | 0.502 | 0.507 | 0.509 | 0.801 |
| 2020 | 29772 | 0.507 | 0.503 | 0.496 | 0.846 |
| 2021 | 28631 | 0.497 | 0.491 | 0.485 | 0.853 |
| 2022 | 22868 | 0.502 | 0.508 | 0.509 | 0.818 |

Top features (importance): news_sentiment_mean(0.512), news_count(0.488)

### M5_todas — AUC por fold

| test year | n | AUC | acc | prec | recall |
|---|---|---|---|---|---|
| 2017 | 26241 | 0.514 | 0.507 | 0.521 | 0.417 |
| 2018 | 56280 | 0.499 | 0.501 | 0.506 | 0.712 |
| 2019 | 23398 | 0.499 | 0.508 | 0.512 | 0.672 |
| 2020 | 29772 | 0.515 | 0.508 | 0.498 | 0.530 |
| 2021 | 28631 | 0.496 | 0.500 | 0.482 | 0.451 |
| 2022 | 22868 | 0.490 | 0.501 | 0.506 | 0.666 |

Top features (importance): macd_hist(0.076), bb_width(0.074), volatility_20d(0.074), rsi(0.068), bb_pct(0.067), mention_count(0.064), total_count(0.064), total_comments(0.060), avg_score(0.060), bullish_count(0.060)

### M6_sin_stocktwits — AUC por fold

| test year | n | AUC | acc | prec | recall |
|---|---|---|---|---|---|
| 2017 | 26241 | 0.501 | 0.494 | 0.506 | 0.340 |
| 2018 | 56280 | 0.501 | 0.501 | 0.506 | 0.648 |
| 2019 | 23398 | 0.498 | 0.488 | 0.489 | 0.199 |
| 2020 | 29772 | 0.504 | 0.502 | 0.491 | 0.460 |
| 2021 | 28631 | 0.498 | 0.499 | 0.483 | 0.497 |
| 2022 | 22868 | 0.494 | 0.492 | 0.500 | 0.470 |

Top features (importance): volatility_20d(0.105), macd_hist(0.102), bb_pct(0.097), rsi(0.094), bb_width(0.093), total_comments(0.090), volume_ratio(0.090), avg_score(0.084), mention_count(0.083), news_sentiment_mean(0.083)

## Mejor modelo: **M5_todas**

Backtest (long top-decil, rebalanceo 5d no solapado):

| | retorno acum. | Sharpe | max drawdown | periodos |
|---|---|---|---|---|
| estrategia | 183.58% | 0.75 | -50.25% | 302 |
| SPY buy&hold | 87.73% | 0.74 | -27.37% | 302 |
