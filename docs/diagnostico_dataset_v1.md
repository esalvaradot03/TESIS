# Diagnóstico estructural — dataset_v1

## 8.1 Integridad

- **Verificación del target:** target == (excess_return_5d_forward > 0) en **231305/231305** filas (0 discrepancias).

20 (ticker, fecha) aleatorios:

| ticker | fecha | excess_5d | target | ok |
|---|---|---|---|---|
| HPQ | 2012-11-13 | -0.1190 | 0 | ✓ |
| MSFT | 2018-01-04 | -0.0055 | 0 | ✓ |
| MAR | 2020-02-26 | -0.0159 | 0 | ✓ |
| VEEV | 2020-06-10 | +0.0337 | 1 | ✓ |
| PG | 2020-08-20 | -0.0198 | 0 | ✓ |
| GOOG | 2015-07-28 | +0.0018 | 1 | ✓ |
| BRK-B | 2018-03-14 | +0.0033 | 1 | ✓ |
| PFE | 2020-01-15 | -0.0098 | 0 | ✓ |
| KO | 2021-12-30 | +0.0460 | 1 | ✓ |
| LMT | 2020-02-20 | +0.0199 | 1 | ✓ |
| NFLX | 2020-01-07 | +0.0094 | 1 | ✓ |
| TSLA | 2017-03-24 | +0.0495 | 1 | ✓ |
| DVN | 2022-11-09 | +0.0116 | 1 | ✓ |
| FSLR | 2018-02-23 | +0.0616 | 1 | ✓ |
| PYPL | 2022-02-11 | -0.0868 | 0 | ✓ |
| BBY | 2017-03-10 | +0.0194 | 1 | ✓ |
| PM | 2018-05-23 | -0.0176 | 0 | ✓ |
| LOW | 2018-11-27 | -0.0212 | 0 | ✓ |
| AVGO | 2018-05-31 | +0.0263 | 1 | ✓ |
| WYNN | 2017-11-20 | -0.0143 | 0 | ✓ |

- Outliers: retornos extremos=131, sentimiento fuera de rango=0, fechas malas=0, duplicados=48285. Detalle en `docs/outliers_dataset_v1.md`.

## 8.2 Cobertura por ticker

- Tickers totales: **475** | <100 filas: 240 | 100-1000: 148 | >1000: 87
- Mediana filas/ticker: 96 | máx: 3769 (AAPL) | mín: 1

Top 20 por filas:

| ticker | filas |  | ticker | filas |
|---|---|---|---|---|
| AAPL | 3769 |  | CRL | 2 |
| GOOG | 3495 |  | VRSK | 2 |
| META | 3471 |  | CEG | 2 |
| NFLX | 3443 |  | CPT | 2 |
| AMZN | 3383 |  | RVTY | 2 |
| BAC | 3381 |  | AJG | 1 |
| AMD | 3189 |  | REG | 1 |
| MSFT | 3057 |  | AVB | 1 |
| TSLA | 3057 |  | PFG | 1 |
| F | 2965 |  | ERIE | 1 |
| GS | 2812 |  | DOC | 1 |
| MU | 2766 |  | UDR | 1 |
| DIS | 2710 |  | EME | 1 |
| GOOGL | 2670 |  | WRB | 1 |
| GILD | 2617 |  | AEE | 1 |
| INTC | 2563 |  | MTD | 1 |
| JPM | 2505 |  | LNT | 1 |
| CMG | 2498 |  | EQR | 1 |
| SBUX | 2498 |  | INVH | 1 |
| NVDA | 2384 |  | COR | 1 |

**Concentración:** los 20 tickers con más datos acumulan 25.6% de las filas → la señal está razonablemente distribuida.

## 8.3 Cobertura temporal

| año | filas |
|---|---|
| 2010 | 88 |
| 2011 | 271 |
| 2012 | 3,210 |
| 2013 | 8,918 |
| 2014 | 8,535 |
| 2015 | 11,063 |
| 2016 | 12,020 |
| 2017 | 26,241 |
| 2018 | 56,280 |
| 2019 | 23,398 |
| 2020 | 29,772 |
| 2021 | 28,631 |
| 2022 | 22,868 |
| 2023 | 10 |

- Meses sin datos (gaps): 0 (ninguno)

## 8.4 Correlaciones Spearman feature ↔ target

Con n=231,305 se exige **p < 0.001** para considerar señal.

### Global  (n=231,305)

| # | feature | ρ | p | señal (p<0.001) |
|---|---|---|---|---|
| 1 | news_count | -0.0103 | 6.58e-07 | no |
| 2 | macd_hist | -0.0096 | 4.24e-06 | no |
| 3 | bb_pct | -0.0086 | 3.58e-05 | no |
| 4 | news_sentiment_mean | +0.0066 | 2.38e-02 | no |
| 5 | rsi | -0.0051 | 1.38e-02 | no |
| 6 | bullish_count | -0.0033 | 1.07e-01 | no |
| 7 | volume_ratio | -0.0033 | 1.12e-01 | no |
| 8 | total_count | -0.0030 | 1.53e-01 | no |
| 9 | total_comments | +0.0030 | 1.54e-01 | no |
| 10 | unique_users | -0.0030 | 1.55e-01 | no |
| 11 | bullish_ratio | -0.0029 | 1.60e-01 | no |
| 12 | mention_count | +0.0024 | 2.41e-01 | no |
| 13 | avg_score | -0.0020 | 7.27e-01 | no |
| 14 | bb_width | +0.0017 | 4.05e-01 | no |
| 15 | bearish_count | +0.0006 | 7.71e-01 | no |

### Década 2010-2015  (n=32,085)

| # | feature | ρ | p | señal (p<0.001) |
|---|---|---|---|---|
| 1 | avg_score | -0.0550 | 1.49e-01 | no |
| 2 | macd_hist | -0.0186 | 8.56e-04 | no |
| 3 | bearish_count | -0.0137 | 1.45e-02 | no |
| 4 | unique_users | -0.0128 | 2.14e-02 | no |
| 5 | bb_pct | -0.0104 | 6.23e-02 | no |
| 6 | bullish_ratio | +0.0100 | 7.21e-02 | no |
| 7 | total_comments | -0.0100 | 7.38e-02 | no |
| 8 | bb_width | +0.0090 | 1.08e-01 | no |
| 9 | volume_ratio | +0.0086 | 1.25e-01 | no |
| 10 | total_count | -0.0079 | 1.57e-01 | no |
| 11 | news_count | +0.0069 | 2.14e-01 | no |
| 12 | volatility_20d | +0.0065 | 2.46e-01 | no |
| 13 | mention_count | -0.0054 | 3.33e-01 | no |
| 14 | bullish_count | -0.0050 | 3.67e-01 | no |
| 15 | rsi | -0.0020 | 7.26e-01 | no |

### Década 2016-2020  (n=147,711)

| # | feature | ρ | p | señal (p<0.001) |
|---|---|---|---|---|
| 1 | news_count | -0.0203 | 6.95e-15 | no |
| 2 | bb_pct | -0.0137 | 1.36e-07 | no |
| 3 | macd_hist | -0.0135 | 2.20e-07 | no |
| 4 | bearish_count | +0.0127 | 9.69e-07 | no |
| 5 | volume_ratio | -0.0121 | 3.35e-06 | no |
| 6 | bullish_ratio | -0.0100 | 1.18e-04 | no |
| 7 | rsi | -0.0087 | 8.13e-04 | no |
| 8 | unique_users | +0.0086 | 9.12e-04 | no |
| 9 | news_sentiment_mean | +0.0083 | 1.51e-02 | no |
| 10 | total_count | +0.0070 | 7.44e-03 | no |
| 11 | total_comments | +0.0068 | 8.53e-03 | no |
| 12 | bullish_count | +0.0054 | 3.73e-02 | no |
| 13 | volatility_20d | +0.0048 | 6.37e-02 | no |
| 14 | mention_count | +0.0048 | 6.62e-02 | no |
| 15 | bb_width | +0.0037 | 1.57e-01 | no |

### Década 2021-2023  (n=51,509)

| # | feature | ρ | p | señal (p<0.001) |
|---|---|---|---|---|
| 1 | unique_users | -0.0288 | 6.59e-11 | no |
| 2 | total_count | -0.0274 | 4.76e-10 | no |
| 3 | avg_score | -0.0272 | 8.19e-02 | no |
| 4 | bullish_count | -0.0266 | 1.62e-09 | no |
| 5 | bearish_count | -0.0249 | 1.61e-08 | no |
| 6 | volatility_20d | -0.0166 | 1.69e-04 | no |
| 7 | volume_ratio | +0.0148 | 8.00e-04 | no |
| 8 | total_comments | -0.0120 | 6.55e-03 | no |
| 9 | bullish_ratio | +0.0094 | 3.28e-02 | no |
| 10 | mention_count | -0.0079 | 7.43e-02 | no |
| 11 | bb_width | -0.0069 | 1.17e-01 | no |
| 12 | news_sentiment_mean | +0.0047 | 4.90e-01 | no |
| 13 | bb_pct | +0.0044 | 3.16e-01 | no |
| 14 | news_count | +0.0029 | 5.05e-01 | no |
| 15 | macd_hist | +0.0019 | 6.59e-01 | no |

### Por liquidez (cuartiles de combined_mentions)

### Liquidez Q1  (n=67,048)

| # | feature | ρ | p | señal (p<0.001) |
|---|---|---|---|---|
| 1 | avg_score | -0.0163 | 4.53e-01 | no |
| 2 | news_sentiment_mean | +0.0061 | 2.62e-01 | no |
| 3 | total_count | -0.0060 | 1.22e-01 | no |
| 4 | bb_width | -0.0039 | 3.10e-01 | no |
| 5 | unique_users | +0.0036 | 3.55e-01 | no |
| 6 | mention_count | -0.0034 | 3.72e-01 | no |
| 7 | rsi | -0.0034 | 3.80e-01 | no |
| 8 | bb_pct | -0.0029 | 4.46e-01 | no |
| 9 | news_count | +0.0027 | 4.90e-01 | no |
| 10 | bullish_count | -0.0025 | 5.25e-01 | no |
| 11 | volatility_20d | +0.0022 | 5.73e-01 | no |
| 12 | macd_hist | +0.0012 | 7.48e-01 | no |
| 13 | total_comments | -0.0011 | 7.68e-01 | no |
| 14 | volume_ratio | +0.0009 | 8.17e-01 | no |
| 15 | bullish_ratio | -0.0009 | 8.21e-01 | no |

### Liquidez Q2  (n=49,824)

| # | feature | ρ | p | señal (p<0.001) |
|---|---|---|---|---|
| 1 | avg_score | -0.0705 | 3.50e-04 | **sí** |
| 2 | macd_hist | -0.0197 | 1.16e-05 | no |
| 3 | bb_pct | -0.0195 | 1.42e-05 | no |
| 4 | rsi | -0.0147 | 1.04e-03 | no |
| 5 | volume_ratio | +0.0132 | 3.21e-03 | no |
| 6 | volatility_20d | +0.0115 | 1.03e-02 | no |
| 7 | bb_width | +0.0111 | 1.34e-02 | no |
| 8 | unique_users | -0.0066 | 1.39e-01 | no |
| 9 | bearish_count | +0.0062 | 1.67e-01 | no |
| 10 | bullish_ratio | -0.0059 | 1.89e-01 | no |
| 11 | news_count | -0.0055 | 2.23e-01 | no |
| 12 | total_count | +0.0046 | 3.07e-01 | no |
| 13 | bullish_count | -0.0022 | 6.29e-01 | no |
| 14 | total_comments | +0.0015 | 7.36e-01 | no |
| 15 | mention_count | +0.0014 | 7.55e-01 | no |

### Liquidez Q3  (n=56,634)

| # | feature | ρ | p | señal (p<0.001) |
|---|---|---|---|---|
| 1 | avg_score | -0.0368 | 7.48e-03 | no |
| 2 | news_count | -0.0161 | 1.26e-04 | no |
| 3 | bb_pct | -0.0126 | 2.68e-03 | no |
| 4 | unique_users | +0.0097 | 2.07e-02 | no |
| 5 | total_count | +0.0096 | 2.30e-02 | no |
| 6 | bullish_count | +0.0092 | 2.81e-02 | no |
| 7 | macd_hist | -0.0091 | 2.97e-02 | no |
| 8 | volume_ratio | -0.0091 | 2.99e-02 | no |
| 9 | rsi | -0.0089 | 3.50e-02 | no |
| 10 | volatility_20d | -0.0051 | 2.25e-01 | no |
| 11 | bb_width | -0.0035 | 4.03e-01 | no |
| 12 | bullish_ratio | +0.0023 | 5.81e-01 | no |
| 13 | mention_count | +0.0020 | 6.39e-01 | no |
| 14 | total_comments | +0.0011 | 7.90e-01 | no |
| 15 | bearish_count | -0.0006 | 8.79e-01 | no |

### Liquidez Q4  (n=57,799)

| # | feature | ρ | p | señal (p<0.001) |
|---|---|---|---|---|
| 1 | news_sentiment_mean | +0.0220 | 3.42e-04 | no |
| 2 | news_count | -0.0215 | 2.38e-07 | no |
| 3 | avg_score | +0.0170 | 1.72e-02 | no |
| 4 | bullish_ratio | -0.0160 | 1.17e-04 | no |
| 5 | volume_ratio | -0.0148 | 3.96e-04 | no |
| 6 | bearish_count | +0.0140 | 7.92e-04 | no |
| 7 | macd_hist | -0.0135 | 1.18e-03 | no |
| 8 | total_comments | +0.0133 | 1.36e-03 | no |
| 9 | mention_count | +0.0130 | 1.80e-03 | no |
| 10 | bb_width | +0.0096 | 2.06e-02 | no |
| 11 | total_count | +0.0043 | 2.96e-01 | no |
| 12 | rsi | +0.0034 | 4.10e-01 | no |
| 13 | bb_pct | -0.0030 | 4.78e-01 | no |
| 14 | volatility_20d | -0.0016 | 7.05e-01 | no |
| 15 | unique_users | +0.0015 | 7.27e-01 | no |

## Sign flips entre sub-grupos

| feature | global | 2010-2015 | 2016-2020 | 2021-2023 | liq_Q1 | liq_Q2 | liq_Q3 | liq_Q4 | ¿flip? |
|---|---|---|---|---|---|---|---|---|---|
| bullish_count | -0.003 | -0.005 | +0.005 | -0.027 | -0.002 | -0.002 | +0.009 | +0.001 | no |
| bearish_count | +0.001 | -0.014 | +0.013 | -0.025 | +0.000 | +0.006 | -0.001 | +0.014 | no |
| total_count | -0.003 | -0.008 | +0.007 | -0.027 | -0.006 | +0.005 | +0.010 | +0.004 | no |
| unique_users | -0.003 | -0.013 | +0.009 | -0.029 | +0.004 | -0.007 | +0.010 | +0.001 | no |
| bullish_ratio | -0.003 | +0.010 | -0.010 | +0.009 | -0.001 | -0.006 | +0.002 | -0.016 | no |
| mention_count | +0.002 | -0.005 | +0.005 | -0.008 | -0.003 | +0.001 | +0.002 | +0.013 | no |
| avg_score | -0.002 | -0.055 | +0.002 | -0.027 | -0.016 | -0.070 | -0.037 | +0.017 | no |
| total_comments | +0.003 | -0.010 | +0.007 | -0.012 | -0.001 | +0.002 | +0.001 | +0.013 | no |
| news_count | -0.010 | +0.007 | -0.020 | +0.003 | +0.003 | -0.005 | -0.016 | -0.021 | no |
| news_sentiment_mean | +0.007 | -0.000 | +0.008 | +0.005 | +0.006 | +0.001 | -0.000 | +0.022 | no |

**0/10 features de sentimiento cambian de signo entre sub-grupos.**
