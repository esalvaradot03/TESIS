# Outliers — dataset_v1

## Retornos extremos (|return_5d_forward| > 50%)

- 131 filas (0.1%). Top 10 por |retorno|:

| ticker | fecha | return_5d | total_count | mention_count |
|---|---|---|---|---|
| NCLH | 2020-03-18 | +1.183 | 72 | 1 |
| CVNA | 2020-03-18 | +1.090 | 16 | 1 |
| PCG | 2019-01-16 | +0.984 | 979 | 3 |
| PCG | 2019-10-28 | +0.913 | 858 | 6 |
| NCLH | 2020-03-19 | +0.909 | 56 | 1 |
| RCL | 2020-03-18 | +0.899 | 165 | 7 |
| APA | 2020-04-01 | +0.886 | 21 | 0 |
| CVNA | 2020-03-19 | +0.884 | 9 | 2 |
| PCG | 2019-01-17 | +0.851 | 844 | 0 |
| BA | 2020-03-19 | +0.848 | 648 | 5 |

## Sentimientos fuera de rango

- bullish_ratio fuera de [0,1]: **0**
- news_sentiment_mean fuera de [-1,1]: **0**

## Fechas fuera de 2010-2023

- 0 filas fuera de rango.

## Duplicados por (ticker, fecha)

- 48285 duplicados.

## Nota: precios
- dataset_v1 no contiene columna de precio absoluto (solo retornos y features estacionarias), así que no aplica el chequeo 'precio < 0'.
