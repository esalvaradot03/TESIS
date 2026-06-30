# Sanity check — target de EXCESO sobre SPY (H=5) + features estacionarias

> Números crudos, sin interpretación.

- Filas tras shift+merge SPY: **1843** | rango 2020-09-11 → 2022-02-28
- SPY: 424 cierres cargados.
- Features técnicas estacionarias (4): ['rsi', 'macd_hist', 'bb_pct', 'bb_width']
- Features de sentimiento (15): ['mention_count', 'net_sentiment_mean', 'net_sentiment_max', 'net_sentiment_min', 'net_sentiment_std', 'positive_ratio', 'negative_ratio', 'neutral_ratio', 'prob_positive_mean', 'prob_negative_mean', 'prob_neutral_mean', 'weighted_sentiment', 'total_upvotes', 'total_comments', 'stocktwits_native_sentiment']
- Híbrido = 19 features. Eliminadas de nivel: close, volume, macd, macd_signal, bb_upper, bb_middle, bb_lower.

## 1. Balance del target (excess > 0) por split

| split | n | % target=1 (bate al mercado) | % target=0 |
|---|---|---|---|
| train (core) | 1180 | 47.3% | 52.7% |
| val | 295 | 56.3% | 43.7% |
| test | 368 | 44.3% | 55.7% |

_split temporal: train ≤ 2021-11-10, test ≥ 2021-11-11._

## 2. AUC test — técnico-puro vs híbrido

| modelo | features | AUC test | accuracy | % pred=1 | CM [TN,FP,FN,TP] |
|---|---|---|---|---|---|
| técnico-puro | 4 | 0.5916 | 0.5625 | 41.8% | [129, 76, 85, 78] |
| híbrido | 19 | 0.6054 | 0.5815 | 49.2% | [119, 86, 68, 95] |

## 3. Top 5 Spearman feature[T] vs EXCESO[T+5]

| # | feature | ρ | ρ² | p | tipo |
|---|---|---|---|---|---|
| 1 | stocktwits_native_sentiment | -0.0602 | 0.4% | 0.010 | sentimiento |
| 2 | bb_pct | +0.0334 | 0.1% | 0.151 | técnica |
| 3 | net_sentiment_mean | -0.0328 | 0.1% | 0.159 | sentimiento |
| 4 | bb_width | -0.0290 | 0.1% | 0.214 | técnica |
| 5 | net_sentiment_min | +0.0270 | 0.1% | 0.246 | sentimiento |

## 4. Walk-forward (entrena una vez al 50%, predice día a día)

### híbrido

- origen entrenamiento: ≤ 2021-06-04 (920 filas) | forward: 923 filas
- **AUC agregado: 0.5175** | accuracy: 0.5146 | % pred=1: 48.0% | base rate target=1: 50.8%

| mes | n | % target=1 | accuracy | AUC | % pred=1 |
|---|---|---|---|---|---|
| 2021-06 | 90 | 77.8% | 0.522 | 0.541 | 45.6% |
| 2021-07 | 105 | 43.8% | 0.543 | 0.469 | 21.0% |
| 2021-08 | 110 | 58.2% | 0.491 | 0.562 | 40.0% |
| 2021-09 | 105 | 38.1% | 0.486 | 0.445 | 53.3% |
| 2021-10 | 105 | 61.0% | 0.524 | 0.488 | 62.9% |
| 2021-11 | 105 | 60.0% | 0.524 | 0.496 | 61.9% |
| 2021-12 | 110 | 39.1% | 0.545 | 0.535 | 35.5% |
| 2022-01 | 100 | 36.0% | 0.510 | 0.565 | 63.0% |
| 2022-02 | 93 | 46.2% | 0.484 | 0.527 | 50.5% |

### técnico-puro

- origen entrenamiento: ≤ 2021-06-04 (920 filas) | forward: 923 filas
- **AUC agregado: 0.5047** | accuracy: 0.5200 | % pred=1: 47.5% | base rate target=1: 50.8%

| mes | n | % target=1 | accuracy | AUC | % pred=1 |
|---|---|---|---|---|---|
| 2021-06 | 90 | 77.8% | 0.533 | 0.526 | 46.7% |
| 2021-07 | 105 | 43.8% | 0.514 | 0.353 | 20.0% |
| 2021-08 | 110 | 58.2% | 0.500 | 0.455 | 37.3% |
| 2021-09 | 105 | 38.1% | 0.476 | 0.453 | 41.0% |
| 2021-10 | 105 | 61.0% | 0.467 | 0.344 | 80.0% |
| 2021-11 | 105 | 60.0% | 0.533 | 0.519 | 55.2% |
| 2021-12 | 110 | 39.1% | 0.618 | 0.572 | 31.8% |
| 2022-01 | 100 | 36.0% | 0.520 | 0.615 | 66.0% |
| 2022-02 | 93 | 46.2% | 0.516 | 0.563 | 51.6% |
