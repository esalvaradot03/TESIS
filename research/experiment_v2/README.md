# Experimento v2 — hipótesis small/mid-cap (survivors-only)

## PRE-REGISTRO (escrito ANTES de correr)

> Este test es survivors-only. El universo excluye por construcción los fracasos
> reales (pump-and-dumps colapsados, quiebras), irrecuperables porque FNSPID no
> los cubre. Por tanto está sesgado A FAVOR de encontrar señal (cota superior
> optimista). Interpretación comprometida de antemano: (a) si NO se cumple el
> criterio de éxito → negativo fuerte, la hipótesis small-cap se rechaza incluso
> en el mejor caso sesgado; (b) si SÍ se cumple → resultado NO confiable,
> atribuible a survivorship bias, no promocionable a model_v1.

## Universo
- `quoteType == EQUITY` (los 475 survivors reales; se excluyen 58 ETF y 132 UNKNOWN).
- Filtro de operabilidad (sustituye al de market cap, no reconstruible point-in-time):
  **dollar volume mediano móvil, ventana 60d, point-in-time, en banda $200K–$50M.**

## Construcción dataset_v2
Idéntica a Fase 1: target de exceso sobre SPY a 5d (balanceado), features técnicas +
sentimiento (StockTwits + WSB + noticias) + aceleración, sin leakage, walk-forward
expanding por año (test 2017-2023).

## Experimentos (exactamente 4)
- F1: 5d, universo survivors completo
- F2: 5d, solo días evento (z-score menciones >= 2)
- F3 (control): 5d, solo técnico
- F4 (control): F1 con sentimiento+aceleración permutados (placebo)

## Criterio de éxito (pre-registrado)
Un experimento "tiene señal" si: **precisión decil superior >= 55% Y Sharpe del
backtest > 0.5 (costos 10 bps/lado) Y el placebo F4 NO cumple lo mismo.**
Sin experimentos adicionales, sin promover nada.
