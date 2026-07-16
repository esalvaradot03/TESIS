# Etapa A — Auditoría de factibilidad (small/mid-caps fuera del S&P 500)

Fuente de menciones: StockTwits NYU `symbol_sentiments/` (misma que v1). Fuente de precios: FNSPID `full_history/` (local, hasta 2023-12). Market cap: yfinance fast_info (actual, estático).

## Números

- Tickers **fuera de los 475 de v1** con actividad (2015-2023): **22,181**
- Con >= 200 días de >=5 menciones: **1,890**
- Sin datos de precio (FNSPID): **880**
- Deslistados (precio termina antes de 2023): **345**
- Con precio reciente (survivors): **665**
- **Survivorship bias potencial** (sin precio + deslistados): **64.8%**
- **UNIVERSO CANDIDATO FINAL** (cap $200M-$10B + precio + >=200d): **217**
- Filas estimadas del dataset (suma de días activos): **~163,205**

## Distribución de market cap del universo final

- min $201M | p10 $311M | p50 $1.35B | p90 $6.15B | max $9.90B
