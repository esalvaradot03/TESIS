# Experimento v3 — hipótesis contrarian de sentimiento (Baker-Wurgler / Stambaugh-Yu-Yuan)

Asset pricing / formación de carteras. NO es clasificación; no se entrena ningún modelo.

## PRE-REGISTRO (escrito ANTES de correr nada)

> Hipótesis: acciones en el decil superior de sentimiento bullish rinden POR DEBAJO del
> mercado en 1-3 meses (contrarian, signo comprometido a priori). Éxito: la cartera
> long-short (short decil alto, long decil bajo) tiene alfa mensual con t>2 sobre
> Fama-French 3 factores, con el signo predicho, Y el placebo (sentimiento permutado) NO.
> Caveat: survivors-only; el survivorship sesga EN CONTRA del efecto short, así que un
> positivo es creíble y un negativo es débil (atribuible al sesgo).

## Universo
- 475 equities survivors reales de la Fase 2 (`quoteType==EQUITY`, `research/experiment_v2/universe_v2.parquet`).
- Mismo filtro de operabilidad point-in-time: dollar volume mediano 60d en $200K–$50M.

## Métrica de sentimiento mensual
- **Bullishness nativa de StockTwits** = bullish / (bullish + bearish) por (ticker, mes),
  usando el label nativo Bullish/Bearish de `symbol_sentiments`.
- Justificación: es el sentimiento retail EXPRESADO directamente (lo que miden los índices
  tipo Baker-Wurgler), disponible point-in-time, sin requerir FinBERT (no computado para
  small-caps). Es el input natural de la hipótesis contrarian.
- Filtro: mínimo **20 menciones (labeled) en el mes** para que el ticker entre ese mes.

## Retornos
- Forward 1m y 3m, exceso sobre SPY. Sin leakage: sentimiento del mes t → retorno de t+1 en adelante.

## Carteras (rebalanceo mensual)
- Ordenar el universo del mes por bullishness; deciles (o quintiles si hay pocos nombres/mes).
- Cartera de prueba: **SHORT decil superior (sentimiento alto), LONG decil inferior.**
- Reportar el retorno de cada decil por separado (monotonicidad).

## Evaluación
a. Retorno mensual medio por decil + long-short, con t-stat.
b. Regresión del long-short sobre Fama-French 3 factores (Mkt-RF, SMB, HML) → **alfa + t-stat**.
c. Exposición tamaño/vol por decil (¿la pata short es small/volátil?).
d. Placebo: permutar el sentimiento entre tickers dentro de cada mes (×100), distribución del
   alfa placebo vs el alfa real.

## Criterio (pre-registrado)
Éxito si **alfa real t>2 con el signo predicho Y el alfa real cae fuera de la distribución
placebo (p<0.05)**. Si el alfa desaparece al controlar factores o el placebo lo replica →
negativo. Una sola corrida, sin optimizar umbrales.

## Adenda — re-corrida con higiene de datos (winsorización)
La primera corrida (cruda) arrojó magnitudes ininterpretables (un decil rindió +1413%/mes)
por **outliers extremos de FNSPID / posibles splits sin ajustar** en small-caps. Se re-corre
UNA vez aplicando higiene de datos ESTÁNDAR en la literatura de referencia (Baker-Wurgler,
Stambaugh-Yu-Yuan winsorizan por defecto). **No cambia la hipótesis, ni el criterio, ni el
umbral de éxito; NO se persigue un resultado.** El veredicto esperado sigue siendo el mismo
negativo — solo cambian las magnitudes a valores creíbles. Cambios (solo estos):
1. Excluir ticker-mes con precio de cierre < $1 (penny stocks).
2. Winsorizar retornos mensuales al 1% / 99% dentro de cada mes antes de formar carteras.
3. Reportar cuántos ticker-mes tenían |retorno| > 200% antes de winsorizar (ruido de datos).
Todo lo demás idéntico (universo, sentimiento, deciles, long-short, FF3, placebo ×100, criterio).
