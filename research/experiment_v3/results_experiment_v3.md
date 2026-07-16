# Resultados Fase 3 — hipótesis contrarian (asset pricing) — VERSIÓN WINSORIZADA

> **Pre-registro (README):** high-sentiment rinde por debajo (contrarian). Éxito: alfa FF3 del long-short (long bajo − short alto) t>2 con signo positivo Y fuera del placebo (p<0.05). Survivors-only sesga EN CONTRA del short → positivo creíble, negativo débil.

> **Higiene de datos (estándar Baker-Wurgler / Stambaugh-Yu-Yuan), NO cambia hipótesis ni criterio:** excluir precio<$1, winsorizar retornos 1%/99% por mes.

Universo survivors-only. Grupos: **deciles (10)**. Métrica: bullishness nativa (bull/(bull+bear)), >=20 menciones/mes.

## Higiene de datos aplicada

- Panel crudo: 18,213 → penny stocks (<$1) excluidos: **754** → panel limpio: **17,459**
- Ticker-mes con **|retorno| > 200% ANTES de winsorizar: 81** (ruido de datos / posibles splits sin ajustar)

## (a) Retorno mensual medio por grupo (0=sent bajo … 9=sent alto)

| grupo | ret medio (%/mes) | t-stat |
|---|---|---|
| 0 (bajo) | +32.884 | +1.04 |
| 1 | +38.571 | +1.02 |
| 2 | -0.449 | -0.27 |
| 3 | +2.823 | +0.90 |
| 4 | +2.666 | +0.78 |
| 5 | +5.758 | +1.07 |
| 6 | +31.052 | +1.01 |
| 7 | +2.973 | +1.01 |
| 8 | +10.025 | +1.27 |
| 9 (alto) | +5.958 | +1.16 |

**Long-short (bajo − alto): +26.927 %/mes, t=+1.01** (positivo = contrarian se cumple)

## (b) Alfa Fama-French 3 factores (HAC t-stats)

- **Alfa long-short: +19.3318 %/mes | t = +1.03** (obs=96 meses)

## (c) Exposición tamaño/vol por grupo

| grupo | dollar vol mediano | vol diaria mediana | n |
|---|---|---|---|
| 0 | $9.6M | 0.0414 | 1785 |
| 1 | $6.7M | 0.0451 | 1745 |
| 2 | $5.5M | 0.0451 | 1730 |
| 3 | $5.2M | 0.0467 | 1743 |
| 4 | $4.3M | 0.0477 | 1751 |
| 5 | $4.2M | 0.0481 | 1721 |
| 6 | $3.6M | 0.0482 | 1731 |
| 7 | $3.2M | 0.0474 | 1742 |
| 8 | $2.7M | 0.0465 | 1734 |
| 9 | $1.6M | 0.0430 | 1777 |

## (d) Alfa real vs distribución placebo (100 permutaciones)

- Placebo alfa: media +1.8265% ± 14.6782% | **p-value (real fuera del placebo): 0.11**

```
  [-39.393%] #
  [-35.263%] #
  [-31.132%] 
  [-27.001%] ######
  [-22.870%] ##########
  [-18.739%] ###
  [-14.608%] 
  [-10.477%] #######
  [-6.346%] ################
  [-2.215%] ########################################
  [+1.915%] ###################
  [+6.046%] ###
  [+10.177%] ####
  [+14.308%] #########
  [+18.439%] #################  <== ALFA REAL
  [+22.570%] ###
  [+26.701%] #
  [+30.832%] 
  [+34.963%] 
  [+39.093%] ###
```

## Veredicto (criterio pre-registrado)

**NEGATIVO:** alfa t=1.03 no supera 2; placebo lo replica (p=0.11). Por el caveat pre-registrado, un negativo aquí es **débil** (survivorship sesga en contra del efecto short) — pero no hay evidencia de alfa contrarian explotable con estos datos.

## Winsorizado vs crudo

- **Crudo:** alfa +48.26%/mes, t=1.01, placebo p=0.30 → NEGATIVO (magnitudes ininterpretables).
- **Winsorizado:** alfa +19.3318%/mes, t=1.03, placebo p=0.11 → NEGATIVO (magnitudes creíbles).
- **Veredicto coincide: sí** — la higiene de datos solo corrige las magnitudes, no altera la conclusión.
