# Sanity check pre-adaptativo

> Solo diagnóstico. No se modifica `model_v0`. El modelo técnico de la sección 2 se entrena al vuelo y no se persiste.

## 1. ¿El target está bien construido?

Regla esperada: `target[T] = 1` ⇔ `close[T+1] > close[T]` (siguiente día de trading del MISMO ticker).

| # | ticker | fecha (T) | close[T] | fecha (T+1) | close[T+1] | sube? | target | ✓ |
|---|---|---|---|---|---|---|---|---|
| 1 | AAPL | 2021-08-12 | 145.34 | 2021-08-13 | 145.52 | sí | 1 | ✓ |
| 2 | AAPL | 2021-10-08 | 139.59 | 2021-10-11 | 139.40 | no | 0 | ✓ |
| 3 | AAPL | 2021-11-15 | 146.64 | 2021-11-16 | 147.62 | sí | 1 | ✓ |
| 4 | AMZN | 2021-03-09 | 153.07 | 2021-03-10 | 152.81 | no | 0 | ✓ |
| 5 | AMZN | 2021-03-18 | 151.48 | 2021-03-19 | 153.78 | sí | 1 | ✓ |
| 6 | FB | 2021-05-27 | 325.88 | 2021-05-28 | 322.23 | no | 0 | ✓ |
| 7 | NVDA | 2020-11-17 | 13.36 | 2020-11-18 | 13.39 | sí | 1 | ✓ |
| 8 | NVDA | 2021-06-23 | 18.99 | 2021-06-24 | 19.15 | sí | 1 | ✓ |
| 9 | TSLA | 2021-06-30 | 226.61 | 2021-07-01 | 225.99 | no | 0 | ✓ |
| 10 | TSLA | 2021-11-15 | 337.82 | 2021-11-16 | 351.51 | sí | 1 | ✓ |

**Resultado:** los 10 casos correctos. Sobre TODO el dataset, el target del pipeline coincide con la verificación independiente en **1863/1863** filas (0 discrepancias). 
**Interpretación:** el target está bien construido (sin off-by-one ni fuga entre tickers); el problema de skill no viene de la etiqueta.

## 2. Modelo "boba": solo features técnicas (mismo split)

Features técnicas usadas (11): ['close', 'volume', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_pct']

| split | accuracy | AUC | % pred sube | CM [TN,FP,FN,TP] |
|---|---|---|---|---|
| val | 0.5033 | 0.4893 | 67.7% | [38, 90, 59, 113] |
| test | 0.5282 | 0.5445 | 53.1% | [104, 105, 71, 93] |

**Interpretación:** el modelo SOLO técnico da AUC = **0.544** en test (≈ azar). El problema **NO es el sentimiento**: esta ventana (5 tickers, horizonte de 1 día, este set de indicadores) simplemente **no es predecible** con estas features. El sentimiento no está 'rompiendo' una señal técnica que tampoco existe.

## 3. ¿Hay señal? Spearman feature[T] ↔ retorno[T+1]

Top 10 por |ρ| (sobre todo el dataset, n = 1863 filas):

| # | feature | Spearman ρ | p-value | tipo |
|---|---|---|---|---|
| 1 | prob_negative_mean | +0.0554 | 0.017 | sentimiento |
| 2 | negative_ratio | +0.0554 | 0.017 | sentimiento |
| 3 | net_sentiment_mean | -0.0540 | 0.020 | sentimiento |
| 4 | bb_middle | -0.0486 | 0.036 | técnica |
| 5 | close | -0.0475 | 0.041 | técnica |
| 6 | bb_lower | -0.0474 | 0.041 | técnica |
| 7 | bb_upper | -0.0450 | 0.052 | técnica |
| 8 | rsi | -0.0299 | 0.197 | técnica |
| 9 | neutral_ratio | -0.0298 | 0.199 | sentimiento |
| 10 | macd_hist | -0.0266 | 0.251 | técnica |

**Resultado:** la correlación máxima en |ρ| es **0.0554** (prob_negative_mean), que explica solo **ρ² ≈ 0.3%** de la varianza del retorno. 3 de 25 features cruzan |ρ| = 0.05, todas en la banda 0.05–0.06.

**Interpretación:** aunque 3 features de sentimiento cruzan por poco el 0.05 (significativas con p≈0.02 solo porque n=1863 es grande), el **tamaño de efecto es despreciable**: un |ρ| de ~0.055 explica ~0.3% de la varianza. En la práctica **no hay señal explotable a 1 día**, ni de sentimiento ni técnica. La significancia estadística aquí no implica utilidad de trading.

## Veredicto

- Target: correcto ✓
- Modelo solo-técnico AUC test: 0.544 (sin señal técnica)
- Spearman máx |ρ|: 0.0554 (ρ² ≈ 0.3%) (señal despreciable)

**Conclusión:** el target es correcto y **no hay señal predictiva útil a 1 día** en este conjunto (ni técnica ni de sentimiento; máx ρ²≈0.3%). El bajo AUC NO es culpa del sentimiento ni de un bug: es un **problema de fondo de predictibilidad** para este horizonte/universo. Antes de invertir en el adaptativo conviene **cambiar el problema**: horizonte más largo (p. ej. 3-5 días), target distinto (magnitud, o exceso sobre el mercado/sector), o más tickers para tener más señal. **Un modelo adaptativo sobre una señal inexistente seguirá sin discriminar** — el adaptativo ayuda contra el cambio de régimen, no contra la ausencia de señal.
