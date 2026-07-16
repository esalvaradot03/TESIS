# Resultados experimentos v1 (protocolo pre-registrado)

Criterio de éxito (fijo): precisión decil superior >= 55% Y Sharpe backtest > 0.5 Y >= 100 trades Y el placebo E8 NO cumple lo mismo.

- E8 (placebo) cumpliría el umbral por sí solo: **no** (si SÍ, invalida cualquier 'señal' que dependa del sentimiento).

## Tabla de métricas (walk-forward, test 2017-2023)

| Exp | config | AUC agg | %meses>0.55 | prec. decil | Sharpe (bt) | cum exceso | trades | veredicto |
|-----|--------|--------|-------------|-------------|-------------|-----------|--------|-----------|
| E1 | 1d, universo completo, híbrido | 0.5036 | 0% | 0.505 | -1.45 | -83.58% | 20570 | no |
| E2 | 1d, ex-top50, híbrido | 0.4992 | 1% | 0.493 | -1.51 | -92.21% | 13773 | no |
| E3 | 5d, ex-top50, híbrido | 0.5027 | 7% | 0.504 | 0.09 | -1.37% | 3302 | no |
| E4 | 10d, ex-top50, híbrido | 0.4996 | 7% | 0.499 | -0.16 | -27.38% | 1832 | no |
| E5 | 1d, ex-top50, evento, híbrido | 0.4994 | 22% | 0.467 | -2.09 | -70.44% | 634 | no |
| E6 | 5d, ex-top50, evento, híbrido | 0.4982 | 32% | 0.473 | -1.70 | -33.30% | 156 | no |
| E7 | 5d, ex-top50, SOLO técnico (control) | 0.4987 | 7% | 0.507 | -0.00 | -15.01% | 2843 | no |
| E8 | 5d, ex-top50, placebo (sent permutado) | 0.5045 | 6% | 0.504 | -0.11 | -29.86% | 2834 | no |

## E7 (solo técnico) vs experimentos con sentimiento

- E7 (solo técnico, 5d): AUC 0.4987, prec.decil 0.507, Sharpe -0.00.
- E3 (con sentimiento): AUC 0.5027 (Δ vs E7 = +0.0040), prec.decil 0.504, Sharpe 0.09.
- E6 (con sentimiento): AUC 0.4982 (Δ vs E7 = -0.0006), prec.decil 0.473, Sharpe -1.70.

Si los experimentos con sentimiento NO superan a E7, el sentimiento no aporta sobre lo técnico (respuesta a la pregunta de la tesis).
