# Resultados Fase 2 — Etapa B (survivors-only small/mid-cap)

> **Pre-registro (ver README):** test survivors-only, sesgado A FAVOR de encontrar señal. (a) Si NO se cumple el criterio → negativo fuerte, hipótesis small-cap rechazada incluso en el mejor caso. (b) Si SÍ se cumple → NO confiable, atribuible a survivorship, no promocionable.

Criterio: prec. decil >= 55% Y Sharpe > 0.5 (10 bps/lado) Y F4 (placebo) no cumple.

- F4 (placebo) cumpliría por sí solo: **no**

## Tabla de métricas (walk-forward, test 2017-2022)

| Exp | config | AUC agg | %meses>0.55 | prec. decil | Sharpe (bt) | cum exceso | trades | veredicto |
|-----|--------|--------|-------------|-------------|-------------|-----------|--------|-----------|
| F1 | 5d, survivors completo, híbrido | 0.5260 | 22% | 0.465 | 0.97 | 11543.56% | 3809 | no |
| F2 | 5d, survivors, solo evento, híbrido | 0.5455 | 46% | 0.465 | 0.80 | 620.61% | 138 | no |
| F3 | 5d, survivors, SOLO técnico (control) | 0.5196 | 18% | 0.463 | 0.57 | 87219.11% | 4344 | no |
| F4 | 5d, survivors, placebo (sent permutado) | 0.5215 | 15% | 0.462 | 0.67 | 103686.70% | 3820 | no |

## Conexión con la interpretación pre-registrada

**Ningún experimento cumple el criterio** → por el pre-registro (a): **negativo fuerte. La hipótesis small-cap se rechaza incluso en el mejor caso sesgado a favor (survivors-only). El sentimiento retail no predice de forma explotable ni en small/mid-caps survivors.**
