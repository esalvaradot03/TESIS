# Fase 5.1 — Correlación contemporánea sentimiento ↔ retorno mismo día

> **Criterio (REGISTRY.md):** CONFIRMA ⇔ Spearman p<0.01 completo, ρ>0, signo estable (train y test ambos ρ>0 con p<0.01), y supera placebo permutado (p emp<0.01). Diagnóstico de contenido, NO predicción. Correlaciones sobre días con actividad real de la fuente (`*_has==1`).

**Pares que CONFIRMAN H5.1: 8 / 10.**

## Correlación mismo día (Spearman ρ / p)

| activo | fuente | n | ρ completo | p | Pearson r | ρ train | ρ test | placebo p | react ρ(net,ret_ayer) | veredicto |
|--------|--------|---|-----------|---|-----------|---------|--------|-----------|-----------------------|-----------|
| TSLA | A_StockTwits | 2012 | 0.5138 | 5.5e-136 | 0.4783 | 0.5227 | 0.5148 | 0.005 | 0.2572 (9.5e-32) | **CONFIRMA** |
| TSLA | B_Noticias | 398 | 0.4257 | 6.0e-19 | 0.3494 | 0.4419 | 0.3344 | 0.005 | 0.1259 (0.012) | **CONFIRMA** |
| AMD | A_StockTwits | 2008 | 0.3599 | 1.8e-62 | 0.3239 | 0.3414 | 0.4373 | 0.005 | 0.1301 (5.0e-09) | **CONFIRMA** |
| AMD | B_Noticias | 1444 | 0.1166 | 9.0e-06 | 0.1191 | 0.1004 | 0.1428 | 0.005 | 0.0599 (0.023) | **CONFIRMA** |
| DIS | A_StockTwits | 2012 | 0.2897 | 3.3e-40 | 0.2319 | 0.2992 | 0.2752 | 0.005 | 0.1336 (1.8e-09) | **CONFIRMA** |
| DIS | B_Noticias | 891 | 0.1130 | 7.2e-04 | 0.1047 | 0.1009 | 0.1224 | 0.005 | 0.0383 (0.253) | no |
| BA | A_StockTwits | 2000 | 0.2946 | 2.4e-41 | 0.2454 | 0.2957 | 0.3536 | 0.005 | 0.1318 (3.3e-09) | **CONFIRMA** |
| BA | B_Noticias | 869 | 0.1777 | 1.3e-07 | 0.1787 | 0.1700 | 0.1918 | 0.005 | 0.0549 (0.106) | **CONFIRMA** |
| GILD | A_StockTwits | 2008 | 0.2575 | 8.9e-32 | 0.2322 | 0.2881 | 0.1758 | 0.005 | 0.1254 (1.7e-08) | **CONFIRMA** |
| GILD | B_Noticias | 1851 | 0.1018 | 1.1e-05 | 0.0795 | 0.1086 | 0.0721 | 0.005 | 0.0382 (0.100) | no |

## Interpretación

- Correlación contemporánea positiva y significativa (p<0.01, completo) en **10/10** pares.
- La columna `react ρ(net,ret_ayer)` mide si el sentimiento de hoy correlaciona con el retorno de AYER (reactividad al precio). Si es del mismo orden que la correlación contemporánea, el sentimiento refleja el movimiento ya ocurrido más que aportar información nueva.
