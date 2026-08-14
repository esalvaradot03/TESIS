# Fase 5.2 — Sentimiento → actividad (volumen/rango anormal de mañana)

> **Criterio (REGISTRY.md):** ÉXITO ⇔ AUC(base+sent) − AUC(base) > 0.02 en test Y AUC(base+sent) > AUC(placebo). Baseline = autorregresivo (relvol/range rezagados 1-5). Sentimiento (10 features StockTwits+noticias) debe agregar valor SOBRE ese baseline. Placebo = sentimiento permutado, autorregresivas intactas.

**Settings ÉXITO: 0 / 10** (5 activos × 2 targets).

| activo | target | n test | AUC base | AUC base+sent | ΔAUC | AUC placebo | top features (gain) | veredicto |
|--------|--------|--------|----------|---------------|------|-------------|---------------------|-----------|
| TSLA | volumen | 502 | 0.7356 | 0.7486 | +0.0131 | 0.7304 | relvol_l1=18.2, relvol_l2=6.7, range_l1=5.6, st_vol=5.1 | neg |
| TSLA | rango | 502 | 0.7871 | 0.8038 | +0.0167 | 0.7924 | range_l1=15.1, st_vol=10.1, range_l3=7.8, range_l5=6.2 | neg |
| AMD | volumen | 502 | 0.7939 | 0.7926 | -0.0013 | 0.7859 | relvol_l1=25.2, relvol_l2=8.4, nw_net_3d=6.5, relvol_l3=6.0 | neg |
| AMD | rango | 502 | 0.7458 | 0.7428 | -0.0030 | 0.7460 | range_l1=24.2, range_l2=10.9, range_l3=9.9, relvol_l1=8.2 | neg |
| DIS | volumen | 502 | 0.7561 | 0.7485 | -0.0075 | 0.7476 | relvol_l1=18.0, relvol_l2=7.8, relvol_l4=5.9, range_l5=5.0 | neg |
| DIS | rango | 502 | 0.8220 | 0.8200 | -0.0019 | 0.8020 | range_l1=11.7, range_l4=9.7, nw_has=6.9, range_l2=6.8 | neg |
| BA | volumen | 502 | 0.7089 | 0.7018 | -0.0071 | 0.7103 | relvol_l1=20.6, relvol_l2=8.7, nw_mom=5.5, range_l1=5.4 | neg |
| BA | rango | 502 | 0.7902 | 0.7750 | -0.0153 | 0.7898 | range_l1=14.3, nw_has=12.4, st_vol=7.5, range_l3=6.3 | neg |
| GILD | volumen | 502 | 0.6995 | 0.7005 | +0.0010 | 0.7027 | relvol_l1=16.5, relvol_l2=6.3, st_vol=5.8, relvol_l3=5.7 | neg |
| GILD | rango | 502 | 0.6243 | 0.6108 | -0.0135 | 0.6163 | range_l1=13.2, range_l3=12.0, range_l2=9.1, range_l4=6.1 | neg |

## Lectura

- AUC baseline autorregresivo medio: **0.7463** (volumen y rango son fuertemente autocorrelacionados, como se esperaba).
- Mejora media al añadir sentimiento (ΔAUC): **-0.0019**.
- **Ningún setting supera el umbral incremental de +0.02 sobre el baseline batiendo al placebo.** El sentimiento no aporta valor predictivo sobre la actividad de mañana más allá del propio pasado autorregresivo del activo.
