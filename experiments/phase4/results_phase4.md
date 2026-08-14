# Fase 4 — Resultados (10 modelos + 10 placebos)

> **Criterio pre-registrado (REGISTRY.md):** ÉXITO ⇔ AUC test > 0.55 Y AUC test > AUC placebo. Sin ajuste post-hoc. Modelo A = StockTwits (label nativo), Modelo B = Noticias FNSPID (FinBERT sobre titulares). Solo features de sentimiento.

**Experimentos ÉXITO: 0 / 10.** AUC media — A (StockTwits): 0.5085 | B (Noticias): 0.4887. Placebo medio — A: 0.5178 | B: 0.4846.

## Tabla resumen

| activo | fuente | n test | AUC | acc | prec sube | prec baja | AUC placebo | top features (gain) | veredicto |
|--------|--------|--------|-----|-----|-----------|-----------|-------------|---------------------|-----------|
| TSLA | A_StockTwits | 502 | 0.4876 | 0.5100 | 0.5115 | 0.4800 | 0.4952 | st_net=3.7, st_mom=3.5, st_net_3d=3.5 | neg |
| TSLA | B_Noticias | 502 | 0.4943 | 0.5080 | 0.5101 | 0.3750 | 0.4844 | nw_net_3d=2.5, nw_mom=1.8, nw_vol=1.8 | neg |
| AMD | A_StockTwits | 502 | 0.4861 | 0.4781 | 0.4781 | 0.0000 | 0.5311 | st_vol=3.9, st_mom=3.8, st_net=3.6 | neg |
| AMD | B_Noticias | 502 | 0.4932 | 0.4841 | 0.4740 | 0.5109 | 0.4650 | nw_net_3d=2.7, nw_net=2.5, nw_mom=2.4 | neg |
| DIS | A_StockTwits | 502 | 0.5283 | 0.4622 | 0.4622 | 0.0000 | 0.5202 | st_net=3.8, st_mom=3.8, st_net_3d=3.4 | neg |
| DIS | B_Noticias | 502 | 0.4847 | 0.4622 | 0.4622 | 0.0000 | 0.5051 | nw_vol=2.0, nw_net_3d=1.9, nw_net=1.7 | neg |
| BA | A_StockTwits | 502 | 0.5132 | 0.5179 | 0.4965 | 0.6364 | 0.5005 | st_net=3.3, st_vol=3.2, st_net_3d=3.2 | neg |
| BA | B_Noticias | 502 | 0.4865 | 0.4861 | 0.4798 | 0.5806 | 0.4736 | nw_mom=3.1, nw_net=3.0, nw_net_3d=2.8 | neg |
| GILD | A_StockTwits | 502 | 0.5272 | 0.5120 | 0.5120 | 0.0000 | 0.5418 | st_net_3d=3.9, st_net=3.8, st_mom=3.6 | neg |
| GILD | B_Noticias | 502 | 0.4848 | 0.5120 | 0.5120 | 0.0000 | 0.4948 | nw_net=3.6, nw_net_3d=3.2, nw_vol=3.2 | neg |

## Veredicto agregado

**NEGATIVO en los 10 experimentos.** Ni el sentimiento de StockTwits ni el de noticias FNSPID predice la dirección del retorno del día siguiente por encima del azar (AUC>0.55) y del placebo, en ninguno de los 5 activos. Consistente con el resultado negativo de las Fases 1-3 sobre el universo amplio: concentrarse en 5 activos de alta actividad y añadir noticias no revierte la conclusión.
