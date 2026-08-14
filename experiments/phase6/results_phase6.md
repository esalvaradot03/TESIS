# Fase 6 — Resultados: sentimiento intradía (30 y 60 min)

> **Criterio (REGISTRY.md):** ÉXITO ⇔ ΔAUC(base+sent − base) > 0.02 en test Y AUC(full) > AUC(placebo). Baseline autorregresivo (ret/range lags 1-5 + n_gap_lags). NET etiquetado principal. Rango 2020-08→2022-12 (IEX); test 2022 = régimen bajista distinto al train alcista (limitación aceptada).

**Settings ÉXITO: 1 / 4.**

## Modelo predictivo (split principal: train 2020-08→2021-12 / test 2022)

| activo | ventana | n test | AUC base | AUC base+sent | ΔAUC | AUC placebo | veredicto |
|--------|---------|--------|----------|---------------|------|-------------|-----------|
| TSLA | 30min | 3,007 | 0.4886 | 0.4902 | +0.0017 | 0.4896 | neg |
| AMD | 30min | 3,006 | 0.4768 | 0.5020 | +0.0251 | 0.4926 | **ÉXITO** |
| TSLA | 60min | 1,503 | 0.5073 | 0.5027 | -0.0046 | 0.5082 | neg |
| AMD | 60min | 1,503 | 0.5223 | 0.5038 | -0.0185 | 0.5082 | neg |

## Diagnóstico de estructura temporal (Spearman NET[t-30,t] vs ret, en test)

Mapea si el sentimiento **coincide**, **sigue** (reactivo) o **anticipa** al precio a 30 min.

| activo | ventana | horizonte | Spearman | p | n |
|--------|---------|-----------|----------|---|---|
| TSLA | 30min | coincidente[t-30,t] | +0.4734 | 8.4e-168 | 3,007 |
| TSLA | 30min | reactivo[t-60,t-30] | +0.3693 | 8.0e-98 | 3,007 |
| TSLA | 30min | anticipatorio[t,t+30] | +0.0108 | 0.555 | 3,007 |
| AMD | 30min | coincidente[t-30,t] | +0.2911 | 8.9e-60 | 3,006 |
| AMD | 30min | reactivo[t-60,t-30] | +0.2467 | 6.5e-43 | 3,006 |
| AMD | 30min | anticipatorio[t,t+30] | -0.0247 | 0.176 | 3,006 |
| TSLA | 60min | coincidente[t-30,t] | +0.5443 | 1.1e-116 | 1,503 |
| TSLA | 60min | reactivo[t-60,t-30] | +0.3387 | 1.1e-41 | 1,503 |
| TSLA | 60min | anticipatorio[t,t+30] | +0.0137 | 0.597 | 1,503 |
| AMD | 60min | coincidente[t-30,t] | +0.3559 | 4.1e-46 | 1,503 |
| AMD | 60min | reactivo[t-60,t-30] | +0.2087 | 3.0e-16 | 1,503 |
| AMD | 60min | anticipatorio[t,t+30] | -0.0361 | 0.162 | 1,503 |

## Robustez walk-forward (30 min, expansiva, test por trimestre)

| activo | trimestre | n test | AUC base | AUC full | ΔAUC | AUC placebo |
|--------|-----------|--------|----------|----------|------|-------------|
| TSLA | 2020Q4 | 756 | 0.5128 | 0.4928 | -0.0200 | 0.5140 |
| TSLA | 2021Q1 | 732 | 0.5091 | 0.5176 | +0.0085 | 0.5016 |
| TSLA | 2021Q2 | 756 | 0.5448 | 0.4993 | -0.0455 | 0.5358 |
| TSLA | 2021Q3 | 768 | 0.4962 | 0.5069 | +0.0107 | 0.4962 |
| TSLA | 2021Q4 | 763 | 0.4808 | 0.5336 | +0.0528 | 0.4632 |
| TSLA | 2022Q1 | 744 | 0.5071 | 0.5136 | +0.0066 | 0.4990 |
| TSLA | 2022Q2 | 744 | 0.5103 | 0.5278 | +0.0175 | 0.5339 |
| TSLA | 2022Q3 | 768 | 0.5148 | 0.5158 | +0.0010 | 0.4942 |
| TSLA | 2022Q4 | 751 | 0.5114 | 0.4619 | -0.0496 | 0.4561 |
| AMD | 2020Q4 | 756 | 0.5394 | 0.5331 | -0.0064 | 0.5173 |
| AMD | 2021Q1 | 732 | 0.4889 | 0.4919 | +0.0030 | 0.4680 |
| AMD | 2021Q2 | 756 | 0.5299 | 0.5226 | -0.0072 | 0.5457 |
| AMD | 2021Q3 | 768 | 0.5295 | 0.5172 | -0.0123 | 0.5291 |
| AMD | 2021Q4 | 762 | 0.4976 | 0.5115 | +0.0139 | 0.4829 |
| AMD | 2022Q1 | 744 | 0.4667 | 0.4471 | -0.0196 | 0.4308 |
| AMD | 2022Q2 | 744 | 0.5128 | 0.5030 | -0.0098 | 0.4990 |
| AMD | 2022Q3 | 768 | 0.5065 | 0.4592 | -0.0473 | 0.5086 |
| AMD | 2022Q4 | 750 | 0.4989 | 0.5004 | +0.0015 | 0.4695 |

## Lectura

- ΔAUC medio (split principal): **+0.0009**; AUC baseline medio: **0.4988** (≈0.50: la dirección a 30 min es casi un martingala incluso desde su propio pasado).
- **Diagnóstico (resultado robusto):** correlación **coincidente** media |ρ|=0.416 (p≈1e-60…1e-168), **reactiva** |ρ|=0.291 (fuerte y significativa), pero **anticipatoria** |ρ|=0.021 (NO significativa, p>0.15 en los 4 settings).
- **Criterio principal:** 1/4 lo cumple (AMD 30min) — pero con AUC absoluto ≈0.50 y baseline anómalamente bajo.
  - **Robustez de AMD 30min (walk-forward):** ΔAUC trimestral medio **-0.0094**, con solo **0/9** trimestres superando +0.02. El 'éxito' del split único **NO se replica** fuera de muestra → atribuible al split particular de 2022, no a un efecto estable.
- **Conclusión Fase 6:** la hipótesis de que la dinámica predictiva vive dentro del día y la agregación diaria la destruye **queda refutada**. A 30 y 60 min el sentimiento sigue siendo **coincidente y reactivo pero NO anticipatorio** — la misma firma que a frecuencia diaria (5.1). No hay poder predictivo intradía estable sobre el baseline autorregresivo.
