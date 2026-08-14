# Fase 6 — dataset intradía construido (cortes exactos)

Par `[t-30,t]→[t,t+30]` estrictamente intradía (1a ventana objetivo de cada día excluida). Features AR (ret/range lags 1-5 + n_gap_lags) + sentimiento (net_lag1 principal, accel, vol). Split train 2020-08→2021-12 / test 2022. Ver REGISTRY.md.

| activo | ventana | train n | test n | train | test | %sube train | %sube test |
|--------|---------|---------|--------|-------|------|-------------|------------|
| TSLA | 30min | 4,279 | 3,007 | 2020-08-03→2021-12-31 | 2022-01-03→2022-12-30 | 49.5 | 49.4 |
| AMD | 30min | 4,278 | 3,006 | 2020-08-03→2021-12-31 | 2022-01-03→2022-12-30 | 50.8 | 49.4 |
| TSLA | 60min | 2,140 | 1,503 | 2020-08-03→2021-12-31 | 2022-01-03→2022-12-30 | 49.4 | 47.6 |
| AMD | 60min | 2,139 | 1,503 | 2020-08-03→2021-12-31 | 2022-01-03→2022-12-30 | 50.3 | 48.3 |
