# Fase 4 — Tarea 2: dataset construido (cortes exactos)

Un parquet por activo en `experiments/phase4/dataset_phase4/`. Features solo de sentimiento (5 StockTwits `st_*` + 5 Noticias `nw_*`), target = dirección del retorno del día siguiente. Días sin actividad → neutro/0 + binaria `*_has` (ver REGISTRY.md).

| activo | días | train n | test n | train | test | frontera excl | %días c/ST | %días c/news | %sube train | %sube test |
|--------|------|---------|--------|-------|------|---------------|-----------|-------------|-------------|------------|
| TSLA | 2013 | 1510 | 502 | 2015-01-02→2020-12-30 | 2021-01-04→2022-12-29 | 2020-12-31 | 100.0 | 19.8 | 51.7 | 51.2 |
| AMD | 2013 | 1510 | 502 | 2015-01-02→2020-12-30 | 2021-01-04→2022-12-29 | 2020-12-31 | 99.8 | 71.7 | 51.2 | 47.8 |
| DIS | 2013 | 1510 | 502 | 2015-01-02→2020-12-30 | 2021-01-04→2022-12-29 | 2020-12-31 | 100.0 | 44.3 | 51.4 | 46.2 |
| BA | 2013 | 1510 | 502 | 2015-01-02→2020-12-30 | 2021-01-04→2022-12-29 | 2020-12-31 | 99.4 | 43.2 | 52.1 | 47.6 |
| GILD | 2013 | 1510 | 502 | 2015-01-02→2020-12-30 | 2021-01-04→2022-12-29 | 2020-12-31 | 99.8 | 92.0 | 50.3 | 51.2 |
