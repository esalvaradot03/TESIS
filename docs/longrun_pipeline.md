# Pipeline de larga duración — dataset maestro

Construye `D:\trading-data\master\dataset_v1.parquet` uniendo, por (ticker, día)
sobre el S&P 500: sentimiento StockTwits (NYU) + menciones WSB + noticias FNSPID
+ indicadores técnicos sobre precios Yahoo. Pensado para correr **desatendido
varios días** y sobrevivir crashes/cortes de luz/archivos corruptos.

Todo el código vive en `src/data/longrun/`. Todos los datos viven en
`D:\trading-data\` (fuera del repo).

## Arquitectura (5 jobs + unión + orquestador)

| Job | Módulo | Entrada → Salida |
|-----|--------|------------------|
| 1 | `process_stocktwits_nyu.py` | `stocktwits_nyu/symbol_sentiments/*.csv` → `master/stocktwits_sentiment_daily.parquet` (particionado por año) |
| 2 | `process_wsb.py` | `wsb_kaggle/unanimad/*.csv` + `kevinwang313/*.sql` → `master/wsb_mentions_daily.parquet` |
| 3 | `process_fnspid.py` | `fnspid/nasdaq_exteral_data.csv` (23 GB) → `master/fnspid_news_daily.parquet` |
| 4 | `download_prices_yahoo.py` | yfinance (S&P500 + SPY) → `master/prices/<TICKER>.parquet` + `master/prices_2008_2024.parquet` |
| 5 | `compute_indicators_long.py` | precios → `master/indicators_daily.parquet` (features estacionarias + retornos forward) |
| 6 | `build_master_dataset.py` | une 1+2+3+5 → `master/dataset_v1.parquet` |
| — | `run_longrun.py` | orquesta: 1-4 en paralelo → 5 → 6 |

### Robustez (requisito crítico)
- **Reanudable:** checkpoint JSON atómico por archivo/chunk en
  `logs/checkpoints/<job>.json`. Al re-correr, salta lo ya hecho.
- **Atómico:** todo Parquet/checkpoint se escribe a `.tmp` y se renombra → un
  corte de luz no deja archivos a medias.
- **Aislado:** `try/except` por archivo/chunk/ticker; un dato corrupto se loguea
  y NO mata el job. Si un job entero falla, los demás siguen y el Job 6 se
  construye con lo disponible (reporta qué faltó).
- **Lock + DONE:** el orquestador toma un lock por PID (no corre dos instancias)
  y al completar escribe `logs/ORCHESTRATOR_DONE` para no re-ejecutar en vano.
- **Logs:** uno por job en `D:\trading-data\logs\<job>.log` + `orchestrator.log`,
  con timestamp por mensaje. Reporte final en `logs/REPORT.md`.

## Cómo correr

```powershell
.\.venv\Scripts\Activate.ps1
# Smoke test (subconjuntos) — valida el flujo en miniatura:
python -m src.data.longrun.run_longrun --smoke
# Corrida real:
python -m src.data.longrun.run_longrun
# Forzar re-corrida tras completar (ignora el marcador DONE):
python -m src.data.longrun.run_longrun --force
```

Si crashea o hay corte de luz: **volvé a correr el mismo comando** y retoma donde
quedó (los jobs ya completados se saltan; las consolidaciones se rehacen).

### Auto-resume tras reinicio (opcional, requiere admin)
Para que se reanude solo al encender la máquina y cada 2 h, registrá la tarea
(PowerShell **como administrador**):

```powershell
$a = New-ScheduledTaskAction -Execute "C:\dev\Tesis\scripts\resume_longrun.cmd"
$t1 = New-ScheduledTaskTrigger -AtLogOn
$t2 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(5) `
      -RepetitionInterval (New-TimeSpan -Hours 2) -RepetitionDuration (New-TimeSpan -Days 5)
$s = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
Register-ScheduledTask -TaskName "TesisLongrunResume" -Action $a -Trigger $t1,$t2 -Settings $s -Force
```

El lock + DONE hacen que estas invocaciones sean idempotentes (no-op si ya corre
o ya completó).

## dataset_v1 — esquema

Por (ticker, fecha):
- **Sentimiento StockTwits:** bullish_count, bearish_count, total_count,
  bullish_ratio, unique_users.
- **WSB:** mention_count, avg_score, total_comments.
- **Noticias FNSPID:** news_count, news_sentiment_mean.
- **Técnico (estacionario):** rsi, macd_hist, bb_pct, bb_width, volatility_20d,
  volume_ratio.
- **Retornos:** return_5d_forward, return_excess_over_spy_5d_forward.
- **Target:** `1` si return_excess_over_spy_5d_forward > 0.
- Filtro de actividad: solo filas con `combined_mentions` (StockTwits + WSB) ≥ 5.

## Decisiones / desviaciones documentadas

1. **FNSPID NO traía sentimiento precalculado** (solo título + resúmenes). Para no
   usar FinBERT (sin GPU) se calcula `news_sentiment_mean` con **VADER** (léxico,
   CPU) sobre el título de las filas ya filtradas a S&P 500. Sustituto barato y honesto.
2. **kevin son dumps mysqldump**, no corren en SQLite (dialecto + escapes `\'`).
   Se parsean con un tokenizer propio de `INSERT ... VALUES`. La tabla
   `reddit_posts` ya trae columna `symbol` (ticker) → no hace falta regex.
   Solo ~8 % de los posts traen ticker; el resto se descarta. Si el dump falla,
   se loguea y se sigue con unanimad.
3. **S&P 500 = lista ACTUAL** (Wikipedia/sp500_loader). LIMITACIÓN: no incluye
   miembros históricos que salieron del índice 2008-2024; esas menciones se
   descartan. Ampliable con una lista histórica.
4. **yfinance 0.2.43 fallaba** (`YFTzMissingError` con la API actual). Se subió a
   **1.4.1**. Además, si yfinance falla para un ticker, hay **fallback sin red** a
   `fnspid/full_history/<TICKER>.csv` (OHLCV ~1980-2023; no cubre 2024).
5. **Dask vs pandas:** se priorizó resumibilidad por archivo/chunk (pandas en
   streaming, checkpoints atómicos) sobre "todo Dask". Dask se usa para la
   consolidación out-of-core (groupby final de Jobs 1 y 3). `nunique` se calcula
   como operación Dask aparte (no está soportado en `groupby.agg`).

## Salidas en `D:\trading-data\`
- `master/*.parquet` — salidas intermedias + `dataset_v1.parquet`.
- `master/prices/<TICKER>.parquet` — un Parquet por ticker (reanudable).
- `staging/<job>/` — parciales por archivo/chunk (intermedios).
- `logs/<job>.log`, `logs/orchestrator.log`, `logs/REPORT.md`.
