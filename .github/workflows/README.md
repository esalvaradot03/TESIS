# Workflows de GitHub Actions — Sistema de Trading

Dos workflows automatizan el paper trading en vivo y el reporte diario.

## `trading.yml` — Live Paper Trading
- **Schedule:** cada 30 min, lun–vie, `0,30 13-21 * * 1-5` (UTC). Cubre el horario
  de mercado 9:30–16:00 ET tanto en EST como en EDT. El cron "sobre-dispara" y el
  chequeo `get_clock().is_open` de Alpaca (que ya contempla feriados) decide si
  realmente se opera → no-op fuera de horario.
- **Flujo:** `scraper_stocktwits` (trending ∩ S&P500) → `preprocessor` →
  `finbert_scorer` → `indicators` → `feature_engine` → predicción XGBoost
  (`models/model_v0`) → `alpaca_executor`. Todo en `src/trading/live_runner.py`.
- **Run logs:** cada corrida escribe `data/live/run_<ts>.parquet` (features +
  predicción + precio) y los **commitea de vuelta al repo** para acumular el
  dataset de reentrenamiento.
- **Caché:** pip (vía `setup-python`) y el modelo FinBERT (~438 MB) en
  `~/.cache/huggingface` con clave estable, así no se re-descarga en cada run.

## `eod_report.yml` — Reporte End-of-Day
- **Schedule:** `30 21 * * 1-5` (UTC), tras el cierre. Genera `scripts/eod_report.py`:
  equity, P&L del día y posiciones abiertas con P&L no realizado.
- Escribe `data/live/reports/eod_<fecha>.{md,json}`, los commitea y los añade al
  *step summary* del run.

## Secrets requeridos (Settings → Secrets and variables → Actions)
StockTwits NO requiere key (endpoint público read-only). Solo Alpaca:

| Secret | Ejemplo |
|--------|---------|
| `ALPACA_API_KEY` | `PK...` |
| `ALPACA_API_SECRET` | `...` |
| `ALPACA_BASE_URL` | `https://paper-api.alpaca.markets` (sin `/v2`) |
| `ALPACA_DATA_URL` | `https://data.alpaca.markets` |
| `ALPACA_FEED` | `iex` |

## Probar manualmente (workflow_dispatch)
Desde la pestaña **Actions** → *Live Paper Trading* → **Run workflow**:
- `dry_run = true` → ejecuta todo el flujo (scrape, FinBERT, features, predicción,
  run log) **sin enviar órdenes** a Alpaca. Ideal para validar el pipeline.
- `force = true` → opera aunque el mercado esté cerrado (para probar fuera de
  horario; útil combinado con `dry_run`).
- `max_symbols` → cuántos trending symbols operar.

O por CLI con `gh`:
```bash
gh workflow run trading.yml -f dry_run=true -f force=true -f max_symbols=10
gh workflow run eod_report.yml
```

Localmente (con `.env` y el venv activo):
```powershell
python -m src.trading.live_runner --dry-run --force --max-symbols 10
python scripts/eod_report.py
```

## Permisos
Ambos workflows usan `permissions: contents: write` para commitear run logs y
reportes. El `.gitignore` re-incluye explícitamente `data/live/**` y
`models/model_v0/**` (ignorados por las reglas generales de `data/`, `models/` y
`*.parquet`).
