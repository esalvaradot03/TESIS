"""
Fase 6 — Tarea 1: auditoría de barras intradía de Alpaca (IEX) para TSLA y AMD.

Descarga barras de 30 min 2016-2022 (feed IEX gratuito) y audita cobertura:
  - desde qué fecha hay datos consistentes,
  - % de ventanas de mercado (09:30–16:00 ET, 13 ventanas/día hábil) con barra,
  - gaps (ventanas de mercado sin barra) por año.

Salida: experiments/phase6/cache/alpaca_30min.parquet (ticker, t [ET], open..volume)
        + stats por consola. Define además el universo de ventanas de mercado que
        audit_report.py usa para el %<5 de StockTwits.
"""

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import ALPACA_API_KEY, ALPACA_API_SECRET, ALPACA_FEED  # noqa: E402

_OUT = ROOT / "experiments" / "phase6"
_CACHE = _OUT / "cache"
TICKERS = ["TSLA", "AMD"]
_START, _END = "2016-01-01", "2022-12-31"


def download(log) -> pd.DataFrame:
    cache = _CACHE / "alpaca_30min.parquet"
    if cache.exists():
        log.info("Alpaca barras: cache %s", cache.name)
        return pd.read_parquet(cache)

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

    client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_API_SECRET)
    feed = ALPACA_FEED
    rows = []
    for t in TICKERS:
        req = StockBarsRequest(
            symbol_or_symbols=t,
            timeframe=TimeFrame(30, TimeFrameUnit.Minute),
            start=pd.Timestamp(_START, tz="America/New_York"),
            end=pd.Timestamp(_END + " 23:59:59", tz="America/New_York"),
            feed=feed,
        )
        bars = client.get_stock_bars(req).df
        if bars.empty:
            log.warning("  %s: sin barras", t)
            continue
        bars = bars.reset_index()
        bars["t"] = pd.to_datetime(bars["timestamp"], utc=True).dt.tz_convert("America/New_York")
        bars["ticker"] = t
        rows.append(bars[["ticker", "t", "open", "high", "low", "close", "volume"]])
        log.info("  %s: %d barras (%s..%s)", t, len(bars),
                 bars["t"].min().date(), bars["t"].max().date())
    allb = pd.concat(rows, ignore_index=True)
    _CACHE.mkdir(parents=True, exist_ok=True)
    allb.to_parquet(cache, index=False)
    return allb


def audit(bars: pd.DataFrame, log) -> pd.DataFrame:
    """Restringe a ventanas de mercado y calcula cobertura por año/ticker."""
    from datetime import time as dtime
    b = bars.copy()
    tod = b["t"].dt.time
    b = b[(tod >= dtime(9, 30)) & (tod < dtime(16, 0)) & (b["t"].dt.weekday < 5)]
    b["date"] = b["t"].dt.normalize()
    b["year"] = b["t"].dt.year
    stats = []
    for t in TICKERS:
        sub = b[b["ticker"] == t]
        for yr in range(2016, 2023):
            sy = sub[sub["year"] == yr]
            n_days = sy["date"].nunique()
            # 13 ventanas de mercado por día hábil (09:30..15:30)
            expected = n_days * 13
            got = len(sy)
            stats.append({"ticker": t, "year": yr, "dias_con_barras": n_days,
                          "barras": got, "ventanas_esperadas": expected,
                          "cobertura_%": round(100 * got / expected, 1) if expected else 0.0,
                          "primera": str(sy["t"].min()) if got else "—"})
    sd = pd.DataFrame(stats)
    sd.to_parquet(_OUT / "alpaca_coverage.parquet", index=False)
    return sd


def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("phase6_alpaca_audit")
    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        log.error("Faltan credenciales Alpaca en .env")
        return
    bars = download(log)
    sd = audit(bars, log)
    log.info("Cobertura barras 30min (IEX):\n%s", sd.to_string(index=False))
    return sd


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run()
