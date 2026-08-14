"""
Fase 7 — preparación de datos (caches). No hace análisis; solo materializa paneles.

Produce en experiments/phase7/cache/:
  - daily_sentiment_5.parquet : (date, ticker, bull, bear, n_msgs) de symbol_sentiments,
    5 activos de Fase 4, 2015-2022 (label date-only, sirve para diario).
  - daily_ohlcv_5.parquet     : OHLCV diario yfinance, 5 activos, 2015-2022.
  - earnings_5.parquet        : fechas de earnings (yfinance) por activo.
  - intraday_all_5.parquet    : (message_id, et, ticker) de feature_wo_messages para
    TSLA/AMD, TODAS las horas (incl. pre/post-market) 2020-08→2022-12 — para la
    dimensión OVERNIGHT (nunca usada; en Fase 6 se excluyó fuera de horario).
Reutiliza de Fase 6: alpaca_30min.parquet (barras) y labeled_msgids.parquet (labels).
"""

import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import (  # noqa: E402
    EXTERNAL_DATA_ROOT,
    STOCKTWITS_NYU_SYMBOL_SENTIMENTS,
)

_FWM = EXTERNAL_DATA_ROOT / "stocktwits_nyu" / "feature_wo_messages"
_OUT = ROOT / "experiments" / "phase7"
_CACHE = _OUT / "cache"
_P6 = ROOT / "experiments" / "phase6" / "cache"

ASSETS5 = ["TSLA", "AMD", "DIS", "BA", "GILD"]
INTRADAY = ["TSLA", "AMD"]
_S, _E = pd.Timestamp("2015-01-01"), pd.Timestamp("2022-12-31")
_IS = pd.Timestamp("2020-08-01", tz="America/New_York")
_IE = pd.Timestamp("2022-12-31 23:59:59", tz="America/New_York")
_CHUNK = 1_000_000
_RE5 = re.compile(r"'(" + "|".join(ASSETS5) + r")'")
_HAS5 = re.compile(r"'(?:" + "|".join(ASSETS5) + r")'")
_REI = re.compile(r"'(TSLA|AMD)'")
_HASI = re.compile(r"'(?:TSLA|AMD)'")


def scan_daily_sentiment(log) -> pd.DataFrame:
    cache = _CACHE / "daily_sentiment_5.parquet"
    if cache.exists():
        log.info("daily_sentiment: cache"); return pd.read_parquet(cache)
    files = sorted(STOCKTWITS_NYU_SYMBOL_SENTIMENTS.glob("*.csv"))
    tset, parts, t0 = set(ASSETS5), [], time.time()
    for fi, f in enumerate(files, 1):
        reader = pd.read_csv(f, usecols=["created_at", "sentiment", "symbol_list"],
                             dtype="string", chunksize=_CHUNK, on_bad_lines="skip",
                             encoding="utf-8", encoding_errors="replace")
        for chunk in reader:
            hit = chunk["symbol_list"].str.contains(_HAS5, na=False)
            if not hit.any():
                continue
            sub = chunk[hit]
            d = pd.to_datetime(sub["created_at"], errors="coerce")
            s = pd.to_numeric(sub["sentiment"], errors="coerce")
            df = pd.DataFrame({"date": d, "sent": s,
                               "ticker": sub["symbol_list"].map(_RE5.findall)})
            df = df.explode("ticker").dropna(subset=["date", "ticker"])
            df = df[(df["date"] >= _S) & (df["date"] <= _E)]
            if df.empty:
                continue
            df["date"] = df["date"].dt.normalize()
            g = df.groupby(["ticker", "date"])
            parts.append(pd.DataFrame({
                "bull": g["sent"].apply(lambda x: int((x > 0).sum())),
                "bear": g["sent"].apply(lambda x: int((x < 0).sum())),
                "n_msgs": g.size()}).reset_index())
        if fi % 10 == 0:
            parts = [pd.concat(parts, ignore_index=True).groupby(["ticker", "date"], as_index=False).sum()]
            log.info("  daily_sent %d/%d (%.0fs)", fi, len(files), time.time() - t0)
    res = pd.concat(parts, ignore_index=True).groupby(["ticker", "date"], as_index=False).sum()
    _CACHE.mkdir(parents=True, exist_ok=True)
    res.to_parquet(cache, index=False)
    log.info("daily_sentiment: %d filas", len(res)); return res


def fetch_daily_ohlcv(log) -> pd.DataFrame:
    cache = _CACHE / "daily_ohlcv_5.parquet"
    if cache.exists():
        log.info("daily_ohlcv: cache"); return pd.read_parquet(cache)
    import yfinance as yf
    rows = []
    for t in ASSETS5:
        h = yf.Ticker(t).history(start="2015-01-01", end="2023-01-01", auto_adjust=True)
        h = h.reset_index()[["Date", "Open", "High", "Low", "Close", "Volume"]]
        h.columns = ["date", "open", "high", "low", "close", "volume"]
        h["date"] = pd.to_datetime(h["date"]).dt.tz_localize(None).dt.normalize()
        h["ticker"] = t
        rows.append(h); time.sleep(0.3)
        log.info("  ohlcv %s: %d", t, len(h))
    res = pd.concat(rows, ignore_index=True)
    res.to_parquet(cache, index=False); return res


def fetch_earnings(log) -> pd.DataFrame:
    cache = _CACHE / "earnings_5.parquet"
    if cache.exists():
        log.info("earnings: cache"); return pd.read_parquet(cache)
    import yfinance as yf
    rows = []
    for t in ASSETS5:
        try:
            ed = yf.Ticker(t).get_earnings_dates(limit=100)
            if ed is not None and len(ed):
                dts = pd.to_datetime(ed.index).tz_localize(None).normalize().unique()
                for d in dts:
                    rows.append({"ticker": t, "earnings_date": d})
        except Exception as e:  # noqa: BLE001
            log.warning("  earnings %s: %s", t, e)
        time.sleep(0.3)
    res = pd.DataFrame(rows)
    res = res[(res["earnings_date"] >= _S) & (res["earnings_date"] <= _E)]
    res.to_parquet(cache, index=False)
    log.info("earnings: %d fechas (%d por activo aprox)", len(res),
             len(res) // max(1, res["ticker"].nunique()) if len(res) else 0)
    return res


def scan_intraday_all(log) -> pd.DataFrame:
    """feature_wo_messages TSLA/AMD, TODAS las horas, 2020-08→2022-12 (para overnight)."""
    cache = _CACHE / "intraday_all_5.parquet"
    if cache.exists():
        log.info("intraday_all: cache"); return pd.read_parquet(cache)
    files = sorted(_FWM.glob("*.csv"))
    parts, t0 = [], time.time()
    for fi, f in enumerate(files, 1):
        try:
            reader = pd.read_csv(f, usecols=["message_id", "created_at", "symbol_list"],
                                 dtype="string", chunksize=_CHUNK, on_bad_lines="skip",
                                 encoding="utf-8", encoding_errors="replace")
        except Exception:  # noqa: BLE001
            continue
        for chunk in reader:
            hit = chunk["symbol_list"].str.contains(_HASI, na=False)
            if not hit.any():
                continue
            sub = chunk[hit]
            t = pd.to_datetime(sub["created_at"], errors="coerce", utc=True)
            d = pd.DataFrame({"message_id": sub["message_id"],
                              "et": t.dt.tz_convert("America/New_York"),
                              "ticker": sub["symbol_list"].map(_REI.findall)})
            d = d.explode("ticker").dropna(subset=["et", "ticker"])
            d = d[(d["et"] >= _IS) & (d["et"] <= _IE)]
            if not d.empty:
                parts.append(d)
        if fi % 40 == 0:
            log.info("  intraday_all %d/%d (%.0fs, %d)", fi, len(files),
                     time.time() - t0, sum(len(p) for p in parts))
    res = pd.concat(parts, ignore_index=True)
    res.to_parquet(cache, index=False)
    log.info("intraday_all: %d filas", len(res)); return res


def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("phase7_prep")
    scan_daily_sentiment(log)
    fetch_daily_ohlcv(log)
    fetch_earnings(log)
    scan_intraday_all(log)
    log.info("Prep Fase 7 completo.")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run()
