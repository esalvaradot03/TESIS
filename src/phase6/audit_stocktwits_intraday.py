"""
Fase 6 — Tarea 1: auditoría intradía de StockTwits (TSLA, AMD, 2016-2022).

Los timestamps intradía viven en `feature_wo_messages` (created_at ISO-8601 con Z =
UTC), NO en symbol_sentiments (que es fecha-solo). Este script escanea esos ~29GB,
filtra TSLA/AMD, convierte UTC→ET (America/New_York, maneja DST), filtra horario de
mercado (09:30–16:00 ET, días hábiles) y cuenta mensajes por ventana de 30 min.

Salida: experiments/phase6/cache/st_intraday_counts.parquet
        (ticker, w30 [inicio ventana 30min ET], n_msgs, n_labeled)
Las ventanas de 60 min, percentiles, %<5 y desglose por año se derivan en audit_report.py.
"""

import logging
import re
import sys
import time
from datetime import time as dtime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import EXTERNAL_DATA_ROOT  # noqa: E402

_FWM = EXTERNAL_DATA_ROOT / "stocktwits_nyu" / "feature_wo_messages"
_OUT = ROOT / "experiments" / "phase6"
_CACHE = _OUT / "cache"

TICKERS = ["TSLA", "AMD"]
_WIN_S = pd.Timestamp("2016-01-01", tz="America/New_York")
_WIN_E = pd.Timestamp("2022-12-31 23:59:59", tz="America/New_York")
_MKT_OPEN, _MKT_CLOSE = dtime(9, 30), dtime(16, 0)
_CHUNK = 1_000_000
_TICK_RE = re.compile(r"'(TSLA|AMD)'")
_HAS_RE = re.compile(r"'(?:TSLA|AMD)'")


def run() -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("phase6_st_audit")
    cache = _CACHE / "st_intraday_counts.parquet"
    if cache.exists():
        log.info("Cache existe: %s", cache.name)
        return pd.read_parquet(cache)

    files = sorted(_FWM.glob("*.csv"))
    log.info("feature_wo_messages: %d archivos (~29GB).", len(files))
    parts, t0 = [], time.time()
    for fi, f in enumerate(files, 1):
        try:
            reader = pd.read_csv(f, usecols=["created_at", "sentiment", "symbol_list"],
                                 dtype={"created_at": "string", "symbol_list": "string"},
                                 chunksize=_CHUNK, on_bad_lines="skip",
                                 encoding="utf-8", encoding_errors="replace")
        except Exception:  # noqa: BLE001
            continue
        for chunk in reader:
            hit = chunk["symbol_list"].str.contains(_HAS_RE, na=False)
            if not hit.any():
                continue
            sub = chunk[hit]
            t = pd.to_datetime(sub["created_at"], errors="coerce", utc=True)
            d = pd.DataFrame({
                "et": t.dt.tz_convert("America/New_York"),
                "ticker": sub["symbol_list"].map(_TICK_RE.findall),
                "labeled": pd.to_numeric(sub["sentiment"], errors="coerce").notna(),
            })
            d = d.explode("ticker").dropna(subset=["et", "ticker"])
            # rango + horario de mercado + días hábiles
            d = d[(d["et"] >= _WIN_S) & (d["et"] <= _WIN_E)]
            tod = d["et"].dt.time
            d = d[(tod >= _MKT_OPEN) & (tod < _MKT_CLOSE) & (d["et"].dt.weekday < 5)]
            if d.empty:
                continue
            d["w30"] = d["et"].dt.floor("30min")
            g = d.groupby(["ticker", "w30"])
            parts.append(pd.DataFrame({
                "n_msgs": g.size(),
                "n_labeled": g["labeled"].sum(),
            }).reset_index())
        if fi % 20 == 0:
            parts = [pd.concat(parts, ignore_index=True)
                     .groupby(["ticker", "w30"], as_index=False).sum()]
            log.info("  %d/%d archivos (%.0fs, %d ventanas)", fi, len(files),
                     time.time() - t0, len(parts[0]))
    res = (pd.concat(parts, ignore_index=True)
           .groupby(["ticker", "w30"], as_index=False).sum()
           .sort_values(["ticker", "w30"]))
    _CACHE.mkdir(parents=True, exist_ok=True)
    res.to_parquet(cache, index=False)
    log.info("StockTwits intradía: %d ventanas-30min con mensajes -> %s", len(res), cache.name)
    for tk in TICKERS:
        sub = res[res["ticker"] == tk]
        log.info("  %s: %d ventanas c/msgs, total %d msgs (%d labeled)",
                 tk, len(sub), int(sub["n_msgs"].sum()), int(sub["n_labeled"].sum()))
    return res


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run()
