"""
JOB 5 — Indicadores técnicos ESTACIONARIOS + retornos forward (depende del Job 4).

Entrada: D:\\trading-data\\master\\prices_2008_2024.parquet
Salida : D:\\trading-data\\master\\indicators_daily.parquet

Conforme al diagnóstico previo, SOLO features estacionarias (sin niveles
absolutos de precio): rsi, macd_hist, bb_pct, bb_width, volatility_20d,
volume_ratio. Más:
  - return_5d_forward = adj_close[t+5]/adj_close[t] - 1
  - return_excess_over_spy_5d_forward = return_5d_forward(ticker) - return_5d_forward(SPY)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.longrun.common import MASTER_DIR, safe_to_parquet, setup_logger

JOB = "job5_indicators"
_INPUT = MASTER_DIR / "prices_2008_2024.parquet"
_OUTPUT = MASTER_DIR / "indicators_daily.parquet"

_RSI_W, _MACD_F, _MACD_S, _MACD_SIG, _BB_W, _BB_STD, _VOL_W, _FWD = 14, 12, 26, 9, 20, 2, 20, 5


def _price_col(g: pd.DataFrame) -> pd.Series:
    c = g["adj_close"] if "adj_close" in g.columns else g["close"]
    c = pd.to_numeric(c, errors="coerce")
    return c.where(c.notna(), pd.to_numeric(g["close"], errors="coerce"))


def _indicators_for_ticker(g: pd.DataFrame) -> pd.DataFrame:
    from ta.momentum import RSIIndicator
    from ta.trend import MACD as MACDIndicator
    from ta.volatility import BollingerBands

    g = g.sort_values("date").copy()
    close = pd.to_numeric(g["close"], errors="coerce")
    px = _price_col(g)
    vol = pd.to_numeric(g["volume"], errors="coerce")

    g["rsi"] = RSIIndicator(close=close, window=_RSI_W, fillna=False).rsi()
    macd = MACDIndicator(close=close, window_fast=_MACD_F, window_slow=_MACD_S,
                         window_sign=_MACD_SIG, fillna=False)
    g["macd_hist"] = macd.macd_diff()
    bb = BollingerBands(close=close, window=_BB_W, window_dev=_BB_STD, fillna=False)
    upper, lower, mid = bb.bollinger_hband(), bb.bollinger_lband(), bb.bollinger_mavg()
    band = upper - lower
    g["bb_pct"] = np.where(band != 0, (close - lower) / band, np.nan)
    g["bb_width"] = np.where(mid != 0, band / mid, np.nan)

    ret = px.pct_change()
    g["volatility_20d"] = ret.rolling(_VOL_W).std()
    g["volume_ratio"] = vol / vol.rolling(_VOL_W).mean()
    g["return_5d_forward"] = px.shift(-_FWD) / px - 1.0
    return g


def run(smoke: bool = False, limit: int | None = None) -> dict:
    log = setup_logger(JOB)
    if not _INPUT.exists():
        log.error("Job 5: no existe %s (¿corrió el Job 4?).", _INPUT)
        return {"job": JOB, "error": "prices missing"}

    prices = pd.read_parquet(_INPUT)
    prices["date"] = pd.to_datetime(prices["date"])
    log.info("Job 5: %d filas de precios, %d tickers.", len(prices), prices["ticker"].nunique())

    tickers = sorted(prices["ticker"].unique())
    if smoke:
        keep = set((["AAPL", "MSFT", "NVDA", "SPY", "AMZN"])[: (limit or 5)]) | {"SPY"}
        tickers = [t for t in tickers if t in keep]

    parts = []
    for t in tickers:
        g = prices[prices["ticker"] == t]
        if len(g) < _MACD_S + _FWD:
            continue
        try:
            parts.append(_indicators_for_ticker(g))
        except Exception as exc:  # noqa: BLE001 — un ticker no mata el job
            log.error("  ✗ %s indicadores: %s", t, exc)
    if not parts:
        log.error("Job 5: ningún ticker produjo indicadores.")
        return {"job": JOB, "error": "no indicators"}

    df = pd.concat(parts, ignore_index=True)

    # Retorno forward de SPY por fecha → exceso de mercado
    spy = df[df["ticker"] == "SPY"][["date", "return_5d_forward"]].rename(
        columns={"return_5d_forward": "spy_ret5"})
    df = df.merge(spy, on="date", how="left")
    df["return_excess_over_spy_5d_forward"] = df["return_5d_forward"] - df["spy_ret5"]

    cols = ["ticker", "date", "rsi", "macd_hist", "bb_pct", "bb_width",
            "volatility_20d", "volume_ratio",
            "return_5d_forward", "return_excess_over_spy_5d_forward"]
    out = df[cols].copy()
    out["date"] = pd.to_datetime(out["date"]).dt.date
    safe_to_parquet(out, _OUTPUT)
    log.info("Job 5 salida: %s | %d filas | %d tickers.",
             _OUTPUT, len(out), out["ticker"].nunique())
    return {"job": JOB, "rows": len(out), "tickers": int(out["ticker"].nunique()),
            "output": str(_OUTPUT)}


if __name__ == "__main__":
    run(smoke="--smoke" in sys.argv)
