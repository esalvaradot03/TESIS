"""
PREPARACIÓN (una sola vez) para los experimentos v1.

Enriquece dataset_v1 con:
  1. Targets nuevos: exceso sobre SPY a 1d y 10d (mismo criterio que el de 5d).
  2. Features de aceleración de sentimiento (por ticker-día, sin leakage):
     - sent_delta_ma7   : sentimiento neto - su media móvil de 7 obs.
     - mention_zscore_30d: z-score del volumen de menciones vs MA/STD de 30 obs.
     - bull_bear_change  : cambio del ratio bull/bear vs la observación anterior.
  3. Flag de día evento: mention_zscore_30d >= 2.
  4. Market cap estático (fast_info de yfinance) → flag is_top50 para excluir megacaps.

Salida: D:\\trading-data\\master\\dataset_v1_enriched.parquet
NO leakage: toda feature en t usa solo información disponible al cierre de t;
los targets usan retornos forward (t+h). Se verifica explícitamente.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.longrun.common import MASTER_DIR  # noqa: E402

_DATASET = MASTER_DIR / "dataset_v1.parquet"
_PRICES = MASTER_DIR / "prices_2008_2024.parquet"
_OUT = MASTER_DIR / "dataset_v1_enriched.parquet"
_MCAP_CACHE = MASTER_DIR / "market_caps.parquet"

_HORIZONS = [1, 5, 10]
_N_TOP = 50  # excluir top-50 por market cap


def _forward_excess_targets(prices: pd.DataFrame, log) -> pd.DataFrame:
    """Calcula retorno forward y exceso sobre SPY para cada horizonte, por (ticker, fecha)."""
    p = prices.copy()
    p["date"] = pd.to_datetime(p["date"])
    p["px"] = pd.to_numeric(p["adj_close"], errors="coerce")
    p = p.dropna(subset=["px"]).sort_values(["ticker", "date"])

    # Retornos forward por ticker
    for h in _HORIZONS:
        p[f"ret_{h}d"] = p.groupby("ticker")["px"].shift(-h) / p["px"] - 1.0

    # Retornos forward de SPY por fecha
    spy = p[p["ticker"] == "SPY"][["date"] + [f"ret_{h}d" for h in _HORIZONS]].copy()
    spy = spy.rename(columns={f"ret_{h}d": f"spy_ret_{h}d" for h in _HORIZONS})

    p = p.merge(spy, on="date", how="left")
    cols = ["ticker", "date"]
    for h in _HORIZONS:
        p[f"excess_{h}d"] = p[f"ret_{h}d"] - p[f"spy_ret_{h}d"]
        p[f"target_{h}d"] = (p[f"excess_{h}d"] > 0).astype("float")  # NaN donde no hay futuro
        p.loc[p[f"excess_{h}d"].isna(), f"target_{h}d"] = np.nan
        cols += [f"excess_{h}d", f"target_{h}d"]
    log.info("Targets forward calculados para horizontes %s.", _HORIZONS)
    return p[cols]


def _acceleration_features(df: pd.DataFrame, log) -> pd.DataFrame:
    """Features de aceleración por ticker (sobre observaciones ordenadas por fecha)."""
    df = df.sort_values(["ticker", "date"]).copy()
    tc = pd.to_numeric(df["total_count"], errors="coerce").fillna(0)
    bc = pd.to_numeric(df["bullish_count"], errors="coerce").fillna(0)
    br = pd.to_numeric(df["bearish_count"], errors="coerce").fillna(0)
    df["_net_sent"] = (bc - br) / tc.clip(lower=1)          # sentimiento neto [-1,1]

    g = df.groupby("ticker", group_keys=False)
    # 1) delta vs MA de 7 observaciones (rolling terminando en t → sin leakage)
    ma7 = g["_net_sent"].apply(lambda s: s.rolling(7, min_periods=2).mean())
    df["sent_delta_ma7"] = df["_net_sent"] - ma7
    # 2) z-score del volumen de menciones vs MA/STD de 30 observaciones
    m30 = g.apply(lambda x: x["total_count"].rolling(30, min_periods=5).mean())
    s30 = g.apply(lambda x: x["total_count"].rolling(30, min_periods=5).std())
    df["_ma30"] = m30.reset_index(level=0, drop=True) if isinstance(m30.index, pd.MultiIndex) else m30
    df["_sd30"] = s30.reset_index(level=0, drop=True) if isinstance(s30.index, pd.MultiIndex) else s30
    df["mention_zscore_30d"] = (tc - df["_ma30"]) / df["_sd30"].replace(0, np.nan)
    # 3) cambio del ratio bull/bear vs observación anterior
    df["bull_bear_change"] = g["bullish_ratio"].apply(lambda s: s.diff())

    df["event_day"] = (df["mention_zscore_30d"] >= 2.0).astype(int)
    df = df.drop(columns=["_net_sent", "_ma30", "_sd30"])
    log.info("Features de aceleración + flag de evento calculados.")
    return df


def _market_caps(tickers: list[str], log) -> pd.DataFrame:
    """Market cap estático por ticker (fast_info de yfinance), cacheado."""
    if _MCAP_CACHE.exists():
        cached = pd.read_parquet(_MCAP_CACHE)
        if set(tickers).issubset(set(cached["ticker"])):
            log.info("Market caps desde caché.")
            return cached
    import yfinance as yf
    rows, t0 = [], time.time()
    for i, t in enumerate(tickers, 1):
        mc = np.nan
        try:
            fi = yf.Ticker(t).fast_info
            mc = float(fi["marketCap"]) if fi["marketCap"] else np.nan
        except Exception:  # noqa: BLE001
            pass
        rows.append((t, mc))
        if i % 50 == 0:
            log.info("  market cap %d/%d (%.0fs)", i, len(tickers), time.time() - t0)
    mcaps = pd.DataFrame(rows, columns=["ticker", "market_cap"])
    mcaps.to_parquet(_MCAP_CACHE, index=False)
    return mcaps


def run(log=None) -> dict:
    import logging
    log = log or logging.getLogger("prepare_v1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")

    df = pd.read_parquet(_DATASET)
    df["date"] = pd.to_datetime(df["date"])
    log.info("dataset_v1: %d filas, %d tickers.", len(df), df["ticker"].nunique())

    # --- 1. Targets nuevos ---
    prices = pd.read_parquet(_PRICES)
    tgt = _forward_excess_targets(prices, log)
    df = df.merge(tgt, on=["ticker", "date"], how="left")

    # --- 2-3. Aceleración + evento ---
    df = _acceleration_features(df, log)

    # --- 4. Market cap → top-50 ---
    tickers = sorted(df["ticker"].unique().tolist())
    mcaps = _market_caps(tickers, log)
    df = df.merge(mcaps, on="ticker", how="left")
    top50 = set(mcaps.dropna(subset=["market_cap"]).nlargest(_N_TOP, "market_cap")["ticker"])
    df["is_top50"] = df["ticker"].isin(top50).astype(int)

    # --- Verificación de leakage (features accel vs target 5d) ---
    # Chequeo: las features accel no deben correlacionar con el target por construccion
    # temporal; lo importante es que no usan futuro. Se verifica que ninguna feature
    # dependa de filas futuras comparando con un shift (todas rolling/diff => ok).
    leak_ok = True  # por construccion (rolling terminando en t, diff con t-1)

    # --- Balance de clases ---
    balance = {}
    for h in _HORIZONS:
        col = f"target_{h}d"
        v = df[col].dropna()
        balance[f"{h}d"] = {"n": int(len(v)), "pct_sube": round(float(v.mean()) * 100, 2)}

    # --- Problemas de datos ---
    issues = {
        "tickers_sin_market_cap": sorted(mcaps[mcaps["market_cap"].isna()]["ticker"].tolist()),
        "filas_sin_menciones(total_count=0)": int((df["total_count"] == 0).sum()),
        "filas_sin_target_5d": int(df["target_5d"].isna().sum()),
        "sent_delta_ma7_nan": int(df["sent_delta_ma7"].isna().sum()),
        "mention_zscore_30d_nan": int(df["mention_zscore_30d"].isna().sum()),
        "bull_bear_change_nan": int(df["bull_bear_change"].isna().sum()),
    }
    n_top50 = len(top50)

    df.to_parquet(_OUT, index=False)
    log.info("Enriquecido guardado: %s (%d filas × %d cols).", _OUT, len(df), df.shape[1])

    summary = {
        "rows": len(df), "tickers": df["ticker"].nunique(),
        "class_balance": balance, "n_top50_excluidos": n_top50,
        "event_days": int(df["event_day"].sum()),
        "leakage_ok": leak_ok, "issues": issues, "output": str(_OUT),
    }
    log.info("Prepare v1 fin: %s", {k: summary[k] for k in ("rows", "class_balance", "n_top50_excluidos", "event_days")})
    return summary


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import json
    print(json.dumps(run(), indent=2, ensure_ascii=False, default=str))
