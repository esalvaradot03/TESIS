"""
Construcción del panel MENSUAL para la Fase 3 (contrarian, asset pricing).

Universo: 475 equities survivors (Fase 2) con filtro de operabilidad point-in-time.
Salida: D:\\trading-data\\master\\panel_v3_monthly.parquet con, por (ticker, mes):
  - bullishness (bull/(bull+bear)), n_mentions (labeled), unique_users
  - ret_fwd_1m, ret_fwd_3m (retorno del ticker), spy_fwd_1m/3m, excess_1m/3m
  - size (dollar volume mediano 60d, point-in-time), vol (vol 60d de retornos diarios)
Sin leakage: sentimiento del mes t se alinea con retornos de t+1 en adelante.
"""

import ast
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import EXTERNAL_DATA_ROOT  # noqa: E402
from src.data.longrun.common import MASTER_DIR, normalize_ticker  # noqa: E402

_UNIVERSE = ROOT / "research" / "experiment_v2" / "universe_v2.parquet"
_SS_DIR = EXTERNAL_DATA_ROOT / "stocktwits_nyu" / "symbol_sentiments"
_FH = EXTERNAL_DATA_ROOT / "fnspid" / "full_history"
_OUT = MASTER_DIR / "panel_v3_monthly.parquet"

_WIN_S, _WIN_E = pd.Timestamp("2015-01-01"), pd.Timestamp("2023-12-31")
_DV_LO, _DV_HI = 200_000.0, 50_000_000.0
_MIN_MENTIONS = 20
_CHUNK = 500_000


def _parse_syms(cell):
    if not isinstance(cell, str) or not cell.strip() or cell.strip() == "[]":
        return []
    try:
        v = ast.literal_eval(cell)
    except (ValueError, SyntaxError, TypeError, MemoryError):
        return []
    return [normalize_ticker(x) for x in v] if isinstance(v, (list, tuple)) else []


def monthly_sentiment(universe: set[str], log) -> pd.DataFrame:
    pat = re.compile(r"'(" + "|".join(re.escape(t) for t in universe) + r")'")
    files = sorted(_SS_DIR.glob("*.csv"))
    parts, t0 = [], time.time()
    for fi, f in enumerate(files, 1):
        try:
            reader = pd.read_csv(f, usecols=["user_id", "created_at", "sentiment", "symbol_list"],
                                 dtype="string", chunksize=_CHUNK, on_bad_lines="skip",
                                 encoding="utf-8", encoding_errors="replace")
        except Exception:  # noqa: BLE001
            continue
        for chunk in reader:
            chunk = chunk[chunk["symbol_list"].str.contains(pat, na=False)]
            if chunk.empty:
                continue
            d = pd.to_datetime(chunk["created_at"], errors="coerce")
            m = d.notna() & (d >= _WIN_S) & (d <= _WIN_E)
            chunk = chunk[m]
            if chunk.empty:
                continue
            chunk = chunk.assign(month=d[m].dt.to_period("M").astype(str),
                                 sent=pd.to_numeric(chunk["sentiment"], errors="coerce"),
                                 syms=chunk["symbol_list"].map(_parse_syms))
            chunk = chunk.explode("syms").rename(columns={"syms": "ticker"})
            chunk = chunk[chunk["ticker"].isin(universe)]
            if not chunk.empty:
                parts.append(chunk[["ticker", "month", "sent", "user_id"]])
        if fi % 5 == 0:
            log.info("  StockTwits mensual %d/%d (%.0fs)", fi, len(files), time.time() - t0)
    big = pd.concat(parts, ignore_index=True)
    g = big.groupby(["ticker", "month"])
    agg = pd.DataFrame({
        "n_mentions": g.size(),
        "bull": g["sent"].apply(lambda s: (s > 0).sum()),
        "bear": g["sent"].apply(lambda s: (s < 0).sum()),
        "unique_users": g["user_id"].nunique(),
    }).reset_index()
    agg["bullishness"] = agg["bull"] / (agg["bull"] + agg["bear"]).clip(lower=1)
    log.info("Sentimiento mensual: %d (ticker,mes) crudos.", len(agg))
    return agg


def monthly_prices(universe: list[str], log) -> pd.DataFrame:
    """Retornos mensuales forward + exceso vs SPY + size/vol + operabilidad (point-in-time)."""
    def _load(t):
        p = _FH / f"{t}.csv"
        if not p.exists():
            return None
        try:
            d = pd.read_csv(p, usecols=["date", "close", "adj_close", "volume"])
        except Exception:
            try:
                d = pd.read_csv(p)
                d.columns = [c.lower().strip().replace(" ", "_") for c in d.columns]
            except Exception:  # noqa: BLE001
                return None
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.dropna(subset=["date"]).sort_values("date")
        d = d[(d["date"] >= _WIN_S - pd.Timedelta(days=120)) & (d["date"] <= _WIN_E)]
        return d if len(d) > 60 else None

    # SPY mensual
    spy = _load("SPY")
    spy_m = spy.set_index("date")["adj_close"].resample("ME").last().pct_change().rename("spy_ret")

    rows = []
    for t in universe:
        d = _load(t)
        if d is None:
            continue
        d = d.set_index("date")
        px = pd.to_numeric(d["adj_close"], errors="coerce").fillna(pd.to_numeric(d["close"], errors="coerce"))
        vol = pd.to_numeric(d["volume"], errors="coerce")
        dv60 = (pd.to_numeric(d["close"], errors="coerce") * vol).rolling(60, min_periods=30).median()
        ret_d = px.pct_change()
        vol60 = ret_d.rolling(60, min_periods=30).std()
        me = px.resample("ME").last()
        mret = me.pct_change()
        # size/vol point-in-time al cierre del mes (último valor del mes)
        size_m = dv60.resample("ME").last()
        vol_m = vol60.resample("ME").last()
        df = pd.DataFrame({"mret": mret, "size": size_m, "vol": vol_m})
        df["ret_fwd_1m"] = df["mret"].shift(-1)
        df["ret_fwd_3m"] = (1 + df["mret"].shift(-1)) * (1 + df["mret"].shift(-2)) * (1 + df["mret"].shift(-3)) - 1
        df["ticker"] = t
        df = df.reset_index().rename(columns={"date": "month_end"})
        rows.append(df)
    allp = pd.concat(rows, ignore_index=True)
    allp = allp.merge(spy_m.reset_index().rename(columns={"date": "month_end"}), on="month_end", how="left")
    allp["spy_fwd_1m"] = allp.groupby("ticker")["spy_ret"].shift(-1)
    allp["spy_fwd_3m"] = allp.groupby("ticker").apply(
        lambda g: (1 + g["spy_ret"].shift(-1)) * (1 + g["spy_ret"].shift(-2)) * (1 + g["spy_ret"].shift(-3)) - 1
    ).reset_index(level=0, drop=True)
    allp["excess_1m"] = allp["ret_fwd_1m"] - allp["spy_fwd_1m"]
    allp["excess_3m"] = allp["ret_fwd_3m"] - allp["spy_fwd_3m"]
    allp["month"] = allp["month_end"].dt.to_period("M").astype(str)
    # operabilidad point-in-time
    allp = allp[(allp["size"] >= _DV_LO) & (allp["size"] <= _DV_HI)]
    log.info("Precios mensuales operables: %d filas, %d tickers.", len(allp), allp["ticker"].nunique())
    return allp[["ticker", "month", "ret_fwd_1m", "ret_fwd_3m", "excess_1m", "excess_3m", "size", "vol"]]


def run() -> dict:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("build_v3")

    universe = pd.read_parquet(_UNIVERSE)["ticker"].tolist()
    uni = set(universe)
    log.info("Universo: %d equities.", len(universe))

    sent = monthly_sentiment(uni, log)
    sent = sent[sent["n_mentions"] >= _MIN_MENTIONS].copy()
    log.info("ticker-mes tras filtro >=%d menciones: %d", _MIN_MENTIONS, len(sent))

    px = monthly_prices(universe, log)

    panel = sent.merge(px, on=["ticker", "month"], how="inner")
    panel = panel.dropna(subset=["ret_fwd_1m"])
    panel.to_parquet(_OUT, index=False)

    per_month = panel.groupby("month").size()
    summary = {
        "ticker_mes_tras_filtro_menciones": int(len(sent)),
        "panel_final(con_precio_operable)": int(len(panel)),
        "meses": int(panel["month"].nunique()),
        "nombres_por_mes_mediana": int(per_month.median()),
        "nombres_por_mes_min": int(per_month.min()),
        "nombres_por_mes_max": int(per_month.max()),
        "tickers_unicos": int(panel["ticker"].nunique()),
        "output": str(_OUT),
    }
    log.info("Panel v3 fin: %s", summary)
    return summary


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import json
    print(json.dumps(run(), indent=2, ensure_ascii=False, default=str))
