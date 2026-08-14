"""
Fase 7 — construcción de paneles listos para análisis (con sellado de 2022).

Expone build_daily(), build_intraday(), build_overnight(), load_earnings().
Convención de sellado: SEAL = 2021-12-31. Las features backward (net, z-scores,
relvol) se computan sobre la serie completa (son pasado, seguras). Los targets
forward se acompañan de su fecha (`tgt_date_h`) para que el runner exija, en train,
que target y predictor caigan ambos en ≤ SEAL (nunca tocar 2022 en Etapa 1).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_CACHE = ROOT / "experiments" / "phase7" / "cache"
_P6 = ROOT / "experiments" / "phase6" / "cache"

ASSETS5 = ["TSLA", "AMD", "DIS", "BA", "GILD"]
INTRADAY = ["TSLA", "AMD"]
SEAL = pd.Timestamp("2021-12-31")
TEST_S = pd.Timestamp("2022-01-01")
_HORIZONS = (1, 2, 3)


def _entropy(bull, bear):
    tot = bull + bear
    p = np.where(tot > 0, bull / np.where(tot > 0, tot, 1), 0.5)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return np.where(tot > 0, h, 0.0)


def _map_next_trading(dates: pd.Series, trading: np.ndarray) -> pd.Series:
    idx = np.searchsorted(trading, dates.values, side="left")
    ok = idx < len(trading)
    out = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    out[ok] = trading[idx[ok]]
    return out


def _add_backward(df: pd.DataFrame) -> pd.DataFrame:
    tot = df["bull"] + df["bear"]
    df["net"] = np.where(tot > 0, (df["bull"] - df["bear"]) / tot.replace(0, np.nan), 0.0)
    df["net"] = df["net"].fillna(0.0)
    df["disagreement"] = 1 - df["net"].abs()
    df["entropy"] = _entropy(df["bull"].to_numpy(), df["bear"].to_numpy())
    df["vol"] = np.log1p(df["n_msgs"])
    df["ret"] = df["close"].pct_change()
    df["abs_ret"] = df["ret"].abs()
    df["range"] = (df["high"] - df["low"]) / df["close"]
    df["relvol"] = df["volume"] / df["volume"].rolling(20, min_periods=10).mean()
    for col, z in [("net", "z_net"), ("vol", "z_vol")]:
        m = df[col].rolling(20, min_periods=10).mean()
        s = df[col].rolling(20, min_periods=10).std()
        df[z] = (df[col] - m) / s.replace(0, np.nan)
    return df


def _add_forward(df: pd.DataFrame, date_col: str, same_day: str | None = None) -> pd.DataFrame:
    for h in _HORIZONS:
        for tgt in ["abs_ret", "range", "relvol"]:
            df[f"fwd_{tgt}_{h}"] = df[tgt].shift(-h)
        df[f"tgt_date_{h}"] = df[date_col].shift(-h)
        if same_day is not None:  # invalidar si el target cruza de día (intradía)
            cross = df[same_day].shift(-h) != df[same_day]
            for tgt in ["abs_ret", "range", "relvol"]:
                df.loc[cross, f"fwd_{tgt}_{h}"] = np.nan
    df["fwd_ret_1"] = df["ret"].shift(-1)
    if same_day is not None:
        cross1 = df[same_day].shift(-1) != df[same_day]
        df.loc[cross1, "fwd_ret_1"] = np.nan
    return df


def build_daily() -> dict[str, pd.DataFrame]:
    sent = pd.read_parquet(_CACHE / "daily_sentiment_5.parquet")
    sent["date"] = pd.to_datetime(sent["date"])
    ohlcv = pd.read_parquet(_CACHE / "daily_ohlcv_5.parquet")
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    out = {}
    for tk in ASSETS5:
        px = ohlcv[ohlcv["ticker"] == tk].sort_values("date").reset_index(drop=True)
        trading = px["date"].values
        s = sent[sent["ticker"] == tk].copy()
        s["td"] = _map_next_trading(s["date"], trading)
        s = s.dropna(subset=["td"]).groupby("td", as_index=False)[["bull", "bear", "n_msgs"]].sum()
        df = px.merge(s.rename(columns={"td": "date"}), on="date", how="left")
        for c in ["bull", "bear", "n_msgs"]:
            df[c] = df[c].fillna(0.0)
        df = _add_backward(df)
        df = _add_forward(df, "date")
        out[tk] = df
    return out


def build_intraday() -> dict[str, pd.DataFrame]:
    from datetime import time as dtime
    intr = pd.read_parquet(_P6 / "intraday_msgids.parquet")
    intr["w30"] = pd.to_datetime(intr["w30"])
    lab = pd.read_parquet(_P6 / "labeled_msgids.parquet")
    m = intr.merge(lab, on="message_id", how="left")
    sv = pd.to_numeric(m["sentiment"], errors="coerce")
    m["bull"] = (sv.fillna(0) > 0).astype(int)
    m["bear"] = (sv.fillna(0) < 0).astype(int)
    g = m.groupby(["ticker", "w30"]).agg(bull=("bull", "sum"), bear=("bear", "sum"),
                                         n_msgs=("message_id", "size")).reset_index()
    bars = pd.read_parquet(_P6 / "alpaca_30min.parquet")
    bars["t"] = pd.to_datetime(bars["t"])
    tod = bars["t"].dt.time
    bars = bars[(tod >= dtime(9, 30)) & (tod < dtime(16, 0)) & (bars["t"].dt.weekday < 5)]
    out = {}
    for tk in INTRADAY:
        b = bars[bars["ticker"] == tk].rename(columns={"t": "date"}).sort_values("date")
        b = b.merge(g[g["ticker"] == tk].rename(columns={"w30": "date"})[["date", "bull", "bear", "n_msgs"]],
                    on="date", how="left")
        for c in ["bull", "bear", "n_msgs"]:
            b[c] = b[c].fillna(0.0)
        b["day"] = b["date"].dt.normalize()
        b["date_naive"] = b["date"].dt.tz_localize(None)  # antes de _add_forward
        b = _add_backward(b)
        b = _add_forward(b, "date_naive", same_day="day")
        out[tk] = b.reset_index(drop=True)
    return out


def build_overnight() -> dict[str, pd.DataFrame]:
    """Por día de trading D: sentimiento/volumen acumulados 16:00(D-1)→09:30(D) (incl.
    pre/post-market) y targets gap_ret, gap_abs, first_hour_ret."""
    from datetime import time as dtime
    allm = pd.read_parquet(_CACHE / "intraday_all_5.parquet")
    allm["et"] = pd.to_datetime(allm["et"])
    lab = pd.read_parquet(_P6 / "labeled_msgids.parquet")
    m = allm.merge(lab, on="message_id", how="left")
    sv = pd.to_numeric(m["sentiment"], errors="coerce")
    m["bull"] = (sv.fillna(0) > 0).astype(int)
    m["bear"] = (sv.fillna(0) < 0).astype(int)
    bars = pd.read_parquet(_P6 / "alpaca_30min.parquet")
    bars["t"] = pd.to_datetime(bars["t"])
    tod = bars["t"].dt.time
    bars_mkt = bars[(tod >= dtime(9, 30)) & (tod < dtime(16, 0)) & (bars["t"].dt.weekday < 5)]
    out = {}
    for tk in INTRADAY:
        b = bars_mkt[bars_mkt["ticker"] == tk].sort_values("t").copy()
        b["day"] = b["t"].dt.normalize()  # tz-aware ET, medianoche
        daily = b.groupby("day").agg(open=("open", "first"), close=("close", "last")).reset_index()
        fh = b[b["t"].dt.time.isin([dtime(9, 30), dtime(10, 0)])].groupby("day").agg(
            o=("open", "first"), c=("close", "last")).reset_index()
        fh["fh_ret"] = fh["c"] / fh["o"] - 1
        daily = daily.merge(fh[["day", "fh_ret"]], on="day", how="left")
        daily["prev_close"] = daily["close"].shift(1)
        daily["gap_ret"] = daily["open"] / daily["prev_close"] - 1
        daily["gap_abs"] = daily["gap_ret"].abs()
        # timestamp de apertura (09:30 ET) por día; ambos tz-aware ET
        daily["open_time"] = daily["day"] + pd.Timedelta(hours=9, minutes=30)
        # mensajes overnight = fuera de horario de mercado, asignados al PRÓXIMO open
        mm = m[m["ticker"] == tk].copy()
        et = mm["et"].dt.time
        is_mkt = (et >= dtime(9, 30)) & (et < dtime(16, 0)) & (mm["et"].dt.weekday < 5)
        ov = mm[~is_mkt].sort_values("et").copy()
        opens_df = daily[["day", "open_time"]].sort_values("open_time")
        ov = pd.merge_asof(ov, opens_df, left_on="et", right_on="open_time",
                           direction="forward").dropna(subset=["day"])
        agg = ov.groupby("day").agg(ov_bull=("bull", "sum"), ov_bear=("bear", "sum"),
                                    ov_n=("message_id", "size")).reset_index()
        d = daily.merge(agg, on="day", how="left")
        d["day"] = d["day"].dt.tz_localize(None)
        for c in ["ov_bull", "ov_bear", "ov_n"]:
            d[c] = d[c].fillna(0.0)
        tot = d["ov_bull"] + d["ov_bear"]
        d["ov_net"] = np.where(tot > 0, (d["ov_bull"] - d["ov_bear"]) / tot.replace(0, np.nan), 0.0)
        d["ov_net"] = d["ov_net"].fillna(0.0)
        d["ov_vol"] = np.log1p(d["ov_n"])
        out[tk] = d
    return out


def load_earnings() -> dict[str, np.ndarray]:
    p = _CACHE / "earnings_5.parquet"
    if not p.exists():
        return {t: np.array([], dtype="datetime64[ns]") for t in ASSETS5}
    e = pd.read_parquet(p)
    e["earnings_date"] = pd.to_datetime(e["earnings_date"])
    return {t: np.sort(e[e["ticker"] == t]["earnings_date"].values) for t in ASSETS5}
