"""
Fase 6 — construcción del dataset intradía (30 y 60 min) para TSLA y AMD.

Une barras Alpaca de 30 min con el sentimiento StockTwits por ventana (join de labels
por message_id ya cacheado) y arma, por ventana objetivo `[t, t+30]`:
  - target: signo de ret(t)=close/open-1;
  - features autorregresivas: ret/range de las 1-5 ventanas previas + n_gap_lags;
  - features de sentimiento/actividad de la ventana anterior `[t-30,t]`.
Regla intradía: el par `[t-30,t]→[t,t+30]` nunca cruza el overnight (se excluye la 1a
ventana objetivo de cada día). Ver REGISTRY.md.

Salida: experiments/phase6/dataset_intraday/{TICKER}_{30|60}.parquet
"""

import logging
import sys
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_OUT = ROOT / "experiments" / "phase6"
_CACHE = _OUT / "cache"
_DS = _OUT / "dataset_intraday"
TICKERS = ["TSLA", "AMD"]
_TZ = "America/New_York"
_TRAIN_S, _TRAIN_E = pd.Timestamp("2020-08-01", tz=_TZ), pd.Timestamp("2021-12-31 23:59", tz=_TZ)
_TEST_S, _TEST_E = pd.Timestamp("2022-01-01", tz=_TZ), pd.Timestamp("2022-12-31 23:59", tz=_TZ)

AR_FEATS = [f"ret_lag{k}" for k in range(1, 6)] + [f"range_lag{k}" for k in range(1, 6)] + ["n_gap_lags"]
SENT_FEATS = ["net_lag1", "net_accel", "vol_lag1", "vol_accel", "lab_lag1"]


def _to_w60(w30: pd.Series) -> pd.Series:
    return (w30 - pd.Timedelta(minutes=30)).dt.floor("60min") + pd.Timedelta(minutes=30)


def _sentiment_windows(size: int, log) -> pd.DataFrame:
    """Por (ticker, ventana): n_msgs, n_lab, net = (bull-bear)/(bull+bear)."""
    intr = pd.read_parquet(_CACHE / "intraday_msgids.parquet")
    lab = pd.read_parquet(_CACHE / "labeled_msgids.parquet")
    intr["w30"] = pd.to_datetime(intr["w30"])
    m = intr.merge(lab, on="message_id", how="left")
    s = pd.to_numeric(m["sentiment"], errors="coerce")
    m["bull"] = (s.fillna(0) > 0).astype(int)
    m["bear"] = (s.fillna(0) < 0).astype(int)
    m["lab"] = s.notna().astype(int)
    m["w"] = m["w30"] if size == 30 else _to_w60(m["w30"])
    g = m.groupby(["ticker", "w"]).agg(
        n_msgs=("message_id", "size"), bull=("bull", "sum"),
        bear=("bear", "sum"), n_lab=("lab", "sum")).reset_index()
    denom = (g["bull"] + g["bear"]).replace(0, np.nan)
    g["net"] = ((g["bull"] - g["bear"]) / denom).fillna(0.0)
    return g[["ticker", "w", "n_msgs", "n_lab", "net"]]


def _bars(size: int) -> pd.DataFrame:
    b = pd.read_parquet(_CACHE / "alpaca_30min.parquet")
    b["t"] = pd.to_datetime(b["t"])
    tod = b["t"].dt.time
    b = b[(tod >= dtime(9, 30)) & (tod < dtime(16, 0)) & (b["t"].dt.weekday < 5)].copy()
    if size == 60:
        b["w"] = _to_w60(b["t"].dt.floor("30min"))
        b = b.sort_values(["ticker", "t"]).groupby(["ticker", "w"]).agg(
            open=("open", "first"), high=("high", "max"),
            low=("low", "min"), close=("close", "last")).reset_index()
    else:
        b = b.rename(columns={"t": "w"})[["ticker", "w", "open", "high", "low", "close"]]
    b["ret"] = b["close"] / b["open"] - 1.0
    b["range"] = (b["high"] - b["low"]) / b["open"]
    return b


def build(size: int, log) -> dict:
    sent = _sentiment_windows(size, log)
    bars = _bars(size)
    stats = {}
    _DS.mkdir(parents=True, exist_ok=True)
    for tk in TICKERS:
        b = bars[bars["ticker"] == tk].merge(
            sent[sent["ticker"] == tk], on=["ticker", "w"], how="left").sort_values("w").reset_index(drop=True)
        b["net"] = b["net"].fillna(0.0)
        b["n_msgs"] = b["n_msgs"].fillna(0.0)
        b["n_lab"] = b["n_lab"].fillna(0.0)
        b["vol"] = np.log1p(b["n_msgs"])
        b["lab"] = np.log1p(b["n_lab"])
        b["day"] = b["w"].dt.normalize()

        # lags autorregresivos y de sentimiento (a lo largo de la serie continua)
        for k in range(1, 6):
            b[f"ret_lag{k}"] = b["ret"].shift(k)
            b[f"range_lag{k}"] = b["range"].shift(k)
        b["net_lag1"] = b["net"].shift(1)
        b["net_lag2"] = b["net"].shift(2)
        b["vol_lag1"] = b["vol"].shift(1)
        b["vol_lag2"] = b["vol"].shift(2)
        b["lab_lag1"] = b["lab"].shift(1)
        b["net_accel"] = b["net_lag1"] - b["net_lag2"]
        b["vol_accel"] = b["vol_lag1"] - b["vol_lag2"]
        # gaps overnight en los lags 1-5
        gap = np.zeros(len(b), dtype=int)
        for k in range(1, 6):
            gap += (b["day"].shift(k) != b["day"]).astype(int).to_numpy()
        b["n_gap_lags"] = gap
        b["same_day1"] = (b["day"].shift(1) == b["day"])

        # target y regla intradía (excluir 1a ventana objetivo de cada día)
        b["y"] = (b["ret"] > 0).astype("Int64")
        d = b.dropna(subset=AR_FEATS + SENT_FEATS + ["y"]).copy()
        d = d[d["same_day1"]].copy()  # par [t-30,t]->[t,t+30] mismo día

        d["split"] = np.where((d["w"] >= _TEST_S) & (d["w"] <= _TEST_E), "test",
                       np.where((d["w"] >= _TRAIN_S) & (d["w"] <= _TRAIN_E), "train", "excl"))
        d = d[d["split"] != "excl"].copy()
        # diagnóstico contemporáneo: ret de la ventana de sentimiento y vecinas
        d["ret_coin"] = d["ret_lag1"]          # ret de [t-30,t] (coincidente con net_lag1)
        d["ret_react"] = d["ret_lag2"]         # ret de [t-60,t-30] (sentimiento sigue al precio)
        d["ret_anti"] = d["ret"]               # ret de [t,t+30] (anticipatorio = target continuo)

        cols = (["w", "ticker", "split", "y", "ret",
                 "ret_coin", "ret_react", "ret_anti"] + AR_FEATS + SENT_FEATS)
        d[cols].to_parquet(_DS / f"{tk}_{size}.parquet", index=False)
        tr, te = d[d["split"] == "train"], d[d["split"] == "test"]
        stats[tk] = {"size": size, "n_total": int(len(d)), "train_n": int(len(tr)),
                     "test_n": int(len(te)),
                     "train_desde": str(tr["w"].min()), "train_hasta": str(tr["w"].max()),
                     "test_desde": str(te["w"].min()), "test_hasta": str(te["w"].max()),
                     "y_up_train_%": round(100 * tr["y"].mean(), 1),
                     "y_up_test_%": round(100 * te["y"].mean(), 1),
                     "%_con_msgs_train": round(100 * (tr["vol_lag1"] > 0).mean(), 1)}
        log.info("  %s %dmin: train=%d test=%d y_up(tr/te)=%.1f/%.1f",
                 tk, size, len(tr), len(te), stats[tk]["y_up_train_%"], stats[tk]["y_up_test_%"])
    return stats


def run() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("phase6_build")
    all_stats = {}
    for size in (30, 60):
        all_stats[size] = build(size, log)
    _write_md(all_stats)
    return all_stats


def _write_md(all_stats):
    L = ["# Fase 6 — dataset intradía construido (cortes exactos)\n",
         "Par `[t-30,t]→[t,t+30]` estrictamente intradía (1a ventana objetivo de cada día "
         "excluida). Features AR (ret/range lags 1-5 + n_gap_lags) + sentimiento (net_lag1 "
         "principal, accel, vol). Split train 2020-08→2021-12 / test 2022. Ver REGISTRY.md.\n",
         "| activo | ventana | train n | test n | train | test | %sube train | %sube test |",
         "|--------|---------|---------|--------|-------|------|-------------|------------|"]
    for size in (30, 60):
        for tk in TICKERS:
            s = all_stats[size][tk]
            L.append(f"| {tk} | {size}min | {s['train_n']:,} | {s['test_n']:,} | "
                     f"{s['train_desde'][:10]}→{s['train_hasta'][:10]} | "
                     f"{s['test_desde'][:10]}→{s['test_hasta'][:10]} | "
                     f"{s['y_up_train_%']} | {s['y_up_test_%']} |")
    L.append("")
    (_OUT / "build_intraday.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import json
    print(json.dumps(run(), indent=2, ensure_ascii=False, default=str))
