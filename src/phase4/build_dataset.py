"""
Fase 4 — Tarea 2, construcción del dataset (5 activos fijos).

Para TSLA, AMD, DIS, BA, GILD construye una serie DIARIA por activo con:
  - Features de sentimiento StockTwits (etiqueta nativa Bullish/Bearish).
  - Features de sentimiento de noticias FNSPID (FinBERT sobre titulares).
  - Target binario: dirección del retorno del día siguiente (yfinance).
Sin features técnicas. Manejo de días sin actividad según REGISTRY.md
(neutro/cero + binaria has_*). Split 2015-2020 train / 2021-2022 test.

Todos los extractos pesados se cachean en experiments/phase4/cache/.
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

_NEWS = EXTERNAL_DATA_ROOT / "fnspid" / "nasdaq_exteral_data.csv"
_OUT = ROOT / "experiments" / "phase4"
_CACHE = _OUT / "cache"
_DS = _OUT / "dataset_phase4"

TICKERS = ["TSLA", "AMD", "DIS", "BA", "GILD"]
_WIN_S = pd.Timestamp("2015-01-01")
_WIN_E = pd.Timestamp("2022-12-31")
_TRAIN_END = pd.Timestamp("2020-12-31")
_TEST_START = pd.Timestamp("2021-01-01")
_CHUNK = 1_000_000
_SYM_RE = re.compile(r"'([A-Za-z][A-Za-z0-9.\-]{0,6})'")


# ----------------------------------------------------------- StockTwits nativo
def extract_stocktwits(log) -> pd.DataFrame:
    """Por (ticker, fecha_calendario): bull, bear, n_msgs (etiqueta nativa)."""
    cache = _CACHE / "st_raw_5.parquet"
    if cache.exists():
        log.info("StockTwits 5 activos: cache %s", cache.name)
        return pd.read_parquet(cache)

    tset = set(TICKERS)
    files = sorted(STOCKTWITS_NYU_SYMBOL_SENTIMENTS.glob("*.csv"))
    parts, t0 = [], time.time()
    for fi, f in enumerate(files, 1):
        reader = pd.read_csv(f, usecols=["created_at", "sentiment", "symbol_list"],
                             dtype={"created_at": "string", "symbol_list": "string"},
                             chunksize=_CHUNK, on_bad_lines="skip",
                             encoding="utf-8", encoding_errors="replace")
        for chunk in reader:
            syms = chunk["symbol_list"].map(
                lambda s: [x.upper() for x in _SYM_RE.findall(s)] if isinstance(s, str) else [])
            hit = syms.map(lambda lst: any(t in tset for t in lst))
            if not hit.any():
                continue
            sub = chunk[hit].copy()
            sub["syms"] = syms[hit].map(lambda lst: [t for t in lst if t in tset])
            d = pd.to_datetime(sub["created_at"], errors="coerce")
            sent = pd.to_numeric(sub["sentiment"], errors="coerce")
            sub = pd.DataFrame({"date": d, "sent": sent, "ticker": sub["syms"]})
            sub = sub.explode("ticker").dropna(subset=["date", "ticker"])
            sub = sub[(sub["date"] >= _WIN_S) & (sub["date"] <= _WIN_E)]
            if sub.empty:
                continue
            sub["cal_date"] = sub["date"].dt.normalize()
            g = sub.groupby(["ticker", "cal_date"])
            parts.append(pd.DataFrame({
                "bull": g["sent"].apply(lambda s: int((s > 0).sum())),
                "bear": g["sent"].apply(lambda s: int((s < 0).sum())),
                "n_msgs": g.size(),
            }).reset_index())
        if fi % 5 == 0:
            log.info("  StockTwits %d/%d (%.0fs)", fi, len(files), time.time() - t0)
    raw = pd.concat(parts, ignore_index=True)
    raw = raw.groupby(["ticker", "cal_date"], as_index=False).sum()
    _CACHE.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(cache, index=False)
    log.info("StockTwits 5 activos: %d ticker-dias -> %s", len(raw), cache.name)
    return raw


# ----------------------------------------------------------- Noticias FNSPID
def extract_news_titles(log) -> pd.DataFrame:
    """Titulares de noticias para los 5 activos (una pasada por el CSV de 23GB)."""
    cache = _CACHE / "news_titles_5.parquet"
    if cache.exists():
        log.info("Titulares noticias: cache %s", cache.name)
        return pd.read_parquet(cache)

    tset = set(TICKERS)
    log.info("Titulares noticias: escaneo %s (~23GB, una pasada)...", _NEWS.name)
    parts, t0, rows = [], time.time(), 0
    reader = pd.read_csv(_NEWS, usecols=["Date", "Article_title", "Stock_symbol"],
                         dtype="string", chunksize=_CHUNK, on_bad_lines="skip",
                         encoding="utf-8", encoding_errors="replace")
    for ci, chunk in enumerate(reader, 1):
        rows += len(chunk)
        sym = chunk["Stock_symbol"].str.upper().str.strip()
        m = sym.isin(tset)
        if not m.any():
            continue
        d = pd.to_datetime(chunk["Date"][m].str.replace(" UTC", "", regex=False),
                           errors="coerce")
        sub = pd.DataFrame({"ticker": sym[m], "date": d,
                            "title": chunk["Article_title"][m]})
        sub = sub.dropna(subset=["date"])
        sub = sub[(sub["date"] >= _WIN_S) & (sub["date"] <= _WIN_E)]
        if not sub.empty:
            parts.append(sub)
        if ci % 5 == 0:
            log.info("  Noticias %d chunks / %d filas (%.0fs, %d hits)",
                     ci, rows, time.time() - t0, sum(len(p) for p in parts))
    tit = pd.concat(parts, ignore_index=True)
    tit["cal_date"] = tit["date"].dt.normalize()
    _CACHE.mkdir(parents=True, exist_ok=True)
    tit.to_parquet(cache, index=False)
    log.info("Titulares noticias: %d artículos (5 activos) -> %s", len(tit), cache.name)
    return tit


def score_news_finbert(titles: pd.DataFrame, log) -> pd.DataFrame:
    """FinBERT sobre titulares → por (ticker, fecha): net medio y n_news."""
    cache = _CACHE / "news_sent_5.parquet"
    if cache.exists():
        log.info("Sentimiento noticias: cache %s", cache.name)
        return pd.read_parquet(cache)

    from src.sentiment.finbert_scorer import FinBERTScorer
    scorer = FinBERTScorer()
    txts = titles["title"].fillna("").astype(str).tolist()
    log.info("FinBERT sobre %d titulares...", len(txts))
    scores = scorer.score_texts(txts)
    net = np.array([s["prob_positive"] - s["prob_negative"] for s in scores])
    df = titles[["ticker", "cal_date"]].copy()
    df["net"] = net
    agg = df.groupby(["ticker", "cal_date"], as_index=False).agg(
        nw_net_day=("net", "mean"), n_news=("net", "size"))
    agg.to_parquet(cache, index=False)
    log.info("Sentimiento noticias: %d ticker-dias -> %s", len(agg), cache.name)
    return agg


# ----------------------------------------------------------- precios / calendario
def load_prices(log) -> dict[str, pd.DataFrame]:
    cache = _CACHE / "prices_5.parquet"
    if cache.exists():
        log.info("Precios yfinance: cache %s", cache.name)
        allp = pd.read_parquet(cache)
    else:
        import yfinance as yf
        rows = []
        for t in TICKERS:
            h = yf.Ticker(t).history(start="2015-01-01", end="2023-01-01", auto_adjust=True)
            h = h.reset_index()[["Date", "Close"]].rename(columns={"Date": "date", "Close": "close"})
            h["date"] = pd.to_datetime(h["date"]).dt.tz_localize(None).dt.normalize()
            h["ticker"] = t
            rows.append(h)
            log.info("  precios %s: %d barras (%s..%s)", t, len(h),
                     h["date"].min().date(), h["date"].max().date())
            time.sleep(0.4)
        allp = pd.concat(rows, ignore_index=True)
        _CACHE.mkdir(parents=True, exist_ok=True)
        allp.to_parquet(cache, index=False)
    return {t: g.sort_values("date").reset_index(drop=True) for t, g in allp.groupby("ticker")}


def _map_to_trading_day(cal_dates: pd.Series, trading_days: np.ndarray) -> pd.Series:
    """Cada fecha de calendario → primer día de trading >= esa fecha."""
    idx = np.searchsorted(trading_days, cal_dates.values, side="left")
    ok = idx < len(trading_days)
    out = pd.Series(pd.NaT, index=cal_dates.index, dtype="datetime64[ns]")
    out[ok] = trading_days[idx[ok]]
    return out


def _agg_to_trading(raw: pd.DataFrame, trading_days: np.ndarray, valcols: list[str]) -> pd.DataFrame:
    """Reasigna (cal_date) al día de trading y suma/promedia por día de trading."""
    r = raw.copy()
    r["td"] = _map_to_trading_day(r["cal_date"], trading_days)
    r = r.dropna(subset=["td"])
    return r.groupby("td")[valcols].sum().reset_index()


def _features_from_series(net: pd.Series, count: pd.Series, prefix: str) -> pd.DataFrame:
    """Construye las 5 features (net, net_3d, vol, mom, has) sobre serie diaria rellenada."""
    net = net.fillna(0.0)
    count = count.fillna(0.0)
    net_3d = net.rolling(3, min_periods=1).mean()
    prev3 = net.shift(1).rolling(3, min_periods=1).mean().fillna(0.0)
    mom = net - prev3
    vol = np.log1p(count)
    has = (count > 0).astype(int)
    return pd.DataFrame({
        f"{prefix}_net": net.values, f"{prefix}_net_3d": net_3d.values,
        f"{prefix}_vol": vol.values, f"{prefix}_mom": mom.values,
        f"{prefix}_has": has.values,
    }, index=net.index)


def build_asset(ticker: str, px: pd.DataFrame, st_raw: pd.DataFrame,
                nw_sent: pd.DataFrame, log) -> tuple[pd.DataFrame, dict]:
    px = px[(px["date"] >= _WIN_S) & (px["date"] <= _WIN_E)].copy()
    trading = px["date"].values  # ordenado
    base = px[["date", "close"]].copy().set_index("date")

    # StockTwits del activo → día de trading
    st = st_raw[st_raw["ticker"] == ticker][["cal_date", "bull", "bear", "n_msgs"]]
    st_td = _agg_to_trading(st, trading, ["bull", "bear", "n_msgs"]).set_index("td")
    st_td = st_td.reindex(base.index)
    denom = (st_td["bull"] + st_td["bear"]).replace(0, np.nan)
    st_net = ((st_td["bull"] - st_td["bear"]) / denom).fillna(0.0)
    st_feat = _features_from_series(st_net, st_td["n_msgs"], "st")

    # Noticias del activo → día de trading (net ponderado por nº de noticias del día)
    nw = nw_sent[nw_sent["ticker"] == ticker][["cal_date", "nw_net_day", "n_news"]].copy()
    nw["net_sum"] = nw["nw_net_day"] * nw["n_news"]
    nw_td = _agg_to_trading(nw.rename(columns={"n_news": "cnt"}), trading, ["net_sum", "cnt"]).set_index("td")
    nw_td = nw_td.reindex(base.index)
    nw_net = (nw_td["net_sum"] / nw_td["cnt"].replace(0, np.nan)).fillna(0.0)
    nw_feat = _features_from_series(nw_net, nw_td["cnt"], "nw")

    # target: dirección del día siguiente (último día sin cierre futuro → NaN → se descarta)
    nxt = base["close"].shift(-1)
    y = (nxt > base["close"]).astype("Int64")
    y[nxt.isna()] = pd.NA
    df = pd.concat([base["close"], st_feat, nw_feat], axis=1)
    df["y"] = y
    df = df.reset_index().rename(columns={"index": "date", "td": "date"})
    if "date" not in df.columns:
        df = df.rename(columns={df.columns[0]: "date"})
    df = df.dropna(subset=["y"]).copy()  # descarta último día (sin target)

    # split temporal; excluir último día de trading de 2020 del train
    df["split"] = np.where(df["date"] >= _TEST_START, "test",
                           np.where(df["date"] <= _TRAIN_END, "train", "excl"))
    last_train_day = df.loc[df["split"] == "train", "date"].max()
    df.loc[df["date"] == last_train_day, "split"] = "excl"

    stats = {
        "ticker": ticker,
        "dias_trading": int(len(df)),
        "train_n": int((df["split"] == "train").sum()),
        "test_n": int((df["split"] == "test").sum()),
        "train_desde": str(df.loc[df["split"] == "train", "date"].min().date()),
        "train_hasta": str(df.loc[df["split"] == "train", "date"].max().date()),
        "test_desde": str(df.loc[df["split"] == "test", "date"].min().date()),
        "test_hasta": str(df.loc[df["split"] == "test", "date"].max().date()),
        "excl_frontera": str(last_train_day.date()),
        "pct_dias_con_st": round(100 * df["st_has"].mean(), 1),
        "pct_dias_con_news": round(100 * df["nw_has"].mean(), 1),
        "y_up_train_pct": round(100 * df.loc[df["split"] == "train", "y"].mean(), 1),
        "y_up_test_pct": round(100 * df.loc[df["split"] == "test", "y"].mean(), 1),
    }
    return df, stats


def run() -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("phase4_build")
    _DS.mkdir(parents=True, exist_ok=True)

    st_raw = extract_stocktwits(log)
    titles = extract_news_titles(log)
    nw_sent = score_news_finbert(titles, log)
    prices = load_prices(log)

    all_stats = []
    for t in TICKERS:
        df, stats = build_asset(t, prices[t], st_raw, nw_sent, log)
        df.to_parquet(_DS / f"{t}.parquet", index=False)
        all_stats.append(stats)
        log.info("  %s: train=%d test=%d, %%news=%.1f %%st=%.1f",
                 t, stats["train_n"], stats["test_n"],
                 stats["pct_dias_con_news"], stats["pct_dias_con_st"])

    sd = pd.DataFrame(all_stats)
    sd.to_parquet(_OUT / "build_stats.parquet", index=False)
    _write_build_md(sd)
    return sd


def _write_build_md(sd: pd.DataFrame):
    L = ["# Fase 4 — Tarea 2: dataset construido (cortes exactos)\n",
         "Un parquet por activo en `experiments/phase4/dataset_phase4/`. Features solo de "
         "sentimiento (5 StockTwits `st_*` + 5 Noticias `nw_*`), target = dirección del retorno "
         "del día siguiente. Días sin actividad → neutro/0 + binaria `*_has` (ver REGISTRY.md).\n",
         "| activo | días | train n | test n | train | test | frontera excl | %días c/ST | "
         "%días c/news | %sube train | %sube test |",
         "|--------|------|---------|--------|-------|------|---------------|-----------|"
         "-------------|-------------|------------|"]
    for _, r in sd.iterrows():
        L.append(
            f"| {r['ticker']} | {r['dias_trading']} | {r['train_n']} | {r['test_n']} | "
            f"{r['train_desde']}→{r['train_hasta']} | {r['test_desde']}→{r['test_hasta']} | "
            f"{r['excl_frontera']} | {r['pct_dias_con_st']} | {r['pct_dias_con_news']} | "
            f"{r['y_up_train_pct']} | {r['y_up_test_pct']} |")
    L.append("")
    (_OUT / "build_dataset.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import json
    print(json.dumps(run().to_dict(orient="records"), indent=2, ensure_ascii=False, default=str))
