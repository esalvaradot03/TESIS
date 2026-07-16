"""
Construcción de dataset_v2 (survivors-only small/mid-cap). Mismo esquema que
dataset_v1_enriched: técnicas + sentimiento (StockTwits+WSB+news) + aceleración,
target de exceso sobre SPY a 5d, sin leakage, filtro de operabilidad por liquidez.

Universo: quoteType==EQUITY de los survivors (>=200 días de menciones + precio
reciente en FNSPID). Filtro: dollar volume mediano 60d point-in-time en $200K-$50M.

Salida: D:\\trading-data\\master\\dataset_v2_enriched.parquet
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

_STATS = MASTER_DIR / "v2_mention_stats.parquet"
_FH = EXTERNAL_DATA_ROOT / "fnspid" / "full_history"
_SS_DIR = EXTERNAL_DATA_ROOT / "stocktwits_nyu" / "symbol_sentiments"
_NEWS = EXTERNAL_DATA_ROOT / "fnspid" / "nasdaq_exteral_data.csv"
_UNANIMAD = EXTERNAL_DATA_ROOT / "wsb_kaggle" / "unanimad_reddit_rwallstreetbets"
_KEVIN = EXTERNAL_DATA_ROOT / "wsb_kaggle" / "kevinwang313_wallstreetbets"
_UNIVERSE_CACHE = ROOT / "research" / "experiment_v2" / "universe_v2.parquet"
_OUT = MASTER_DIR / "dataset_v2_enriched.parquet"

_WIN_S, _WIN_E = pd.Timestamp("2015-01-01"), pd.Timestamp("2023-12-31")
_DV_LO, _DV_HI = 200_000.0, 50_000_000.0
_CHUNK = 500_000
_FWD = 5


# ---------------------------------------------------------------------------
# Universo
# ---------------------------------------------------------------------------

def build_universe(log) -> list[str]:
    if _UNIVERSE_CACHE.exists():
        u = pd.read_parquet(_UNIVERSE_CACHE)
        log.info("Universo desde caché: %d tickers EQUITY.", len(u))
        return u["ticker"].tolist()
    import yfinance as yf
    s = pd.read_parquet(_STATS)
    cand = s[s["days_ge5"] >= 200].copy()
    # survivors: FNSPID con precio hasta >=2023
    def _recent(t):
        p = _FH / f"{t}.csv"
        if not p.exists():
            return False
        try:
            d = pd.to_datetime(pd.read_csv(p, usecols=["date"])["date"], errors="coerce").dropna()
            return len(d) > 0 and d.max() >= pd.Timestamp("2023-01-01")
        except Exception:  # noqa: BLE001
            return False
    cand = cand[cand["ticker"].map(_recent)]
    log.info("Clasificando quoteType de %d survivors...", len(cand))
    eq = []
    for i, t in enumerate(cand["ticker"], 1):
        if t.endswith("-X"):
            continue
        try:
            if str(yf.Ticker(t).fast_info["quoteType"]).upper() == "EQUITY":
                eq.append(t)
        except Exception:  # noqa: BLE001
            pass
        if i % 100 == 0:
            log.info("  %d/%d", i, len(cand))
    pd.DataFrame({"ticker": eq}).to_parquet(_UNIVERSE_CACHE, index=False)
    log.info("Universo EQUITY: %d tickers.", len(eq))
    return eq


# ---------------------------------------------------------------------------
# Precios + indicadores + liquidez + target (FNSPID)
# ---------------------------------------------------------------------------

def _indicators(g: pd.DataFrame) -> pd.DataFrame:
    from ta.momentum import RSIIndicator
    from ta.trend import MACD
    from ta.volatility import BollingerBands
    g = g.sort_values("date").copy()
    close = pd.to_numeric(g["close"], errors="coerce")
    px = pd.to_numeric(g["adj_close"], errors="coerce").fillna(close)
    vol = pd.to_numeric(g["volume"], errors="coerce")
    g["rsi"] = RSIIndicator(close, 14, fillna=False).rsi()
    g["macd_hist"] = MACD(close, 12, 26, 9, fillna=False).macd_diff()
    bb = BollingerBands(close, 20, 2, fillna=False)
    band = bb.bollinger_hband() - bb.bollinger_lband()
    mid = bb.bollinger_mavg()
    g["bb_pct"] = np.where(band != 0, (close - bb.bollinger_lband()) / band, np.nan)
    g["bb_width"] = np.where(mid != 0, band / mid, np.nan)
    ret = px.pct_change()
    g["volatility_20d"] = ret.rolling(20).std()
    g["volume_ratio"] = vol / vol.rolling(20).mean()
    g["dollar_vol_60d"] = (close * vol).rolling(60, min_periods=30).median()
    g["ret_5d_fwd"] = px.shift(-_FWD) / px - 1.0
    return g


def prices_tech(universe: list[str], log) -> pd.DataFrame:
    parts = []
    for t in universe + ["SPY"]:
        p = _FH / f"{t}.csv"
        if not p.exists():
            continue
        try:
            d = pd.read_csv(p)
            d.columns = [c.lower().strip().replace(" ", "_") for c in d.columns]
            d["date"] = pd.to_datetime(d["date"], errors="coerce")
            d = d.dropna(subset=["date"])
            d = d[(d["date"] >= _WIN_S - pd.Timedelta(days=120)) & (d["date"] <= _WIN_E)]
            if len(d) < 60:
                continue
            d["ticker"] = t
            parts.append(_indicators(d))
        except Exception as exc:  # noqa: BLE001
            log.warning("  precio %s: %s", t, exc)
    px = pd.concat(parts, ignore_index=True)
    # exceso vs SPY
    spy = px[px["ticker"] == "SPY"][["date", "ret_5d_fwd"]].rename(columns={"ret_5d_fwd": "spy_5d"})
    px = px.merge(spy, on="date", how="left")
    px["excess_5d"] = px["ret_5d_fwd"] - px["spy_5d"]
    px = px[px["ticker"] != "SPY"]
    px = px[(px["date"] >= _WIN_S) & (px["date"] <= _WIN_E)]
    # filtro de operabilidad (liquidez point-in-time)
    px = px[(px["dollar_vol_60d"] >= _DV_LO) & (px["dollar_vol_60d"] <= _DV_HI)]
    cols = ["ticker", "date", "rsi", "macd_hist", "bb_pct", "bb_width", "volatility_20d",
            "volume_ratio", "dollar_vol_60d", "excess_5d"]
    log.info("Precios/técnico + liquidez: %d filas, %d tickers.", len(px), px["ticker"].nunique())
    return px[cols]


# ---------------------------------------------------------------------------
# StockTwits sentiment diario (symbol_sentiments, filtrado al universo)
# ---------------------------------------------------------------------------

def _parse_syms(cell):
    if not isinstance(cell, str) or not cell.strip() or cell.strip() == "[]":
        return []
    try:
        v = ast.literal_eval(cell)
    except (ValueError, SyntaxError, TypeError, MemoryError):
        return []
    return [normalize_ticker(x) for x in v] if isinstance(v, (list, tuple)) else []


def stocktwits_daily(universe: set[str], log) -> pd.DataFrame:
    uni = set(universe)
    # pre-filtro barato: solo parsear filas cuyo symbol_list mencione un ticker del universo
    pat = re.compile(r"'(" + "|".join(re.escape(t) for t in uni) + r")'")
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
            hit = chunk["symbol_list"].str.contains(pat, na=False)
            chunk = chunk[hit]
            if chunk.empty:
                continue
            d = pd.to_datetime(chunk["created_at"], errors="coerce")
            m = d.notna() & (d >= _WIN_S) & (d <= _WIN_E)
            chunk = chunk[m]
            if chunk.empty:
                continue
            chunk = chunk.assign(date=d[m].dt.date,
                                 sent=pd.to_numeric(chunk["sentiment"], errors="coerce"),
                                 syms=chunk["symbol_list"].map(_parse_syms))
            chunk = chunk.explode("syms").rename(columns={"syms": "ticker"})
            chunk = chunk[chunk["ticker"].isin(uni)]
            if chunk.empty:
                continue
            parts.append(chunk[["ticker", "date", "sent", "user_id"]])
        if fi % 5 == 0:
            log.info("  StockTwits %d/%d (%.0fs)", fi, len(files), time.time() - t0)
    if not parts:
        return pd.DataFrame(columns=["ticker", "date"])
    big = pd.concat(parts, ignore_index=True)
    big["date"] = pd.to_datetime(big["date"])
    g = big.groupby(["ticker", "date"])
    agg = pd.DataFrame({
        "total_count": g.size(),
        "bullish_count": g["sent"].apply(lambda s: (s > 0).sum()),
        "bearish_count": g["sent"].apply(lambda s: (s < 0).sum()),
        "unique_users": g["user_id"].nunique(),
    }).reset_index()
    agg["bullish_ratio"] = agg["bullish_count"] / agg["total_count"].clip(lower=1)
    log.info("StockTwits diario: %d filas (ticker,día).", len(agg))
    return agg


# ---------------------------------------------------------------------------
# News diario (FNSPID, VADER) + WSB diario
# ---------------------------------------------------------------------------

def news_daily(universe: set[str], log) -> pd.DataFrame:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        vader = SentimentIntensityAnalyzer()
    except Exception:  # noqa: BLE001
        vader = None
    uni = set(universe)
    parts, t0 = [], time.time()
    reader = pd.read_csv(_NEWS, usecols=["Date", "Stock_symbol", "Article_title"],
                         chunksize=_CHUNK, dtype="string", on_bad_lines="skip",
                         encoding="utf-8", encoding_errors="replace")
    for i, chunk in enumerate(reader):
        chunk["ticker"] = chunk["Stock_symbol"].map(normalize_ticker)
        chunk = chunk[chunk["ticker"].isin(uni)]
        if chunk.empty:
            continue
        d = pd.to_datetime(chunk["Date"], errors="coerce", utc=True).dt.tz_localize(None)
        m = d.notna() & (d >= _WIN_S) & (d <= _WIN_E)
        chunk = chunk[m]
        if chunk.empty:
            continue
        chunk = chunk.assign(date=d[m].dt.date)
        if vader is not None:
            chunk["sent"] = [vader.polarity_scores(str(x))["compound"] for x in chunk["Article_title"].fillna("")]
        else:
            chunk["sent"] = np.nan
        parts.append(chunk[["ticker", "date", "sent"]])
        if i % 20 == 0:
            log.info("  news chunk %d (%.0fs)", i, time.time() - t0)
    if not parts:
        return pd.DataFrame(columns=["ticker", "date"])
    big = pd.concat(parts, ignore_index=True)
    big["date"] = pd.to_datetime(big["date"])
    agg = big.groupby(["ticker", "date"]).agg(
        news_count=("sent", "count"), news_sentiment_mean=("sent", "mean")).reset_index()
    log.info("News diario: %d filas.", len(agg))
    return agg


def wsb_daily(universe: set[str], log) -> pd.DataFrame:
    uni = set(universe)
    rows = []
    cashtag = re.compile(r"\$([A-Z]{1,5})")
    # unanimad: cashtags en title
    for f in sorted(_UNANIMAD.glob("*.csv")):
        try:
            d = pd.read_csv(f, usecols=lambda c: c in ("title", "score", "num_comments", "created_utc"),
                            encoding="utf-8", encoding_errors="replace")
            d["date"] = pd.to_datetime(d["created_utc"], unit="s", errors="coerce").dt.date
            d = d.dropna(subset=["date"])
            d["tk"] = d["title"].astype("string").fillna("").map(
                lambda s: [t for t in cashtag.findall(s.upper()) if t in uni])
            d = d[d["tk"].map(len) > 0].explode("tk").rename(columns={"tk": "ticker"})
            for r in d.itertuples(index=False):
                rows.append((r.ticker, r.date, float(getattr(r, "score", 0) or 0),
                             float(getattr(r, "num_comments", 0) or 0)))
        except Exception as exc:  # noqa: BLE001
            log.warning("  wsb unanimad %s: %s", f.name, exc)
    # kevin: columna symbol
    try:
        from src.data.longrun.process_wsb import _parse_value_tuples, _field, _KEVIN_COLS
        ci = _KEVIN_COLS
        for f in sorted(_KEVIN.glob("*.sql")):
            with open(f, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if not line.startswith("INSERT INTO `reddit_posts`"):
                        continue
                    for flds in _parse_value_tuples(line.split("VALUES ", 1)[1]):
                        if len(flds) <= ci["symbol"]:
                            continue
                        sym = _field(flds[ci["symbol"]]); cr = _field(flds[ci["created_utc"]])
                        if not sym or not cr:
                            continue
                        tk = normalize_ticker(sym)
                        if tk not in uni:
                            continue
                        dt = pd.to_datetime(cr, errors="coerce")
                        if pd.isna(dt):
                            continue
                        rows.append((tk, dt.date(),
                                     float(_field(flds[ci["score"]]) or 0),
                                     float(_field(flds[ci["num_comments"]]) or 0)))
    except Exception as exc:  # noqa: BLE001
        log.warning("  wsb kevin: %s", exc)
    if not rows:
        return pd.DataFrame(columns=["ticker", "date"])
    df = pd.DataFrame(rows, columns=["ticker", "date", "score", "num_comments"])
    df["date"] = pd.to_datetime(df["date"])
    agg = df.groupby(["ticker", "date"]).agg(
        mention_count=("score", "count"), avg_score=("score", "mean"),
        total_comments=("num_comments", "sum")).reset_index()
    log.info("WSB diario: %d filas.", len(agg))
    return agg


# ---------------------------------------------------------------------------
# Ensamble
# ---------------------------------------------------------------------------

def run() -> dict:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("build_v2")

    universe = build_universe(log)
    uni = set(universe)

    base = prices_tech(universe, log)
    st = stocktwits_daily(uni, log)
    news = news_daily(uni, log)
    wsb = wsb_daily(uni, log)

    for name, part in [("stocktwits", st), ("news", news), ("wsb", wsb)]:
        if not part.empty:
            part["date"] = pd.to_datetime(part["date"])
            base = base.merge(part, on=["ticker", "date"], how="left")

    # NaN -> 0 en conteos
    for c in ["bullish_count", "bearish_count", "total_count", "unique_users",
              "mention_count", "total_comments", "news_count"]:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0)
        else:
            base[c] = 0
    for c in ["bullish_ratio", "avg_score", "news_sentiment_mean"]:
        if c not in base.columns:
            base[c] = np.nan
    base["bullish_ratio"] = base["bullish_ratio"].fillna(0.0)

    # combined mentions >= 5 (idéntico a v1)
    base["combined_mentions"] = base["total_count"] + base["mention_count"]
    base = base[base["combined_mentions"] >= 5].copy()

    # aceleración (mismo criterio que prepare.py de fase 1)
    base = base.sort_values(["ticker", "date"])
    tc = base["total_count"].clip(lower=1)
    base["_net"] = (base["bullish_count"] - base["bearish_count"]) / tc
    g = base.groupby("ticker", group_keys=False)
    base["sent_delta_ma7"] = base["_net"] - g["_net"].apply(lambda s: s.rolling(7, min_periods=2).mean())
    ma30 = g["total_count"].apply(lambda s: s.rolling(30, min_periods=5).mean())
    sd30 = g["total_count"].apply(lambda s: s.rolling(30, min_periods=5).std())
    base["mention_zscore_30d"] = (base["total_count"] - ma30) / sd30.replace(0, np.nan)
    base["bull_bear_change"] = g["bullish_ratio"].apply(lambda s: s.diff())
    base["event_day"] = (base["mention_zscore_30d"] >= 2.0).astype(int)
    base["is_top50"] = 0  # n/a en v2 (universo ya es small/mid)

    # target
    base = base.dropna(subset=["excess_5d"])
    base["target_5d"] = (base["excess_5d"] > 0).astype(int)
    base["target"] = base["target_5d"]
    base = base.drop(columns=["_net"])

    base.to_parquet(_OUT, index=False)
    bal = round(float(base["target_5d"].mean()) * 100, 2)
    summary = {"rows": len(base), "tickers": int(base["ticker"].nunique()),
               "target_5d_pct_sube": bal, "event_days": int(base["event_day"].sum()),
               "date_min": str(base["date"].min().date()), "date_max": str(base["date"].max().date()),
               "output": str(_OUT)}
    log.info("dataset_v2 fin: %s", summary)
    return summary


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import json
    print(json.dumps(run(), indent=2, ensure_ascii=False, default=str))
