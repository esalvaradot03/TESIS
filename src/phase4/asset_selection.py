"""
Fase 4 — Tarea 1: selección data-driven de los 5 activos.

Cruza actividad social (StockTwits NYU) con cobertura de noticias (FNSPID) en
2015-2022 y produce un ranking. NO decide los 5 activos: deja la tabla lista
para revisión humana.

Salidas (todas en experiments/phase4/):
  - st_counts.parquet   : mensajes y días activos por ticker (StockTwits)
  - news_counts.parquet : noticias y días activos por ticker (FNSPID)
  - asset_selection.parquet / .md : ranking cruzado + ETFs de referencia

Los escaneos pesados (34 CSV StockTwits ~4.3GB, 1 CSV noticias ~23GB) se cachean
en parquet; si el cache existe se salta el re-escaneo (idempotente).
"""

import ast
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
_FH = EXTERNAL_DATA_ROOT / "fnspid" / "full_history"
_OUT = ROOT / "experiments" / "phase4"

_WIN_S = pd.Timestamp("2015-01-01")
_WIN_E = pd.Timestamp("2022-12-31")
_CHUNK = 1_000_000

# ETFs de referencia que se incluyen aunque no aparezcan en el top social.
_ETFS = ["SPY", "QQQ", "GLD", "USO", "IWM"]

_SYM_RE = re.compile(r"'([A-Za-z][A-Za-z0-9.\-]{0,6})'")


def _consolidate(parts: list[pd.DataFrame]) -> list[pd.DataFrame]:
    """Colapsa (ticker, date) sumando 'msgs' para acotar memoria."""
    big = pd.concat(parts, ignore_index=True)
    big = big.groupby(["ticker", "date"], as_index=False)["msgs"].sum()
    return [big]


# ---------------------------------------------------------------- StockTwits
def scan_stocktwits(log) -> pd.DataFrame:
    cache = _OUT / "st_counts.parquet"
    if cache.exists():
        log.info("StockTwits: uso cache %s", cache.name)
        return pd.read_parquet(cache)

    files = sorted(STOCKTWITS_NYU_SYMBOL_SENTIMENTS.glob("*.csv"))
    log.info("StockTwits: %d archivos a escanear.", len(files))
    parts: list[pd.DataFrame] = []
    t0 = time.time()
    for fi, f in enumerate(files, 1):
        reader = pd.read_csv(f, usecols=["created_at", "symbol_list"], dtype="string",
                             chunksize=_CHUNK, on_bad_lines="skip",
                             encoding="utf-8", encoding_errors="replace")
        for chunk in reader:
            d = pd.to_datetime(chunk["created_at"], errors="coerce")
            m = d.notna() & (d >= _WIN_S) & (d <= _WIN_E)
            if not m.any():
                continue
            sub = pd.DataFrame({
                "date": d[m].dt.strftime("%Y-%m-%d"),
                "syms": chunk["symbol_list"][m].map(
                    lambda s: _SYM_RE.findall(s) if isinstance(s, str) else []),
            })
            sub = sub.explode("syms").dropna(subset=["syms"])
            if sub.empty:
                continue
            sub["ticker"] = sub["syms"].str.upper()
            sub["msgs"] = 1
            parts.append(sub.groupby(["ticker", "date"], as_index=False)["msgs"].sum())
        if fi % 4 == 0:
            parts = _consolidate(parts)
            log.info("  StockTwits %d/%d (%.0fs, %d ticker-dias)",
                     fi, len(files), time.time() - t0, len(parts[0]))
    td = pd.concat(parts, ignore_index=True).groupby(["ticker", "date"], as_index=False)["msgs"].sum()
    agg = td.groupby("ticker").agg(st_msgs=("msgs", "sum"),
                                   st_active_days=("date", "nunique")).reset_index()
    agg["st_msgs_per_day"] = agg["st_msgs"] / agg["st_active_days"]
    agg = agg.sort_values("st_msgs", ascending=False)
    _OUT.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(cache, index=False)
    log.info("StockTwits: %d tickers -> %s", len(agg), cache.name)
    return agg


# ---------------------------------------------------------------- FNSPID news
def scan_news(log) -> pd.DataFrame:
    cache = _OUT / "news_counts.parquet"
    if cache.exists():
        log.info("Noticias: uso cache %s", cache.name)
        return pd.read_parquet(cache)

    log.info("Noticias: escaneo %s (~23GB, una pasada)...", _NEWS.name)
    parts: list[pd.DataFrame] = []
    t0, rows = time.time(), 0
    reader = pd.read_csv(_NEWS, usecols=["Date", "Stock_symbol"], dtype="string",
                         chunksize=_CHUNK, on_bad_lines="skip",
                         encoding="utf-8", encoding_errors="replace")
    for ci, chunk in enumerate(reader, 1):
        rows += len(chunk)
        d = pd.to_datetime(chunk["Date"].str.replace(" UTC", "", regex=False),
                           errors="coerce", utc=False)
        m = d.notna() & (d >= _WIN_S) & (d <= _WIN_E) & chunk["Stock_symbol"].notna()
        if not m.any():
            continue
        sub = pd.DataFrame({
            "ticker": chunk["Stock_symbol"][m].str.upper().str.strip(),
            "date": d[m].dt.strftime("%Y-%m-%d"),
        })
        sub["msgs"] = 1
        parts.append(sub.groupby(["ticker", "date"], as_index=False)["msgs"].sum())
        if ci % 5 == 0:
            parts = _consolidate(parts)
            log.info("  Noticias %d chunks / %d filas (%.0fs, %d ticker-dias)",
                     ci, rows, time.time() - t0, len(parts[0]))
    td = pd.concat(parts, ignore_index=True).groupby(["ticker", "date"], as_index=False)["msgs"].sum()
    agg = td.groupby("ticker").agg(news_total=("msgs", "sum"),
                                   news_active_days=("date", "nunique")).reset_index()
    agg["news_per_day"] = agg["news_total"] / agg["news_active_days"]
    agg = agg.sort_values("news_total", ascending=False)
    agg.to_parquet(cache, index=False)
    log.info("Noticias: %d tickers, %d filas totales -> %s", len(agg), rows, cache.name)
    return agg


# ---------------------------------------------------------------- precios
def _fnspid_price_span(ticker: str) -> tuple[str, str, int]:
    """Rango de fechas y nº de barras en el CSV local de FNSPID (para detectar
    el truncamiento 2020)."""
    p = _FH / f"{ticker}.csv"
    if not p.exists():
        return ("", "", 0)
    try:
        d = pd.read_csv(p, usecols=["date"])
    except Exception:  # noqa: BLE001
        return ("", "", 0)
    dt = pd.to_datetime(d["date"], errors="coerce").dropna()
    if dt.empty:
        return ("", "", 0)
    return (dt.min().strftime("%Y-%m-%d"), dt.max().strftime("%Y-%m-%d"), len(dt))


def _yf_price_ok(tickers: list[str], log) -> pd.DataFrame:
    """Verifica cobertura de precio 2015-2022 vía yfinance para los finalistas."""
    import yfinance as yf
    rows = []
    for t in tickers:
        try:
            h = yf.Ticker(t).history(start="2015-01-01", end="2023-01-01",
                                     auto_adjust=True)
        except Exception as e:  # noqa: BLE001
            rows.append({"ticker": t, "yf_bars": 0, "yf_start": "", "yf_end": "",
                         "yf_ok": False, "yf_err": str(e)[:40]})
            continue
        if h.empty:
            rows.append({"ticker": t, "yf_bars": 0, "yf_start": "", "yf_end": "",
                         "yf_ok": False, "yf_err": "vacio"})
            continue
        idx = h.index
        rows.append({
            "ticker": t, "yf_bars": len(h),
            "yf_start": idx.min().strftime("%Y-%m-%d"),
            "yf_end": idx.max().strftime("%Y-%m-%d"),
            # cobertura completa: empieza <=2015 y llega a >=2022-12
            "yf_ok": bool(idx.min().year <= 2015 and idx.max() >= pd.Timestamp("2022-12-01", tz=idx.tz)),
            "yf_err": "",
        })
        time.sleep(0.3)
    return pd.DataFrame(rows)


def _yf_meta(tickers: list[str], log) -> pd.DataFrame:
    """Market cap aprox, sector y tipo (acción/ETF) vía yfinance."""
    import yfinance as yf
    rows = []
    for t in tickers:
        info = {}
        try:
            info = yf.Ticker(t).info
        except Exception:  # noqa: BLE001
            info = {}
        qt = info.get("quoteType", "")
        rows.append({
            "ticker": t,
            "tipo": "ETF" if qt == "ETF" else ("ACCION" if qt in ("EQUITY", "") else qt),
            "sector": info.get("sector") or info.get("category") or "",
            "mktcap_musd": round((info.get("marketCap") or 0) / 1e6) or np.nan,
            "nombre": (info.get("shortName") or "")[:32],
        })
        time.sleep(0.3)
    return pd.DataFrame(rows)


def run() -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("phase4_select")
    _OUT.mkdir(parents=True, exist_ok=True)

    st = scan_stocktwits(log)
    news = scan_news(log)

    # cruce: solo tickers presentes en ambas fuentes
    xr = st.merge(news, on="ticker", how="inner")
    # ranking combinado por percentil en cada fuente
    xr["st_rank"] = xr["st_msgs"].rank(ascending=False)
    xr["news_rank"] = xr["news_total"].rank(ascending=False)
    xr["combo_rank"] = xr["st_rank"] + xr["news_rank"]
    xr = xr.sort_values("combo_rank")

    # candidatos: top 30 por combo + ETFs de referencia
    top = xr.head(30).copy()
    etf_extra = [e for e in _ETFS if e not in set(top["ticker"])]
    if etf_extra:
        top = pd.concat([top, xr[xr["ticker"].isin(etf_extra)]], ignore_index=True)

    finalists = top["ticker"].tolist()
    log.info("Verificando precios FNSPID + yfinance para %d candidatos...", len(finalists))
    fn = pd.DataFrame(
        [(t, *_fnspid_price_span(t)) for t in finalists],
        columns=["ticker", "fnspid_start", "fnspid_end", "fnspid_bars"])
    yf_px = _yf_price_ok(finalists, log)
    yf_mt = _yf_meta(finalists, log)

    tab = (top.merge(fn, on="ticker", how="left")
              .merge(yf_px, on="ticker", how="left")
              .merge(yf_mt, on="ticker", how="left"))
    # marca del artefacto de truncamiento 2020 en FNSPID
    tab["trunc_2020"] = tab["fnspid_end"].fillna("").str.slice(0, 4).isin(["2020", "2019", ""])
    tab = tab.sort_values("combo_rank")

    tab.to_parquet(_OUT / "asset_selection.parquet", index=False)
    _write_md(tab, xr, log)
    return tab


def _write_md(tab: pd.DataFrame, xr: pd.DataFrame, log):
    def _f(x, nd=0):
        if pd.isna(x):
            return "—"
        return f"{x:,.{nd}f}"

    L = ["# Fase 4 — Tarea 1: selección data-driven de activos\n",
         f"Rango: **2015-01-01 … 2022-12-31**. Universo cruzado StockTwits∩FNSPID: "
         f"**{len(xr):,} tickers**. Se muestran top 30 por rank combinado + ETFs de referencia.\n",
         "Marcas: `trunc_2020` = el precio local FNSPID termina en 2019/2020 (artefacto de "
         "truncamiento); `yf_ok` = yfinance cubre 2015→2022 completo.\n",
         "| # | ticker | tipo | sector | mktcap $M | ST msgs | ST/día | news | news/día | "
         "combo | fnspid_end | yf_ok | trunc2020 |",
         "|---|--------|------|--------|-----------|---------|--------|------|----------|"
         "-------|-----------|-------|-----------|"]
    for i, (_, r) in enumerate(tab.iterrows(), 1):
        L.append(
            f"| {i} | **{r['ticker']}** | {r.get('tipo','')} | {str(r.get('sector',''))[:16]} | "
            f"{_f(r.get('mktcap_musd'))} | {_f(r['st_msgs'])} | {_f(r['st_msgs_per_day'],1)} | "
            f"{_f(r['news_total'])} | {_f(r['news_per_day'],1)} | {_f(r['combo_rank'])} | "
            f"{r.get('fnspid_end') or '—'} | {'sí' if r.get('yf_ok') else 'no'} | "
            f"{'⚠' if r.get('trunc_2020') else ''} |")
    L.append("")
    (_OUT / "asset_selection.md").write_text("\n".join(L), encoding="utf-8")
    log.info("Reporte -> %s", _OUT / "asset_selection.md")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run()
