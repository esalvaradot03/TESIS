"""
FASE 2 — ETAPA A: auditoría de factibilidad (small/mid-caps fuera del S&P 500).

Solo audita, NO construye dataset ni entrena. Pasos:
  1. Escanea StockTwits NYU (symbol_sentiments/, misma fuente que alimentó v1)
     y agrega menciones por (ticker, día), excluyendo los 475 tickers de v1.
     Por ticker: nº días con >=5 menciones (2015-2023), rango de fechas, media.
  2. Cruza contra precios: usa FNSPID full_history (local) como fuente primaria
     (evita miles de llamadas a yfinance). Detecta deslistados (precios que
     terminan antes de 2023) y sin datos → mide survivorship bias potencial.
  3. Universo candidato: >=200 días de actividad (2015-2023) + market cap
     $200M-$10B (yfinance fast_info, solo para candidatos con precio reciente).
  4. Reporta y PARA.

Cachea el escaneo pesado en D:\\trading-data\\master\\v2_mention_stats.parquet.
"""

import ast
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import EXTERNAL_DATA_ROOT  # noqa: E402
from src.data.longrun.common import MASTER_DIR, normalize_ticker  # noqa: E402

_SS_DIR = EXTERNAL_DATA_ROOT / "stocktwits_nyu" / "symbol_sentiments"
_FNSPID_HIST = EXTERNAL_DATA_ROOT / "fnspid" / "full_history"
_STATS_CACHE = MASTER_DIR / "v2_mention_stats.parquet"
_DATASET_V1 = MASTER_DIR / "dataset_v1.parquet"

_WIN_START, _WIN_END = pd.Timestamp("2015-01-01"), pd.Timestamp("2023-12-31")
_MIN_ACTIVE_DAYS = 200
_MCAP_LO, _MCAP_HI = 200e6, 10e9
_CHUNK = 500_000


def _v1_tickers() -> set[str]:
    d = pd.read_parquet(_DATASET_V1, columns=["ticker"])
    return {normalize_ticker(t) for t in d["ticker"].unique()}


def _parse_syms(cell):
    if not isinstance(cell, str) or not cell.strip() or cell.strip() == "[]":
        return []
    try:
        v = ast.literal_eval(cell)
    except (ValueError, SyntaxError, TypeError, MemoryError):
        return []
    return [normalize_ticker(x) for x in v] if isinstance(v, (list, tuple)) else []


def scan_mentions(log) -> pd.DataFrame:
    """Agrega menciones por (ticker, día) sobre symbol_sentiments, excluyendo v1."""
    if _STATS_CACHE.exists():
        log.info("mention stats desde caché %s", _STATS_CACHE)
        return pd.read_parquet(_STATS_CACHE)

    excl = _v1_tickers()
    files = sorted(_SS_DIR.glob("*.csv"))
    log.info("Escaneando %d archivos symbol_sentiments (excluyendo %d tickers v1)...",
             len(files), len(excl))
    partials = []
    t0 = time.time()
    for fi, f in enumerate(files, 1):
        try:
            reader = pd.read_csv(f, usecols=["created_at", "symbol_list"], dtype="string",
                                 chunksize=_CHUNK, on_bad_lines="skip",
                                 encoding="utf-8", encoding_errors="replace")
        except Exception as exc:  # noqa: BLE001
            log.warning("  no se pudo leer %s: %s", f.name, exc)
            continue
        for chunk in reader:
            d = pd.to_datetime(chunk["created_at"], errors="coerce")
            m = d.notna() & (d >= _WIN_START) & (d <= _WIN_END)
            chunk = chunk[m]
            if chunk.empty:
                continue
            dd = d[m].dt.date
            syms = chunk["symbol_list"].map(_parse_syms)
            tmp = pd.DataFrame({"date": dd.values, "syms": syms.values})
            tmp = tmp[tmp["syms"].map(len) > 0].explode("syms").rename(columns={"syms": "ticker"})
            tmp = tmp[~tmp["ticker"].isin(excl)]
            if tmp.empty:
                continue
            partials.append(tmp.groupby(["ticker", "date"]).size().rename("n"))
        if fi % 5 == 0:
            log.info("  %d/%d archivos (%.0fs)", fi, len(files), time.time() - t0)

    if not partials:
        return pd.DataFrame()
    log.info("Consolidando %d parciales...", len(partials))
    counts = pd.concat(partials).groupby(level=[0, 1]).sum().reset_index()
    counts.columns = ["ticker", "date", "n"]

    # Estadísticas por ticker
    counts["active5"] = counts["n"] >= 5
    g = counts.groupby("ticker")
    stats = pd.DataFrame({
        "days_ge5": g["active5"].sum().astype(int),
        "total_days": g.size(),
        "date_min": g["date"].min(),
        "date_max": g["date"].max(),
        "avg_mentions": g["n"].mean().round(2),
        "total_mentions": g["n"].sum().astype(int),
    }).reset_index()
    _STATS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    stats.to_parquet(_STATS_CACHE, index=False)
    log.info("Escaneo listo: %d tickers no-v1 con actividad. Cache: %s", len(stats), _STATS_CACHE)
    return stats


def price_check(tickers: list[str], log) -> pd.DataFrame:
    """Disponibilidad de precios vía FNSPID full_history (local). Detecta deslistados."""
    rows = []
    for t in tickers:
        p = _FNSPID_HIST / f"{t}.csv"
        has, last = False, None
        if p.exists():
            try:
                d = pd.read_csv(p, usecols=["date"])
                dd = pd.to_datetime(d["date"], errors="coerce").dropna()
                if len(dd):
                    has, last = True, dd.max()
            except Exception:  # noqa: BLE001
                pass
        rows.append((t, has, last))
    pc = pd.DataFrame(rows, columns=["ticker", "has_price", "last_price_date"])
    # deslistado = tiene precio pero termina antes de 2023 (el dataset llega a 2023-12)
    pc["delisted"] = pc["has_price"] & (pc["last_price_date"] < pd.Timestamp("2023-01-01"))
    pc["price_recent"] = pc["has_price"] & (pc["last_price_date"] >= pd.Timestamp("2023-01-01"))
    return pc


def market_caps(tickers: list[str], log) -> pd.DataFrame:
    """Market cap (fast_info) solo para candidatos con precio reciente."""
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
    return pd.DataFrame(rows, columns=["ticker", "market_cap"])


def run() -> dict:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("audit_v2")

    # --- 1. Escaneo de menciones ---
    stats = scan_mentions(log)
    if stats.empty:
        return {"error": "sin datos de menciones"}
    stats["date_min"] = pd.to_datetime(stats["date_min"])
    stats["date_max"] = pd.to_datetime(stats["date_max"])

    n_any = len(stats)
    cand_mentions = stats[stats["days_ge5"] >= _MIN_ACTIVE_DAYS].copy()
    log.info("Tickers no-v1 con actividad: %d | con >=%d días >=5 menciones: %d",
             n_any, _MIN_ACTIVE_DAYS, len(cand_mentions))

    # --- 2. Cruce de precios (FNSPID full_history local) ---
    pc = price_check(cand_mentions["ticker"].tolist(), log)
    cand = cand_mentions.merge(pc, on="ticker", how="left")
    no_price = int((~cand["has_price"]).sum())
    delisted = int(cand["delisted"].sum())
    recent = int(cand["price_recent"].sum())
    survivorship_pct = round(100.0 * (no_price + delisted) / len(cand), 1) if len(cand) else 0.0

    # --- 3. Market cap (solo candidatos con precio reciente) → universo final ---
    recent_tickers = cand[cand["price_recent"]]["ticker"].tolist()
    mcaps = market_caps(recent_tickers, log)
    cand = cand.merge(mcaps, on="ticker", how="left")
    in_cap = cand[(cand["market_cap"] >= _MCAP_LO) & (cand["market_cap"] <= _MCAP_HI)].copy()
    est_rows = int(in_cap["days_ge5"].sum())

    # Distribución de market cap del universo final
    mc = in_cap["market_cap"].dropna()
    mc_dist = {
        "p10": float(mc.quantile(0.10)) if len(mc) else np.nan,
        "p50": float(mc.quantile(0.50)) if len(mc) else np.nan,
        "p90": float(mc.quantile(0.90)) if len(mc) else np.nan,
        "min": float(mc.min()) if len(mc) else np.nan,
        "max": float(mc.max()) if len(mc) else np.nan,
    }

    summary = {
        "tickers_no_v1_con_actividad": n_any,
        "candidatos_por_menciones(>=200d)": len(cand_mentions),
        "sin_precio": no_price,
        "deslistados(precio_termina_pre2023)": delisted,
        "con_precio_reciente": recent,
        "survivorship_pct(sin_precio+deslistados)": survivorship_pct,
        "candidatos_con_market_cap": int(cand["market_cap"].notna().sum()),
        "UNIVERSO_FINAL(cap_200M-10B + precio + >=200d)": len(in_cap),
        "filas_estimadas_dataset": est_rows,
        "market_cap_dist": mc_dist,
    }
    _write_report(summary, cand, in_cap, log)
    return summary


def _fmt_b(x):
    return f"${x/1e9:.2f}B" if x >= 1e9 else f"${x/1e6:.0f}M"


def _write_report(s, cand, in_cap, log):
    out = ROOT / "research" / "experiment_v2" / "audit_report.md"
    L = ["# Etapa A — Auditoría de factibilidad (small/mid-caps fuera del S&P 500)\n",
         "Fuente de menciones: StockTwits NYU `symbol_sentiments/` (misma que v1). "
         "Fuente de precios: FNSPID `full_history/` (local, hasta 2023-12). "
         "Market cap: yfinance fast_info (actual, estático).\n",
         "## Números\n",
         f"- Tickers **fuera de los 475 de v1** con actividad (2015-2023): **{s['tickers_no_v1_con_actividad']:,}**",
         f"- Con >= {_MIN_ACTIVE_DAYS} días de >=5 menciones: **{s['candidatos_por_menciones(>=200d)']:,}**",
         f"- Sin datos de precio (FNSPID): **{s['sin_precio']:,}**",
         f"- Deslistados (precio termina antes de 2023): **{s['deslistados(precio_termina_pre2023)']:,}**",
         f"- Con precio reciente (survivors): **{s['con_precio_reciente']:,}**",
         f"- **Survivorship bias potencial** (sin precio + deslistados): **{s['survivorship_pct(sin_precio+deslistados)']}%**",
         f"- **UNIVERSO CANDIDATO FINAL** (cap $200M-$10B + precio + >=200d): **{s['UNIVERSO_FINAL(cap_200M-10B + precio + >=200d)']:,}**",
         f"- Filas estimadas del dataset (suma de días activos): **~{s['filas_estimadas_dataset']:,}**\n",
         "## Distribución de market cap del universo final\n"]
    d = s["market_cap_dist"]
    if not np.isnan(d.get("p50", np.nan)):
        L.append(f"- min {_fmt_b(d['min'])} | p10 {_fmt_b(d['p10'])} | p50 {_fmt_b(d['p50'])} "
                 f"| p90 {_fmt_b(d['p90'])} | max {_fmt_b(d['max'])}")
    L.append("")
    out.write_text("\n".join(L), encoding="utf-8")
    log.info("Reporte en %s", out)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import json
    print(json.dumps(run(), indent=2, ensure_ascii=False, default=str))
