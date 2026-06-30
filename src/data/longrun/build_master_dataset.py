"""
JOB 6 — Une todas las fuentes en el dataset maestro (depende de Jobs 1,2,3,5).

Outer-join por (ticker, fecha) de:
  stocktwits_sentiment_daily + wsb_mentions_daily + fnspid_news_daily + indicators_daily
Maneja NaN explícito, define target = 1 si excess_return_5d_forward > 0, y filtra
filas con < N menciones combinadas (StockTwits + WSB).

Salida: D:\\trading-data\\master\\dataset_v1.parquet
Si falta alguna fuente, se construye con las disponibles y se reporta cuál faltó.
"""

import sys
from pathlib import Path

import pandas as pd

from src.data.longrun.common import MASTER_DIR, safe_to_parquet, setup_logger

JOB = "job6_master"
_OUTPUT = MASTER_DIR / "dataset_v1.parquet"
_MIN_COMBINED_MENTIONS = 5

_SOURCES = {
    "stocktwits": MASTER_DIR / "stocktwits_sentiment_daily.parquet",
    "wsb": MASTER_DIR / "wsb_mentions_daily.parquet",
    "fnspid": MASTER_DIR / "fnspid_news_daily.parquet",
    "indicators": MASTER_DIR / "indicators_daily.parquet",
}

_COUNT_COLS_FILL0 = ["bullish_count", "bearish_count", "total_count", "unique_users",
                     "mention_count", "total_comments", "news_count"]


def _load(path: Path, log):
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date
        if "ticker" in df.columns:
            df["ticker"] = df["ticker"].astype(str).str.upper()
        if "year" in df.columns:
            df = df.drop(columns=["year"])
        return df
    except Exception as exc:  # noqa: BLE001
        log.error("No se pudo leer %s: %s", path, exc)
        return None


def run(smoke: bool = False, limit: int | None = None) -> dict:
    log = setup_logger(JOB)
    loaded, missing = {}, []
    for name, path in _SOURCES.items():
        df = _load(path, log)
        if df is None:
            missing.append(name)
            log.warning("Fuente faltante: %s (%s)", name, path)
        else:
            loaded[name] = df
            log.info("Fuente %s: %d filas.", name, len(df))

    if "indicators" not in loaded:
        log.error("Sin indicators_daily no hay target; abortando Job 6.")
        return {"job": JOB, "error": "indicators missing", "missing": missing}

    master = loaded["indicators"]
    for name in ("stocktwits", "wsb", "fnspid"):
        if name in loaded:
            master = master.merge(loaded[name], on=["ticker", "date"], how="outer")

    # NaN explícito en conteos → 0
    for c in _COUNT_COLS_FILL0:
        if c in master.columns:
            master[c] = pd.to_numeric(master[c], errors="coerce").fillna(0)
    if "bullish_ratio" in master.columns:
        master["bullish_ratio"] = master["bullish_ratio"].fillna(0.0)

    # Menciones combinadas y filtro de actividad
    st = master["total_count"] if "total_count" in master.columns else 0
    wsb = master["mention_count"] if "mention_count" in master.columns else 0
    master["combined_mentions"] = st + wsb

    # target solo donde hay retorno en exceso (necesita indicadores+futuro)
    before = len(master)
    master = master.dropna(subset=["return_excess_over_spy_5d_forward"])
    master["target"] = (master["return_excess_over_spy_5d_forward"] > 0).astype(int)

    master = master[master["combined_mentions"] >= _MIN_COMBINED_MENTIONS].reset_index(drop=True)
    master = master.sort_values(["ticker", "date"]).reset_index(drop=True)

    safe_to_parquet(master, _OUTPUT)

    # --- Reporte ---
    yr = pd.to_datetime(master["date"]).dt.year
    coverage = yr.value_counts().sort_index().to_dict()
    nan_ratio = master.isna().mean().round(4)
    nan_cols = {c: float(v) for c, v in nan_ratio.items() if v > 0}
    summary = {
        "job": JOB,
        "rows": len(master),
        "rows_before_filters": before,
        "tickers": int(master["ticker"].nunique()),
        "date_min": str(master["date"].min()),
        "date_max": str(master["date"].max()),
        "target_pos_rate": round(float(master["target"].mean()), 4),
        "coverage_by_year": {int(k): int(v) for k, v in coverage.items()},
        "nan_ratio_by_col": nan_cols,
        "missing_sources": missing,
        "columns": list(master.columns),
        "output": str(_OUTPUT),
    }
    log.info("Job 6 fin: %d filas | %d tickers | %s→%s | target+=%.3f | faltó: %s",
             summary["rows"], summary["tickers"], summary["date_min"],
             summary["date_max"], summary["target_pos_rate"], missing or "ninguna")
    return summary


if __name__ == "__main__":
    run(smoke="--smoke" in sys.argv)
