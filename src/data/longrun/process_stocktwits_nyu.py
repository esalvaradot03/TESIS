"""
JOB 1 — Procesa StockTwits NYU (symbol_sentiments) → sentimiento diario por ticker.

Entrada : D:\\trading-data\\stocktwits_nyu\\symbol_sentiments\\*.csv
Salida  : D:\\trading-data\\master\\stocktwits_sentiment_daily.parquet (particionado por año)

Esquema de entrada: message_id, user_id, created_at (fecha), sentiment (+1/-1),
symbol_list (LITERAL de Python, p.ej. "['ZNGA','META']").

Robustez: per-archivo a staging (reanudable por checkpoint); try/except por
archivo; consolidación final con Dask (groupby out-of-core).
"""

import ast
import sys
from pathlib import Path

import pandas as pd

from src.data.longrun.common import (
    DATE_END,
    DATE_START,
    MASTER_DIR,
    STAGING_DIR,
    Checkpoint,
    normalize_ticker,
    safe_to_parquet,
    setup_logger,
    sp500_set,
)

JOB = "job1_stocktwits_nyu"
_INPUT_DIR_REL = Path("stocktwits_nyu") / "symbol_sentiments"
_STAGE_DIR = STAGING_DIR / JOB
_OUTPUT = MASTER_DIR / "stocktwits_sentiment_daily.parquet"


def _parse_symbol_list(cell: object) -> list[str]:
    if not isinstance(cell, str):
        return []
    s = cell.strip()
    if not s or s == "[]":
        return []
    try:
        val = ast.literal_eval(s)
    except (ValueError, SyntaxError, TypeError, MemoryError):
        return []
    if isinstance(val, (list, tuple)):
        return [normalize_ticker(x) for x in val if x]
    return []


def _process_file(path: Path, sp500: set[str], log) -> int:
    """Lee un CSV, filtra, explota por ticker y escribe staging. Devuelve nº de filas."""
    df = pd.read_csv(
        path,
        usecols=["user_id", "created_at", "sentiment", "symbol_list"],
        dtype={"user_id": "Int64", "sentiment": "float32", "symbol_list": "string"},
        encoding="utf-8", encoding_errors="replace",
    )
    df["date"] = pd.to_datetime(df["created_at"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])
    df = df[(df["date"] >= DATE_START) & (df["date"] <= DATE_END)]
    if df.empty:
        return 0

    df["symbols"] = df["symbol_list"].map(_parse_symbol_list)
    df = df[df["symbols"].map(len) > 0]
    df = df.explode("symbols").rename(columns={"symbols": "ticker"})
    df = df[df["ticker"].isin(sp500)]
    if df.empty:
        return 0

    out = df[["ticker", "date", "sentiment", "user_id"]].reset_index(drop=True)
    safe_to_parquet(out, _STAGE_DIR / f"{path.stem}.parquet")
    return len(out)


def _consolidate(log) -> None:
    """Consolida staging → agregación diaria por (ticker, fecha), particionada por año."""
    parts = sorted(_STAGE_DIR.glob("*.parquet"))
    if not parts:
        log.warning("Sin staging para consolidar.")
        return
    log.info("Consolidando %d archivos de staging con Dask ...", len(parts))
    try:
        import dask.dataframe as dd
        ddf = dd.read_parquet([str(p) for p in parts])
        ddf["bull"] = (ddf["sentiment"] > 0).astype("int64")
        ddf["bear"] = (ddf["sentiment"] < 0).astype("int64")
        # nunique NO está soportado en groupby.agg de Dask → se calcula aparte
        # (también out-of-core) y se une al resto de agregaciones.
        counts = ddf.groupby(["ticker", "date"]).agg(
            bullish_count=("bull", "sum"),
            bearish_count=("bear", "sum"),
            total_count=("sentiment", "count"),
        ).reset_index().compute()
        uu = (ddf.groupby(["ticker", "date"])["user_id"].nunique()
              .reset_index().rename(columns={"user_id": "unique_users"}).compute())
        agg = counts.merge(uu, on=["ticker", "date"], how="left")
    except Exception as exc:  # noqa: BLE001 — fallback pandas si Dask falla
        log.warning("Dask falló (%s); fallback a pandas por chunks.", exc)
        frames = []
        for p in parts:
            d = pd.read_parquet(p)
            d["bull"] = (d["sentiment"] > 0).astype("int64")
            d["bear"] = (d["sentiment"] < 0).astype("int64")
            frames.append(d)
        big = pd.concat(frames, ignore_index=True)
        agg = big.groupby(["ticker", "date"]).agg(
            bullish_count=("bull", "sum"),
            bearish_count=("bear", "sum"),
            total_count=("sentiment", "count"),
            unique_users=("user_id", "nunique"),
        ).reset_index()

    agg["bullish_ratio"] = agg["bullish_count"] / agg["total_count"].clip(lower=1)
    agg["date"] = pd.to_datetime(agg["date"])
    agg["year"] = agg["date"].dt.year
    agg["date"] = agg["date"].dt.date

    _OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    # Particionado por año (directorio Parquet)
    agg.to_parquet(_OUTPUT, index=False, partition_cols=["year"])
    log.info("Job 1 salida: %s | %d filas (ticker, día) | %d tickers, %d años.",
             _OUTPUT, len(agg), agg["ticker"].nunique(), agg["year"].nunique())


def run(smoke: bool = False, limit: int | None = None) -> dict:
    log = setup_logger(JOB)
    ckpt = Checkpoint(JOB)
    from config.settings import EXTERNAL_DATA_ROOT
    files = sorted((EXTERNAL_DATA_ROOT / _INPUT_DIR_REL).glob("*.csv"))
    if smoke:
        files = files[: (limit or 2)]
    log.info("Job 1 inicio: %d archivos (smoke=%s).", len(files), smoke)

    ok = err = skipped = total_rows = 0
    for f in files:
        if ckpt.is_done(f.name):
            skipped += 1
            continue
        try:
            n = _process_file(f, sp500_set(), log)
            total_rows += n
            ckpt.mark(f.name)
            ok += 1
            log.info("  ✓ %s (%d filas staged)", f.name, n)
        except Exception as exc:  # noqa: BLE001 — un archivo corrupto no mata el job
            err += 1
            log.error("  ✗ %s: %s", f.name, exc)

    _consolidate(log)
    summary = {"job": JOB, "files_ok": ok, "files_err": err, "files_skipped": skipped,
               "staged_rows": total_rows, "output": str(_OUTPUT)}
    log.info("Job 1 fin: %s", summary)
    return summary


if __name__ == "__main__":
    run(smoke="--smoke" in sys.argv)
