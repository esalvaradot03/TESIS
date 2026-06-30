"""
JOB 3 — Procesa FNSPID (noticias) → señal de noticias diaria por ticker.

Entrada: D:\\trading-data\\fnspid\\nasdaq_exteral_data.csv  (≈23 GB, se ignora __MACOSX/)
Salida : D:\\trading-data\\master\\fnspid_news_daily.parquet

IMPORTANTE: contrario al supuesto inicial, este volcado de FNSPID NO trae una
columna de sentimiento precalculado (solo Date, Stock_symbol, Article_title,
resúmenes…). Para NO usar FinBERT (sin GPU) se calcula el sentimiento con VADER
(léxico, CPU, sin descargas) SOBRE EL TÍTULO, y solo de las filas que pasan el
filtro S&P 500 + fechas (mucho más barato). Documentado como desviación.

Robustez: lectura por chunks (no carga 23 GB a RAM), checkpoint por chunk,
try/except por chunk, consolidación con Dask.
"""

import sys
from pathlib import Path

import pandas as pd

from config.settings import EXTERNAL_DATA_ROOT
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

JOB = "job3_fnspid"
_INPUT = EXTERNAL_DATA_ROOT / "fnspid" / "nasdaq_exteral_data.csv"
_STAGE_DIR = STAGING_DIR / JOB
_OUTPUT = MASTER_DIR / "fnspid_news_daily.parquet"
_CHUNK = 500_000


def _get_vader():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except Exception:  # noqa: BLE001
        return None


def _process_chunk(chunk: pd.DataFrame, sp500: set[str], vader, idx: int) -> int:
    chunk = chunk.rename(columns={"Date": "date_raw", "Stock_symbol": "ticker",
                                  "Article_title": "title"})
    chunk["ticker"] = chunk["ticker"].map(normalize_ticker)
    chunk = chunk[chunk["ticker"].isin(sp500)]
    chunk["date"] = pd.to_datetime(chunk["date_raw"], errors="coerce", utc=True).dt.date
    chunk = chunk.dropna(subset=["date"])
    chunk = chunk[(chunk["date"] >= DATE_START) & (chunk["date"] <= DATE_END)]
    if chunk.empty:
        return 0
    if vader is not None:
        titles = chunk["title"].astype("string").fillna("")
        chunk["sentiment"] = [vader.polarity_scores(t)["compound"] for t in titles]
    else:
        chunk["sentiment"] = float("nan")
    out = chunk[["ticker", "date", "sentiment"]].reset_index(drop=True)
    safe_to_parquet(out, _STAGE_DIR / f"chunk_{idx:05d}.parquet")
    return len(out)


def _consolidate(log) -> None:
    parts = sorted(_STAGE_DIR.glob("*.parquet"))
    if not parts:
        log.warning("Job 3: sin staging.")
        return
    log.info("Consolidando %d chunks con Dask ...", len(parts))
    try:
        import dask.dataframe as dd
        ddf = dd.read_parquet([str(p) for p in parts])
        agg = ddf.groupby(["ticker", "date"]).agg(
            news_count=("sentiment", "count"),
            news_sentiment_mean=("sentiment", "mean"),
        ).reset_index().compute()
    except Exception as exc:  # noqa: BLE001
        log.warning("Dask falló (%s); fallback pandas.", exc)
        big = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        agg = big.groupby(["ticker", "date"]).agg(
            news_count=("sentiment", "count"),
            news_sentiment_mean=("sentiment", "mean"),
        ).reset_index()
    safe_to_parquet(agg, _OUTPUT)
    log.info("Job 3 salida: %s | %d filas | %d tickers.",
             _OUTPUT, len(agg), agg["ticker"].nunique())


def run(smoke: bool = False, limit: int | None = None) -> dict:
    log = setup_logger(JOB)
    ckpt = Checkpoint(JOB)
    if not _INPUT.exists():
        log.error("Job 3: no existe %s", _INPUT)
        return {"job": JOB, "error": "input missing"}

    vader = _get_vader()
    if vader is None:
        log.warning("VADER no disponible; news_sentiment_mean quedará NaN (solo news_count).")
    sp500 = sp500_set()
    max_chunks = (limit or 1) if smoke else None

    log.info("Job 3 inicio (smoke=%s, chunk=%d).", smoke, _CHUNK)
    ok = err = skipped = total = 0
    reader = pd.read_csv(
        _INPUT, usecols=["Date", "Stock_symbol", "Article_title"],
        chunksize=_CHUNK, encoding="utf-8", encoding_errors="replace",
        on_bad_lines="skip", dtype="string",
    )
    for idx, chunk in enumerate(reader):
        if max_chunks is not None and idx >= max_chunks:
            break
        key = f"chunk_{idx:05d}"
        if ckpt.is_done(key):
            skipped += 1
            continue
        try:
            n = _process_chunk(chunk, sp500, vader, idx)
            total += n
            ckpt.mark(key)
            ok += 1
            if idx % 10 == 0:
                log.info("  chunk %d ✓ (%d filas S&P500 acumuladas)", idx, total)
        except Exception as exc:  # noqa: BLE001 — chunk corrupto no mata el job
            err += 1
            log.error("  chunk %d ✗: %s", idx, exc)

    _consolidate(log)
    summary = {"job": JOB, "chunks_ok": ok, "chunks_err": err,
               "chunks_skipped": skipped, "rows": total, "output": str(_OUTPUT)}
    log.info("Job 3 fin: %s", summary)
    return summary


if __name__ == "__main__":
    run(smoke="--smoke" in sys.argv)
