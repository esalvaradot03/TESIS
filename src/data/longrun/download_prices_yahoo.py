"""
JOB 4 — Descarga OHLCV diario 2008-2024 con yfinance, un Parquet por ticker.

Salida: D:\\trading-data\\master\\prices\\<TICKER>.parquet  (reanudable por ticker)
        D:\\trading-data\\master\\prices_2008_2024.parquet   (consolidado)

Reanudable: si un <TICKER>.parquet ya existe, se omite. Rate limit ≤ 5 req/s,
retry con backoff exponencial (hasta 3) ante errores o respuestas vacías.
Incluye SPY (necesario para el retorno en exceso del Job 5).
"""

import sys
import time
from pathlib import Path

import pandas as pd

from config.settings import EXTERNAL_DATA_ROOT
from src.data.longrun.common import (
    DATE_END,
    DATE_START,
    MASTER_DIR,
    PRICES_DIR,
    Checkpoint,
    normalize_ticker,
    safe_to_parquet,
    setup_logger,
    sp500_set,
)

JOB = "job4_prices_yahoo"
_OUTPUT = MASTER_DIR / "prices_2008_2024.parquet"
_FNSPID_HISTORY = EXTERNAL_DATA_ROOT / "fnspid" / "full_history"
_MIN_INTERVAL = 0.2          # ≤ 5 req/s
_MAX_RETRIES = 3
_BACKOFF = 2.0


def _fallback_fnspid(ticker: str, log) -> pd.DataFrame | None:
    """
    Fallback SIN red: usa los OHLCV de FNSPID full_history\\<TICKER>.csv.

    Cubre ~1980-2023 (no 2024) y algunos tickers faltan/están truncados, pero
    hace al Job 4 robusto si la API de Yahoo falla durante la corrida desatendida.
    """
    p = _FNSPID_HISTORY / f"{ticker}.csv"
    if not p.exists():
        return None
    try:
        d = pd.read_csv(p, encoding="utf-8", encoding_errors="replace")
        d.columns = [str(c).lower().strip().replace(" ", "_") for c in d.columns]
        d["ticker"] = ticker
        d["date"] = pd.to_datetime(d["date"], errors="coerce").dt.date
        d = d.dropna(subset=["date"])
        d = d[(d["date"] >= DATE_START) & (d["date"] <= DATE_END)]
        keep = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
        for c in keep:
            if c not in d.columns:
                d[c] = pd.NA
        log.info("  ↳ %s: fallback FNSPID full_history (%d filas).", ticker, len(d))
        return d[keep].reset_index(drop=True) if not d.empty else None
    except Exception as exc:  # noqa: BLE001
        log.warning("  fallback FNSPID %s falló: %s", ticker, exc)
        return None


def _download_one(ticker: str, log) -> pd.DataFrame | None:
    import yfinance as yf
    yf_symbol = ticker.replace("-", "-")  # yfinance usa 'BRK-B'
    backoff = _BACKOFF
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            df = yf.download(
                yf_symbol, start=DATE_START.isoformat(), end=DATE_END.isoformat(),
                progress=False, auto_adjust=False, threads=False,
            )
            if df is not None and not df.empty:
                return df
            log.warning("  %s: respuesta vacía (intento %d/%d).", ticker, attempt, _MAX_RETRIES)
        except Exception as exc:  # noqa: BLE001
            log.warning("  %s: error %s (intento %d/%d).", ticker, exc, attempt, _MAX_RETRIES)
        time.sleep(backoff)
        backoff *= 2
    return None


def _normalize_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    df = df.reset_index()
    # yfinance puede devolver MultiIndex de columnas si pasa varios símbolos; aplanar
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df.columns = [str(c).lower().replace(" ", "_") for c in df.columns]
    rename = {"adj_close": "adj_close", "date": "date"}
    df = df.rename(columns=rename)
    df["ticker"] = ticker
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    keep = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
    for c in keep:
        if c not in df.columns:
            df[c] = pd.NA
    return df.dropna(subset=["date"])[keep]


def _consolidate(log) -> None:
    parts = sorted(PRICES_DIR.glob("*.parquet"))
    if not parts:
        log.warning("Job 4: sin precios por ticker para consolidar.")
        return
    log.info("Consolidando %d tickers ...", len(parts))
    frames = []
    for p in parts:
        try:
            frames.append(pd.read_parquet(p))
        except Exception as exc:  # noqa: BLE001
            log.error("  no se pudo leer %s: %s", p.name, exc)
    if not frames:
        return
    big = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])
    safe_to_parquet(big, _OUTPUT)
    log.info("Job 4 salida: %s | %d filas | %d tickers.",
             _OUTPUT, len(big), big["ticker"].nunique())


def run(smoke: bool = False, limit: int | None = None) -> dict:
    log = setup_logger(JOB)
    ckpt = Checkpoint(JOB)
    tickers = sorted(sp500_set() | {"SPY"})
    if smoke:
        tickers = ["AAPL", "MSFT", "NVDA", "SPY", "AMZN"][: (limit or 5)]
    log.info("Job 4 inicio: %d tickers (smoke=%s).", len(tickers), smoke)

    ok = err = skipped = 0
    last_call = 0.0
    for t in tickers:
        t = normalize_ticker(t)
        out_path = PRICES_DIR / f"{t}.parquet"
        if out_path.exists() and not smoke:
            skipped += 1
            continue
        # rate limit
        wait = _MIN_INTERVAL - (time.monotonic() - last_call)
        if wait > 0:
            time.sleep(wait)
        last_call = time.monotonic()

        df = _download_one(t, log)
        try:
            norm = _normalize_df(df, t) if df is not None else _fallback_fnspid(t, log)
            if norm is None or norm.empty:
                err += 1
                log.error("  ✗ %s: sin datos (yfinance ni fallback FNSPID).", t)
                continue
            safe_to_parquet(norm, out_path)
            ckpt.mark(t)
            ok += 1
            if ok % 25 == 0:
                log.info("  progreso: %d descargados, %d omitidos, %d errores.", ok, skipped, err)
        except Exception as exc:  # noqa: BLE001
            err += 1
            log.error("  ✗ %s normalización: %s", t, exc)

    _consolidate(log)
    summary = {"job": JOB, "downloaded": ok, "skipped": skipped, "errors": err,
               "output": str(_OUTPUT)}
    log.info("Job 4 fin: %s", summary)
    return summary


if __name__ == "__main__":
    run(smoke="--smoke" in sys.argv)
