"""
JOB 2 — Procesa WallStreetBets (Kaggle) → menciones diarias por ticker.

Entrada A: wsb_kaggle\\unanimad_reddit_rwallstreetbets\\*.csv  (Reddit, solo 'title')
Entrada B: wsb_kaggle\\kevinwang313_wallstreetbets\\*.sql       (mysqldump)
Salida   : master\\wsb_mentions_daily.parquet

- unanimad: detección de tickers con ticker_detector (cashtags + S&P 500) sobre el título.
- kevin: la tabla `reddit_posts` ya trae columna `symbol` (ticker) y scores de
  sentimiento → se parsea el dump MySQL con un tokenizer propio (SQLite no corre
  mysqldump por dialecto/escapes). Si el parseo falla, se loguea y se SIGUE con unanimad.

Agregación (ticker, fecha): mention_count, avg_score, total_comments.
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

JOB = "job2_wsb"
_UNANIMAD_DIR = EXTERNAL_DATA_ROOT / "wsb_kaggle" / "unanimad_reddit_rwallstreetbets"
_KEVIN_DIR = EXTERNAL_DATA_ROOT / "wsb_kaggle" / "kevinwang313_wallstreetbets"
_STAGE_DIR = STAGING_DIR / JOB
_OUTPUT = MASTER_DIR / "wsb_mentions_daily.parquet"

# Índices de columnas en la tabla reddit_posts (orden del CREATE TABLE)
_KEVIN_COLS = {"created_utc": 12, "num_comments": 13, "score": 14, "symbol": 15}


# ---------------------------------------------------------------------------
# unanimad (CSV)
# ---------------------------------------------------------------------------

def _process_unanimad(log) -> int:
    from src.sentiment.ticker_detector import TickerDetector

    files = sorted(_UNANIMAD_DIR.glob("*.csv"))
    if not files:
        log.warning("unanimad: sin CSVs.")
        return 0
    detector = TickerDetector()
    total = 0
    for f in files:
        df = pd.read_csv(
            f, usecols=lambda c: c in ("title", "score", "num_comments", "created_utc"),
            encoding="utf-8", encoding_errors="replace",
        )
        df["date"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce").dt.date
        df = df.dropna(subset=["date"])
        df = df[(df["date"] >= DATE_START) & (df["date"] <= DATE_END)]
        df["score"] = pd.to_numeric(df.get("score"), errors="coerce").fillna(0)
        df["num_comments"] = pd.to_numeric(df.get("num_comments"), errors="coerce").fillna(0)
        df["title"] = df["title"].astype("string").fillna("")
        df["tickers"] = [detector.detect(t) for t in df["title"]]
        df = df[df["tickers"].map(len) > 0].explode("tickers").rename(columns={"tickers": "ticker"})
        if df.empty:
            continue
        out = df[["ticker", "date", "score", "num_comments"]].reset_index(drop=True)
        safe_to_parquet(out, _STAGE_DIR / f"unanimad_{f.stem}.parquet")
        total += len(out)
        log.info("  unanimad ✓ %s (%d filas)", f.name, len(out))
    return total


# ---------------------------------------------------------------------------
# kevin (mysqldump → parser propio)
# ---------------------------------------------------------------------------

def _parse_value_tuples(s: str):
    """Tokeniza la parte VALUES de un INSERT de mysqldump en listas de campos.

    Maneja strings con comillas simples y escapes de barra invertida (\\' \\\\ \\n),
    NULL y números. Ignora comas/paréntesis dentro de strings.
    """
    i, n = 0, len(s)
    while i < n:
        while i < n and s[i] != "(":
            i += 1
        if i >= n:
            break
        i += 1  # pasar '('
        fields: list = []
        cur: list[str] = []
        in_str = False
        quoted = False
        while i < n:
            c = s[i]
            if in_str:
                if c == "\\" and i + 1 < n:
                    cur.append(s[i + 1]); i += 2; continue
                if c == "'":
                    in_str = False; i += 1; continue
                cur.append(c); i += 1; continue
            if c == "'":
                in_str = True; quoted = True; i += 1; continue
            if c == ",":
                fields.append(("".join(cur), quoted)); cur = []; quoted = False; i += 1; continue
            if c == ")":
                fields.append(("".join(cur), quoted)); i += 1
                break
            cur.append(c); i += 1
        yield fields


def _field(val_quoted: tuple) -> object:
    text, quoted = val_quoted
    if not quoted and text.strip().upper() == "NULL":
        return None
    return text


def _process_kevin(log) -> int:
    sp500 = sp500_set()
    total = 0
    for f in sorted(_KEVIN_DIR.glob("*.sql")):
        # Solo el dump con la tabla de datos reddit_posts trae menciones
        try:
            rows = []
            ci = _KEVIN_COLS
            with open(f, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if not line.startswith("INSERT INTO `reddit_posts`"):
                        continue
                    vals = line.split("VALUES ", 1)[1]
                    for fields in _parse_value_tuples(vals):
                        if len(fields) <= ci["symbol"]:
                            continue
                        symbol = _field(fields[ci["symbol"]])
                        created = _field(fields[ci["created_utc"]])
                        if not symbol or not created:
                            continue
                        rows.append((
                            normalize_ticker(symbol), created,
                            _field(fields[ci["score"]]), _field(fields[ci["num_comments"]]),
                        ))
            if not rows:
                log.info("  kevin: %s sin filas reddit_posts (ok, otra tabla).", f.name)
                continue
            d = pd.DataFrame(rows, columns=["ticker", "created_utc", "score", "num_comments"])
            d["date"] = pd.to_datetime(d["created_utc"], errors="coerce").dt.date
            d = d.dropna(subset=["date"])
            d = d[(d["date"] >= DATE_START) & (d["date"] <= DATE_END)]
            d = d[d["ticker"].isin(sp500)]
            d["score"] = pd.to_numeric(d["score"], errors="coerce").fillna(0)
            d["num_comments"] = pd.to_numeric(d["num_comments"], errors="coerce").fillna(0)
            if d.empty:
                continue
            out = d[["ticker", "date", "score", "num_comments"]].reset_index(drop=True)
            safe_to_parquet(out, _STAGE_DIR / f"kevin_{f.stem}.parquet")
            total += len(out)
            log.info("  kevin ✓ %s (%d filas)", f.name, len(out))
        except Exception as exc:  # noqa: BLE001 — dump corrupto/dialecto: seguir
            log.error("  kevin ✗ %s: %s (se omite, continúa con el resto)", f.name, exc)
    return total


# ---------------------------------------------------------------------------
# Consolidación
# ---------------------------------------------------------------------------

def _consolidate(log) -> None:
    parts = sorted(_STAGE_DIR.glob("*.parquet"))
    if not parts:
        log.warning("Job 2: sin staging para consolidar.")
        return
    frames = [pd.read_parquet(p) for p in parts]
    big = pd.concat(frames, ignore_index=True)
    agg = big.groupby(["ticker", "date"]).agg(
        mention_count=("score", "count"),
        avg_score=("score", "mean"),
        total_comments=("num_comments", "sum"),
    ).reset_index()
    safe_to_parquet(agg, _OUTPUT)
    log.info("Job 2 salida: %s | %d filas | %d tickers.",
             _OUTPUT, len(agg), agg["ticker"].nunique())


def run(smoke: bool = False, limit: int | None = None) -> dict:
    log = setup_logger(JOB)
    ckpt = Checkpoint(JOB)
    log.info("Job 2 inicio (smoke=%s).", smoke)

    n_un = n_kv = 0
    if not ckpt.is_done("unanimad"):
        try:
            n_un = _process_unanimad(log)
            ckpt.mark("unanimad")
        except Exception as exc:  # noqa: BLE001
            log.error("unanimad falló: %s", exc)
    else:
        log.info("unanimad ya procesado (checkpoint).")

    if smoke:
        log.info("smoke: se omite kevin SQL (parseo largo); solo se valida unanimad.")
    elif not ckpt.is_done("kevin"):
        try:
            n_kv = _process_kevin(log)
            ckpt.mark("kevin")
        except Exception as exc:  # noqa: BLE001
            log.error("kevin falló global: %s", exc)
    else:
        log.info("kevin ya procesado (checkpoint).")

    _consolidate(log)
    summary = {"job": JOB, "unanimad_rows": n_un, "kevin_rows": n_kv, "output": str(_OUTPUT)}
    log.info("Job 2 fin: %s", summary)
    return summary


if __name__ == "__main__":
    run(smoke="--smoke" in sys.argv)
