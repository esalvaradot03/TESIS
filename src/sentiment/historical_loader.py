"""
Cargador de datasets históricos de sentimiento.

Acepta dos familias de formato y detecta cuál es por sus columnas:

  1. StockTwits (fuente principal del proyecto). CSV con columnas de mensaje
     (id, body, created_at, symbols, sentiment, likes, ...). Se normaliza al
     esquema StockTwits, idéntico al de scraper_stocktwits.py:

        post_id, timestamp, title, body, score, num_comments,
        detected_tickers, stocktwits_sentiment

     Al traer ya los tickers en `detected_tickers`, este path SALTA
     ticker_detector y va directo a preprocessor.

  2. Reddit / WSB (legacy, para datasets históricos de Kaggle/Pushshift).
     Se normaliza al esquema de scraper_reddit.py:

        post_id, timestamp, title, body, score, num_comments

     Este path SÍ requiere pasar luego por ticker_detector.

Uso desde línea de comandos:
    python -m src.sentiment.historical_loader ruta/dataset.csv
    python -m src.sentiment.historical_loader ruta/dataset.csv --start 2024-01-01 --end 2024-06-30
    python -m src.sentiment.historical_loader  # modo demo con muestra interna
"""

import ast
import json
import logging
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config.settings import RAW_DIR

logger = logging.getLogger(__name__)

_OUTPUT_FILE = RAW_DIR / "wsb_historical.csv"
_STOCKTWITS_OUTPUT_FILE = RAW_DIR / "stocktwits_historical.csv"

# Tickers presentes en el dataset Kaggle frankcaoyun/stocktwits-2020-2022-raw.
# FB es correcto para la época (pre-rebranding a META en jun-2022).
_KAGGLE_TARGET_TICKERS = ["AAPL", "AMZN", "FB", "NVDA", "TSLA"]

# Columnas de salida — mismo esquema que scraper_reddit.py
_OUTPUT_COLUMNS = ["post_id", "timestamp", "title", "body", "score", "num_comments"]

# Columnas de salida StockTwits — mismo esquema que scraper_stocktwits.py
_STOCKTWITS_OUTPUT_COLUMNS = [
    "post_id", "timestamp", "title", "body", "score", "num_comments",
    "detected_tickers", "stocktwits_sentiment",
]

# Mapeos de columnas de entrada para datasets StockTwits → nombre interno
_STOCKTWITS_COL_MAP: dict[str, list[str]] = {
    "post_id":    ["id", "message_id", "post_id"],
    "created_at": ["created_at", "created_utc", "timestamp", "date", "datetime", "time"],
    "body":       ["body", "text", "message", "content"],
    "symbols":    ["symbols", "symbol", "tickers", "ticker", "cashtags"],
    "sentiment":  ["sentiment", "basic", "entity_sentiment", "stocktwits_sentiment"],
    "likes":      ["likes", "likes_total", "like_count", "reshares", "score"],
}

# Mapeos de nombres de columnas de entrada → nombre interno
# Se prueban en orden; se usa el primero que exista en el CSV
_COL_MAP: dict[str, list[str]] = {
    "post_id":      ["id", "post_id", "link_id", "name"],
    "title":        ["title"],
    "body":         ["selftext", "body", "text", "content"],
    "created_utc":  ["created_utc", "timestamp", "created", "date", "datetime"],
    "score":        ["score", "ups", "upvotes", "likes"],
    "num_comments": ["num_comments", "comments", "num_comment", "comment_count"],
    "subreddit":    ["subreddit", "subreddit_name_prefixed"],
}

_WSB_NAMES = {"wallstreetbets", "r/wallstreetbets"}


# ---------------------------------------------------------------------------
# Detección y mapeo de columnas
# ---------------------------------------------------------------------------

def _resolve_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Devuelve el primer nombre de candidato que exista en el DataFrame."""
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _build_column_mapping(df: pd.DataFrame) -> dict[str, str]:
    """
    Construye el mapa {nombre_interno: nombre_real_en_csv} para las columnas
    encontradas. Lanza ValueError si alguna columna obligatoria no existe.
    """
    mapping: dict[str, str] = {}
    required = ["post_id", "title", "created_utc"]

    for internal, candidates in _COL_MAP.items():
        found = _resolve_column(df, candidates)
        if found:
            mapping[internal] = found
        elif internal in required:
            raise ValueError(
                f"Columna requerida '{internal}' no encontrada en el CSV. "
                f"Candidatos buscados: {candidates}. "
                f"Columnas disponibles: {list(df.columns)}"
            )

    return mapping


# ---------------------------------------------------------------------------
# Detección de formato
# ---------------------------------------------------------------------------

def _detect_format(df: pd.DataFrame) -> str:
    """
    Decide si el CSV es un dataset de StockTwits o de Reddit/WSB.

    Heurística: StockTwits trae los tickers en una columna de símbolos
    (`symbols`/`cashtags`/...) o tiene la firma de mensaje (body + created_at
    sin title/selftext). En cualquier otro caso se asume Reddit.

    Returns:
        "stocktwits" o "reddit".
    """
    cols = {c.lower() for c in df.columns}

    symbol_cols = {"symbols", "cashtags", "tickers"}
    if cols & symbol_cols:
        return "stocktwits"

    reddit_signature = {"title", "selftext", "subreddit", "num_comments"}
    has_message_shape = "body" in cols and bool(
        cols & {"created_at", "created_utc", "timestamp", "date"}
    )
    if has_message_shape and not (cols & reddit_signature):
        return "stocktwits"

    return "reddit"


# ---------------------------------------------------------------------------
# Normalización StockTwits
# ---------------------------------------------------------------------------

# Tras strip de '$', un ticker válido empieza por letra y sigue con letras/dígitos/-
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9-]*$")


def _clean_symbols(items: list) -> list[str]:
    """Normaliza una lista de símbolos crudos a tickers en mayúsculas (BRK.B → BRK-B)."""
    out: list[str] = []
    for it in items:
        tok = str(it).strip().lstrip("$").upper().replace(".", "-")
        if tok and _TICKER_RE.match(tok):
            out.append(tok)
    return sorted(set(out))


def _parse_symbols_cell(cell: object) -> list[str]:
    """
    Extrae la lista de tickers de una celda de la columna de símbolos.

    Acepta varios formatos de serialización habituales en dumps de StockTwits:
      - JSON de objetos:  '[{"symbol": "AAPL"}, {"symbol": "NVDA"}]'
      - JSON de strings:  '["AAPL", "NVDA"]'
      - Texto separado:   'AAPL, NVDA' / '$AAPL $NVDA' / 'AAPL;NVDA'
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    s = str(cell).strip()
    if not s or s.lower() == "nan":
        return []

    # Intento 1: JSON
    try:
        val = json.loads(s)
        if isinstance(val, list):
            raw: list = []
            for item in val:
                if isinstance(item, dict):
                    sym = item.get("symbol") or item.get("ticker")
                    if sym:
                        raw.append(sym)
                else:
                    raw.append(item)
            return _clean_symbols(raw)
        if isinstance(val, dict):
            sym = val.get("symbol") or val.get("ticker")
            return _clean_symbols([sym]) if sym else []
    except (ValueError, TypeError):
        pass

    # Intento 2: texto separado por comas/espacios/punto y coma
    parts = re.split(r"[,;\s]+", s)
    return _clean_symbols(parts)


def _normalize_native_sentiment(value: object) -> str:
    """Normaliza el label nativo a 'Bullish' / 'Bearish' / 'None'."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "None"
    v = str(value).strip().lower()
    if v in ("bullish", "bull", "1", "+1", "positive"):
        return "Bullish"
    if v in ("bearish", "bear", "-1", "negative"):
        return "Bearish"
    return "None"


def _normalize_stocktwits(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """
    Renombra y normaliza un chunk de dataset StockTwits al esquema interno.

    Args:
        df: DataFrame crudo del dataset StockTwits.
        mapping: Mapa {nombre_interno: nombre_en_csv}.

    Returns:
        DataFrame con columnas _STOCKTWITS_OUTPUT_COLUMNS.
    """
    out = pd.DataFrame()

    out["post_id"] = df[mapping["post_id"]].astype(str).str.strip()
    out["timestamp"] = _parse_timestamp(df[mapping["created_at"]])
    out["title"] = df[mapping["body"]].astype(str).fillna("").str.strip()
    out["body"] = ""

    out["score"] = (
        pd.to_numeric(df[mapping["likes"]], errors="coerce").fillna(0).astype(int)
        if "likes" in mapping else 0
    )
    out["num_comments"] = 0

    out["detected_tickers"] = [
        json.dumps(_parse_symbols_cell(cell)) for cell in df[mapping["symbols"]]
    ]

    if "sentiment" in mapping:
        out["stocktwits_sentiment"] = df[mapping["sentiment"]].map(_normalize_native_sentiment)
    else:
        out["stocktwits_sentiment"] = "None"

    # Descartar mensajes sin ticker (no hay a quién asignar el sentimiento)
    out = out[out["detected_tickers"] != "[]"]
    return out


def _build_stocktwits_mapping(df: pd.DataFrame) -> dict[str, str]:
    """Construye el mapa de columnas para un dataset StockTwits. Requiere id, created_at, body, symbols."""
    mapping: dict[str, str] = {}
    required = ["post_id", "created_at", "body", "symbols"]
    for internal, candidates in _STOCKTWITS_COL_MAP.items():
        found = _resolve_column(df, candidates)
        if found:
            mapping[internal] = found
        elif internal in required:
            raise ValueError(
                f"Columna requerida '{internal}' no encontrada en el CSV StockTwits. "
                f"Candidatos buscados: {candidates}. "
                f"Columnas disponibles: {list(df.columns)}"
            )
    return mapping


# ---------------------------------------------------------------------------
# Normalización Reddit
# ---------------------------------------------------------------------------

def _parse_timestamp(series: pd.Series) -> pd.Series:
    """
    Convierte la columna de tiempo al formato ISO 8601 UTC.

    Acepta:
      - Unix timestamp en segundos (entero o float)
      - Strings en cualquier formato que pandas pueda inferir
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")

    parsed = pd.to_datetime(series, utc=True, infer_datetime_format=True, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _normalize(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """
    Renombra columnas al esquema interno y normaliza tipos.

    Args:
        df: DataFrame crudo con los nombres originales del CSV.
        mapping: Mapa {nombre_interno: nombre_en_csv}.

    Returns:
        DataFrame con columnas _OUTPUT_COLUMNS, tipos limpios.
    """
    out = pd.DataFrame()

    out["post_id"] = df[mapping["post_id"]].astype(str).str.strip()

    out["timestamp"] = _parse_timestamp(df[mapping["created_utc"]])

    out["title"] = df[mapping["title"]].astype(str).fillna("").str.strip()

    if "body" in mapping:
        body = df[mapping["body"]].astype(str).fillna("")
        # Pushshift almacena posts eliminados como "[removed]" o "[deleted]"
        out["body"] = body.where(~body.isin(["[removed]", "[deleted]", "nan"]), "")
    else:
        out["body"] = ""

    out["score"] = (
        pd.to_numeric(df[mapping["score"]], errors="coerce").fillna(0).astype(int)
        if "score" in mapping else 0
    )

    out["num_comments"] = (
        pd.to_numeric(df[mapping["num_comments"]], errors="coerce").fillna(0).astype(int)
        if "num_comments" in mapping else 0
    )

    return out


# ---------------------------------------------------------------------------
# Filtros
# ---------------------------------------------------------------------------

def _filter_subreddit(df_raw: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """
    Mantiene solo posts de r/WallStreetBets si la columna subreddit existe.
    Si no existe, asume que todo el CSV es de WSB (muchos datasets ya vienen filtrados).
    """
    if "subreddit" not in mapping:
        logger.info("Columna 'subreddit' no encontrada — asumiendo que el CSV es solo WSB.")
        return df_raw

    col = mapping["subreddit"]
    mask = df_raw[col].str.lower().str.strip().isin(_WSB_NAMES)
    filtered = df_raw[mask]
    dropped = len(df_raw) - len(filtered)
    if dropped:
        logger.info("Eliminados %d posts de otros subreddits.", dropped)
    return filtered


def _filter_date_range(
    df: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    """
    Filtra posts cuyo timestamp caiga fuera del rango [start_date, end_date].
    Las fechas se interpretan como inicio de día UTC.
    """
    if not start_date and not end_date:
        return df

    ts = pd.to_datetime(df["timestamp"], utc=True)

    if start_date:
        start_dt = datetime.fromisoformat(start_date).replace(tzinfo=timezone.utc)
        df = df[ts >= start_dt]

    if end_date:
        end_dt = datetime.fromisoformat(end_date).replace(
            hour=23, minute=59, second=59, tzinfo=timezone.utc
        )
        df = df[ts <= end_dt]

    return df


# ---------------------------------------------------------------------------
# Punto de entrada principal
# ---------------------------------------------------------------------------

def load_historical(
    input_path: Path,
    start_date: str | None = None,
    end_date: str | None = None,
    output_path: Path | None = None,
    chunksize: int = 100_000,
) -> Path:
    """
    Carga un dataset histórico de sentimiento, lo normaliza y lo guarda en disco.

    Detecta automáticamente si el CSV es StockTwits o Reddit/WSB y aplica la
    normalización correspondiente. Procesa el CSV en chunks para manejar
    archivos grandes (los dumps pueden tener millones de filas) sin agotar memoria.

    Args:
        input_path: Ruta al CSV de entrada (StockTwits o Kaggle/Pushshift).
        start_date: Filtro de fecha inicio 'YYYY-MM-DD' (opcional).
        end_date: Filtro de fecha fin 'YYYY-MM-DD' (opcional).
        output_path: Destino del CSV normalizado. Si None, se elige según el
                     formato detectado (stocktwits_historical.csv / wsb_historical.csv).
        chunksize: Filas por chunk al leer el CSV.

    Returns:
        Ruta al CSV de salida normalizado.
    """
    logger.info("Cargando dataset histórico desde %s...", input_path)

    # Leer primera muestra para detectar formato y columnas
    sample = pd.read_csv(input_path, nrows=5)
    fmt = _detect_format(sample)
    logger.info("Formato detectado: %s", fmt)

    if fmt == "stocktwits":
        mapping = _build_stocktwits_mapping(sample)
        out_cols = _STOCKTWITS_OUTPUT_COLUMNS
        if output_path is None:
            output_path = _STOCKTWITS_OUTPUT_FILE
    else:
        mapping = _build_column_mapping(sample)
        out_cols = _OUTPUT_COLUMNS
        if output_path is None:
            output_path = _OUTPUT_FILE

    logger.info("Mapeo de columnas detectado: %s", dict(mapping))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_written = 0
    write_header = True

    for chunk_idx, chunk in enumerate(
        pd.read_csv(input_path, chunksize=chunksize, low_memory=False)
    ):
        if fmt == "stocktwits":
            normalized = _normalize_stocktwits(chunk, mapping)
        else:
            chunk_filtered = _filter_subreddit(chunk, mapping)
            if chunk_filtered.empty:
                continue
            normalized = _normalize(chunk_filtered, mapping)

        normalized = normalized.dropna(subset=["timestamp"])
        normalized = _filter_date_range(normalized, start_date, end_date)

        if normalized.empty:
            continue

        normalized[out_cols].to_csv(
            output_path, mode="a" if not write_header else "w",
            header=write_header, index=False, encoding="utf-8",
        )
        write_header = False
        total_written += len(normalized)

        if (chunk_idx + 1) % 10 == 0:
            logger.info("Chunks procesados: %d | Filas escritas: %d", chunk_idx + 1, total_written)

    if total_written == 0:
        logger.warning(
            "No se escribieron filas. Verificá que el CSV tenga mensajes válidos "
            "en el rango de fechas indicado."
        )
    else:
        logger.info(
            "Dataset histórico normalizado (%s): %d filas → %s",
            fmt, total_written, output_path,
        )

    return output_path


# ---------------------------------------------------------------------------
# Dataset Kaggle frankcaoyun/stocktwits-2020-2022-raw (directorio de 362 CSVs)
# ---------------------------------------------------------------------------
# Particularidades de este dataset:
#   - 5 carpetas con nombres INCONSISTENTES (ej. "AMZN2019-2022" sin guion) →
#     el ticker NO se infiere de la carpeta, sino del campo `symbols`.
#   - Columnas `symbols` y `entities` son LITERALES de Python (comillas simples),
#     NO JSON → se parsean con ast.literal_eval (seguro), nunca con eval/json.loads.
#   - `likes` viene como dict {'total': N, ...} o NaN.
#   - El sentimiento nativo está anidado en entities['sentiment'] y suele/puede
#     ser None; cuando existe es {'basic': 'Bullish'|'Bearish'}.

def _safe_literal_eval(cell: object):
    """
    Evalúa de forma segura un literal de Python contenido en una celda.

    Devuelve None si la celda está vacía, es NaN o no es un literal válido.
    Nunca usa eval(): solo ast.literal_eval, que no ejecuta código arbitrario.
    """
    if not isinstance(cell, str):
        return None
    s = cell.strip()
    if not s or s.lower() in ("nan", "none", "null"):
        return None
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None


def _kaggle_extract_symbols(cell: object, allowed: set[str]) -> list[str]:
    """
    Extrae los tickers de la columna `symbols`, filtrados al conjunto permitido.

    `symbols` es una lista de dicts como [{'id': 686, 'symbol': 'AAPL', ...}].
    Se toma el campo 'symbol' de cada elemento (NO se infiere de la carpeta).
    """
    parsed = _safe_literal_eval(cell)
    if not isinstance(parsed, list):
        return []
    out: list[str] = []
    for item in parsed:
        if isinstance(item, dict):
            sym = item.get("symbol")
            if sym:
                tok = str(sym).strip().upper().replace(".", "-")
                if tok in allowed:
                    out.append(tok)
    return sorted(set(out))


def _kaggle_extract_native(cell: object) -> str:
    """
    Extrae el label nativo de entities['sentiment'] de forma segura.

    Suele ser None; cuando existe es {'basic': 'Bullish'|'Bearish'}.
    Devuelve 'Bullish', 'Bearish' o 'None' (no asume que siempre venga).
    """
    parsed = _safe_literal_eval(cell)
    if isinstance(parsed, dict):
        sentiment = parsed.get("sentiment")
        if isinstance(sentiment, dict):
            basic = sentiment.get("basic")
            if basic in ("Bullish", "Bearish"):
                return basic
    return "None"


def _kaggle_extract_likes(cell: object) -> int:
    """Extrae likes['total'] de la columna `likes` (dict o NaN)."""
    parsed = _safe_literal_eval(cell)
    if isinstance(parsed, dict):
        try:
            return int(parsed.get("total", 0) or 0)
        except (TypeError, ValueError):
            return 0
    return 0


def load_kaggle_stocktwits_dataset(
    root_dir: Path,
    start_date: str = "2020-01-01",
    end_date: str = "2022-03-31",
    tickers: list[str] | None = None,
    max_per_ticker_day: int = 40,
    output_path: Path = _STOCKTWITS_OUTPUT_FILE,
    read_chunksize: int = 100_000,
) -> tuple[Path, dict]:
    """
    Normaliza el dataset Kaggle frankcaoyun/stocktwits-2020-2022-raw al esquema interno.

    Recorre los 362 CSVs (os.walk), parsea `symbols`/`entities`/`likes` con
    ast.literal_eval, filtra a los tickers objetivo y al rango de fechas, y
    aplica un cap por (ticker, día) para acotar el costo posterior de FinBERT
    sin perder la señal diaria (>> MIN_MENTIONS_PER_DAY).

    Cada mensaje conserva TODOS sus tickers objetivo en `detected_tickers`
    (igual que el path en vivo); el explode por ticker lo hace finbert_scorer.

    Args:
        root_dir: Carpeta raíz del dataset descargado por kagglehub.
        start_date / end_date: Rango de fechas inclusivo 'YYYY-MM-DD'.
        tickers: Tickers objetivo. Si None, usa _KAGGLE_TARGET_TICKERS.
        max_per_ticker_day: Cap de mensajes muestreados por (ticker, día).
        output_path: CSV de salida normalizado.
        read_chunksize: Filas por chunk al leer cada CSV.

    Returns:
        (ruta_salida, stats) donde stats incluye los conteos para reportar la
        trampa #1 (% de mensajes con label nativo no-nulo) sobre TODO el dataset.
    """
    allowed = set(tickers) if tickers else set(_KAGGLE_TARGET_TICKERS)
    start_ts = pd.Timestamp(start_date + "T00:00:00", tz="UTC")
    end_ts = pd.Timestamp(end_date + "T23:59:59", tz="UTC")

    # Descubrir todos los CSVs bajo root_dir
    csv_files: list[str] = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for name in sorted(filenames):
            if name.lower().endswith(".csv"):
                csv_files.append(os.path.join(dirpath, name))
    csv_files.sort()
    logger.info("Dataset Kaggle: %d archivos CSV bajo %s", len(csv_files), root_dir)

    # Estado de muestreo y estadísticas (sobre TODO el dataset, antes del cap)
    bucket: Counter = Counter()           # (ticker, date) -> nº muestreado
    seen_ids: set[str] = set()            # dedup por post_id (mensajes repetidos entre carpetas)
    kept_rows: list[dict] = []
    total_target_msgs = 0                 # mensajes en rango con ≥1 ticker objetivo
    total_native_nonnull = 0             # de esos, cuántos traen label nativo
    native_label_counts: Counter = Counter()
    usecols = ["id", "body", "created_at", "symbols", "entities", "likes"]

    for fi, path in enumerate(csv_files, 1):
        try:
            reader = pd.read_csv(
                path, usecols=lambda c: c in usecols,
                chunksize=read_chunksize, low_memory=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("No se pudo leer %s: %s. Se omite.", path, exc)
            continue

        for chunk in reader:
            # 1) Filtro de fecha vectorizado (recorta antes de literal_eval costoso)
            ts = pd.to_datetime(chunk["created_at"], utc=True, errors="coerce")
            in_range = ts.notna() & (ts >= start_ts) & (ts <= end_ts)
            chunk = chunk[in_range]
            ts = ts[in_range]
            if chunk.empty:
                continue

            # 2) Parseo por fila de symbols / entities / likes
            syms_series = chunk["symbols"].map(lambda c: _kaggle_extract_symbols(c, allowed))
            has_target = syms_series.map(len) > 0
            if not has_target.any():
                continue

            chunk = chunk[has_target]
            ts = ts[has_target]
            syms_series = syms_series[has_target]
            natives = chunk["entities"].map(_kaggle_extract_native)
            likes = chunk["likes"].map(_kaggle_extract_likes)
            ids = chunk["id"].astype(str)
            bodies = chunk["body"].astype(str)
            iso = ts.dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")
            dates = ts.dt.date

            # 3) Estadísticas sobre TODO el dataset (antes de cap/dedup)
            total_target_msgs += len(chunk)
            nn = (natives != "None")
            total_native_nonnull += int(nn.sum())
            native_label_counts.update(natives.tolist())

            # 4) Muestreo con cap por (ticker, día) + dedup por post_id
            for pid, body, ts_iso, d, msg_syms, native, like in zip(
                ids, bodies, iso, dates, syms_series, natives, likes
            ):
                if pid in seen_ids:
                    continue
                if not any(bucket[(t, d)] < max_per_ticker_day for t in msg_syms):
                    continue
                seen_ids.add(pid)
                for t in msg_syms:
                    bucket[(t, d)] += 1
                kept_rows.append({
                    "post_id": pid,
                    "timestamp": ts_iso,
                    "title": body.strip(),
                    "body": "",
                    "score": int(like),
                    "num_comments": 0,
                    "detected_tickers": json.dumps(list(msg_syms)),
                    "stocktwits_sentiment": native,
                })

        if fi % 25 == 0:
            logger.info(
                "Progreso: %d/%d archivos | %d msgs objetivo | %d muestreados.",
                fi, len(csv_files), total_target_msgs, len(kept_rows),
            )

    # --- Persistencia ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = pd.DataFrame(kept_rows, columns=_STOCKTWITS_OUTPUT_COLUMNS)
    out_df.to_csv(output_path, index=False, encoding="utf-8")

    native_pct = (
        100.0 * total_native_nonnull / total_target_msgs if total_target_msgs else 0.0
    )
    stats = {
        "files": len(csv_files),
        "total_target_msgs": total_target_msgs,
        "native_nonnull": total_native_nonnull,
        "native_nonnull_pct": round(native_pct, 2),
        "native_label_counts": dict(native_label_counts),
        "sampled_rows": len(out_df),
        "max_per_ticker_day": max_per_ticker_day,
        "ticker_distribution": _ticker_distribution(out_df),
        "date_range": (start_date, end_date),
    }
    logger.info(
        "Kaggle StockTwits normalizado: %d msgs objetivo en rango, "
        "%.2f%% con label nativo. Muestreados (cap=%d/día): %d filas → %s",
        total_target_msgs, native_pct, max_per_ticker_day, len(out_df), output_path,
    )
    return output_path, stats


def _ticker_distribution(df: pd.DataFrame) -> dict[str, int]:
    """Cuenta mensajes muestreados por ticker objetivo (un mensaje puede sumar a varios)."""
    counts: Counter = Counter()
    for cell in df["detected_tickers"]:
        try:
            for t in json.loads(cell):
                counts[t] += 1
        except (ValueError, TypeError):
            continue
    return dict(counts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Normaliza un dataset histórico (StockTwits o WSB) al esquema interno."
    )
    parser.add_argument(
        "input", nargs="?", default=None,
        help="Ruta al CSV de entrada (StockTwits o Kaggle/Pushshift). Sin argumento: modo demo.",
    )
    parser.add_argument("--start", default=None, help="Fecha inicio YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="Fecha fin YYYY-MM-DD")
    parser.add_argument(
        "--output", default=None,
        help="Ruta CSV de salida. Por defecto se elige según el formato detectado.",
    )
    parser.add_argument(
        "--max-per-ticker-day", type=int, default=40,
        help="(Solo dataset Kaggle / directorio) cap de mensajes por (ticker, día).",
    )

    args = parser.parse_args()

    # Si el input es un DIRECTORIO → dataset Kaggle frankcaoyun/stocktwits-2020-2022-raw
    if args.input is not None and Path(args.input).is_dir():
        out_path, stats = load_kaggle_stocktwits_dataset(
            root_dir=Path(args.input),
            start_date=args.start or "2020-01-01",
            end_date=args.end or "2022-03-31",
            max_per_ticker_day=args.max_per_ticker_day,
            output_path=Path(args.output) if args.output else _STOCKTWITS_OUTPUT_FILE,
        )
        print("\n=== Stats dataset Kaggle ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")
        result = pd.read_csv(out_path)
        print(f"\nNormalizado: {len(result)} filas → {out_path}")
        print(result.head(5).to_string(index=False))
        import sys as _sys
        _sys.exit(0)

    if args.input is None:
        # Modo demo: muestra ambos formatos soportados → esquema interno
        import io

        st_csv = """id,body,created_at,symbols,sentiment,likes
600001,$AAPL breaking out calls printing,2024-06-20T14:32:10Z,"[{""symbol"": ""AAPL""}]",Bullish,12
600002,$TSLA overextended taking profits,2024-06-20T15:01:44Z,"[{""symbol"": ""TSLA""}]",Bearish,3
600003,Rotating from $AAPL into $NVDA,2024-06-20T16:05:33Z,"AAPL, NVDA",Bullish,27
600004,$GME no position just watching,2024-06-20T17:10:00Z,$GME,,1
"""
        st_df = pd.read_csv(io.StringIO(st_csv))
        print("Formato detectado en demo StockTwits:", _detect_format(st_df))
        st_map = _build_stocktwits_mapping(st_df)
        st_norm = _normalize_stocktwits(st_df, st_map)
        print("\nDemo StockTwits (mensaje -> esquema interno):\n")
        print(st_norm[_STOCKTWITS_OUTPUT_COLUMNS].to_string(index=False))
        print(f"\nColumnas de salida StockTwits: {_STOCKTWITS_OUTPUT_COLUMNS}")

        demo_csv = """id,title,selftext,created_utc,score,num_comments,subreddit
abc1234,TSLA to the moon - loaded on calls,Not financial advice. RSI oversold.,1704153600,4521,312,wallstreetbets
def5678,Why NVDA will hit $1000 EOY - my DD,Fundamentals are insane. Revenue up 30% YoY.,1704240000,8932,891,wallstreetbets
ghi9012,Lost everything on GME puts,Market is completely rigged. Down 80% today.,1704326400,2341,445,wallstreetbets
stu2222,Short TSLA thesis,Overvalued by 40%. Management risk.,1704672000,987,87,learnprogramming
"""
        demo_df = pd.read_csv(io.StringIO(demo_csv))
        print("\nFormato detectado en demo Reddit:", _detect_format(demo_df))
        mapping = _build_column_mapping(demo_df)
        filtered = _filter_subreddit(demo_df, mapping)
        normalized = _normalize(filtered, mapping)

        print("\nDemo Reddit/WSB (Pushshift -> esquema interno):\n")
        print(normalized[_OUTPUT_COLUMNS].to_string(index=False))
        print(f"\nColumnas de salida Reddit: {_OUTPUT_COLUMNS}")
        print("\nPara usar con un CSV real:")
        print("  python -m src.sentiment.historical_loader mi_dataset.csv --start 2024-01-01 --end 2024-06-30")
    else:
        out = load_historical(
            input_path=Path(args.input),
            start_date=args.start,
            end_date=args.end,
            output_path=Path(args.output) if args.output else None,
        )
        # Mostrar primeras filas del resultado
        result = pd.read_csv(out)
        print(f"\nNormalizado: {len(result)} posts → {out}")
        print(result.head(5).to_string(index=False))
