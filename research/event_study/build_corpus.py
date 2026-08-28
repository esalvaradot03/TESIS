"""
Genera lotes de muestreo de posts de StockTwits para etiquetado manual, con
la columna target_ticker ya asignada — la unidad de anotación del corpus del
event study es el par (post, target_ticker), no el post.

Dos fuentes, según cobertura de cada uno de los 6 tickers del event study:

  - **NYU** (`D:\\trading-data\\stocktwits_nyu\\messages\\`): texto crudo
    histórico. Confirmado con cobertura para DIS y CMG (ver
    docs/external_data.md y process_stocktwits_nyu.py, que ya procesa el
    bucket hermano `symbol_sentiments/` — sin texto). Este bucket `messages/`
    nunca se leyó antes en este repo, así que el schema se detecta en
    tiempo de ejecución probando nombres de columna candidatos conocidos
    del dataset StockTwits (ver _detect_nyu_schema) en vez de asumirlo.

  - **StockTwits en vivo** (`src.sentiment.scraper_stocktwits`): cubre los
    6 tickers, pero el endpoint público solo expone el stream reciente
    (últimos días/semanas), no búsqueda histórica. Para NCLH, CRWD, TGT y
    DDOG —sin cobertura NYU confirmada— esto da una muestra de "ahora", NO
    un sample estratificado por año como pide el plan de muestreo LR. Ver
    el warning que emite _load_live_messages.

Uso:
    python -m research.event_study.build_corpus --batch l0 --n 100 --min-per-ticker 10
"""

import ast
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import DATA_DIR, SEED, STOCKTWITS_NYU_MESSAGES
from src.sentiment.preprocessor import build_input_text, clean_text

logger = logging.getLogger(__name__)

# Los 6 tickers del event study (decididos, ver docs del pivote — no cambiar).
TICKERS: list[str] = ["NCLH", "DIS", "CRWD", "TGT", "CMG", "DDOG"]

# Tickers con cobertura histórica confirmada en el bucket NYU. Los otros 4
# solo tienen la fuente en vivo (ver docstring del módulo).
NYU_COVERED_TICKERS: set[str] = {"DIS", "CMG"}

_MANUAL_LABELS_DIR = DATA_DIR / "manual_labels"

# Columnas del pool combinado (formato de trabajo interno, con procedencia).
_POOL_COLUMNS: list[str] = [
    "post_id", "target_ticker", "tickers_detectados", "timestamp", "clean_text", "source",
]

# Columnas del CSV de anotación (una fila por par post/target_ticker).
_ANNOTATION_COLUMNS: list[str] = [
    "post_id", "target_ticker", "tickers_detectados", "clean_text",
    "label", "confianza", "base", "nota",
]

# Nombres de columna candidatos para el bucket NYU messages/, en orden de
# probabilidad (mismo dataset que symbol_sentiments/, ver process_stocktwits_nyu.py:
# message_id, user_id, created_at, sentiment, symbol_list — 'body' es el nombre
# nativo del campo de texto en la API de StockTwits).
_TEXT_COLUMN_CANDIDATES: list[str] = ["body", "text", "message", "message_body"]
_TIMESTAMP_COLUMN_CANDIDATES: list[str] = ["created_at", "timestamp", "date"]
_SYMBOL_COLUMN_CANDIDATES: list[str] = ["symbol_list", "symbols"]
_ID_COLUMN_CANDIDATES: list[str] = ["message_id", "id"]


# ---------------------------------------------------------------------------
# Fuente NYU (histórica, solo DIS/CMG confirmados)
# ---------------------------------------------------------------------------

def _parse_symbol_list(cell: object) -> list[str]:
    """
    Parsea la columna de símbolos del dataset NYU (literal de Python, e.g.
    "['ZNGA','META']"), mismo formato que _parse_symbol_list en
    process_stocktwits_nyu.py. Normaliza a mayúsculas y '.' → '-'.
    """
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
        return sorted({str(x).strip().upper().replace(".", "-") for x in val if x})
    return []


def _require_column(columns: list[str], candidates: list[str], role: str, path: Path) -> str:
    """Devuelve la primera columna candidata presente, o falla con las columnas reales."""
    for c in candidates:
        if c in columns:
            return c
    raise ValueError(
        f"No se encontró columna de {role} en {path}. Candidatas probadas: {candidates}. "
        f"Columnas reales: {columns}. Ajustá las constantes *_COLUMN_CANDIDATES en "
        "build_corpus.py si el dataset NYU usa otro nombre de columna."
    )


def _detect_nyu_schema(sample_path: Path) -> dict[str, str | None]:
    """
    Detecta las columnas de id/timestamp/símbolos/texto del bucket NYU
    messages/ probando nombres candidatos, en vez de asumir un nombre fijo
    (este bucket nunca se leyó antes en el repo). 'id' es opcional (se
    sintetiza un post_id si no aparece); timestamp/símbolos/texto son
    obligatorias y la función falla con las columnas reales si no matchean.
    """
    columns = list(pd.read_csv(sample_path, nrows=1).columns)
    return {
        "id": next((c for c in _ID_COLUMN_CANDIDATES if c in columns), None),
        "timestamp": _require_column(columns, _TIMESTAMP_COLUMN_CANDIDATES, "timestamp", sample_path),
        "symbols": _require_column(columns, _SYMBOL_COLUMN_CANDIDATES, "símbolos", sample_path),
        "text": _require_column(columns, _TEXT_COLUMN_CANDIDATES, "texto", sample_path),
    }


def _read_nyu_file(path: Path, schema: dict[str, str | None], tickers: set[str]) -> pd.DataFrame:
    """Lee un CSV NYU, filtra a `tickers` y explota a una fila por (post, target_ticker)."""
    usecols = [c for c in {schema["id"], schema["timestamp"], schema["symbols"], schema["text"]} if c]
    df = pd.read_csv(path, usecols=usecols, dtype=str, encoding="utf-8", encoding_errors="replace")

    df["_symbols"] = df[schema["symbols"]].map(_parse_symbol_list)
    df = df[df["_symbols"].map(len) > 0]
    if df.empty:
        return pd.DataFrame(columns=_POOL_COLUMNS)

    df["_matched"] = df["_symbols"].map(lambda syms: [s for s in syms if s in tickers])
    df = df[df["_matched"].map(len) > 0]
    if df.empty:
        return pd.DataFrame(columns=_POOL_COLUMNS)

    post_id = df[schema["id"]].astype(str) if schema["id"] else pd.Series(
        [f"{path.stem}_{i}" for i in df.index], index=df.index
    )

    rows: list[dict] = []
    for idx, row in df.iterrows():
        clean = clean_text(build_input_text(str(row[schema["text"]]), ""))
        symbols_json = json.dumps(row["_symbols"])
        for ticker in row["_matched"]:
            rows.append({
                "post_id": post_id.loc[idx],
                "target_ticker": ticker,
                "tickers_detectados": symbols_json,
                "timestamp": row[schema["timestamp"]],
                "clean_text": clean,
                "source": "nyu",
            })
    return pd.DataFrame(rows, columns=_POOL_COLUMNS)


def _load_nyu_messages(
    tickers: set[str],
    messages_dir: Path = STOCKTWITS_NYU_MESSAGES,
) -> pd.DataFrame:
    """
    Carga y filtra el bucket NYU messages/ a los tickers pedidos.

    Devuelve un DataFrame vacío (con warning) si el directorio no existe
    (típicamente porque D:\\trading-data\\ no está montado en esta sesión)
    o si no hay archivos .csv. Un archivo individual corrupto no aborta el
    resto (se loguea el error y se continúa con los demás).
    """
    if not tickers:
        return pd.DataFrame(columns=_POOL_COLUMNS)
    if not messages_dir.exists():
        logger.warning(
            "No existe %s (¿está montado D:\\trading-data\\?). Se omite la fuente NYU.",
            messages_dir,
        )
        return pd.DataFrame(columns=_POOL_COLUMNS)

    files = sorted(messages_dir.glob("*.csv"))
    if not files:
        logger.warning("Sin archivos .csv en %s. Se omite la fuente NYU.", messages_dir)
        return pd.DataFrame(columns=_POOL_COLUMNS)

    schema = _detect_nyu_schema(files[0])
    logger.info("Schema NYU detectado en %s: %s", files[0].name, schema)

    frames: list[pd.DataFrame] = []
    for path in files:
        try:
            frame = _read_nyu_file(path, schema, tickers)
        except Exception as exc:  # noqa: BLE001 — un archivo corrupto no mata el job
            logger.error("Archivo NYU %s falló (%s); se omite.", path, exc)
            continue
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=_POOL_COLUMNS)
    result = pd.concat(frames, ignore_index=True)
    logger.info("Fuente NYU: %d pares (post, ticker) sobre %d archivo(s).", len(result), len(files))
    return result


# ---------------------------------------------------------------------------
# Fuente StockTwits en vivo (recientes, los 6 tickers)
# ---------------------------------------------------------------------------

def _load_live_messages(tickers: list[str], max_per_symbol: int) -> pd.DataFrame:
    """
    Trae mensajes recientes vía src.sentiment.scraper_stocktwits.scrape().

    ADVERTENCIA metodológica: el endpoint público de StockTwits solo expone
    el stream reciente (últimos días/semanas), no búsqueda histórica. Para
    los tickers sin cobertura NYU (NCLH, CRWD, TGT, DDOG) esta fuente da una
    muestra de "ahora mismo", no un sample estratificado por año — no cumple
    por sí sola el plan de muestreo LR para esos 4 tickers.
    """
    logger.warning(
        "Fuente en vivo: muestra del stream RECIENTE de StockTwits, no histórica. "
        "Para %s (sin cobertura NYU) esto NO es un sample estratificado por año.",
        sorted(set(tickers) - NYU_COVERED_TICKERS),
    )
    try:
        from src.sentiment.scraper_stocktwits import scrape
    except ImportError as exc:
        logger.warning("No se pudo importar el scraper de StockTwits (%s). Se omite la fuente en vivo.", exc)
        return pd.DataFrame(columns=_POOL_COLUMNS)

    try:
        raw_path = scrape(symbols=tickers, max_messages_per_symbol=max_per_symbol)
    except Exception as exc:  # noqa: BLE001 — la ausencia de red no debe abortar el build
        logger.warning("Scraping en vivo falló (%s). Se omite la fuente en vivo.", exc)
        return pd.DataFrame(columns=_POOL_COLUMNS)

    raw = pd.read_csv(raw_path, dtype=str).fillna("")
    universe = set(tickers)

    rows: list[dict] = []
    for _, r in raw.iterrows():
        detected = json.loads(r["detected_tickers"]) if r["detected_tickers"] else []
        matched = [t for t in detected if t in universe]
        if not matched:
            continue
        clean = clean_text(build_input_text(r["title"], r["body"]))
        for ticker in matched:
            rows.append({
                "post_id": r["post_id"],
                "target_ticker": ticker,
                "tickers_detectados": r["detected_tickers"],
                "timestamp": r["timestamp"],
                "clean_text": clean,
                "source": "live",
            })

    logger.info("Fuente en vivo: %d pares (post, ticker) sobre %d tickers pedidos.", len(rows), len(tickers))
    return pd.DataFrame(rows, columns=_POOL_COLUMNS)


# ---------------------------------------------------------------------------
# Pool combinado y muestreo
# ---------------------------------------------------------------------------

def build_pool(
    tickers: list[str] = TICKERS,
    nyu_tickers: set[str] = NYU_COVERED_TICKERS,
    max_live_per_symbol: int = 60,
) -> pd.DataFrame:
    """
    Combina la fuente NYU (tickers con cobertura confirmada) y la fuente en
    vivo (los 6 tickers) en un solo pool, deduplicado por (post_id, target_ticker).

    Args:
        tickers: Universo de tickers del event study.
        nyu_tickers: Subconjunto de `tickers` con cobertura NYU confirmada.
        max_live_per_symbol: Tope de mensajes por símbolo para el scraping en vivo.

    Returns:
        DataFrame con columnas _POOL_COLUMNS. Vacío si ninguna fuente devolvió datos.
    """
    frames: list[pd.DataFrame] = []

    nyu_df = _load_nyu_messages(nyu_tickers & set(tickers))
    if not nyu_df.empty:
        frames.append(nyu_df)

    live_df = _load_live_messages(tickers, max_live_per_symbol)
    if not live_df.empty:
        frames.append(live_df)

    if not frames:
        logger.warning("Pool vacío: sin datos de NYU ni de StockTwits en vivo.")
        return pd.DataFrame(columns=_POOL_COLUMNS)

    pool = pd.concat(frames, ignore_index=True)
    pool = pool.drop_duplicates(subset=["post_id", "target_ticker"], keep="first").reset_index(drop=True)

    logger.info(
        "Pool combinado: %d pares (post, ticker) | por fuente: %s | tickers: %s",
        len(pool),
        pool["source"].value_counts().to_dict(),
        sorted(pool["target_ticker"].unique()),
    )
    return pool


def sample_random(
    pool: pd.DataFrame,
    n_total: int = 100,
    min_per_ticker: int = 10,
    tickers: list[str] = TICKERS,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Sample aleatorio con mínimo garantizado por ticker (usado para el lote L0
    de calibración; reusable para LR con estratificación adicional por año).

    Primero toma hasta `min_per_ticker` filas al azar de cada ticker; luego
    completa hasta `n_total` con filas aleatorias del resto del pool. Si un
    ticker no tiene suficientes filas disponibles, se loguea un warning y se
    toman todas las que haya.

    Args:
        pool: DataFrame de build_pool(), sin duplicados de (post_id, target_ticker).
        n_total: Tamaño total del lote.
        min_per_ticker: Mínimo de filas por ticker a garantizar.
        tickers: Universo de tickers sobre el que aplicar el mínimo.
        seed: Semilla aleatoria (reproducibilidad — SEED=42 del proyecto).

    Returns:
        DataFrame con las filas seleccionadas, orden aleatorizado.
    """
    rng = np.random.default_rng(seed)
    pool = pool.drop_duplicates(subset=["post_id", "target_ticker"]).reset_index(drop=True)

    selected_idx: list[int] = []
    for ticker in tickers:
        candidates = pool.index[pool["target_ticker"] == ticker].to_numpy()
        if len(candidates) < min_per_ticker:
            logger.warning(
                "%s: solo %d posts disponibles en el pool (mínimo pedido %d). Se toman todos.",
                ticker, len(candidates), min_per_ticker,
            )
        take = min(min_per_ticker, len(candidates))
        if take:
            selected_idx.extend(rng.choice(candidates, size=take, replace=False).tolist())

    remaining_needed = n_total - len(selected_idx)
    if remaining_needed > 0:
        remaining_pool = pool.index.difference(selected_idx).to_numpy()
        if len(remaining_pool) < remaining_needed:
            logger.warning(
                "Pool insuficiente para completar %d posts: %d disponibles en total.",
                n_total, len(selected_idx) + len(remaining_pool),
            )
            remaining_needed = len(remaining_pool)
        if remaining_needed > 0:
            selected_idx.extend(rng.choice(remaining_pool, size=remaining_needed, replace=False).tolist())

    sample = pool.loc[selected_idx].sample(frac=1, random_state=seed).reset_index(drop=True)
    logger.info(
        "Sample: %d posts, %d tickers representados (pedido: %d posts, %d tickers).",
        len(sample), sample["target_ticker"].nunique(), n_total, len(tickers),
    )
    return sample


def write_annotation_batch(
    sample: pd.DataFrame,
    batch_name: str,
    annotators: list[str],
    output_dir: Path = _MANUAL_LABELS_DIR,
) -> list[Path]:
    """
    Escribe un CSV de anotación idéntico por cada etiquetador.

    Args:
        sample: DataFrame de sample_random() (o cualquier subset del pool).
        batch_name: Nombre del lote (e.g. "l0", "lr_aleatorio", "lu1").
        annotators: Lista de identificadores de etiquetador (e.g. ["esteban", "camilo"]).
        output_dir: Directorio de salida.

    Returns:
        Lista de rutas escritas, una por etiquetador: {output_dir}/{batch_name}_{annotator}.csv.
    """
    out = sample[["post_id", "target_ticker", "tickers_detectados", "clean_text"]].copy()
    for col in ("label", "confianza", "base", "nota"):
        out[col] = ""
    out = out[_ANNOTATION_COLUMNS]

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for annotator in annotators:
        path = output_dir / f"{batch_name}_{annotator}.csv"
        out.to_csv(path, index=False, encoding="utf-8")
        paths.append(path)
        logger.info("Lote '%s' para %s: %d filas → %s", batch_name, annotator, len(out), path)
    return paths


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
        description="Genera un lote de muestreo para etiquetado manual (event study, 6 tickers)."
    )
    parser.add_argument("--batch", default="l0", help="Nombre del lote (default: l0).")
    parser.add_argument("--n", type=int, default=100, help="Tamaño total del lote (default: 100).")
    parser.add_argument("--min-per-ticker", type=int, default=10, help="Mínimo por ticker (default: 10).")
    parser.add_argument(
        "--annotators", nargs="+", default=["esteban", "camilo"],
        help="Identificadores de etiquetador (default: esteban camilo).",
    )
    parser.add_argument(
        "--max-live-per-symbol", type=int, default=60,
        help="Tope de mensajes por símbolo para la fuente en vivo (default: 60).",
    )
    args = parser.parse_args()

    pool = build_pool(max_live_per_symbol=args.max_live_per_symbol)
    if pool.empty:
        print(
            "Pool vacío: no se pudo leer NYU (¿D:\\trading-data\\ montado?) ni "
            "StockTwits en vivo (¿red disponible?). Nada que muestrear."
        )
        sys.exit(1)

    sample = sample_random(pool, n_total=args.n, min_per_ticker=args.min_per_ticker)
    result_paths = write_annotation_batch(sample, args.batch, args.annotators)
    for p in result_paths:
        print(f"Escrito → {p}")
