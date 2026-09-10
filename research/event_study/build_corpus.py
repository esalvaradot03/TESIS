"""
Genera los lotes de etiquetado manual del corpus del event study.

La unidad de muestreo es el par **(post, ticker objetivo)**: un post que
menciona dos tickers del estudio produce dos filas independientes, una por
ticker, porque el sentimiento hacia cada ticker puede ser opuesto en el
mismo texto ("vendo $DIS para entrar a $CMG").

Fuentes, según cobertura de cada ticker:
  - **NYU** (`D:\\trading-data\\stocktwits_nyu\\messages\\`): texto crudo
    histórico. Cobertura confirmada solo para DIS y CMG. Este bucket nunca
    se había leído en el repo (solo `symbol_sentiments/`, sin texto), así
    que el schema se detecta en ejecución probando nombres candidatos y
    falla con las columnas reales si ninguno matchea.
  - **StockTwits en vivo**: cubre los 6 tickers pero el endpoint público
    solo sirve el stream reciente, no búsqueda histórica — no aporta
    profundidad por año para NCLH/CRWD/TGT/DDOG.

Diseño de muestreo (pre-registrado, SEED=42):
  1000 pares, estratificados por (ticker, año), mínimo 10 por ticker,
  partidos en 4 lotes:
    l0     100 pares — calibración, AMBOS anotadores etiquetan los mismos
    doble  200 pares — kappa primario, AMBOS etiquetan los mismos
    test   300 pares — 150/150, sin solapamiento
    train  400 pares — 200/200, sin solapamiento
  Cada anotador recibe 650 pares. El orden de filas se randomiza por
  (lote, anotador) con semillas derivadas de SEED para evitar sesgo de orden.

Uso:
    python -m research.event_study.build_corpus
    python -m research.event_study.build_corpus --total 1000 --min-per-ticker 10
"""

import ast
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import DATA_DIR, SEED, STOCKTWITS_NYU_MESSAGES
from src.sentiment.preprocessor import build_input_text, clean_text

logger = logging.getLogger(__name__)

# Los 6 tickers del event study (decididos, no cambiar).
TICKERS: list[str] = ["NCLH", "DIS", "CRWD", "TGT", "CMG", "DDOG"]

# Tickers con cobertura histórica confirmada en el bucket NYU.
NYU_COVERED_TICKERS: set[str] = {"DIS", "CMG"}

ANNOTATORS: list[str] = ["camilo", "esteban"]

TOTAL_PAIRS: int = 1000
MIN_PER_TICKER: int = 10

# lote -> (tamaño, compartido entre ambos anotadores)
_BATCHES: dict[str, tuple[int, bool]] = {
    "l0": (100, True),
    "doble": (200, True),
    "test": (300, False),
    "train": (400, False),
}

_MANUAL_LABELS_DIR = DATA_DIR / "manual_labels"

# Columnas del pool interno (incluye procedencia y año para estratificar).
_POOL_COLUMNS: list[str] = [
    "post_id", "target_ticker", "tickers_detectados", "fecha", "clean_text", "year", "source",
]

# Columnas del CSV que recibe el anotador (las 4 últimas van vacías).
_OUTPUT_COLUMNS: list[str] = [
    "post_id", "target_ticker", "tickers_detectados", "fecha", "clean_text",
    "label", "confianza", "base", "nota",
]

# Nombres candidatos del bucket NYU messages/ (mismo dataset que
# symbol_sentiments/: message_id, user_id, created_at, sentiment, symbol_list;
# 'body' es el nombre nativo del campo de texto en la API de StockTwits).
_TEXT_COLUMN_CANDIDATES: list[str] = ["body", "text", "message", "message_body"]
_TIMESTAMP_COLUMN_CANDIDATES: list[str] = ["created_at", "timestamp", "date"]
_SYMBOL_COLUMN_CANDIDATES: list[str] = ["symbol_list", "symbols"]
_ID_COLUMN_CANDIDATES: list[str] = ["message_id", "id"]


# ---------------------------------------------------------------------------
# Fuente NYU (histórica; solo DIS/CMG confirmados)
# ---------------------------------------------------------------------------

def _parse_symbol_list(cell: object) -> list[str]:
    """
    Parsea la columna de símbolos del dataset NYU (literal de Python, e.g.
    "['ZNGA','META']"), igual que process_stocktwits_nyu.py. Normaliza a
    mayúsculas y '.' → '-'.
    """
    if not isinstance(cell, str):
        return []
    raw = cell.strip()
    if not raw or raw == "[]":
        return []
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError, TypeError, MemoryError):
        return []
    if isinstance(parsed, (list, tuple)):
        return sorted({str(x).strip().upper().replace(".", "-") for x in parsed if x})
    return []


def _require_column(columns: list[str], candidates: list[str], role: str, path: Path) -> str:
    """Devuelve la primera columna candidata presente, o falla listando las reales."""
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(
        f"No se encontró columna de {role} en {path}. Candidatas probadas: {candidates}. "
        f"Columnas reales: {columns}. Ajustá las constantes *_COLUMN_CANDIDATES en "
        "build_corpus.py si el dataset NYU usa otro nombre."
    )


def detect_nyu_schema(sample_path: Path) -> dict[str, str | None]:
    """
    Detecta las columnas id/timestamp/símbolos/texto del bucket NYU messages/.

    Args:
        sample_path: Un CSV cualquiera del bucket, para leer solo su cabecera.

    Returns:
        Dict con las claves id (opcional, None si no aparece), timestamp,
        symbols y text.

    Raises:
        ValueError: si falta alguna columna obligatoria (timestamp/symbols/text).
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
    df["_matched"] = df["_symbols"].map(lambda syms: [s for s in syms if s in tickers])
    df = df[df["_matched"].map(len) > 0]
    if df.empty:
        return pd.DataFrame(columns=_POOL_COLUMNS)

    ids = df[schema["id"]].astype(str) if schema["id"] else pd.Series(
        [f"{path.stem}_{i}" for i in df.index], index=df.index
    )
    fechas = pd.to_datetime(df[schema["timestamp"]], errors="coerce", utc=True)

    rows: list[dict] = []
    for idx, row in df.iterrows():
        fecha = fechas.loc[idx]
        if pd.isna(fecha):
            continue
        text = clean_text(build_input_text(str(row[schema["text"]]), ""))
        symbols_json = json.dumps(row["_symbols"])
        for ticker in row["_matched"]:
            rows.append({
                "post_id": ids.loc[idx],
                "target_ticker": ticker,
                "tickers_detectados": symbols_json,
                "fecha": fecha.date().isoformat(),
                "clean_text": text,
                "year": int(fecha.year),
                "source": "nyu",
            })
    return pd.DataFrame(rows, columns=_POOL_COLUMNS)


def load_nyu_pairs(
    tickers: set[str],
    messages_dir: Path = STOCKTWITS_NYU_MESSAGES,
) -> pd.DataFrame:
    """
    Carga pares (post, target_ticker) históricos del bucket NYU messages/.

    Args:
        tickers: Tickers a conservar.
        messages_dir: Directorio del bucket NYU con los CSV crudos.

    Returns:
        DataFrame con _POOL_COLUMNS. Vacío (con warning) si el directorio no
        existe — típicamente porque D:\\trading-data\\ no está montado.
    """
    if not tickers:
        return pd.DataFrame(columns=_POOL_COLUMNS)
    if not messages_dir.exists():
        logger.warning(
            "No existe %s (¿está montado D:\\trading-data\\?). Sin fuente histórica NYU.",
            messages_dir,
        )
        return pd.DataFrame(columns=_POOL_COLUMNS)

    files = sorted(messages_dir.glob("*.csv"))
    if not files:
        logger.warning("Sin archivos .csv en %s. Sin fuente histórica NYU.", messages_dir)
        return pd.DataFrame(columns=_POOL_COLUMNS)

    schema = detect_nyu_schema(files[0])
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
    logger.info("Fuente NYU: %d pares sobre %d archivo(s).", len(result), len(files))
    return result


# ---------------------------------------------------------------------------
# Fuente StockTwits en vivo (recientes; los 6 tickers)
# ---------------------------------------------------------------------------

def load_live_pairs(tickers: list[str], max_per_symbol: int = 60) -> pd.DataFrame:
    """
    Trae pares (post, target_ticker) del stream reciente de StockTwits.

    ADVERTENCIA metodológica: el endpoint público solo expone mensajes
    recientes, no búsqueda histórica. Para los tickers sin cobertura NYU
    esta fuente concentra todo el muestreo en el año en curso y NO permite
    estratificar por año.

    Args:
        tickers: Tickers a consultar.
        max_per_symbol: Tope de mensajes por símbolo.

    Returns:
        DataFrame con _POOL_COLUMNS. Vacío si el scraping falla.
    """
    logger.warning(
        "Fuente en vivo: stream RECIENTE de StockTwits, no histórico. "
        "Para %s (sin cobertura NYU) no aporta profundidad por año.",
        sorted(set(tickers) - NYU_COVERED_TICKERS),
    )
    try:
        from src.sentiment.scraper_stocktwits import scrape
    except ImportError as exc:
        logger.warning("No se pudo importar el scraper (%s). Se omite la fuente en vivo.", exc)
        return pd.DataFrame(columns=_POOL_COLUMNS)

    try:
        raw_path = scrape(symbols=tickers, max_messages_per_symbol=max_per_symbol)
    except Exception as exc:  # noqa: BLE001 — sin red el build sigue con lo que haya
        logger.warning("Scraping en vivo falló (%s). Se omite la fuente en vivo.", exc)
        return pd.DataFrame(columns=_POOL_COLUMNS)

    raw = pd.read_csv(raw_path, dtype=str).fillna("")
    universe = set(tickers)

    rows: list[dict] = []
    for _, row in raw.iterrows():
        detected = json.loads(row["detected_tickers"]) if row["detected_tickers"] else []
        matched = [t for t in detected if t in universe]
        if not matched:
            continue
        fecha = pd.to_datetime(row["timestamp"], errors="coerce", utc=True)
        if pd.isna(fecha):
            continue
        text = clean_text(build_input_text(row["title"], row["body"]))
        for ticker in matched:
            rows.append({
                "post_id": row["post_id"],
                "target_ticker": ticker,
                "tickers_detectados": row["detected_tickers"],
                "fecha": fecha.date().isoformat(),
                "clean_text": text,
                "year": int(fecha.year),
                "source": "live",
            })

    logger.info("Fuente en vivo: %d pares sobre %d tickers.", len(rows), len(tickers))
    return pd.DataFrame(rows, columns=_POOL_COLUMNS)


def build_pool(
    tickers: list[str] = TICKERS,
    nyu_tickers: set[str] = NYU_COVERED_TICKERS,
    max_live_per_symbol: int = 60,
) -> pd.DataFrame:
    """
    Combina las dos fuentes en un pool deduplicado por (post_id, target_ticker).

    Args:
        tickers: Universo de tickers del estudio.
        nyu_tickers: Subconjunto con cobertura histórica NYU.
        max_live_per_symbol: Tope de mensajes por símbolo en la fuente en vivo.

    Returns:
        DataFrame con _POOL_COLUMNS, sin filas de texto vacío.
    """
    frames = [
        load_nyu_pairs(nyu_tickers & set(tickers)),
        load_live_pairs(tickers, max_live_per_symbol),
    ]
    frames = [f for f in frames if not f.empty]
    if not frames:
        logger.warning("Pool vacío: ninguna fuente devolvió datos.")
        return pd.DataFrame(columns=_POOL_COLUMNS)

    pool = pd.concat(frames, ignore_index=True)
    pool = pool[pool["clean_text"].str.strip() != ""]
    pool = pool.drop_duplicates(subset=["post_id", "target_ticker"], keep="first").reset_index(drop=True)

    logger.info(
        "Pool: %d pares | fuentes: %s | tickers: %s | años: %s",
        len(pool),
        pool["source"].value_counts().to_dict(),
        sorted(pool["target_ticker"].unique()),
        sorted(pool["year"].unique().tolist()),
    )
    return pool


# ---------------------------------------------------------------------------
# Muestreo estratificado por (ticker, año)
# ---------------------------------------------------------------------------

def _largest_remainder(weights: dict, total: int, caps: dict) -> dict:
    """
    Reparte `total` unidades entre las llaves de `weights` proporcionalmente,
    respetando el tope `caps` de cada llave (método del resto mayor).

    Args:
        weights: Peso relativo de cada llave (típicamente su disponibilidad).
        total: Unidades a repartir.
        caps: Tope por llave.

    Returns:
        Dict llave → unidades asignadas. La suma es min(total, sum(caps)).
    """
    total_weight = sum(weights.values())
    if total <= 0 or total_weight <= 0:
        return {k: 0 for k in weights}

    exact = {k: total * w / total_weight for k, w in weights.items()}
    alloc = {k: min(int(v), caps[k]) for k, v in exact.items()}

    # El sobrante se reparte por parte fraccionaria descendente, saltando topes.
    order = sorted(exact, key=lambda k: exact[k] - int(exact[k]), reverse=True)
    while sum(alloc.values()) < total:
        progressed = False
        for key in order:
            if sum(alloc.values()) >= total:
                break
            if alloc[key] < caps[key]:
                alloc[key] += 1
                progressed = True
        if not progressed:  # todas las llaves llegaron a su tope
            break
    return alloc


def allocate_by_cell(
    pool: pd.DataFrame,
    total: int = TOTAL_PAIRS,
    min_per_ticker: int = MIN_PER_TICKER,
    tickers: list[str] = TICKERS,
) -> dict[tuple[str, int], int]:
    """
    Decide cuántos pares tomar de cada celda (ticker, año).

    Primero garantiza `min_per_ticker` por ticker (repartido entre sus años),
    y recién después reparte el resto proporcionalmente a la disponibilidad
    de cada celda. Así un ticker con poca cobertura no queda fuera del corpus.

    Args:
        pool: Pool de build_pool().
        total: Pares a muestrear en total.
        min_per_ticker: Mínimo garantizado por ticker.
        tickers: Universo de tickers.

    Returns:
        Dict (ticker, año) → cantidad a muestrear.
    """
    available = pool.groupby(["target_ticker", "year"]).size().to_dict()
    ticker_caps = {t: sum(n for (tk, _), n in available.items() if tk == t) for t in tickers}
    ticker_caps = {t: n for t, n in ticker_caps.items() if n > 0}

    missing = [t for t in tickers if t not in ticker_caps]
    if missing:
        logger.warning("Tickers sin ningún par en el pool: %s", missing)

    # 1) mínimo garantizado por ticker
    base = {t: min(min_per_ticker, cap) for t, cap in ticker_caps.items()}
    remaining = total - sum(base.values())
    if remaining < 0:
        raise ValueError(
            f"El mínimo por ticker ({min_per_ticker} × {len(base)} tickers) supera "
            f"el total pedido ({total})."
        )

    # 2) el resto, proporcional a la disponibilidad residual
    residual = {t: ticker_caps[t] - base[t] for t in base}
    extra = _largest_remainder(residual, remaining, residual)
    per_ticker = {t: base[t] + extra.get(t, 0) for t in base}

    # 3) dentro de cada ticker, repartir por año
    cells: dict[tuple[str, int], int] = {}
    for ticker, n_ticker in per_ticker.items():
        year_caps = {y: n for (tk, y), n in available.items() if tk == ticker}
        per_year = _largest_remainder(year_caps, n_ticker, year_caps)
        for year, n_year in per_year.items():
            if n_year > 0:
                cells[(ticker, year)] = n_year
    return cells


def sample_stratified(
    pool: pd.DataFrame,
    total: int = TOTAL_PAIRS,
    min_per_ticker: int = MIN_PER_TICKER,
    tickers: list[str] = TICKERS,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Muestrea `total` pares estratificados por (ticker, año), con SEED fijo.

    Args:
        pool: Pool de build_pool().
        total: Pares a muestrear.
        min_per_ticker: Mínimo garantizado por ticker.
        tickers: Universo de tickers.
        seed: Semilla (SEED=42 del proyecto).

    Returns:
        DataFrame barajado con los pares seleccionados (columnas _POOL_COLUMNS).
    """
    if pool.empty:
        return pool.copy()

    cells = allocate_by_cell(pool, total, min_per_ticker, tickers)
    rng = np.random.default_rng(seed)

    picked: list[int] = []
    for (ticker, year), n_cell in sorted(cells.items()):
        candidates = pool.index[
            (pool["target_ticker"] == ticker) & (pool["year"] == year)
        ].to_numpy()
        picked.extend(rng.choice(candidates, size=n_cell, replace=False).tolist())

    sample = pool.loc[picked].sample(frac=1, random_state=seed).reset_index(drop=True)
    logger.info(
        "Muestreo: %d pares de %d pedidos (%d celdas ticker×año).",
        len(sample), total, len(cells),
    )
    return sample


# ---------------------------------------------------------------------------
# Partición en lotes y reparto entre anotadores
# ---------------------------------------------------------------------------

def partition(sample: pd.DataFrame, batches: dict[str, tuple[int, bool]] = _BATCHES) -> dict[str, pd.DataFrame]:
    """
    Corta el sample en los lotes l0 / doble / test / train, en ese orden.

    Args:
        sample: Salida de sample_stratified() (ya barajada).
        batches: Definición lote → (tamaño, compartido).

    Returns:
        Dict lote → DataFrame.

    Raises:
        ValueError: si el sample no alcanza para cubrir todos los lotes.
    """
    needed = sum(size for size, _ in batches.values())
    if len(sample) < needed:
        raise ValueError(
            f"El sample tiene {len(sample)} pares pero los lotes requieren {needed}. "
            "Ampliá el pool (montá D:\\trading-data\\ o acumulá más scraping) o "
            "corré con --allow-partial para generar lotes proporcionales más chicos."
        )

    result: dict[str, pd.DataFrame] = {}
    start = 0
    for name, (size, _) in batches.items():
        result[name] = sample.iloc[start: start + size].reset_index(drop=True)
        start += size
    return result


def assign_to_annotators(
    batches: dict[str, pd.DataFrame],
    definitions: dict[str, tuple[int, bool]] = _BATCHES,
    annotators: list[str] = ANNOTATORS,
) -> dict[tuple[str, str], pd.DataFrame]:
    """
    Reparte cada lote entre los anotadores.

    Los lotes compartidos (l0, doble) van completos a ambos — son la base del
    kappa. Los demás se parten en bloques disjuntos del mismo tamaño.

    Args:
        batches: Salida de partition().
        definitions: Definición lote → (tamaño, compartido).
        annotators: Nombres de los anotadores.

    Returns:
        Dict (lote, anotador) → DataFrame.
    """
    result: dict[tuple[str, str], pd.DataFrame] = {}
    for name, frame in batches.items():
        shared = definitions[name][1]
        if shared:
            for annotator in annotators:
                result[(name, annotator)] = frame.copy()
            continue
        chunk = len(frame) // len(annotators)
        for i, annotator in enumerate(annotators):
            result[(name, annotator)] = frame.iloc[i * chunk: (i + 1) * chunk].reset_index(drop=True)
    return result


def _annotator_seed(seed: int, batch: str, annotator: str) -> int:
    """Semilla determinista y distinta por (lote, anotador), derivada de `seed`."""
    digest = hashlib.sha256(f"{seed}|{batch}|{annotator}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def write_batches(
    assignments: dict[tuple[str, str], pd.DataFrame],
    output_dir: Path = _MANUAL_LABELS_DIR,
    seed: int = SEED,
) -> dict[str, Path]:
    """
    Escribe un CSV por (lote, anotador), con las columnas de anotación vacías.

    El orden de filas se randomiza con una semilla derivada distinta por
    (lote, anotador): los lotes compartidos llevan los MISMOS pares en orden
    DISTINTO para cada anotador, reduciendo sesgo de orden sin romper el
    merge posterior (la llave es (post_id, target_ticker)).

    Args:
        assignments: Salida de assign_to_annotators().
        output_dir: Directorio de salida.
        seed: Semilla base.

    Returns:
        Dict nombre_archivo → ruta escrita.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for (batch, annotator), frame in sorted(assignments.items()):
        shuffled = frame.sample(frac=1, random_state=_annotator_seed(seed, batch, annotator))
        out = shuffled[["post_id", "target_ticker", "tickers_detectados", "fecha", "clean_text"]].copy()
        for column in ("label", "confianza", "base", "nota"):
            out[column] = ""
        out = out[_OUTPUT_COLUMNS]

        path = output_dir / f"{batch}_{annotator}.csv"
        out.to_csv(path, index=False, encoding="utf-8")
        written[path.name] = path
        logger.info("Lote %s/%s: %d filas → %s", batch, annotator, len(out), path)

    return written


# ---------------------------------------------------------------------------
# Manifiesto de trazabilidad
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    """SHA-256 del contenido de un archivo."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest(
    sample: pd.DataFrame,
    assignments: dict[tuple[str, str], pd.DataFrame],
    written: dict[str, Path],
    seed: int = SEED,
    output_dir: Path = _MANUAL_LABELS_DIR,
) -> Path:
    """
    Escribe manifest.json con la trazabilidad del muestreo (pre-registro).

    Args:
        sample: Pares muestreados.
        assignments: Reparto (lote, anotador) → filas.
        written: Archivos escritos, para hashear.
        seed: Semilla usada.
        output_dir: Directorio de salida.

    Returns:
        Ruta al manifest.json.
    """
    por_ticker_y_anio: dict[str, dict[str, int]] = {}
    for (ticker, year), count in sample.groupby(["target_ticker", "year"]).size().items():
        por_ticker_y_anio.setdefault(str(ticker), {})[str(year)] = int(count)

    manifest = {
        "generado": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "total_pares": int(len(sample)),
        "min_por_ticker": MIN_PER_TICKER,
        "unidad_de_muestreo": "(post_id, target_ticker)",
        "lotes": {
            batch: {
                "pares": int(size),
                "compartido": shared,
                "por_anotador": {
                    annotator: int(len(assignments[(batch, annotator)]))
                    for annotator in ANNOTATORS
                    if (batch, annotator) in assignments
                },
            }
            for batch, (size, shared) in _BATCHES.items()
        },
        "por_ticker": {
            str(t): int(n) for t, n in sample["target_ticker"].value_counts().sort_index().items()
        },
        "por_ticker_y_anio": por_ticker_y_anio,
        "por_fuente": {str(s): int(n) for s, n in sample["source"].value_counts().items()},
        "archivos": {
            name: {"filas": int(sum(1 for _ in path.open(encoding="utf-8")) - 1), "sha256": _sha256(path)}
            for name, path in sorted(written.items())
        },
    }

    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Manifiesto escrito en %s", path)
    return path


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
        description="Genera los lotes de etiquetado del corpus del event study (6 tickers)."
    )
    parser.add_argument("--total", type=int, default=TOTAL_PAIRS, help="Pares a muestrear (default: 1000).")
    parser.add_argument("--min-per-ticker", type=int, default=MIN_PER_TICKER, help="Mínimo por ticker (default: 10).")
    parser.add_argument("--max-live-per-symbol", type=int, default=60, help="Tope de mensajes por símbolo en vivo.")
    parser.add_argument("--output-dir", type=Path, default=_MANUAL_LABELS_DIR, help="Directorio de salida.")
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Permite generar lotes proporcionalmente más chicos si el pool no alcanza "
             "(por defecto falla, para no romper el diseño pre-registrado).",
    )
    args = parser.parse_args()

    pool = build_pool(max_live_per_symbol=args.max_live_per_symbol)
    if pool.empty:
        print(
            "Pool vacío: no se pudo leer NYU (¿D:\\trading-data\\ montado?) ni "
            "StockTwits en vivo (¿red disponible?). No se generó ningún lote."
        )
        sys.exit(1)

    sample = sample_stratified(pool, total=args.total, min_per_ticker=args.min_per_ticker)

    definitions = _BATCHES
    if len(sample) < args.total:
        message = (
            f"El pool solo permitió muestrear {len(sample)} de {args.total} pares pedidos."
        )
        if not args.allow_partial:
            print(f"{message} Abortado para no romper el diseño pre-registrado. "
                  "Usá --allow-partial si querés lotes más chicos igual.")
            sys.exit(1)
        scale = len(sample) / args.total
        definitions = {
            name: (max(1, int(size * scale)), shared) for name, (size, shared) in _BATCHES.items()
        }
        logger.warning("%s Se generan lotes proporcionales: %s", message,
                       {k: v[0] for k, v in definitions.items()})

    batches = partition(sample, definitions)
    assignments = assign_to_annotators(batches, definitions)
    written = write_batches(assignments, args.output_dir)
    manifest_path = write_manifest(sample, assignments, written, output_dir=args.output_dir)

    print(f"\n{len(written)} archivos escritos en {args.output_dir}")
    for name in sorted(written):
        print(f"  {name}")
    print(f"Manifiesto: {manifest_path}")
