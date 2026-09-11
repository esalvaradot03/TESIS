"""
Genera los lotes de etiquetado manual del corpus del event study.

La unidad de muestreo es el par **(post, ticker objetivo)**: un post que
menciona dos tickers del estudio produce dos filas independientes, una por
ticker, porque el sentimiento hacia cada ticker puede ser opuesto en el
mismo texto ("vendo $DIS para entrar a $CMG").

Fuentes:
  - **NYU** (`D:\\trading-data\\stocktwits_nyu\\`), fuente única por defecto.
    Cubre los 6 tickers entre 2010 y 2022 (CRWD y DDOG desde su IPO en 2019;
    NCLH casi todo desde 2020). El bucket viene partido en dos:
    `symbol_sentiments/` trae (message_id, created_at, symbol_list) y
    `messages/` trae solo (message_id, message_body). El pool combina:
      - etiquetados (label nativo Bullish/Bearish): ticker y fecha reales de
        symbol_sentiments/, unidos al texto por message_id;
      - sin etiqueta: ticker por cashtag en el texto (regla validada contra
        symbol_list: precisión y recall 100%) y fecha interpolada desde el
        message_id con las anclas de symbol_sentiments/ (holdout 20%: 99,997%
        en el día correcto, error máximo 1 día).
    Sin los no etiquetados la clase neutral quedaría excluida por el filtro de
    origen y la comparación de enfoques sería circular.
  - **StockTwits en vivo** (opcional, `--include-live`): solo el stream
    reciente, sin búsqueda histórica, y el sample cambia según cuándo se
    corra. Fuera del diseño pre-registrado.

Diseño de muestreo (pre-registrado, SEED=42):
  1000 pares con cuotas iguales por ticker (~167; un ticker corto se reporta,
  no se rellena), estratificados por año dentro de cada ticker, partidos en
  4 lotes disjuntos:
    l0     100 pares — calibración, AMBOS anotadores etiquetan los mismos;
                       mínimo 10 por ticker
    doble  200 pares — kappa primario, AMBOS etiquetan los mismos;
                       mínimo 10 por ticker
    test   300 pares — 150/150, sin solapamiento
    train  400 pares — 200/200, sin solapamiento
  Todos los pares de un mismo post caen en el mismo lote (sin fuga de texto
  entre train y test). Cada anotador recibe 650 pares. El orden de filas se
  randomiza por (lote, anotador) con semillas derivadas de SEED para evitar
  sesgo de orden.

Uso:
    python -m research.event_study.build_corpus
    python -m research.event_study.build_corpus --total 1000 --min-per-ticker 10
"""

import ast
import csv
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import DATA_DIR, SEED, STOCKTWITS_NYU_MESSAGES, STOCKTWITS_NYU_SYMBOL_SENTIMENTS
from src.sentiment.preprocessor import build_input_text, clean_text

logger = logging.getLogger(__name__)

# Los 6 tickers del event study (decididos, no cambiar).
TICKERS: list[str] = ["NCLH", "DIS", "CRWD", "TGT", "CMG", "DDOG"]

# Tickers con cobertura histórica en el bucket NYU (los 6, verificado sep-2026).
NYU_COVERED_TICKERS: set[str] = set(TICKERS)

ANNOTATORS: list[str] = ["camilo", "esteban"]

TOTAL_PAIRS: int = 1000
MIN_PER_TICKER: int = 10

# Reparto entre tickers: cuotas iguales (total / n_tickers). La pregunta es si
# la captura de sentimiento varía por segmento de liquidez, así que se necesita
# precisión comparable en los seis; muestrear en proporción a los posts
# importaría al corpus el sesgo de atención del foro (DIS sería ~50%). Un
# ticker sin material para su cuota se reporta como faltante, no se rellena.
EQUAL_QUOTAS: bool = True

# lote -> (tamaño, compartido entre ambos anotadores)
_BATCHES: dict[str, tuple[int, bool]] = {
    "l0": (100, True),
    "doble": (200, True),
    "test": (300, False),
    "train": (400, False),
}

# Mínimo de pares por ticker dentro de cada lote. Los lotes sin entrada no
# tienen mínimo: test y train se llenan al azar (no por fecha).
_BATCH_MIN_PER_TICKER: dict[str, int] = {"l0": 10, "doble": 10}

_MANUAL_LABELS_DIR = DATA_DIR / "manual_labels"

# Caché del join NYU (escanear messages/ son ~55 GB). Ignorado por git.
_NYU_CACHE_DIR = DATA_DIR / "processed"
_NYU_CACHE_COLUMNS: list[str] = ["message_id", "created_at", "symbol_list", "message_body", "origen"]
_EPOCH = np.datetime64("1970-01-01", "D")

# Columnas del pool interno (incluye procedencia y año para estratificar).
_POOL_COLUMNS: list[str] = [
    "post_id", "target_ticker", "tickers_detectados", "fecha", "clean_text", "year", "source",
]

# Columnas del CSV que recibe el anotador (las 4 últimas van vacías).
_OUTPUT_COLUMNS: list[str] = [
    "post_id", "target_ticker", "tickers_detectados", "fecha", "clean_text",
    "label", "confianza", "base", "nota",
]


# ---------------------------------------------------------------------------
# Fuente NYU (histórica; los 6 tickers)
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


def _load_nyu_metadata(tickers: set[str], sentiments_dir: Path) -> pd.DataFrame:
    """
    Lee symbol_sentiments/ y devuelve los mensajes que mencionan algún ticker.

    Args:
        tickers: Tickers a conservar.
        sentiments_dir: Directorio NYU symbol_sentiments/.

    Returns:
        DataFrame (message_id, created_at, symbol_list) con symbol_list como
        JSON array normalizado, un registro por message_id.
    """
    pattern = "|".join(sorted(tickers))
    frames: list[pd.DataFrame] = []
    for path in sorted(sentiments_dir.glob("*.csv")):
        for chunk in pd.read_csv(path, usecols=["message_id", "created_at", "symbol_list"],
                                 dtype=str, chunksize=500_000):
            # Prefiltro barato por substring; el filtro exacto va tras parsear.
            chunk = chunk[chunk["symbol_list"].str.contains(pattern, na=False)]
            if not chunk.empty:
                frames.append(chunk)
    if not frames:
        return pd.DataFrame(columns=["message_id", "created_at", "symbol_list"])

    meta = pd.concat(frames, ignore_index=True)
    symbols = meta["symbol_list"].map(_parse_symbol_list)
    keep = symbols.map(lambda syms: any(s in tickers for s in syms))
    meta = meta[keep].assign(symbol_list=symbols[keep].map(json.dumps))
    return meta.drop_duplicates("message_id").reset_index(drop=True)


# Cashtags de cualquier símbolo (para tickers_detectados) y de los tickers del
# estudio (para decidir si un mensaje sin etiqueta entra al pool). La regla se
# validó contra symbol_list en mensajes etiquetados: precisión y recall 100%
# (3 archivos de messages/, 6.211 pares, sep-2026).
_ANY_CASHTAG_RE = re.compile(r"\$([A-Za-z]{1,6}(?:[.\-][A-Za-z]{1,2})?)(?![A-Za-z0-9])")


def _study_cashtag_re(tickers: set[str]) -> re.Pattern:
    """Regex de cashtags de los tickers del estudio (sin grupos de captura, sin distinguir mayúsculas)."""
    return re.compile(r"\$(?:" + "|".join(sorted(tickers)) + r")(?![A-Za-z0-9])", re.IGNORECASE)


def _load_id_day_anchors(sentiments_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Lee (message_id, día) de TODOS los mensajes de symbol_sentiments/.

    Son las anclas para fechar por interpolación los mensajes sin etiqueta
    (messages/ no trae fecha). Validado con holdout aleatorio del 20%
    (SEED=42, 21,2M mensajes): 99,997% cae en el día correcto y el error
    máximo es 1 día; 100% en el día correcto para 2019–2022.

    Returns:
        (ids int64 ordenados y sin duplicados, días desde 1970-01-01 como int32).
    """
    ids_parts: list[np.ndarray] = []
    day_parts: list[np.ndarray] = []
    for path in sorted(sentiments_dir.glob("*.csv")):
        for chunk in pd.read_csv(path, usecols=["message_id", "created_at"], dtype=str, chunksize=2_000_000):
            mid = pd.to_numeric(chunk["message_id"], errors="coerce")
            day = pd.to_datetime(chunk["created_at"], errors="coerce", format="%Y-%m-%d")
            ok = mid.notna() & day.notna()
            ids_parts.append(mid[ok].to_numpy(np.int64))
            day_parts.append((day[ok].to_numpy("datetime64[D]") - _EPOCH).astype(np.int32))
    ids = np.concatenate(ids_parts)
    days = np.concatenate(day_parts)
    order = np.argsort(ids, kind="stable")
    ids, days = ids[order], days[order]
    keep = np.r_[True, ids[1:] != ids[:-1]]
    return ids[keep], days[keep]


def _in_sorted(values: np.ndarray, sorted_ref: np.ndarray) -> np.ndarray:
    """Pertenencia a un array ordenado vía searchsorted (np.isin con 100M+ refs es lento)."""
    pos = np.clip(np.searchsorted(sorted_ref, values), 0, len(sorted_ref) - 1)
    return sorted_ref[pos] == values


def _scan_messages_file(
    path: Path, labeled_ids: set[str], anchor_ids: np.ndarray, study_re: re.Pattern,
) -> pd.DataFrame:
    """
    Devuelve (message_id, message_body) de un CSV de messages/ para:
      - los mensajes etiquetados que mencionan un ticker del estudio (`labeled_ids`), y
      - los mensajes SIN etiqueta (id fuera de `anchor_ids`) con cashtag de un
        ticker del estudio en el texto.

    Varios archivos del bucket tienen comillas mal cerradas que tumban el
    parser C de pandas ("Buffer overflow caught"); en ese caso se relee el
    archivo completo con el módulo csv, más lento pero tolerante.
    """
    def keep(frame: pd.DataFrame) -> pd.DataFrame:
        numeric = pd.to_numeric(frame["message_id"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
        unlabeled = ~_in_sorted(numeric, anchor_ids)
        mentions = frame["message_body"].fillna("").str.contains(study_re, na=False).to_numpy()
        mask = frame["message_id"].isin(labeled_ids).to_numpy() | (unlabeled & mentions)
        return frame.loc[mask, ["message_id", "message_body"]]

    try:
        hits = [
            keep(chunk)
            for chunk in pd.read_csv(path, dtype=str, chunksize=1_000_000,
                                     encoding="utf-8", encoding_errors="replace")
        ]
        return pd.concat(hits, ignore_index=True)
    except pd.errors.ParserError as exc:
        logger.warning("%s: parser C falló (%s); se relee con el módulo csv.", path.name, str(exc)[:80])

    csv.field_size_limit(2**31 - 1)
    ids: list[str] = []
    bodies: list[str] = []
    with path.open(encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if len(row) >= 2:
                ids.append(row[0])
                bodies.append(row[1])
    return keep(pd.DataFrame({"message_id": ids, "message_body": bodies})).reset_index(drop=True)


def _build_nyu_cache(
    tickers: set[str], sentiments_dir: Path, messages_dir: Path, cache_path: Path,
) -> pd.DataFrame:
    """
    Arma el pool NYU y lo guarda en `cache_path`:
      - etiquetados: ticker y fecha reales de symbol_sentiments/, texto de messages/;
      - sin etiqueta: ticker por cashtag en el texto, fecha interpolada desde el
        message_id con las anclas de symbol_sentiments/. Los ids fuera del
        rango de anclas (antes de 2010-06 o después de 2022-12) se descartan:
        ahí la fecha sería una extrapolación.

    Returns:
        DataFrame con _NYU_CACHE_COLUMNS (origen = etiquetado | sin_etiqueta).
    """
    meta = _load_nyu_metadata(tickers, sentiments_dir)
    labeled_ids = set(meta["message_id"])
    anchor_ids, anchor_days = _load_id_day_anchors(sentiments_dir)
    study_re = _study_cashtag_re(tickers)
    logger.info(
        "symbol_sentiments/: %d etiquetados mencionan %s; %d anclas de fecha (ids %d–%d).",
        len(labeled_ids), sorted(tickers), len(anchor_ids), anchor_ids[0], anchor_ids[-1],
    )

    files = sorted(messages_dir.glob("*.csv"))
    texts: list[pd.DataFrame] = []
    for i, path in enumerate(files, start=1):
        texts.append(_scan_messages_file(path, labeled_ids, anchor_ids, study_re))
        logger.info("messages/ %d/%d (%s): %d textos.", i, len(files), path.name, len(texts[-1]))
    text = pd.concat(texts, ignore_index=True).drop_duplicates("message_id")

    is_labeled = text["message_id"].isin(labeled_ids)
    labeled = meta.merge(text[is_labeled], on="message_id", how="inner").assign(origen="etiquetado")

    unlabeled = text[~is_labeled]
    ids = pd.to_numeric(unlabeled["message_id"], errors="coerce")
    unlabeled, ids = unlabeled[ids.notna()], ids[ids.notna()].astype(np.int64).to_numpy()
    in_range = (ids >= anchor_ids[0]) & (ids <= anchor_ids[-1])
    logger.info(
        "Sin etiqueta con cashtag: %d; fuera del rango de anclas (descartados): %d.",
        len(ids), int((~in_range).sum()),
    )
    unlabeled, ids = unlabeled[in_range].copy(), ids[in_range]
    days = np.rint(np.interp(ids, anchor_ids, anchor_days)).astype(np.int64)
    unlabeled["created_at"] = (_EPOCH + days.astype("timedelta64[D]")).astype(str)
    unlabeled["symbol_list"] = unlabeled["message_body"].map(
        lambda body: json.dumps(sorted(
            {m.upper().replace(".", "-") for m in _ANY_CASHTAG_RE.findall(body)}
            | {m[1:].upper() for m in study_re.findall(body)}
        ))
    )
    unlabeled["origen"] = "sin_etiqueta"

    joined = pd.concat([labeled, unlabeled], ignore_index=True)[_NYU_CACHE_COLUMNS]
    logger.info(
        "Pool NYU: %d mensajes | %s", len(joined), joined["origen"].value_counts().to_dict(),
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(cache_path, index=False)
    logger.info("Caché NYU escrito en %s", cache_path)
    return joined


def _explode_nyu_pairs(raw: pd.DataFrame, tickers: set[str]) -> pd.DataFrame:
    """Convierte el pool NYU en pares (post, target_ticker) con _POOL_COLUMNS."""
    fechas = pd.to_datetime(raw["created_at"], errors="coerce", utc=True)
    origenes = raw["origen"] if "origen" in raw else pd.Series("etiquetado", index=raw.index)
    rows: list[dict] = []
    for post_id, fecha, symbols_json, body, origen in zip(
        raw["message_id"], fechas, raw["symbol_list"], raw["message_body"], origenes
    ):
        if pd.isna(fecha):
            continue
        symbols = json.loads(symbols_json)
        text = clean_text(build_input_text(str(body), ""))
        for ticker in symbols:
            if ticker not in tickers:
                continue
            rows.append({
                "post_id": str(post_id),
                "target_ticker": ticker,
                "tickers_detectados": symbols_json,
                "fecha": fecha.date().isoformat(),
                "clean_text": text,
                "year": int(fecha.year),
                "source": f"nyu_{origen}",
            })
    return pd.DataFrame(rows, columns=_POOL_COLUMNS)


def load_nyu_pairs(
    tickers: set[str],
    sentiments_dir: Path = STOCKTWITS_NYU_SYMBOL_SENTIMENTS,
    messages_dir: Path = STOCKTWITS_NYU_MESSAGES,
    cache_dir: Path = _NYU_CACHE_DIR,
    rebuild: bool = False,
) -> pd.DataFrame:
    """
    Carga pares (post, target_ticker) históricos del bucket NYU.

    El escaneo de messages/ (~55 GB) se cachea en
    data/processed/nyu_pool_v2_<tickers>.parquet; se reutiliza salvo `rebuild`.
    (v1 = solo etiquetados; v2 = etiquetados + sin etiqueta con fecha interpolada.)

    Args:
        tickers: Tickers a conservar.
        sentiments_dir: Directorio NYU symbol_sentiments/ (ticker y fecha).
        messages_dir: Directorio NYU messages/ (texto).
        cache_dir: Directorio del caché del join.
        rebuild: Ignora el caché y rehace el join.

    Returns:
        DataFrame con _POOL_COLUMNS. Vacío (con warning) si no hay caché y el
        bucket no está disponible — típicamente D:\\trading-data\\ sin montar.
    """
    if not tickers:
        return pd.DataFrame(columns=_POOL_COLUMNS)

    cache_path = cache_dir / f"nyu_pool_v2_{'_'.join(sorted(tickers))}.parquet"
    if cache_path.exists() and not rebuild:
        raw = pd.read_parquet(cache_path)
        logger.info("Caché NYU: %d mensajes desde %s", len(raw), cache_path)
    else:
        missing = [d for d in (sentiments_dir, messages_dir) if not d.exists()]
        if missing:
            logger.warning(
                "No existe %s (¿está montado D:\\trading-data\\?). Sin fuente histórica NYU.",
                missing,
            )
            return pd.DataFrame(columns=_POOL_COLUMNS)
        raw = _build_nyu_cache(tickers, sentiments_dir, messages_dir, cache_path)

    result = _explode_nyu_pairs(raw, tickers)
    logger.info("Fuente NYU: %d pares.", len(result))
    return result


# ---------------------------------------------------------------------------
# Fuente StockTwits en vivo (recientes; los 6 tickers)
# ---------------------------------------------------------------------------

def load_live_pairs(tickers: list[str], max_per_symbol: int = 60) -> pd.DataFrame:
    """
    Trae pares (post, target_ticker) del stream reciente de StockTwits.

    ADVERTENCIA metodológica: el endpoint público solo expone mensajes
    recientes, no búsqueda histórica, así que esta fuente concentra el
    muestreo en el año en curso y el sample cambia según cuándo se corra.

    Args:
        tickers: Tickers a consultar.
        max_per_symbol: Tope de mensajes por símbolo.

    Returns:
        DataFrame con _POOL_COLUMNS. Vacío si el scraping falla.
    """
    logger.warning(
        "Fuente en vivo: stream RECIENTE de StockTwits, no histórico; "
        "el sample deja de ser reproducible."
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
    include_live: bool = False,
    max_live_per_symbol: int = 60,
    rebuild_nyu_cache: bool = False,
) -> pd.DataFrame:
    """
    Arma el pool de pares deduplicado por (post_id, target_ticker).

    Args:
        tickers: Universo de tickers del estudio.
        nyu_tickers: Subconjunto con cobertura histórica NYU.
        include_live: Suma la fuente en vivo (fuera del diseño pre-registrado).
        max_live_per_symbol: Tope de mensajes por símbolo en la fuente en vivo.
        rebuild_nyu_cache: Rehace el join NYU aunque exista el caché.

    Returns:
        DataFrame con _POOL_COLUMNS, sin filas de texto vacío.
    """
    frames = [load_nyu_pairs(nyu_tickers & set(tickers), rebuild=rebuild_nyu_cache)]
    if include_live:
        frames.append(load_live_pairs(tickers, max_live_per_symbol))
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
    equal_quotas: bool = EQUAL_QUOTAS,
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
        equal_quotas: Si True, cada ticker recibe total / n_tickers pares
            (`min_per_ticker` no aplica); el faltante de un ticker corto se
            reporta y NO se rellena con otros. Si False, reparto con mínimo
            `min_per_ticker` y el resto proporcional a la disponibilidad.

    Returns:
        Dict (ticker, año) → cantidad a muestrear.
    """
    available = pool.groupby(["target_ticker", "year"]).size().to_dict()
    ticker_caps = {t: sum(n for (tk, _), n in available.items() if tk == t) for t in tickers}
    ticker_caps = {t: n for t, n in ticker_caps.items() if n > 0}

    missing = [t for t in tickers if t not in ticker_caps]
    if missing:
        logger.warning("Tickers sin ningún par en el pool: %s", missing)

    if equal_quotas:
        # Cuotas iguales sobre los tickers del estudio; el faltante no se rellena.
        quota = _largest_remainder({t: 1 for t in tickers}, total, {t: total for t in tickers})
        per_ticker = {t: min(quota[t], ticker_caps.get(t, 0)) for t in tickers}
        short = {t: quota[t] - n for t, n in per_ticker.items() if n < quota[t]}
        if short:
            logger.warning("Tickers sin material para su cuota (faltante, no se rellena): %s", short)
    else:
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
    equal_quotas: bool = EQUAL_QUOTAS,
) -> pd.DataFrame:
    """
    Muestrea `total` pares estratificados por (ticker, año), con SEED fijo.

    Args:
        pool: Pool de build_pool().
        total: Pares a muestrear.
        min_per_ticker: Mínimo garantizado por ticker.
        tickers: Universo de tickers.
        seed: Semilla (SEED=42 del proyecto).
        equal_quotas: Cuotas iguales por ticker (ver allocate_by_cell).

    Returns:
        DataFrame barajado con los pares seleccionados (columnas _POOL_COLUMNS).
    """
    if pool.empty:
        return pool.copy()

    cells = allocate_by_cell(pool, total, min_per_ticker, tickers, equal_quotas)
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

def partition(
    sample: pd.DataFrame,
    batches: dict[str, tuple[int, bool]] = _BATCHES,
    batch_mins: dict[str, int] = _BATCH_MIN_PER_TICKER,
    tickers: list[str] = TICKERS,
) -> dict[str, pd.DataFrame]:
    """
    Corta el sample en los lotes l0 / doble / test / train, en ese orden.

    Todos los pares de un mismo post_id van al mismo lote, para que el mismo
    texto no quede a la vez en train y en test. Para cada lote, primero se
    cubre su mínimo por ticker (`batch_mins`) con los primeros posts del
    sample que lo satisfacen, y después se completa en el orden aleatorio
    del sample con los posts que caben; el post que no cabe pasa al lote
    siguiente, así los tamaños quedan exactos.

    Args:
        sample: Salida de sample_stratified() (ya barajada).
        batches: Definición lote → (tamaño, compartido).
        batch_mins: Mínimo de pares por ticker dentro de cada lote.
        tickers: Tickers sobre los que aplica el mínimo.

    Returns:
        Dict lote → DataFrame.

    Raises:
        ValueError: si el sample no alcanza para cubrir los lotes o sus mínimos.
    """
    needed = sum(size for size, _ in batches.values())
    if len(sample) < needed:
        raise ValueError(
            f"El sample tiene {len(sample)} pares pero los lotes requieren {needed}. "
            "Ampliá el pool (montá D:\\trading-data\\ o acumulá más scraping) o "
            "corré con --allow-partial para generar lotes proporcionales más chicos."
        )

    positions = sample.groupby("post_id").indices
    pending = [positions[post_id] for post_id in pd.unique(sample["post_id"])]
    ticker_of = sample["target_ticker"].to_numpy()

    result: dict[str, pd.DataFrame] = {}
    for name, (size, _) in batches.items():
        taken = [False] * len(pending)
        count = 0

        # 1) mínimo por ticker del lote
        minimum = batch_mins.get(name, 0)
        for ticker in tickers if minimum else []:
            have = sum(int((ticker_of[g] == ticker).sum()) for g, t in zip(pending, taken) if t)
            for i, group in enumerate(pending):
                if have >= minimum:
                    break
                if taken[i] or count + len(group) > size or not (ticker_of[group] == ticker).any():
                    continue
                taken[i] = True
                count += len(group)
                have += int((ticker_of[group] == ticker).sum())
            if have < minimum:
                raise ValueError(
                    f"Lote '{name}': solo {have} pares de {ticker} disponibles (mínimo {minimum})."
                )

        # 2) el resto, en el orden aleatorio del sample
        for i, group in enumerate(pending):
            if count >= size:
                break
            if not taken[i] and count + len(group) <= size:
                taken[i] = True
                count += len(group)
        if count < size:
            raise ValueError(
                f"No se pudo completar el lote '{name}' ({count}/{size}) sin partir posts."
            )

        chosen = [g for g, t in zip(pending, taken) if t]
        result[name] = sample.iloc[np.concatenate(chosen)].reset_index(drop=True)
        pending = [g for g, t in zip(pending, taken) if not t]
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


def _pool_sha256(pool: pd.DataFrame) -> str:
    """SHA-256 del pool, independiente del orden de filas (ordenado por la llave del par)."""
    canon = pool[["post_id", "target_ticker", "fecha", "clean_text"]].sort_values(["post_id", "target_ticker"])
    return hashlib.sha256(canon.to_csv(index=False).encode("utf-8")).hexdigest()


def write_manifest(
    sample: pd.DataFrame,
    assignments: dict[tuple[str, str], pd.DataFrame],
    written: dict[str, Path],
    pool: pd.DataFrame | None = None,
    seed: int = SEED,
    output_dir: Path = _MANUAL_LABELS_DIR,
    equal_quotas: bool = EQUAL_QUOTAS,
) -> Path:
    """
    Escribe manifest.json con la trazabilidad del muestreo (pre-registro).

    Args:
        sample: Pares muestreados.
        assignments: Reparto (lote, anotador) → filas.
        written: Archivos escritos, para hashear.
        pool: Pool del que se muestreó (para registrar su tamaño por ticker).
        seed: Semilla usada.
        output_dir: Directorio de salida.

    Returns:
        Ruta al manifest.json.
    """
    por_ticker_y_anio: dict[str, dict[str, int]] = {}
    for (ticker, year), count in sample.groupby(["target_ticker", "year"]).size().items():
        por_ticker_y_anio.setdefault(str(ticker), {})[str(year)] = int(count)

    ids_por_lote: dict[str, dict[str, list[list[str]]]] = {}
    por_lote_y_ticker: dict[str, dict[str, int]] = {}
    for batch in _BATCHES:
        frames = {a: assignments[(batch, a)] for a in ANNOTATORS if (batch, a) in assignments}
        ids_por_lote[batch] = {
            a: sorted([str(p), str(t)] for p, t in zip(f["post_id"], f["target_ticker"]))
            for a, f in frames.items()
        }
        union = pd.concat(frames.values()).drop_duplicates(["post_id", "target_ticker"])
        por_lote_y_ticker[batch] = {
            str(t): int(n) for t, n in union["target_ticker"].value_counts().sort_index().items()
        }

    manifest = {
        "generado": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "total_pares": int(len(sample)),
        "min_por_ticker_sample": max(MIN_PER_TICKER, sum(_BATCH_MIN_PER_TICKER.values())),
        "min_por_ticker_por_lote": _BATCH_MIN_PER_TICKER,
        "reparto_test_train": "aleatorio (SEED), no por fecha",
        "pool_sha256": _pool_sha256(pool) if pool is not None else None,
        "unidad_de_muestreo": "(post_id, target_ticker)",
        "particion_agrupada_por_post": True,
        "fuente_nyu": (
            "etiquetados: symbol_sentiments ⋈ messages por message_id (fecha real); "
            "sin_etiqueta: cashtag en messages/, fecha interpolada desde message_id "
            "(holdout 20%: 99,997% mismo día, error máx 1 día)"
        ),
        "reparto_por_ticker": (
            "cuotas iguales (faltantes se reportan, no se rellenan)"
            if equal_quotas else "proporcional a la disponibilidad"
        ),
        "pool_por_fuente": (
            {str(s): int(n) for s, n in pool["source"].value_counts().sort_index().items()}
            if pool is not None else None
        ),
        "pool_por_ticker": (
            {str(t): int(n) for t, n in pool["target_ticker"].value_counts().sort_index().items()}
            if pool is not None else None
        ),
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
        "por_lote_y_ticker": por_lote_y_ticker,
        "ids_por_lote": ids_por_lote,
        "archivos": {
            name: {"filas": int(len(pd.read_csv(path, dtype=str))), "sha256": _sha256(path)}
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
    parser.add_argument(
        "--include-live", action="store_true",
        help="Suma el stream reciente de StockTwits al pool (fuera del diseño pre-registrado).",
    )
    parser.add_argument("--max-live-per-symbol", type=int, default=60, help="Tope de mensajes por símbolo en vivo.")
    parser.add_argument(
        "--rebuild-nyu-cache", action="store_true",
        help="Rehace el join NYU (~55 GB de messages/) aunque exista el caché.",
    )
    parser.add_argument(
        "--proportional", action="store_true",
        help="Reparto proporcional a la disponibilidad en vez de cuotas iguales por ticker.",
    )
    parser.add_argument("--output-dir", type=Path, default=_MANUAL_LABELS_DIR, help="Directorio de salida.")
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Permite generar lotes proporcionalmente más chicos si el pool no alcanza "
             "(por defecto falla, para no romper el diseño pre-registrado).",
    )
    args = parser.parse_args()

    pool = build_pool(
        include_live=args.include_live,
        max_live_per_symbol=args.max_live_per_symbol,
        rebuild_nyu_cache=args.rebuild_nyu_cache,
    )
    if pool.empty:
        print(
            "Pool vacío: no se pudo leer NYU (¿D:\\trading-data\\ montado?)"
            + (" ni StockTwits en vivo (¿red disponible?)" if args.include_live else "")
            + ". No se generó ningún lote."
        )
        sys.exit(1)

    # El sample necesita al menos la suma de los mínimos por lote de cada ticker.
    sample_min = max(args.min_per_ticker, sum(_BATCH_MIN_PER_TICKER.values()))
    equal_quotas = not args.proportional
    sample = sample_stratified(
        pool, total=args.total, min_per_ticker=sample_min, equal_quotas=equal_quotas,
    )

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
    manifest_path = write_manifest(
        sample, assignments, written, pool=pool, output_dir=args.output_dir,
        equal_quotas=equal_quotas,
    )

    print(f"\n{len(written)} archivos escritos en {args.output_dir}")
    for name in sorted(written):
        print(f"  {name}")
    print(f"Manifiesto: {manifest_path}")
