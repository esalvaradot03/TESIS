"""
Persistencia de labels manuales de sentimiento (subsistema de active learning).

Schema del CSV acumulativo (data/labeled/labeled_posts.csv):
  post_id, text, label (positive|negative|neutral), labeler, timestamp

A diferencia de sentiment_scores.csv (que es solo append por caché de post_id),
este store admite re-etiquetado: si el mismo (post_id, labeler) se etiqueta de
nuevo, se conserva la versión con el timestamp más reciente.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config.settings import LABELED_STORE_FILE

logger = logging.getLogger(__name__)

_COLUMNS: list[str] = ["post_id", "text", "label", "labeler", "timestamp"]
_VALID_LABELS: set[str] = {"positive", "negative", "neutral"}


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def load(path: Path = LABELED_STORE_FILE) -> pd.DataFrame:
    """
    Carga el CSV acumulativo de labels manuales.

    Args:
        path: Ruta al CSV de labels. Por defecto LABELED_STORE_FILE.

    Returns:
        DataFrame con columnas _COLUMNS. Vacío (0 filas) si el archivo
        todavía no existe, para que los llamadores no tengan que chequear
        existencia por separado.
    """
    if not path.exists():
        logger.info("No existe %s todavía. Devolviendo store vacío.", path)
        return pd.DataFrame(columns=_COLUMNS)

    df = pd.read_csv(path, dtype=str).fillna("")
    logger.info("Store de labels cargado: %d filas desde %s.", len(df), path)
    return df


# ---------------------------------------------------------------------------
# Escritura
# ---------------------------------------------------------------------------

def append_labels(labels: pd.DataFrame, path: Path = LABELED_STORE_FILE) -> Path:
    """
    Agrega nuevas labels manuales al store, con soporte de re-etiquetado.

    Valida que 'label' solo contenga valores en _VALID_LABELS. Si falta la
    columna 'timestamp' en `labels`, se completa con la hora actual (UTC).
    El store completo se reescribe (no append en modo texto) porque el
    merge con el histórico requiere deduplicar por (post_id, labeler)
    quedándose con el timestamp más reciente ante un re-etiquetado.

    Args:
        labels: DataFrame nuevo con al menos post_id, text, label, labeler.
        path: Ruta al CSV acumulativo de labels.

    Returns:
        Ruta al CSV actualizado.

    Raises:
        ValueError: si faltan columnas requeridas o hay labels fuera de
            _VALID_LABELS.
    """
    required = {"post_id", "text", "label", "labeler"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(
            f"Columnas faltantes en labels: {missing}. "
            f"Columnas encontradas: {list(labels.columns)}"
        )

    new_labels = labels.copy()
    if "timestamp" not in new_labels.columns:
        new_labels["timestamp"] = datetime.now(timezone.utc).isoformat()
    new_labels = new_labels[_COLUMNS].astype(str)

    invalid = new_labels[~new_labels["label"].isin(_VALID_LABELS)]
    if not invalid.empty:
        raise ValueError(
            f"{len(invalid)} filas con label fuera de {_VALID_LABELS}: "
            f"{invalid['label'].unique().tolist()}"
        )

    existing = load(path)
    combined = pd.concat([existing, new_labels], ignore_index=True)

    # Re-etiquetado: ante duplicados de (post_id, labeler), gana el timestamp
    # más reciente.
    combined = combined.sort_values("timestamp")
    before = len(combined)
    combined = combined.drop_duplicates(subset=["post_id", "labeler"], keep="last")
    resolved = before - len(combined)
    if resolved:
        logger.info(
            "%d re-etiquetados resueltos (se conservó el timestamp más reciente).",
            resolved,
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    combined = combined.sort_values(["post_id", "labeler"]).reset_index(drop=True)
    combined.to_csv(path, index=False, encoding="utf-8")

    logger.info(
        "Store actualizado: %d labels nuevas, %d filas totales en %s.",
        len(new_labels), len(combined), path,
    )
    return path


# ---------------------------------------------------------------------------
# Unión con scores de un scorer (base, finetuned o lexicon)
# ---------------------------------------------------------------------------

def merge_with_scores(labels: pd.DataFrame, scores_path: Path) -> pd.DataFrame:
    """
    Une labels manuales con las predicciones de un sentiment_scores*.csv.

    Los CSV de scores están explotados por (post_id, ticker), así que antes
    de unir se deduplica por post_id quedándose con la primera ocurrencia
    (el label manual es a nivel de post, no de ticker).

    Args:
        labels: DataFrame de labels manuales (schema de _COLUMNS).
        scores_path: Ruta a un CSV de scores (sentiment_scores.csv,
            sentiment_scores_finetuned.csv o sentiment_scores_lexicon.csv).

    Returns:
        DataFrame con una fila por post_id presente en ambas fuentes,
        columnas de labels sufijadas '_label' y de scores sufijadas '_post'
        donde hay colisión de nombres (timestamp).
    """
    scores = pd.read_csv(scores_path, dtype=str).fillna("")
    before = len(scores)
    scores = scores.drop_duplicates(subset=["post_id"], keep="first")
    if before - len(scores):
        logger.info(
            "%d filas colapsadas por post_id duplicado (multi-ticker) en %s.",
            before - len(scores), scores_path,
        )

    merged = pd.merge(
        labels, scores, on="post_id", how="inner", suffixes=("_label", "_post")
    )
    logger.info(
        "merge_with_scores: %d posts etiquetados, %d con match en %s.",
        len(labels), len(merged), scores_path,
    )
    return merged
