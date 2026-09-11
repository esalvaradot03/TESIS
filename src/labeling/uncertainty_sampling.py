"""
Active learning por muestreo de incertidumbre (entropía) para priorizar qué
posts etiquetar a mano.

Flujo:
  1. Lee sentiment_scores.csv (salida del scorer baseline FinBERT).
  2. Deduplica por post_id (el CSV está explotado por (post_id, ticker) y
     la entropía es una propiedad del post, no del ticker).
  3. Excluye posts que ya tienen label manual en LABELED_STORE_FILE.
  4. Calcula la entropía de Shannon sobre [prob_positive, prob_negative,
     prob_neutral] por post y devuelve los N posts con entropía más alta
     (los más "inciertos" para el modelo, y por lo tanto los más
     informativos para etiquetar).

Nota: sentiment_scores.csv no trae el texto del post (solo post_id). Para
poder mostrarle el texto a quien etiqueta hay que pasar `text_source_path`
apuntando al CSV limpio del preprocessor (columna clean_text).
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import LABELED_STORE_FILE, PROCESSED_DIR, UNCERTAINTY_SAMPLE_SIZE
from src.labeling.labeled_store import load as load_labels

logger = logging.getLogger(__name__)

_PROB_COLUMNS: list[str] = ["prob_positive", "prob_negative", "prob_neutral"]
_SAMPLE_FILE = PROCESSED_DIR / "uncertainty_sample.csv"


# ---------------------------------------------------------------------------
# Entropía
# ---------------------------------------------------------------------------

def compute_entropy(probs: np.ndarray) -> np.ndarray:
    """
    Calcula la entropía de Shannon (en nats) fila por fila.

    Args:
        probs: Array [n, k] de distribuciones de probabilidad (cada fila
            suma ~1). En este módulo k=3 (positive, negative, neutral).

    Returns:
        Array [n] con la entropía de cada fila. 0 si la distribución está
        concentrada en una sola clase; máxima cuando es uniforme.
    """
    safe_log = np.where(probs > 0, np.log(np.clip(probs, 1e-12, None)), 0.0)
    return -np.sum(probs * safe_log, axis=1)


# ---------------------------------------------------------------------------
# Selección de posts
# ---------------------------------------------------------------------------

def select_uncertain_posts(
    scores_path: Path,
    text_source_path: Path | None = None,
    n: int = UNCERTAINTY_SAMPLE_SIZE,
    labeled_store_path: Path = LABELED_STORE_FILE,
) -> pd.DataFrame:
    """
    Selecciona los N posts con mayor entropía de predicción para etiquetar.

    Args:
        scores_path: Ruta a un sentiment_scores*.csv (post_id, prob_positive,
            prob_negative, prob_neutral, ...).
        text_source_path: CSV limpio del preprocessor con columnas post_id y
            clean_text. Si se provee, se une para incluir el texto a mostrar
            al etiquetador. Si es None, el resultado no trae texto.
        n: Cantidad de posts a devolver (los de mayor entropía).
        labeled_store_path: Ruta al store de labels manuales, para excluir
            posts ya etiquetados.

    Returns:
        DataFrame ordenado por entropía descendente con columnas post_id,
        entropy, prob_positive, prob_negative, prob_neutral y, si se pasó
        text_source_path, clean_text.
    """
    logger.info("Cargando scores desde %s...", scores_path)
    scores = pd.read_csv(scores_path, dtype=str).fillna("")

    required = {"post_id", *_PROB_COLUMNS}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(
            f"Columnas faltantes en {scores_path}: {missing}. "
            f"Columnas encontradas: {list(scores.columns)}"
        )

    for col in _PROB_COLUMNS:
        scores[col] = pd.to_numeric(scores[col], errors="coerce").fillna(0.0)

    before = len(scores)
    scores = scores.drop_duplicates(subset=["post_id"], keep="first").reset_index(drop=True)
    if before - len(scores):
        logger.info(
            "%d filas colapsadas por post_id duplicado (multi-ticker).",
            before - len(scores),
        )

    labeled = load_labels(labeled_store_path)
    labeled_ids = set(labeled["post_id"])
    if labeled_ids:
        before_labeled_filter = len(scores)
        scores = scores[~scores["post_id"].isin(labeled_ids)]
        excluded = before_labeled_filter - len(scores)
        if excluded:
            logger.info(
                "%d posts excluidos por ya tener label manual (de %d etiquetados en total).",
                excluded, len(labeled_ids),
            )

    scores["entropy"] = compute_entropy(scores[_PROB_COLUMNS].to_numpy())
    scores = scores.sort_values("entropy", ascending=False)

    sample = scores.head(n)[["post_id", "entropy", *_PROB_COLUMNS]].reset_index(drop=True)

    if text_source_path is not None:
        text_df = pd.read_csv(text_source_path, dtype=str, usecols=["post_id", "clean_text"])
        sample = pd.merge(sample, text_df, on="post_id", how="left")
        missing_text = sample["clean_text"].isna().sum()
        if missing_text:
            logger.warning(
                "%d posts del sample no tienen match de texto en %s.",
                missing_text, text_source_path,
            )
    else:
        logger.warning(
            "No se proveyó text_source_path: el sample no incluye texto. "
            "Agregalo desde el CSV *_clean.csv del preprocessor antes de etiquetar."
        )

    logger.info(
        "Sample de incertidumbre: %d posts seleccionados (de %d candidatos).",
        len(sample), len(scores),
    )
    return sample


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if len(sys.argv) < 2:
        print(
            "Uso: python -m src.labeling.uncertainty_sampling <sentiment_scores.csv> "
            "[text_source_clean.csv] [n]\n"
            "Ejemplo: python -m src.labeling.uncertainty_sampling "
            "data/processed/sentiment_scores.csv data/processed/stocktwits_historical_clean.csv 100"
        )
        sys.exit(1)

    scores_csv = Path(sys.argv[1])
    text_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    sample_n = int(sys.argv[3]) if len(sys.argv) > 3 else UNCERTAINTY_SAMPLE_SIZE

    result = select_uncertain_posts(scores_csv, text_source_path=text_csv, n=sample_n)
    _SAMPLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(_SAMPLE_FILE, index=False, encoding="utf-8")
    print(f"Sample de {len(result)} posts guardado en → {_SAMPLE_FILE}")
    print(result.head(10).to_string(index=False))
