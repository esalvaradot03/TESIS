"""
Comparación de los tres enfoques de sentimiento (FinBERT baseline, FinBERT
con linear probing, VADER léxico) sobre el golden set de labels manuales.

El golden set es el store de labeled_store.py. Si un mismo post_id fue
etiquetado por más de un labeler con labels distintas, se registra como
desacuerdo (ver inter_rater.py para cuantificarlo) y se usa la etiqueta más
reciente como referencia única para las métricas de este módulo.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from config.settings import LABELED_STORE_FILE, PROCESSED_DIR
from src.labeling.labeled_store import load as load_labels

logger = logging.getLogger(__name__)

_DEFAULT_SCORE_PATHS: dict[str, Path] = {
    "finbert_base": PROCESSED_DIR / "sentiment_scores.csv",
    "finbert_finetuned": PROCESSED_DIR / "sentiment_scores_finetuned.csv",
    "lexicon_vader": PROCESSED_DIR / "sentiment_scores_lexicon.csv",
}

_LABELS_ORDER: list[str] = ["positive", "negative", "neutral"]


# ---------------------------------------------------------------------------
# Carga y normalización
# ---------------------------------------------------------------------------

def _load_golden_labels(golden_path: Path) -> pd.DataFrame:
    """
    Carga el store de labels manuales y lo reduce a un label por post_id.

    Args:
        golden_path: Ruta al CSV acumulativo de labels manuales.

    Returns:
        DataFrame con columnas post_id, gold_label.

    Raises:
        ValueError: si el store está vacío.
    """
    labels = load_labels(golden_path)
    if labels.empty:
        raise ValueError(
            f"No hay labels manuales en {golden_path}. Etiquetá posts primero "
            "(ver src/labeling/uncertainty_sampling.py)."
        )

    n_labelers = labels.groupby("post_id")["labeler"].nunique()
    contested_ids = n_labelers[n_labelers > 1].index
    distinct_labels = labels[labels["post_id"].isin(contested_ids)].groupby("post_id")["label"].nunique()
    disagreements = int((distinct_labels > 1).sum())
    if disagreements:
        logger.warning(
            "%d posts tienen labels distintas entre etiquetadores; se usa la más "
            "reciente. Corré src/labeling/inter_rater.py para cuantificar el acuerdo.",
            disagreements,
        )

    golden = labels.sort_values("timestamp").drop_duplicates(subset=["post_id"], keep="last")
    return golden[["post_id", "label"]].rename(columns={"label": "gold_label"})


def _load_predictions(scores_path: Path) -> pd.DataFrame:
    """
    Carga un CSV de sentiment_scores*.csv y lo reduce a un label por post_id.

    Args:
        scores_path: Ruta a sentiment_scores.csv, sentiment_scores_finetuned.csv
            o sentiment_scores_lexicon.csv.

    Returns:
        DataFrame con columnas post_id, pred_label.
    """
    scores = pd.read_csv(scores_path, dtype=str).fillna("")
    required = {"post_id", "sentiment_label"}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(
            f"Columnas faltantes en {scores_path}: {missing}. "
            f"Columnas encontradas: {list(scores.columns)}"
        )

    before = len(scores)
    scores = scores.drop_duplicates(subset=["post_id"], keep="first")
    if before - len(scores):
        logger.info(
            "%d filas colapsadas por post_id duplicado (multi-ticker) en %s.",
            before - len(scores), scores_path,
        )

    return scores[["post_id", "sentiment_label"]].rename(columns={"sentiment_label": "pred_label"})


# ---------------------------------------------------------------------------
# Comparación
# ---------------------------------------------------------------------------

def compare_approaches(
    golden_path: Path = LABELED_STORE_FILE,
    score_paths: dict[str, Path] = _DEFAULT_SCORE_PATHS,
) -> pd.DataFrame:
    """
    Compara accuracy y macro-F1 de cada enfoque contra el golden set.

    Enfoques cuyo CSV de scores no exista se omiten con un warning (por
    ejemplo, si todavía no se corrió finbert_finetuned_scorer.py).

    Args:
        golden_path: Ruta al store de labels manuales.
        score_paths: Dict {nombre_enfoque: ruta_csv_scores}.

    Returns:
        DataFrame indexado por nombre de enfoque, columnas n_posts, accuracy, macro_f1.
    """
    golden = _load_golden_labels(golden_path)

    rows: list[dict] = []
    for name, path in score_paths.items():
        if not path.exists():
            logger.warning("Scores no encontrados para '%s' en %s; se omite.", name, path)
            continue

        preds = _load_predictions(path)
        merged = pd.merge(golden, preds, on="post_id", how="inner")
        if merged.empty:
            logger.warning("Sin posts en común entre el golden set y '%s'.", name)
            continue

        accuracy = accuracy_score(merged["gold_label"], merged["pred_label"])
        macro_f1 = f1_score(
            merged["gold_label"], merged["pred_label"],
            labels=_LABELS_ORDER, average="macro", zero_division=0,
        )
        rows.append({
            "approach": name,
            "n_posts": len(merged),
            "accuracy": accuracy,
            "macro_f1": macro_f1,
        })

    result = pd.DataFrame(rows).set_index("approach") if rows else pd.DataFrame(
        columns=["n_posts", "accuracy", "macro_f1"]
    )
    logger.info("Comparación de enfoques:\n%s", result.to_string())
    return result


def confusion_matrices(
    golden_path: Path = LABELED_STORE_FILE,
    score_paths: dict[str, Path] = _DEFAULT_SCORE_PATHS,
) -> dict[str, pd.DataFrame]:
    """
    Calcula la matriz de confusión de cada enfoque contra el golden set.

    Args:
        golden_path: Ruta al store de labels manuales.
        score_paths: Dict {nombre_enfoque: ruta_csv_scores}.

    Returns:
        Dict {nombre_enfoque: DataFrame [3x3]} indexado y con columnas en
        _LABELS_ORDER (filas = gold, columnas = predicho).
    """
    golden = _load_golden_labels(golden_path)

    matrices: dict[str, pd.DataFrame] = {}
    for name, path in score_paths.items():
        if not path.exists():
            continue

        preds = _load_predictions(path)
        merged = pd.merge(golden, preds, on="post_id", how="inner")
        if merged.empty:
            continue

        cm = confusion_matrix(merged["gold_label"], merged["pred_label"], labels=_LABELS_ORDER)
        matrices[name] = pd.DataFrame(cm, index=_LABELS_ORDER, columns=_LABELS_ORDER)

    return matrices


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    summary = compare_approaches()
    print("\n=== Accuracy / Macro-F1 por enfoque ===")
    print(summary.to_string())

    matrices = confusion_matrices()
    for approach_name, matrix in matrices.items():
        print(f"\n=== Matriz de confusión: {approach_name} (filas=gold, cols=pred) ===")
        print(matrix.to_string())
