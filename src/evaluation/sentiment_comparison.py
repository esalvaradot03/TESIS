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


def _merge_by_approach(
    golden_path: Path,
    score_paths: dict[str, Path],
) -> dict[str, pd.DataFrame]:
    """
    Golden set unido con las predicciones de cada enfoque, una sola vez.

    Punto de entrada compartido de compare_approaches() y
    confusion_matrices(): ambas derivan sus métricas del mismo merge
    (golden × predicciones por post_id), así que calcularlo una sola vez y
    pasarlo a las dos evita releer y re-unir los mismos CSV dos veces (ver
    el bloque CLI, que llama a ambas funciones).

    Args:
        golden_path: Ruta al store de labels manuales.
        score_paths: Dict {nombre_enfoque: ruta_csv_scores}.

    Returns:
        Dict {nombre_enfoque: DataFrame con post_id, gold_label, pred_label}.
        Enfoques sin CSV de scores, o sin posts en común con el golden set,
        se omiten (con warning).
    """
    golden = _load_golden_labels(golden_path)

    merged_by_approach: dict[str, pd.DataFrame] = {}
    for name, path in score_paths.items():
        if not path.exists():
            logger.warning("Scores no encontrados para '%s' en %s; se omite.", name, path)
            continue

        preds = _load_predictions(path)
        merged = pd.merge(golden, preds, on="post_id", how="inner")
        if merged.empty:
            logger.warning("Sin posts en común entre el golden set y '%s'.", name)
            continue

        merged_by_approach[name] = merged
    return merged_by_approach


# ---------------------------------------------------------------------------
# Comparación
# ---------------------------------------------------------------------------

def compare_approaches(
    golden_path: Path = LABELED_STORE_FILE,
    score_paths: dict[str, Path] = _DEFAULT_SCORE_PATHS,
    merged_by_approach: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    Compara accuracy y macro-F1 de cada enfoque contra el golden set.

    Enfoques cuyo CSV de scores no exista se omiten con un warning (por
    ejemplo, si todavía no se corrió finbert_finetuned_scorer.py).

    Args:
        golden_path: Ruta al store de labels manuales. Ignorado si se pasa
            merged_by_approach.
        score_paths: Dict {nombre_enfoque: ruta_csv_scores}. Ignorado si se
            pasa merged_by_approach.
        merged_by_approach: Resultado ya calculado de _merge_by_approach(),
            para reusarlo entre esta función y confusion_matrices() sin
            releer/re-unir los CSV (ver bloque CLI). Si es None (default),
            se calcula internamente.

    Returns:
        DataFrame indexado por nombre de enfoque, columnas n_posts, accuracy, macro_f1.
    """
    if merged_by_approach is None:
        merged_by_approach = _merge_by_approach(golden_path, score_paths)

    rows: list[dict] = []
    for name, merged in merged_by_approach.items():
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
    merged_by_approach: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Calcula la matriz de confusión de cada enfoque contra el golden set.

    Args:
        golden_path: Ruta al store de labels manuales. Ignorado si se pasa
            merged_by_approach.
        score_paths: Dict {nombre_enfoque: ruta_csv_scores}. Ignorado si se
            pasa merged_by_approach.
        merged_by_approach: Resultado ya calculado de _merge_by_approach()
            (ver compare_approaches). Si es None (default), se calcula
            internamente.

    Returns:
        Dict {nombre_enfoque: DataFrame [3x3]} indexado y con columnas en
        _LABELS_ORDER (filas = gold, columnas = predicho).
    """
    if merged_by_approach is None:
        merged_by_approach = _merge_by_approach(golden_path, score_paths)

    matrices: dict[str, pd.DataFrame] = {}
    for name, merged in merged_by_approach.items():
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

    merged = _merge_by_approach(LABELED_STORE_FILE, _DEFAULT_SCORE_PATHS)

    summary = compare_approaches(merged_by_approach=merged)
    print("\n=== Accuracy / Macro-F1 por enfoque ===")
    print(summary.to_string())

    matrices = confusion_matrices(merged_by_approach=merged)
    for approach_name, matrix in matrices.items():
        print(f"\n=== Matriz de confusión: {approach_name} (filas=gold, cols=pred) ===")
        print(matrix.to_string())
