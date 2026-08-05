"""
Acuerdo entre etiquetadores (inter-rater agreement) para el store de labels
manuales, usando Cohen's kappa cuando dos personas anotan los mismos posts.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from config.settings import LABELED_STORE_FILE
from src.labeling.labeled_store import load as load_labels

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cálculo de kappa
# ---------------------------------------------------------------------------

def cohen_kappa(labels_a: pd.Series, labels_b: pd.Series) -> float:
    """
    Calcula Cohen's kappa entre dos series de labels alineadas por posición.

    Args:
        labels_a: Labels del primer etiquetador.
        labels_b: Labels del segundo etiquetador, mismo orden y longitud.

    Returns:
        Kappa en [-1, 1]. NaN si las series están vacías.

    Raises:
        ValueError: si las series tienen longitudes distintas.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError(
            f"labels_a y labels_b deben tener la misma longitud "
            f"({len(labels_a)} != {len(labels_b)})"
        )
    if len(labels_a) == 0:
        return float("nan")
    return float(cohen_kappa_score(labels_a, labels_b))


# ---------------------------------------------------------------------------
# Acuerdo sobre el store completo
# ---------------------------------------------------------------------------

def compute_agreement(
    labeler_a: str,
    labeler_b: str,
    store_path: Path = LABELED_STORE_FILE,
) -> dict:
    """
    Calcula el acuerdo entre dos etiquetadores sobre los posts que ambos
    etiquetaron.

    Args:
        labeler_a: Identificador del primer etiquetador (columna 'labeler').
        labeler_b: Identificador del segundo etiquetador.
        store_path: Ruta al CSV acumulativo de labels manuales.

    Returns:
        Dict con n_common (posts etiquetados por ambos), kappa (Cohen's
        kappa) y pct_agreement (fracción de coincidencia exacta). kappa y
        pct_agreement son NaN si n_common es 0.
    """
    store = load_labels(store_path)

    labels_a = store[store["labeler"] == labeler_a][["post_id", "label"]].rename(
        columns={"label": "label_a"}
    )
    labels_b = store[store["labeler"] == labeler_b][["post_id", "label"]].rename(
        columns={"label": "label_b"}
    )
    common = pd.merge(labels_a, labels_b, on="post_id", how="inner")

    if common.empty:
        logger.warning(
            "Sin posts en común entre '%s' y '%s' en %s.",
            labeler_a, labeler_b, store_path,
        )
        return {"n_common": 0, "kappa": float("nan"), "pct_agreement": float("nan")}

    kappa = cohen_kappa(common["label_a"], common["label_b"])
    pct_agreement = float((common["label_a"] == common["label_b"]).mean())

    logger.info(
        "Acuerdo '%s' vs '%s': %d posts en común, kappa=%.3f, acuerdo exacto=%.1f%%.",
        labeler_a, labeler_b, len(common), kappa, pct_agreement * 100,
    )
    return {
        "n_common": len(common),
        "kappa": kappa,
        "pct_agreement": pct_agreement,
    }


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

    if len(sys.argv) < 3:
        print(
            "Uso: python -m src.labeling.inter_rater <labeler_a> <labeler_b> [store_path]\n"
            "Ejemplo: python -m src.labeling.inter_rater camilo ana"
        )
        sys.exit(1)

    store_csv = Path(sys.argv[3]) if len(sys.argv) > 3 else LABELED_STORE_FILE
    result = compute_agreement(sys.argv[1], sys.argv[2], store_path=store_csv)
    print(result)
