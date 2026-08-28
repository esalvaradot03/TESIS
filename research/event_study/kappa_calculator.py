"""
Cohen's kappa entre dos CSVs de etiquetado manual del event study.

A diferencia de src/labeling/inter_rater.py (que opera sobre un store único
con granularidad post_id + columna 'labeler'), este módulo compara dos CSVs
independientes —uno por etiquetador— con granularidad (post_id,
target_ticker), que es la unidad de anotación del corpus del event study
(ver research/event_study/build_corpus.py).

El kappa primario se calcula excluyendo 'unusable': un desacuerdo sobre si
un post es etiquetable o no es una pregunta distinta de un desacuerdo de
polaridad (bullish/bearish/neutral), y mezclarlas infla artificialmente el
acuerdo. Reportar el kappa sobre LR (aleatorio); LU (uncertainty sampling)
es más difícil por construcción y no debe usarse como kappa primario.
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

logger = logging.getLogger(__name__)

_MERGE_KEYS: list[str] = ["post_id", "target_ticker"]

# Excluida del kappa principal: es un desacuerdo de "¿es etiquetable?", no
# de polaridad de sentimiento (ver docstring del módulo).
_EXCLUDED_FROM_KAPPA: set[str] = {"unusable"}


# ---------------------------------------------------------------------------
# Carga y merge
# ---------------------------------------------------------------------------

def load_annotations(path: Path) -> pd.DataFrame:
    """
    Carga un CSV de anotación de un etiquetador.

    Args:
        path: Ruta al CSV (schema de build_corpus.write_annotation_batch:
            post_id, target_ticker, tickers_detectados, clean_text, label,
            confianza, base, nota).

    Returns:
        DataFrame con dtype=str (los vacíos quedan como "").

    Raises:
        ValueError: si faltan post_id, target_ticker o label.
    """
    df = pd.read_csv(path, dtype=str).fillna("")
    required = {"post_id", "target_ticker", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Columnas faltantes en {path}: {missing}. Columnas encontradas: {list(df.columns)}"
        )
    return df


def merge_annotations(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    labeler_a: str,
    labeler_b: str,
) -> pd.DataFrame:
    """
    Une dos CSVs de anotación por (post_id, target_ticker).

    Args:
        df_a: Anotaciones del primer etiquetador (load_annotations()).
        df_b: Anotaciones del segundo etiquetador.
        labeler_a: Identificador del primer etiquetador (sufijo de columnas).
        labeler_b: Identificador del segundo etiquetador.

    Returns:
        DataFrame con columnas sufijadas _<labeler_a> / _<labeler_b> donde
        colisionan (label, confianza, base, nota, clean_text, tickers_detectados).
    """
    merged = pd.merge(
        df_a, df_b, on=_MERGE_KEYS, how="inner", suffixes=(f"_{labeler_a}", f"_{labeler_b}")
    )
    if merged.empty:
        logger.warning(
            "Sin pares (post_id, target_ticker) en común entre '%s' y '%s'.",
            labeler_a, labeler_b,
        )
    return merged


# ---------------------------------------------------------------------------
# Kappa
# ---------------------------------------------------------------------------

def compute_kappa(
    merged: pd.DataFrame,
    labeler_a: str,
    labeler_b: str,
    exclude_labels: set[str] = _EXCLUDED_FROM_KAPPA,
) -> dict:
    """
    Calcula Cohen's kappa sobre la columna label, excluyendo `exclude_labels`
    del cálculo principal.

    Args:
        merged: DataFrame de merge_annotations().
        labeler_a: Identificador del primer etiquetador.
        labeler_b: Identificador del segundo etiquetador.
        exclude_labels: Labels a excluir del cálculo principal (default: 'unusable').

    Returns:
        Dict con n_common (pares en común), n_scored (tras excluir), kappa,
        pct_agreement. kappa/pct_agreement son NaN si n_scored es 0.
    """
    label_a_col, label_b_col = f"label_{labeler_a}", f"label_{labeler_b}"
    n_common = len(merged)
    if n_common == 0:
        return {"n_common": 0, "n_scored": 0, "kappa": float("nan"), "pct_agreement": float("nan")}

    scored = merged[
        ~merged[label_a_col].isin(exclude_labels) & ~merged[label_b_col].isin(exclude_labels)
    ]
    if scored.empty:
        logger.warning("Todos los pares en común caen en %s para al menos un etiquetador.", exclude_labels)
        return {"n_common": n_common, "n_scored": 0, "kappa": float("nan"), "pct_agreement": float("nan")}

    kappa = float(cohen_kappa_score(scored[label_a_col], scored[label_b_col]))
    pct_agreement = float((scored[label_a_col] == scored[label_b_col]).mean())

    logger.info(
        "Kappa '%s' vs '%s': %d pares en común, %d puntuables (excl. %s). kappa=%.3f, acuerdo=%.1f%%.",
        labeler_a, labeler_b, n_common, len(scored), exclude_labels, kappa, pct_agreement * 100,
    )
    return {"n_common": n_common, "n_scored": len(scored), "kappa": kappa, "pct_agreement": pct_agreement}


# ---------------------------------------------------------------------------
# Desacuerdos para adjudicación
# ---------------------------------------------------------------------------

def list_disagreements(merged: pd.DataFrame, labeler_a: str, labeler_b: str) -> pd.DataFrame:
    """
    Lista los pares donde el label difiere entre los dos etiquetadores.

    Incluye la columna confianza de cada etiquetador (si está presente) y
    ordena por confianza combinada descendente: los desacuerdos donde ambos
    marcaron alta confianza son los más urgentes de adjudicar (indican una
    ambigüedad real del codebook, no un descuido).

    Args:
        merged: DataFrame de merge_annotations().
        labeler_a: Identificador del primer etiquetador.
        labeler_b: Identificador del segundo etiquetador.

    Returns:
        DataFrame con post_id, target_ticker, label_<a>, label_<b> y, si
        existen, confianza_<a>, confianza_<b>. Vacío si no hay desacuerdos.
    """
    label_a_col, label_b_col = f"label_{labeler_a}", f"label_{labeler_b}"
    disagreements = merged[merged[label_a_col] != merged[label_b_col]].copy()
    if disagreements.empty:
        logger.info("Sin desacuerdos entre '%s' y '%s'.", labeler_a, labeler_b)
        return disagreements

    cols = [*_MERGE_KEYS, label_a_col, label_b_col]
    conf_a_col, conf_b_col = f"confianza_{labeler_a}", f"confianza_{labeler_b}"
    if conf_a_col in disagreements.columns and conf_b_col in disagreements.columns:
        conf_sum = (
            pd.to_numeric(disagreements[conf_a_col], errors="coerce").fillna(0)
            + pd.to_numeric(disagreements[conf_b_col], errors="coerce").fillna(0)
        )
        disagreements = disagreements.assign(_conf_sum=conf_sum).sort_values(
            "_conf_sum", ascending=False
        ).drop(columns="_conf_sum")
        cols += [conf_a_col, conf_b_col]

    logger.info("%d desacuerdos de %d pares en común.", len(disagreements), len(merged))
    return disagreements[cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if len(sys.argv) < 3:
        print(
            "Uso: python -m research.event_study.kappa_calculator <csv_etiquetador_a> "
            "<csv_etiquetador_b> [nombre_a] [nombre_b]\n"
            "Ejemplo: python -m research.event_study.kappa_calculator "
            "data/manual_labels/l0_esteban.csv data/manual_labels/l0_camilo.csv esteban camilo"
        )
        sys.exit(1)

    path_a, path_b = Path(sys.argv[1]), Path(sys.argv[2])
    name_a = sys.argv[3] if len(sys.argv) > 3 else path_a.stem
    name_b = sys.argv[4] if len(sys.argv) > 4 else path_b.stem

    annotations_a = load_annotations(path_a)
    annotations_b = load_annotations(path_b)
    merged_df = merge_annotations(annotations_a, annotations_b, name_a, name_b)

    result = compute_kappa(merged_df, name_a, name_b)
    print(f"n_common={result['n_common']}  n_scored={result['n_scored']}  "
          f"kappa={result['kappa']:.3f}  acuerdo={result['pct_agreement'] * 100:.1f}%")

    disagreements_df = list_disagreements(merged_df, name_a, name_b)
    if not disagreements_df.empty:
        out_path = path_a.parent / f"desacuerdos_{name_a}_vs_{name_b}.csv"
        disagreements_df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"\n{len(disagreements_df)} desacuerdos guardados en → {out_path}")
        print(disagreements_df.head(10).to_string(index=False))
