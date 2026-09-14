"""
Comparación de los tres enfoques de sentimiento sobre el corpus etiquetado.

Enfoques: FinBERT off-the-shelf, linear probing y lexicón VADER. Se añaden dos
referencias obligatorias para leer cualquier número:

  - **Baseline mayoritaria:** responder siempre la clase más frecuente del
    train. Un enfoque que no le gane no está aportando nada.
  - **Placebo** (regla 2 del pre-registro): la misma cabeza entrenada con las
    etiquetas del train permutadas. Si el modelo real no se separa del placebo,
    el resultado es del montaje, no de la señal.

Además del test completo, cada métrica se reporta por mitad de anotador. Los
lotes `test` y `train` son disjuntos entre anotadores, así que evaluar al probe
de un anotador sobre la mitad del otro mide **transferencia de criterio**.

Uso:
    python -m research.event_study.evaluate_approaches
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from config.settings import SEED
from research.event_study.linear_probe import (
    ANOTADORES,
    CLASES,
    cargar_split,
    entrenar,
    predecir,
)
from src.sentiment.finbert_scorer import FinBERTScorer
from src.sentiment.lexicon_scorer import LexiconScorer

logger = logging.getLogger(__name__)

_SALIDA = Path("research/event_study/comparacion_enfoques.csv")


def metricas(nombre: str, gold: pd.DataFrame, predicciones: list[str]) -> list[dict]:
    """
    Calcula accuracy y macro-F1 sobre el test completo y sobre cada mitad.

    Args:
        nombre: Nombre del enfoque, tal como va en el reporte.
        gold: DataFrame de cargar_split("test"), con columnas `clase` y `anotador`.
        predicciones: Clases predichas, alineadas posicionalmente con `gold`.

    Returns:
        Una fila por subconjunto evaluado.
    """
    filas: list[dict] = []
    subconjuntos = [("test completo", gold)]
    subconjuntos += [(f"test {quien}", gold[gold["anotador"] == quien]) for quien in ANOTADORES]

    for etiqueta, subset in subconjuntos:
        if subset.empty:
            continue
        reales = subset["clase"].tolist()
        predichas = [predicciones[i] for i in subset.index]
        filas.append({
            "enfoque": nombre,
            "evaluado_en": etiqueta,
            "n": len(reales),
            "accuracy": round(float(accuracy_score(reales, predichas)), 4),
            "macro_f1": round(float(f1_score(reales, predichas, labels=CLASES,
                                             average="macro", zero_division=0)), 4),
        })
    return filas


def comparar(salida: Path = _SALIDA, seed: int = SEED) -> pd.DataFrame:
    """
    Corre los tres enfoques más baseline y placebos, y escribe el CSV de resultados.

    Args:
        salida: Ruta del CSV de resultados.
        seed: Semilla para la permutación del placebo.

    Returns:
        DataFrame con una fila por (enfoque, subconjunto).
    """
    train, test = cargar_split("train"), cargar_split("test")
    logger.info("train: %d pares | test: %d pares", len(train), len(test))

    textos_test = test["clean_text"].tolist()
    resultados: list[dict] = []

    resultados += metricas(
        "FinBERT off-the-shelf",
        test,
        [r["sentiment_label"] for r in FinBERTScorer().score_texts(textos_test)],
    )
    resultados += metricas(
        "VADER",
        test,
        [r["sentiment_label"] for r in LexiconScorer().score_texts(textos_test)],
    )

    mayoritaria = train["clase"].value_counts().idxmax()
    resultados += metricas(f"Baseline mayoritaria ({mayoritaria})", test, [mayoritaria] * len(test))

    rng = np.random.default_rng(seed)
    variantes = {f"probe (train={quien})": train[train["anotador"] == quien] for quien in ANOTADORES}
    variantes["probe (train=ambos)"] = train

    for nombre, subset in variantes.items():
        textos = subset["clean_text"].tolist()
        for sufijo, clases in (
            (nombre, subset["clase"].tolist()),
            (f"PLACEBO {nombre}", rng.permutation(subset["clase"].to_numpy()).tolist()),
        ):
            trainer, _ = entrenar(textos, clases)
            predicciones = predecir(trainer, textos_test)
            resultados += metricas(sufijo, test, predicciones)
            if sufijo == nombre:
                matriz = pd.DataFrame(
                    confusion_matrix(test["clase"], predicciones, labels=CLASES),
                    index=CLASES, columns=CLASES,
                )
                logger.info("Matriz de %s sobre el test completo:\n%s", nombre, matriz.to_string())

    tabla = pd.DataFrame(resultados)
    salida.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(salida, index=False, encoding="utf-8")
    logger.info("Resultados escritos en %s", salida)
    return tabla


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

    parser = argparse.ArgumentParser(description="Compara los 3 enfoques de sentimiento.")
    parser.add_argument("--salida", type=Path, default=_SALIDA, help="CSV de resultados.")
    args = parser.parse_args()

    tabla = comparar(salida=args.salida)
    print("\n=== macro-F1 (azar en 3 clases: accuracy 0.33) ===")
    print(tabla.pivot(index="enfoque", columns="evaluado_en", values="macro_f1").to_string())
    print("\n=== accuracy ===")
    print(tabla.pivot(index="enfoque", columns="evaluado_en", values="accuracy").to_string())
    print(f"\nGuardado en {args.salida}")
