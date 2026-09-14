"""
¿Etiquetar más mejora? Curva de aprendizaje de la cabeza lineal.

Entrena con fracciones crecientes del corpus de train (muestreo estratificado
por clase) y evalúa siempre sobre el mismo test. Si la curva sigue subiendo al
100%, etiquetar más rinde; si se aplana, el cuello de botella está en otra
parte: desacuerdo entre anotadores, calidad del texto o capacidad del modelo.

Resultado al 2026-09-13 (363 pares de train, 266 de test, 3 semillas):
la curva sube de 91 a 272 pares (+0.028 macro-F1) y ahí se aplana; el tramo
272→363 da −0.005, tres veces menor que la desviación entre semillas (±0.016).

Los embeddings del backbone congelado se calculan UNA vez y se reutilizan en
todos los puntos de la curva: lo caro es el backbone, no la cabeza.

Uso:
    python -m research.event_study.curva_aprendizaje
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from research.event_study.linear_probe import CLASES, HPARAMS_PREREGISTRADOS, cargar_split
from src.sentiment.finbert_finetune import FinBERTHeadTrainer, _IDX_TO_LABEL

logger = logging.getLogger(__name__)

FRACCIONES: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
SEMILLAS: tuple[int, ...] = (42, 43, 44)
_SALIDA = Path("research/event_study/curva_aprendizaje.csv")


def correr(
    fracciones: tuple[float, ...] = FRACCIONES,
    semillas: tuple[int, ...] = SEMILLAS,
    salida: Path = _SALIDA,
) -> pd.DataFrame:
    """
    Corre la curva de aprendizaje y escribe el CSV con un punto por (fracción, semilla).

    Args:
        fracciones: Fracciones del train a usar.
        semillas: Semillas del muestreo y del entrenamiento.
        salida: CSV de resultados.

    Returns:
        DataFrame con accuracy y macro-F1 de cada punto.
    """
    train, test = cargar_split("train"), cargar_split("test")
    logger.info("train: %d pares | test: %d pares", len(train), len(test))

    base = FinBERTHeadTrainer(hparams=HPARAMS_PREREGISTRADOS)
    embeddings_test = base._embed_all(test["clean_text"].tolist())
    etiquetas_train = train["clase"].to_numpy()
    etiquetas_test = test["clase"].tolist()

    filas: list[dict] = []
    for fraccion in fracciones:
        for semilla in semillas:
            n = int(round(len(train) * fraccion))
            if fraccion >= 1.0:
                indices = np.arange(len(train))
            else:
                indices, _ = train_test_split(
                    np.arange(len(train)), train_size=n, random_state=semilla,
                    stratify=etiquetas_train,
                )

            hparams = type(HPARAMS_PREREGISTRADOS)(
                **{**HPARAMS_PREREGISTRADOS.__dict__, "seed": semilla}
            )
            trainer = FinBERTHeadTrainer(hparams=hparams)
            trainer.train(
                [train["clean_text"].iloc[i] for i in indices],
                etiquetas_train[indices].tolist(),
            )
            trainer.head.eval()
            with torch.no_grad():
                predichas = [
                    _IDX_TO_LABEL[int(i)]
                    for i in trainer.head(embeddings_test).argmax(dim=-1).cpu().numpy()
                ]

            filas.append({
                "fraccion": fraccion,
                "n_train": len(indices),
                "semilla": semilla,
                "accuracy": accuracy_score(etiquetas_test, predichas),
                "macro_f1": f1_score(etiquetas_test, predichas, labels=CLASES,
                                     average="macro", zero_division=0),
            })
            logger.info("%3d%% (n=%3d, seed=%d): accuracy %.3f | macro-F1 %.3f",
                        int(fraccion * 100), len(indices), semilla,
                        filas[-1]["accuracy"], filas[-1]["macro_f1"])

    tabla = pd.DataFrame(filas)
    salida.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(salida, index=False, encoding="utf-8")
    return tabla


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    tabla = correr()
    resumen = tabla.groupby(["fraccion", "n_train"]).agg(
        accuracy_media=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
        macro_f1_media=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std"),
    ).round(4)
    print("\n=== CURVA DE APRENDIZAJE (promedio de semillas) ===")
    print(resumen.to_string())

    ganancia = resumen["macro_f1_media"].iloc[-1] - resumen["macro_f1_media"].iloc[-2]
    print(f"\nGanancia del último tramo: {ganancia:+.4f} macro-F1 "
          f"(desviación entre semillas: ±{resumen['macro_f1_sd'].iloc[-1]:.4f})")
    print("Si la ganancia es menor que esa desviación, la curva ya se aplanó: "
          "etiquetar más del mismo tipo no rinde.")
