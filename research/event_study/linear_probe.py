"""
Linear probing sobre el corpus etiquetado del event study.

Envuelve `src.sentiment.finbert_finetune` (backbone FinBERT congelado + cabeza
lineal sobre el embedding [CLS]) para el schema propio de este corpus, que
etiqueta por par **(post_id, target_ticker)** y no por post.

Dos diferencias con el subsistema anterior:

  - **Vocabulario.** El corpus usa bullish/bearish/neutral/unusable. Se mapea
    a positive/negative/neutral para que las métricas sean comparables con los
    otros dos enfoques; `unusable` se excluye, igual que en el kappa primario.
  - **Hiperparámetros.** Los defaults del repo (3 épocas, lr 1e-4) están
    pensados para miles de ejemplos: con ~300 pares dejan la cabeza en azar
    (train_loss 1.097 vs ln(3)=1.0986, verificado 2026-09-13). Acá se usan los
    valores congelados en docs/pre_registro_event_study.md.

Uso:
    python -m research.event_study.linear_probe --train ambos
"""

import logging
from pathlib import Path

import pandas as pd
import torch

from config.settings import DATA_DIR, SEED
from src.sentiment.finbert_finetune import (
    FineTuneHyperparams,
    FinBERTHeadTrainer,
    _IDX_TO_LABEL,
)

logger = logging.getLogger(__name__)

MANUAL_LABELS_DIR = DATA_DIR / "manual_labels"
ANOTADORES: tuple[str, ...] = ("camilo", "esteban")

# Corpus → vocabulario de los scorers. `unusable` no aparece: se excluye.
LABEL_A_CLASE: dict[str, str] = {"bullish": "positive", "bearish": "negative", "neutral": "neutral"}
CLASES: list[str] = ["positive", "negative", "neutral"]

# Hiperparámetros congelados en el pre-registro (sección 4.3).
HPARAMS_PREREGISTRADOS = FineTuneHyperparams(
    epochs=200, lr_head=1e-3, batch_size=16, val_split=0.15, seed=SEED,
)

_ARCHIVO_ALTERADOS = MANUAL_LABELS_DIR / "_filas_texto_alterado.csv"


def pares_excluidos(path: Path = _ARCHIVO_ALTERADOS) -> set[tuple[str, str]]:
    """
    Pares (post_id, target_ticker) etiquetados sobre un texto que Excel alteró.

    Su label se puso mirando un texto distinto al del corpus, así que no son
    válidos para entrenar ni evaluar.

    Args:
        path: CSV generado al normalizar los lotes.

    Returns:
        Conjunto de llaves a excluir; vacío si el archivo no existe.
    """
    if not path.exists():
        return set()
    df = pd.read_csv(path, dtype=str)
    return set(zip(df["post_id"], df["target_ticker"]))


def cargar_lote(
    lote: str,
    anotador: str,
    excluir: set[tuple[str, str]] | None = None,
    labels_dir: Path = MANUAL_LABELS_DIR,
) -> pd.DataFrame:
    """
    Carga un lote etiquetado y lo deja listo para entrenar o evaluar.

    Args:
        lote: Nombre del lote (l0, doble, test, train).
        anotador: Identificador del anotador (camilo, esteban).
        excluir: Llaves a descartar; por defecto, las de pares_excluidos().
        labels_dir: Directorio de los CSV de etiquetado.

    Returns:
        DataFrame con las filas usables y una columna `clase`
        (positive/negative/neutral) más `anotador`.

    Raises:
        FileNotFoundError: si no existe el CSV del lote.
    """
    path = labels_dir / f"{lote}_{anotador}.csv"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}")

    excluir = pares_excluidos() if excluir is None else excluir
    df = pd.read_csv(path, dtype=str).fillna("")
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(LABEL_A_CLASE)]
    if excluir:
        llaves = list(zip(df["post_id"], df["target_ticker"]))
        df = df[[k not in excluir for k in llaves]]

    df = df.assign(clase=df["label"].map(LABEL_A_CLASE), anotador=anotador)
    logger.info("%s_%s: %d pares usables.", lote, anotador, len(df))
    return df.reset_index(drop=True)


def cargar_split(lote: str, anotadores: tuple[str, ...] = ANOTADORES) -> pd.DataFrame:
    """
    Une el mismo lote de varios anotadores (test y train son disjuntos por diseño).

    Args:
        lote: Nombre del lote.
        anotadores: Anotadores a incluir.

    Returns:
        DataFrame concatenado, con índice reiniciado.
    """
    partes = [cargar_lote(lote, quien) for quien in anotadores]
    return pd.concat(partes, ignore_index=True)


def entrenar(
    textos: list[str],
    clases: list[str],
    hparams: FineTuneHyperparams = HPARAMS_PREREGISTRADOS,
) -> tuple[FinBERTHeadTrainer, dict]:
    """
    Entrena la cabeza lineal sobre el backbone congelado.

    Args:
        textos: Textos limpios (columna clean_text del corpus).
        clases: Clases en el vocabulario de CLASES, una por texto.
        hparams: Hiperparámetros; por defecto los del pre-registro.

    Returns:
        (trainer entrenado, historial de entrenamiento).
    """
    trainer = FinBERTHeadTrainer(hparams=hparams)
    historial = trainer.train(textos, clases)
    return trainer, historial


def predecir(trainer: FinBERTHeadTrainer, textos: list[str]) -> list[str]:
    """
    Predice la clase de cada texto con la cabeza ya entrenada.

    Args:
        trainer: Trainer devuelto por entrenar() o FinBERTHeadTrainer.load_head().
        textos: Textos limpios a clasificar.

    Returns:
        Lista de clases predichas, en el orden de `textos`.
    """
    embeddings = trainer._embed_all(textos)
    trainer.head.eval()
    with torch.no_grad():
        indices = trainer.head(embeddings).argmax(dim=-1).cpu().numpy()
    return [_IDX_TO_LABEL[int(i)] for i in indices]


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

    parser = argparse.ArgumentParser(description="Entrena la cabeza lineal del event study.")
    parser.add_argument(
        "--train", default="ambos", choices=[*ANOTADORES, "ambos"],
        help="Anotador cuyas etiquetas se usan para entrenar (default: ambos).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("models/event_study_probe"),
        help="Directorio donde se persiste la cabeza entrenada.",
    )
    args = parser.parse_args()

    anotadores = ANOTADORES if args.train == "ambos" else (args.train,)
    train = cargar_split("train", anotadores)
    print(f"Entrenando con {len(train)} pares ({args.train}): {train['clase'].value_counts().to_dict()}")

    trainer, historial = entrenar(train["clean_text"].tolist(), train["clase"].tolist())
    destino = trainer.save(args.output_dir)
    print(f"Cabeza persistida en {destino}")
    print(f"val_accuracy final: {historial['val_accuracy'][-1]:.3f} | "
          f"train_loss final: {historial['train_loss'][-1]:.4f}")
