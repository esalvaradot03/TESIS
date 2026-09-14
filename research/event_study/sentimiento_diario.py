"""
Sentimiento diario por (ticker, día) con los tres enfoques, para el dashboard.

El estudio de eventos solo puntúa los mensajes de las ventanas [-5,+5]. El
calendario de concurrencia necesita **todos** los días, así que este módulo
recorre el pool NYU completo de 2020 a 2022 y agrega por (ticker, día).

Costo: con tope de 20 mensajes por (ticker, día) son ~114.000 textos y dos
pasadas del backbone congelado (FinBERT off-the-shelf y linear probing usan
modelos distintos), del orden de 5-6 horas en CPU. Por eso el trabajo
**guarda avances**: cada bloque puntuado se anexa a un parquet parcial y, si
el proceso se corta, al relanzarlo retoma donde iba.

Salidas:
  - data/processed/_sentimiento_diario_parcial.parquet  (por mensaje, reanudable)
  - research/event_study/sentimiento_diario.csv         (agregado por ticker/día)

Uso:
    python -m research.event_study.sentimiento_diario
    python -m research.event_study.sentimiento_diario --max-por-dia 20 --desde 2020 --hasta 2022
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config.settings import DATA_DIR, SEED
from research.event_study.linear_probe import HPARAMS_PREREGISTRADOS, cargar_split
from src.sentiment.finbert_finetune import FinBERTHeadTrainer, _LABEL_TO_IDX
from src.sentiment.finbert_scorer import FinBERTScorer
from src.sentiment.lexicon_scorer import LexiconScorer

logger = logging.getLogger(__name__)

TICKERS: set[str] = {"NCLH", "DIS", "CRWD", "TGT", "CMG", "DDOG"}
ENFOQUES: tuple[str, ...] = ("finbert", "probe", "vader")

_POOL = DATA_DIR / "processed" / "nyu_pool_v2_CMG_CRWD_DDOG_DIS_NCLH_TGT.parquet"
_PARCIAL = DATA_DIR / "processed" / "_sentimiento_diario_parcial.parquet"
_SALIDA = Path("research/event_study/sentimiento_diario.csv")

BLOQUE = 4_000  # textos por bloque; cada bloque se persiste antes de seguir


def seleccionar_mensajes(
    desde: str = "2020", hasta: str = "2022", max_por_dia: int = 20, seed: int = SEED,
) -> pd.DataFrame:
    """
    Toma del pool NYU los mensajes de los 6 tickers en el rango de años pedido.

    Args:
        desde: Año inicial inclusive (formato "YYYY").
        hasta: Año final inclusive.
        max_por_dia: Tope de mensajes por (ticker, día); acota el costo.
        seed: Semilla del muestreo cuando hay más mensajes que el tope.

    Returns:
        DataFrame con ticker, dia, message_id y texto.
    """
    # Primero el índice (ticker, día, id) SIN el texto: materializar los ~586k
    # cuerpos antes de aplicar el tope dispara la memoria sin necesidad, porque
    # el tope descarta cuatro de cada cinco.
    indice = pd.read_parquet(_POOL, columns=["message_id", "created_at", "symbol_list"])
    indice["dia"] = indice["created_at"].str[:10]
    indice = indice[indice["dia"].str[:4].between(desde, hasta)]

    filas: list[tuple[str, str, str]] = []
    for mid, dia, symbols_json in zip(indice["message_id"], indice["dia"], indice["symbol_list"]):
        for ticker in json.loads(symbols_json):
            if ticker in TICKERS:
                filas.append((ticker, dia, mid))
    df = pd.DataFrame(filas, columns=["ticker", "dia", "message_id"])
    logger.info("Mensajes %s-%s de los 6 tickers: %d", desde, hasta, len(df))

    if max_por_dia > 0 and not df.empty:
        df = (df.groupby(["ticker", "dia"], group_keys=False)
                .apply(lambda g: g.sample(min(len(g), max_por_dia), random_state=seed)))
        logger.info("Tras el tope de %d por (ticker, día): %d mensajes.", max_por_dia, len(df))

    textos = pd.read_parquet(_POOL, columns=["message_id", "message_body"])
    df = df.merge(textos, on="message_id", how="left").rename(columns={"message_body": "texto"})
    return df.reset_index(drop=True)


def _entrenar_probe() -> FinBERTHeadTrainer:
    """Entrena la cabeza lineal sobre el corpus etiquetado (hiperparámetros del pre-registro)."""
    train = cargar_split("train")
    trainer = FinBERTHeadTrainer(hparams=HPARAMS_PREREGISTRADOS)
    trainer.train(train["clean_text"].tolist(), train["clase"].tolist())
    logger.info("Cabeza lineal entrenada con %d pares.", len(train))
    return trainer


def _net_probe(textos: list[str], trainer: FinBERTHeadTrainer) -> np.ndarray:
    """net_sentiment de la cabeza entrenada: p(positive) − p(negative)."""
    embeddings = trainer._embed_all(textos)
    trainer.head.eval()
    with torch.no_grad():
        probs = torch.softmax(trainer.head(embeddings), dim=-1).cpu().numpy()
    return probs[:, _LABEL_TO_IDX["positive"]] - probs[:, _LABEL_TO_IDX["negative"]]


def puntuar_pendientes(mensajes: pd.DataFrame, parcial: Path = _PARCIAL) -> pd.DataFrame:
    """
    Puntúa los mensajes que falten, guardando avances por bloque.

    Args:
        mensajes: Salida de seleccionar_mensajes().
        parcial: Parquet donde se acumulan los puntajes por mensaje.

    Returns:
        DataFrame con ticker, dia, message_id y una columna por enfoque.
    """
    hechos = pd.DataFrame(columns=["message_id"])
    if parcial.exists():
        hechos = pd.read_parquet(parcial)
        logger.info("Avance previo: %d mensajes ya puntuados.", len(hechos))

    faltan = mensajes[~mensajes["message_id"].isin(set(hechos.get("message_id", [])))]
    if faltan.empty:
        logger.info("No falta nada por puntuar.")
        return hechos

    logger.info("Faltan %d mensajes. Cargando modelos...", len(faltan))
    finbert, lexicon = FinBERTScorer(), LexiconScorer()
    trainer = _entrenar_probe()

    acumulado = [hechos] if len(hechos) else []
    inicio = time.time()
    total_bloques = (len(faltan) + BLOQUE - 1) // BLOQUE

    for i in range(0, len(faltan), BLOQUE):
        bloque = faltan.iloc[i: i + BLOQUE]
        textos = bloque["texto"].fillna("").astype(str).tolist()

        puntajes = bloque[["ticker", "dia", "message_id"]].copy()
        puntajes["finbert"] = [r["prob_positive"] - r["prob_negative"] for r in finbert.score_texts(textos)]
        puntajes["vader"] = [r["prob_positive"] - r["prob_negative"] for r in lexicon.score_texts(textos)]
        puntajes["probe"] = _net_probe(textos, trainer)

        acumulado.append(puntajes)
        pd.concat(acumulado, ignore_index=True).to_parquet(parcial, index=False)

        n = i // BLOQUE + 1
        transcurrido = time.time() - inicio
        restante = transcurrido / n * (total_bloques - n)
        logger.info("Bloque %d/%d listo (%.1f min transcurridos, ~%.1f min restantes).",
                    n, total_bloques, transcurrido / 60, restante / 60)

    return pd.concat(acumulado, ignore_index=True)


def agregar(puntajes: pd.DataFrame, salida: Path = _SALIDA) -> pd.DataFrame:
    """
    Promedia por (ticker, día) y escribe el CSV que consume el dashboard.

    Args:
        puntajes: Puntajes por mensaje.
        salida: CSV de salida.

    Returns:
        DataFrame agregado con n_mensajes y el promedio de cada enfoque.
    """
    agregado = (puntajes.groupby(["ticker", "dia"])
                .agg(n_mensajes=("message_id", "count"),
                     **{e: (e, "mean") for e in ENFOQUES})
                .reset_index()
                .sort_values(["ticker", "dia"]))
    for enfoque in ENFOQUES:
        agregado[enfoque] = agregado[enfoque].round(4)
    salida.parent.mkdir(parents=True, exist_ok=True)
    agregado.to_csv(salida, index=False, encoding="utf-8")
    logger.info("Agregado: %d filas (ticker, día) → %s", len(agregado), salida)
    return agregado


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

    parser = argparse.ArgumentParser(description="Sentimiento diario por ticker con los 3 enfoques.")
    parser.add_argument("--desde", default="2020", help="Año inicial (default: 2020).")
    parser.add_argument("--hasta", default="2022", help="Año final (default: 2022).")
    parser.add_argument("--max-por-dia", type=int, default=20, help="Tope por (ticker, día) (default: 20).")
    args = parser.parse_args()

    mensajes = seleccionar_mensajes(args.desde, args.hasta, args.max_por_dia)
    puntajes = puntuar_pendientes(mensajes)
    agregado = agregar(puntajes)

    print("\n=== RESUMEN ===")
    print(f"Mensajes puntuados: {len(puntajes):,}")
    print(f"Pares (ticker, día): {len(agregado):,}")
    print(agregado.groupby("ticker")["n_mensajes"].agg(["count", "sum"]).to_string())
