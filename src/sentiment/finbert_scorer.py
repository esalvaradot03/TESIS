"""
Scoring de sentimiento financiero con FinBERT (ProsusAI/finbert).

Flujo:
  1. Lee el CSV limpio (salida de preprocessor.py).
  2. Filtra posts ya procesados comparando post_id contra el CSV de salida (caché).
  3. Pasa los textos nuevos por FinBERT en batches de FINBERT_BATCH_SIZE.
  4. Explota la columna detected_tickers → un registro por (post_id, ticker).
  5. Añade las nuevas filas al CSV acumulativo data/processed/sentiment_scores.csv.

Columnas de salida:
  post_id, timestamp, ticker, sentiment_label, sentiment_score,
  prob_positive, prob_negative, prob_neutral, score, num_comments
"""

import json
import logging
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config.settings import (
    FINBERT_BATCH_SIZE,
    FINBERT_MAX_LENGTH,
    FINBERT_MODEL_NAME,
    PROCESSED_DIR,
)

logger = logging.getLogger(__name__)

_OUTPUT_FILE = PROCESSED_DIR / "sentiment_scores.csv"

# Mapeo de índice de clase a etiqueta según el config de ProsusAI/finbert.
# El orden (positive=0, negative=1, neutral=2) está definido en el model card.
_IDX_TO_LABEL: dict[int, str] = {0: "positive", 1: "negative", 2: "neutral"}

# Resultado asignado a textos vacíos sin pasar por el modelo
_NEUTRAL_SCORE: dict = {
    "sentiment_label": "neutral",
    "sentiment_score": 1.0,
    "prob_positive": 0.0,
    "prob_negative": 0.0,
    "prob_neutral": 1.0,
}

# Columnas que se escriben en el CSV de salida (orden canónico).
# `stocktwits_sentiment` es opcional: solo aparece cuando la fuente es StockTwits
# (lo emite scraper_stocktwits.py). Con datos de Reddit la columna no existe y
# simplemente se omite, ya que la selección filtra por las columnas presentes.
_OUTPUT_COLUMNS: list[str] = [
    "post_id",
    "timestamp",
    "ticker",
    "sentiment_label",
    "sentiment_score",
    "prob_positive",
    "prob_negative",
    "prob_neutral",
    "score",
    "num_comments",
    "stocktwits_sentiment",
]


# ---------------------------------------------------------------------------
# Clase principal
# ---------------------------------------------------------------------------

class FinBERTScorer:
    """
    Encapsula la carga del modelo FinBERT y el scoring en batch.

    El modelo y el tokenizador se cargan una sola vez en el constructor
    y se reutilizan en todas las llamadas a score_texts().
    """

    def __init__(
        self,
        model_name: str = FINBERT_MODEL_NAME,
        batch_size: int = FINBERT_BATCH_SIZE,
        max_length: int = FINBERT_MAX_LENGTH,
    ) -> None:
        """
        Carga el modelo y lo mueve al dispositivo disponible.

        Args:
            model_name: Identificador HuggingFace del modelo (por defecto ProsusAI/finbert).
            batch_size: Número de textos por batch de inferencia.
            max_length: Longitud máxima de tokens (trunca textos más largos).
        """
        self._batch_size = batch_size
        self._max_length = max_length
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(
            "Cargando FinBERT ('%s') en %s...", model_name, self._device
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.to(self._device)
        self._model.eval()
        logger.info("FinBERT listo. Batch size: %d.", self._batch_size)

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _infer_batch(self, texts: list[str]) -> list[dict]:
        """
        Ejecuta inferencia sobre un batch ya filtrado de textos no vacíos.

        Args:
            texts: Lista de strings limpios de longitud <= batch_size.

        Returns:
            Lista de dicts con las cinco métricas de sentimiento.
        """
        encoding = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length,
        )
        encoding = {k: v.to(self._device) for k, v in encoding.items()}

        with torch.no_grad():
            logits = self._model(**encoding).logits

        # softmax → probabilidades [batch, 3]
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        results: list[dict] = []
        for row in probs:
            idx = int(row.argmax())
            results.append({
                "sentiment_label": _IDX_TO_LABEL[idx],
                "sentiment_score": float(row[idx]),
                "prob_positive": float(row[0]),
                "prob_negative": float(row[1]),
                "prob_neutral": float(row[2]),
            })
        return results

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def score_texts(self, texts: list[str]) -> list[dict]:
        """
        Puntúa una lista de textos en batches y devuelve un resultado por texto.

        Los textos vacíos se asignan directamente como neutral (score 1.0)
        sin consumir recursos de inferencia.

        Args:
            texts: Lista de textos limpios (salida de preprocessor.clean_text).

        Returns:
            Lista de dicts con campos de sentimiento, misma longitud y orden que texts.
        """
        n = len(texts)
        results: list[dict | None] = [None] * n

        non_empty_idx = [i for i, t in enumerate(texts) if t.strip()]
        empty_idx = [i for i, t in enumerate(texts) if not t.strip()]

        for i in empty_idx:
            results[i] = _NEUTRAL_SCORE.copy()

        if not non_empty_idx:
            return results  # type: ignore[return-value]

        non_empty_texts = [texts[i] for i in non_empty_idx]
        n_batches = (len(non_empty_texts) + self._batch_size - 1) // self._batch_size

        ptr = 0  # posición en non_empty_idx
        for batch_texts in tqdm(
            _iter_batches(non_empty_texts, self._batch_size),
            total=n_batches,
            desc="FinBERT",
            unit="batch",
            leave=False,
        ):
            batch_scores = self._infer_batch(batch_texts)
            for offset, score in enumerate(batch_scores):
                results[non_empty_idx[ptr + offset]] = score
            ptr += len(batch_texts)

        logger.info(
            "Scoring completado: %d textos procesados, %d vacíos asignados neutral.",
            len(non_empty_idx),
            len(empty_idx),
        )
        return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Helpers de transformación
# ---------------------------------------------------------------------------

def _iter_batches(items: list, size: int):
    """Genera sub-listas de longitud máxima size."""
    for start in range(0, len(items), size):
        yield items[start: start + size]


def _load_cached_post_ids(output_path: Path) -> set[str]:
    """
    Devuelve el conjunto de post_ids ya presentes en el CSV de salida.

    Lee solo la columna post_id para no cargar todo el archivo en memoria.
    Devuelve un set vacío si el archivo no existe o está corrupto.
    """
    if not output_path.exists():
        return set()
    try:
        cached = pd.read_csv(output_path, usecols=["post_id"], dtype=str)
        ids = set(cached["post_id"].dropna().unique())
        logger.info("Caché cargado: %d post_ids ya procesados.", len(ids))
        return ids
    except Exception as exc:
        logger.warning(
            "No se pudo leer caché desde %s (%s). Se reprocesarán todos los posts.",
            output_path,
            exc,
        )
        return set()


def _explode_by_ticker(df: pd.DataFrame, scores: list[dict]) -> pd.DataFrame:
    """
    Combina scores con el DataFrame y lo expande a una fila por (post_id, ticker).

    Posts cuya columna detected_tickers sea una lista vacía se descartan,
    ya que no hay ticker al que asignar el sentimiento.

    Args:
        df: DataFrame de posts nuevos (sin modificar).
        scores: Lista de dicts de sentimiento, misma longitud y orden que df.

    Returns:
        DataFrame con una fila por combinación (post_id, ticker).
    """
    out = df.copy().reset_index(drop=True)
    scores_df = pd.DataFrame(scores, index=out.index)
    out = pd.concat([out, scores_df], axis=1)

    # Parsear la columna JSON → list[str]
    out["detected_tickers"] = out["detected_tickers"].map(
        lambda x: json.loads(x) if isinstance(x, str) else []
    )

    # Descartar posts sin tickers antes del explode (evita filas con ticker=NaN)
    out = out[out["detected_tickers"].map(len) > 0]
    if out.empty:
        return out

    out = out.explode("detected_tickers").rename(
        columns={"detected_tickers": "ticker"}
    )
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Función de orquestación batch
# ---------------------------------------------------------------------------

def score_csv(
    input_path: Path,
    output_path: Path = _OUTPUT_FILE,
    scorer: FinBERTScorer | None = None,
) -> Path:
    """
    Puntúa el sentimiento de un CSV limpio y persiste los resultados.

    Los posts cuyo post_id ya aparece en output_path se omiten (caché).
    Los nuevos se procesan con FinBERT, se explotan por ticker y se añaden
    al CSV acumulativo en modo append.

    Args:
        input_path: CSV limpio (salida de preprocessor.preprocess_csv).
                    Columnas requeridas: post_id, timestamp, score,
                    num_comments, detected_tickers, clean_text.
        output_path: CSV acumulativo de salida (actúa como caché).
        scorer: Instancia de FinBERTScorer reutilizable. Si None, se crea una.

    Returns:
        Ruta al CSV de salida.
    """
    logger.info("Cargando CSV limpio desde %s...", input_path)
    df = pd.read_csv(input_path, dtype=str).fillna("")

    required = {
        "post_id", "timestamp", "score", "num_comments",
        "detected_tickers", "clean_text",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Columnas faltantes en el CSV de entrada: {missing}. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    # --- Filtrado por caché ---
    cached_ids = _load_cached_post_ids(output_path)
    new_df = df[~df["post_id"].isin(cached_ids)].copy()

    if new_df.empty:
        logger.info("Todos los posts ya están en caché. Sin trabajo pendiente.")
        return output_path

    logger.info(
        "%d posts nuevos a procesar (%d ya en caché, %d en el CSV de entrada).",
        len(new_df),
        len(cached_ids),
        len(df),
    )

    # --- Scoring ---
    if scorer is None:
        scorer = FinBERTScorer()

    scores = scorer.score_texts(new_df["clean_text"].tolist())

    # --- Explosión por ticker ---
    result_df = _explode_by_ticker(new_df, scores)

    skipped = len(new_df) - result_df["post_id"].nunique()
    if skipped:
        logger.info(
            "%d posts descartados por no tener tickers detectados.", skipped
        )

    if result_df.empty:
        logger.warning("Sin filas para escribir tras el filtro de tickers.")
        return output_path

    # Conservar solo columnas canónicas; las presentes en el df pueden variar
    available = [c for c in _OUTPUT_COLUMNS if c in result_df.columns]
    result_df = result_df[available]

    # --- Persistencia (append) ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    result_df.to_csv(
        output_path,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8",
    )

    logger.info(
        "Escritas %d filas (post × ticker) en %s. "
        "Posts únicos nuevos: %d. Tickers únicos: %d.",
        len(result_df),
        output_path,
        result_df["post_id"].nunique(),
        result_df["ticker"].nunique(),
    )
    return output_path


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
            "Uso: python -m src.sentiment.finbert_scorer <ruta_csv_limpio> [ruta_salida]\n"
            "Ejemplo: python -m src.sentiment.finbert_scorer "
            "data/processed/wsb_20240101_120000_tickers_clean.csv"
        )
        sys.exit(1)

    input_csv = Path(sys.argv[1])
    output_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else _OUTPUT_FILE

    result_path = score_csv(input_csv, output_csv)
    print(f"Scores guardados en → {result_path}")
