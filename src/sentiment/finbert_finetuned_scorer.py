"""
Scoring de sentimiento con FinBERT + linear probing (hermano de
finbert_scorer.py, mismo esquema de entrada/salida).

A diferencia del baseline (cabeza de clasificación pre-entrenada de
ProsusAI/finbert), este scorer recarga el backbone congelado desde
HuggingFace y le monta la cabeza entrenada con finbert_finetune.py sobre
labels manuales.

Este módulo es deliberadamente independiente de finbert_scorer.py (no
importa sus helpers privados) para no acoplar los dos scorers: solo
comparte con finbert_finetune.py la arquitectura de la cabeza
(FinBERTClassifierHead) y el mapeo de labels (_IDX_TO_LABEL), que son
artefactos públicos necesarios para reconstruir el modelo entrenado.

Columnas de salida (idénticas a sentiment_scores.csv):
  post_id, timestamp, ticker, sentiment_label, sentiment_score,
  prob_positive, prob_negative, prob_neutral, score, num_comments,
  stocktwits_sentiment
"""

import json
import logging
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from config.settings import (
    FINBERT_BATCH_SIZE,
    FINBERT_FINETUNED_MODEL_PATH,
    FINBERT_MAX_LENGTH,
    FINBERT_MODEL_NAME,
    PROCESSED_DIR,
)
from src.sentiment.finbert_finetune import FinBERTClassifierHead, _IDX_TO_LABEL

logger = logging.getLogger(__name__)

_OUTPUT_FILE = PROCESSED_DIR / "sentiment_scores_finetuned.csv"

# Resultado asignado a textos vacíos sin pasar por el modelo
_NEUTRAL_SCORE: dict = {
    "sentiment_label": "neutral",
    "sentiment_score": 1.0,
    "prob_positive": 0.0,
    "prob_negative": 0.0,
    "prob_neutral": 1.0,
}

# Columnas de salida — copia literal de finbert_scorer._OUTPUT_COLUMNS para
# que ambos CSV sean directamente comparables (mismo schema, distinto scorer).
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

class FinBERTFinetunedScorer:
    """
    Encapsula el backbone congelado + la cabeza de linear probing entrenada,
    y expone la misma interfaz pública que FinBERTScorer.
    """

    def __init__(
        self,
        model_dir: Path = FINBERT_FINETUNED_MODEL_PATH,
        model_name: str = FINBERT_MODEL_NAME,
        batch_size: int = FINBERT_BATCH_SIZE,
        max_length: int = FINBERT_MAX_LENGTH,
    ) -> None:
        """
        Carga el backbone desde HuggingFace y la cabeza persistida.

        Args:
            model_dir: Directorio con head.pt y meta.json (salida de
                FinBERTHeadTrainer.save()).
            model_name: Identificador HuggingFace del backbone a recargar.
            batch_size: Número de textos por batch de inferencia.
            max_length: Longitud máxima de tokens usada si meta.json no
                especifica una (fallback; normalmente se respeta la usada
                en entrenamiento para no desalinear la tokenización).
        """
        self._batch_size = batch_size
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        meta_path = model_dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"No se encontró {meta_path}. Corré "
                "`python -m src.sentiment.finbert_finetune` primero para "
                "entrenar y persistir la cabeza."
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self._max_length = meta.get("hparams", {}).get("max_length", max_length)

        logger.info("Cargando backbone FinBERT ('%s') en %s...", model_name, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._backbone = AutoModel.from_pretrained(model_name)
        self._backbone.to(self._device)
        self._backbone.eval()

        self._head = FinBERTClassifierHead(
            hidden_size=meta["hidden_size"], num_labels=len(_IDX_TO_LABEL)
        )
        self._head.load_state_dict(torch.load(model_dir / "head.pt", map_location=self._device))
        self._head.to(self._device)
        self._head.eval()

        logger.info(
            "Scorer finetuned listo (cabeza cargada desde %s). Batch size: %d.",
            model_dir, self._batch_size,
        )

    # ------------------------------------------------------------------
    # Métodos internos
    # ------------------------------------------------------------------

    def _infer_batch(self, texts: list[str]) -> list[dict]:
        """Ejecuta inferencia (backbone + cabeza) sobre un batch de textos no vacíos."""
        encoding = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length,
        )
        encoding = {k: v.to(self._device) for k, v in encoding.items()}

        with torch.no_grad():
            cls_embedding = self._backbone(**encoding).last_hidden_state[:, 0, :]
            logits = self._head(cls_embedding)

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

        ptr = 0
        for batch_texts in tqdm(
            _iter_batches(non_empty_texts, self._batch_size),
            total=n_batches,
            desc="FinBERT-finetuned",
            unit="batch",
            leave=False,
        ):
            batch_scores = self._infer_batch(batch_texts)
            for offset, score in enumerate(batch_scores):
                results[non_empty_idx[ptr + offset]] = score
            ptr += len(batch_texts)

        logger.info(
            "Scoring finetuned completado: %d textos procesados, %d vacíos asignados neutral.",
            len(non_empty_idx),
            len(empty_idx),
        )
        return results  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Helpers de transformación (duplicados de finbert_scorer.py a propósito,
# para que este módulo no dependa del baseline)
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

    out["detected_tickers"] = out["detected_tickers"].map(
        lambda x: json.loads(x) if isinstance(x, str) else []
    )

    out = out[out["detected_tickers"].map(len) > 0]
    if out.empty:
        return out

    out = out.explode("detected_tickers").rename(columns={"detected_tickers": "ticker"})
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Función de orquestación batch
# ---------------------------------------------------------------------------

def score_csv(
    input_path: Path,
    output_path: Path = _OUTPUT_FILE,
    scorer: FinBERTFinetunedScorer | None = None,
) -> Path:
    """
    Puntúa el sentimiento de un CSV limpio con la cabeza finetuned y
    persiste los resultados.

    Los posts cuyo post_id ya aparece en output_path se omiten (caché).
    Los nuevos se procesan, se explotan por ticker y se añaden al CSV
    acumulativo en modo append.

    Args:
        input_path: CSV limpio (salida de preprocessor.preprocess_csv).
                    Columnas requeridas: post_id, timestamp, score,
                    num_comments, detected_tickers, clean_text.
        output_path: CSV acumulativo de salida (actúa como caché).
        scorer: Instancia de FinBERTFinetunedScorer reutilizable. Si None, se crea una.

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

    if scorer is None:
        scorer = FinBERTFinetunedScorer()

    scores = scorer.score_texts(new_df["clean_text"].tolist())

    result_df = _explode_by_ticker(new_df, scores)

    skipped = len(new_df) - result_df["post_id"].nunique()
    if skipped:
        logger.info(
            "%d posts descartados por no tener tickers detectados.", skipped
        )

    if result_df.empty:
        logger.warning("Sin filas para escribir tras el filtro de tickers.")
        return output_path

    available = [c for c in _OUTPUT_COLUMNS if c in result_df.columns]
    result_df = result_df[available]

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
            "Uso: python -m src.sentiment.finbert_finetuned_scorer <ruta_csv_limpio> [ruta_salida]\n"
            "Ejemplo: python -m src.sentiment.finbert_finetuned_scorer "
            "data/processed/stocktwits_historical_clean.csv"
        )
        sys.exit(1)

    input_csv = Path(sys.argv[1])
    output_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else _OUTPUT_FILE

    result_path = score_csv(input_csv, output_csv)
    print(f"Scores guardados en → {result_path}")
