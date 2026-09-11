"""
Baseline léxico de sentimiento con VADER (hermano de finbert_scorer.py,
mismo esquema de entrada/salida).

VADER no requiere entrenamiento ni GPU: es un baseline "neutro" para
comparar contra los dos enfoques basados en FinBERT. No depende de
finbert_scorer.py (helpers duplicados) para mantener los tres scorers
desacoplados entre sí.

Columnas de salida (idénticas a sentiment_scores.csv):
  post_id, timestamp, ticker, sentiment_label, sentiment_score,
  prob_positive, prob_negative, prob_neutral, score, num_comments,
  stocktwits_sentiment
"""

import json
import logging
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from config.settings import PROCESSED_DIR

logger = logging.getLogger(__name__)

_OUTPUT_FILE = PROCESSED_DIR / "sentiment_scores_lexicon.csv"

# Umbrales estándar de VADER sobre el compound score para derivar la
# etiqueta discreta (ver documentación oficial de vaderSentiment).
_POSITIVE_THRESHOLD = 0.05
_NEGATIVE_THRESHOLD = -0.05

# Resultado asignado a textos vacíos sin pasar por el analizador
_NEUTRAL_SCORE: dict = {
    "sentiment_label": "neutral",
    "sentiment_score": 1.0,
    "prob_positive": 0.0,
    "prob_negative": 0.0,
    "prob_neutral": 1.0,
}

# Columnas de salida — copia literal de finbert_scorer._OUTPUT_COLUMNS para
# que los tres CSV de scores sean directamente comparables.
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

class LexiconScorer:
    """Wrapper de VADER con la misma interfaz pública que FinBERTScorer."""

    def __init__(self) -> None:
        self._analyzer = SentimentIntensityAnalyzer()

    def score_texts(self, texts: list[str]) -> list[dict]:
        """
        Puntúa una lista de textos con VADER.

        Convierte polarity_scores() de VADER (pos/neg/neu/compound) a las
        columnas estándar: prob_positive/negative/neutral se toman
        directamente de pos/neg/neu (ya suman ~1, es más fiel que derivarlas
        del compound score). sentiment_label sale de los umbrales estándar
        de VADER sobre compound (convención oficial de la librería), NO del
        argmax de pos/neg/neu — la mayoría de las palabras de un texto son
        neutras, así que pos/neg/neu casi siempre tienen a neu como máximo;
        clasificar por argmax degradaría VADER a un clasificador casi
        siempre "neutral" e inútil como baseline.

        Por eso sentiment_score = prob_positive/negative/neutral[label] NO
        es necesariamente la probabilidad máxima de las tres (a diferencia
        de FinBERTScorer, donde sentiment_label sí sale de argmax y
        sentiment_score sí es siempre el máximo): compound y pos/neg/neu son
        señales relacionadas pero no derivadas una de la otra en VADER. Es
        una propiedad intencional del lexicón, no un bug — no comparar
        sentiment_score entre lexicon y los dos enfoques FinBERT como si
        fuera la misma cantidad; para eso usar sentiment_label (accuracy/F1
        en sentiment_comparison.py) o net_sentiment (prob_positive −
        prob_negative en feature_engine.py), que sí son comparables.

        Args:
            texts: Lista de textos limpios (salida de preprocessor.clean_text).

        Returns:
            Lista de dicts con campos de sentimiento, misma longitud y orden que texts.
        """
        results: list[dict] = []
        for text in texts:
            if not text.strip():
                results.append(_NEUTRAL_SCORE.copy())
                continue

            polarity = self._analyzer.polarity_scores(text)
            probs = {
                "positive": polarity["pos"],
                "negative": polarity["neg"],
                "neutral": polarity["neu"],
            }
            compound = polarity["compound"]
            if compound >= _POSITIVE_THRESHOLD:
                label = "positive"
            elif compound <= _NEGATIVE_THRESHOLD:
                label = "negative"
            else:
                label = "neutral"

            results.append({
                "sentiment_label": label,
                "sentiment_score": probs[label],
                "prob_positive": polarity["pos"],
                "prob_negative": polarity["neg"],
                "prob_neutral": polarity["neu"],
            })

        logger.info("Scoring lexicon completado: %d textos procesados.", len(texts))
        return results


# ---------------------------------------------------------------------------
# Helpers de transformación (duplicados de finbert_scorer.py a propósito,
# para que este módulo no dependa de los scorers basados en FinBERT)
# ---------------------------------------------------------------------------

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
    scorer: LexiconScorer | None = None,
) -> Path:
    """
    Puntúa el sentimiento de un CSV limpio con VADER y persiste los resultados.

    Los posts cuyo post_id ya aparece en output_path se omiten (caché).
    Los nuevos se procesan, se explotan por ticker y se añaden al CSV
    acumulativo en modo append.

    Args:
        input_path: CSV limpio (salida de preprocessor.preprocess_csv).
                    Columnas requeridas: post_id, timestamp, score,
                    num_comments, detected_tickers, clean_text.
        output_path: CSV acumulativo de salida (actúa como caché).
        scorer: Instancia de LexiconScorer reutilizable. Si None, se crea una.

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
        scorer = LexiconScorer()

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
            "Uso: python -m src.sentiment.lexicon_scorer <ruta_csv_limpio> [ruta_salida]\n"
            "Ejemplo: python -m src.sentiment.lexicon_scorer "
            "data/processed/stocktwits_historical_clean.csv"
        )
        sys.exit(1)

    input_csv = Path(sys.argv[1])
    output_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else _OUTPUT_FILE

    result_path = score_csv(input_csv, output_csv)
    print(f"Scores guardados en → {result_path}")
