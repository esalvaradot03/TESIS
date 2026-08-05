"""
Motor de features: combina scores de sentimiento diarios con indicadores técnicos.

Flujo:
  1. Carga sentiment_scores.csv (salida de finbert_scorer.py).
  2. Calcula net_sentiment = prob_positive − prob_negative por post → señal continua
     en [−1, +1] que representa sentimiento neto del post.
  3. Agrega por (ticker, fecha_trading): mention_count, medias, máximos, ratios,
     sentimiento ponderado por upvotes y totales de engagement.
  4. Filtra pares (ticker, fecha) con mention_count < MIN_MENTIONS_PER_DAY.
  5. Asigna posts de fines de semana / festivos al siguiente día de trading
     para no filtrar señal de días sin mercado abierto.
  6. Une (inner join) con el DataFrame de indicadores técnicos.
  7. Añade features derivadas de los indicadores (bb_width, bb_pct, macd_hist).
  8. Devuelve y persiste el DataFrame combinado en data/processed/features.parquet.

Columnas de salida:
  ticker, date
  — sentimiento: mention_count, net_sentiment_mean/max/min/std,
                 positive_ratio, negative_ratio, neutral_ratio,
                 prob_positive_mean, prob_negative_mean, prob_neutral_mean,
                 weighted_sentiment, total_upvotes, total_comments
  — técnicos:    close, volume, rsi, macd, macd_signal, macd_hist,
                 bb_upper, bb_middle, bb_lower, bb_width, bb_pct
"""

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import MIN_MENTIONS_PER_DAY, PROCESSED_DIR

logger = logging.getLogger(__name__)

_SENTIMENT_FILE = PROCESSED_DIR / "sentiment_scores.csv"
_INDICATORS_FILE = PROCESSED_DIR / "indicators.parquet"
_FEATURES_FILE = PROCESSED_DIR / "features.parquet"

# Enfoques de sentimiento disponibles (pivote agosto 2026: comparación de
# FinBERT baseline vs. linear probing vs. lexicón VADER). Cada uno escribe
# su propio CSV de scores con el mismo schema (ver finbert_scorer.py,
# finbert_finetuned_scorer.py, lexicon_scorer.py).
_SENTIMENT_SOURCE_FILES: dict[str, Path] = {
    "base": _SENTIMENT_FILE,
    "finetuned": PROCESSED_DIR / "sentiment_scores_finetuned.csv",
    "lexicon": PROCESSED_DIR / "sentiment_scores_lexicon.csv",
}

# ---------------------------------------------------------------------------
# Columnas canónicas de salida (orden fijo para reproducibilidad)
# ---------------------------------------------------------------------------
_SENTIMENT_FEATURES: list[str] = [
    "mention_count",
    "net_sentiment_mean",
    "net_sentiment_max",
    "net_sentiment_min",
    "net_sentiment_std",
    "positive_ratio",
    "negative_ratio",
    "neutral_ratio",
    "prob_positive_mean",
    "prob_negative_mean",
    "prob_neutral_mean",
    "weighted_sentiment",
    "total_upvotes",
    "total_comments",
    # Opcional — solo presente con datos de StockTwits (label nativo Bullish/Bearish).
    # Si la fuente es Reddit/WSB, la columna no existe y se omite en todo el pipeline.
    "stocktwits_native_sentiment",
]

# Label nativo de StockTwits → valor numérico para promediar por (ticker, día)
_NATIVE_SENTIMENT_MAP: dict[str, float] = {"Bullish": 1.0, "Bearish": -1.0, "None": 0.0}

_TECHNICAL_FEATURES: list[str] = [
    "close",
    "volume",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "bb_upper",
    "bb_middle",
    "bb_lower",
    "bb_width",
    "bb_pct",
]

_OUTPUT_COLUMNS: list[str] = ["ticker", "date"] + _SENTIMENT_FEATURES + _TECHNICAL_FEATURES


# ---------------------------------------------------------------------------
# Helpers de fechas
# ---------------------------------------------------------------------------

def _extract_trading_dates(trading_days: pd.Series) -> set[date]:
    """Devuelve el conjunto de fechas únicas de trading presentes en los indicadores."""
    return set(pd.to_datetime(trading_days).dt.date.unique())


def _next_trading_day(d: date, trading_days_sorted: list[date]) -> date | None:
    """
    Devuelve el primer día de trading igual o posterior a `d`.

    Usa búsqueda binaria para eficiencia con 500+ tickers × meses de datos.
    Devuelve None si `d` supera el último día disponible.
    """
    lo, hi = 0, len(trading_days_sorted)
    while lo < hi:
        mid = (lo + hi) // 2
        if trading_days_sorted[mid] < d:
            lo = mid + 1
        else:
            hi = mid
    return trading_days_sorted[lo] if lo < len(trading_days_sorted) else None


def _map_to_trading_dates(
    dates: pd.Series,
    trading_days_sorted: list[date],
) -> pd.Series:
    """
    Mapea cada fecha calendario al siguiente día de trading.

    Los posts del sábado van al lunes, los del viernes después del cierre
    ya están en el viernes (el modelo los usa con lag natural de 1 día al
    entrenar con la señal del día previo).
    """
    cache: dict[date, date | None] = {}

    def _map(d: date) -> date | None:
        if d not in cache:
            cache[d] = _next_trading_day(d, trading_days_sorted)
        return cache[d]

    return dates.map(_map)


# ---------------------------------------------------------------------------
# Agregación de sentimiento
# ---------------------------------------------------------------------------

def aggregate_sentiment(
    sentiment: pd.DataFrame,
    trading_days_sorted: list[date],
    min_mentions: int = MIN_MENTIONS_PER_DAY,
) -> pd.DataFrame:
    """
    Agrega scores de sentimiento por (ticker, fecha_trading).

    Calcula:
      - mention_count: posts que mencionan el ticker ese día de trading.
      - net_sentiment = prob_positive − prob_negative (señal continua [−1, +1]).
      - Estadísticas de net_sentiment: media, máximo, mínimo, desviación típica.
      - positive/negative/neutral_ratio: fracción de posts con esa etiqueta.
      - prob_*_mean: probabilidad media de cada clase (directamente de FinBERT).
      - weighted_sentiment: media de net_sentiment ponderada por upvotes de Reddit.
      - total_upvotes / total_comments: engagement total del día.

    Filtra pares (ticker, fecha) con mention_count < min_mentions.

    Args:
        sentiment: DataFrame de sentiment_scores.csv ya cargado.
        trading_days_sorted: Lista ordenada de fechas de trading (del universo
                             de indicadores). Usada para mapear fechas no bursátiles.
        min_mentions: Umbral mínimo de menciones para incluir un par (ticker, fecha).

    Returns:
        DataFrame indexado por (ticker, date) con las columnas de sentimiento.
    """
    df = sentiment.copy()

    # --- Señal continua neta ---
    df["net_sentiment"] = df["prob_positive"].astype(float) - df["prob_negative"].astype(float)

    # --- Fecha de trading (asigna fin-de-semana al lunes) ---
    df["date_raw"] = pd.to_datetime(df["timestamp"], utc=True).dt.date
    df["date"] = _map_to_trading_dates(df["date_raw"], trading_days_sorted)
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].astype("object")  # date objects, no datetime64

    # --- Peso para sentimiento ponderado (upvotes; mínimo 1 para evitar div/0) ---
    df["reddit_score"] = pd.to_numeric(df["score"], errors="coerce").fillna(1).clip(lower=1)
    df["net_x_w"] = df["net_sentiment"] * df["reddit_score"]

    # --- Labels como dummies para los ratios ---
    label_dummies = pd.get_dummies(df["sentiment_label"]).reindex(
        columns=["positive", "negative", "neutral"], fill_value=0
    )
    df = pd.concat([df, label_dummies], axis=1)

    # --- Label nativo de StockTwits (opcional) ---
    # Solo está presente si la fuente es StockTwits. Bullish=+1, Bearish=−1, None=0.
    has_native = "stocktwits_sentiment" in df.columns
    if has_native:
        df["_native_sentiment"] = (
            df["stocktwits_sentiment"].map(_NATIVE_SENTIMENT_MAP).fillna(0.0)
        )

    group = df.groupby(["ticker", "date"])

    # Agregaciones en un solo paso (vectorizado, sin apply)
    agg_kwargs: dict = dict(
        mention_count=("post_id", "count"),
        net_sentiment_mean=("net_sentiment", "mean"),
        net_sentiment_max=("net_sentiment", "max"),
        net_sentiment_min=("net_sentiment", "min"),
        net_sentiment_std=("net_sentiment", "std"),
        _pos_count=("positive", "sum"),
        _neg_count=("negative", "sum"),
        _neu_count=("neutral", "sum"),
        prob_positive_mean=("prob_positive", "mean"),
        prob_negative_mean=("prob_negative", "mean"),
        prob_neutral_mean=("prob_neutral", "mean"),
        _net_x_w_sum=("net_x_w", "sum"),
        _w_sum=("reddit_score", "sum"),
        total_upvotes=("reddit_score", "sum"),
        total_comments=("num_comments", "sum"),
    )
    if has_native:
        # Promedio diario del label nativo Bullish/Bearish por (ticker, día)
        agg_kwargs["stocktwits_native_sentiment"] = ("_native_sentiment", "mean")

    agg = group.agg(**agg_kwargs).reset_index()

    # Ratios de etiqueta
    agg["positive_ratio"] = agg["_pos_count"] / agg["mention_count"]
    agg["negative_ratio"] = agg["_neg_count"] / agg["mention_count"]
    agg["neutral_ratio"] = agg["_neu_count"] / agg["mention_count"]

    # Sentimiento ponderado por upvotes
    agg["weighted_sentiment"] = agg["_net_x_w_sum"] / agg["_w_sum"]

    # NaN en std cuando mention_count == 1 → reemplazar por 0
    agg["net_sentiment_std"] = agg["net_sentiment_std"].fillna(0.0)

    # Eliminar columnas auxiliares privadas
    agg = agg.drop(columns=["_pos_count", "_neg_count", "_neu_count", "_net_x_w_sum", "_w_sum"])

    # --- Filtro de menciones mínimas ---
    before = len(agg)
    agg = agg[agg["mention_count"] >= min_mentions]
    filtered = before - len(agg)
    if filtered:
        logger.info(
            "Filtrados %d pares (ticker, fecha) con menos de %d menciones.",
            filtered, min_mentions,
        )

    logger.info(
        "Agregación de sentimiento: %d pares (ticker, fecha) con >= %d menciones.",
        len(agg), min_mentions,
    )
    return agg


# ---------------------------------------------------------------------------
# Features derivadas de indicadores técnicos
# ---------------------------------------------------------------------------

def _add_derived_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade features derivadas directamente calculables de las columnas del
    indicador técnico. No introduce datos futuros.

      - macd_hist  = macd − macd_signal  (histograma MACD)
      - bb_width   = (bb_upper − bb_lower) / bb_middle  (anchura relativa de las bandas)
      - bb_pct     = (close − bb_lower) / (bb_upper − bb_lower)  (posición del precio)
    """
    df = df.copy()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    band_range = df["bb_upper"] - df["bb_lower"]
    df["bb_width"] = np.where(df["bb_middle"] != 0, band_range / df["bb_middle"], np.nan)
    df["bb_pct"] = np.where(band_range != 0, (df["close"] - df["bb_lower"]) / band_range, np.nan)
    return df


# ---------------------------------------------------------------------------
# Punto de entrada principal
# ---------------------------------------------------------------------------

def build_features(
    sentiment_path: Path | None = None,
    sentiment_source: str = "base",
    indicators: pd.DataFrame | None = None,
    indicators_path: Path = _INDICATORS_FILE,
    min_mentions: int = MIN_MENTIONS_PER_DAY,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """
    Construye el DataFrame de features combinando sentimiento e indicadores.

    Args:
        sentiment_path: Ruta al CSV de scores de sentimiento. Si es None, se
                    resuelve a partir de sentiment_source. Si se pasa
                    explícito, tiene prioridad sobre sentiment_source.
        sentiment_source: Enfoque de sentimiento a usar: "base" (FinBERT
                    off-the-shelf), "finetuned" (linear probing) o "lexicon"
                    (VADER). Ver _SENTIMENT_SOURCE_FILES.
        indicators: DataFrame de indicadores ya cargado. Si es None se lee
                    indicators_path (Parquet).
        indicators_path: Ruta al Parquet de indicadores (usado si indicators=None).
        min_mentions: Umbral de menciones diarias por ticker para incluir la fila.
        output_path: Ruta donde se persiste el Parquet de features. Si es
                    None, se resuelve a features.parquet para "base" o
                    features_{sentiment_source}.parquet para los demás
                    enfoques, para no pisar el Parquet de otro enfoque.

    Returns:
        DataFrame con columnas definidas en _OUTPUT_COLUMNS.

    Raises:
        ValueError: si sentiment_source no está en _SENTIMENT_SOURCE_FILES.
    """
    if sentiment_source not in _SENTIMENT_SOURCE_FILES:
        raise ValueError(
            f"sentiment_source desconocido: '{sentiment_source}'. "
            f"Esperado uno de {list(_SENTIMENT_SOURCE_FILES)}."
        )
    if sentiment_path is None:
        sentiment_path = _SENTIMENT_SOURCE_FILES[sentiment_source]
    if output_path is None:
        output_path = (
            _FEATURES_FILE if sentiment_source == "base"
            else PROCESSED_DIR / f"features_{sentiment_source}.parquet"
        )

    # --- Carga de inputs ---
    logger.info("Cargando sentiment (%s) desde %s...", sentiment_source, sentiment_path)
    sentiment = pd.read_csv(sentiment_path, dtype=str)

    required_sent = {"post_id", "timestamp", "ticker", "sentiment_label",
                     "prob_positive", "prob_negative", "prob_neutral",
                     "score", "num_comments"}
    missing = required_sent - set(sentiment.columns)
    if missing:
        raise ValueError(f"Columnas faltantes en sentiment_scores.csv: {missing}")

    # El CSV se leyó con dtype=str; convertir columnas numéricas antes del groupby
    _numeric_sent = ["prob_positive", "prob_negative", "prob_neutral",
                     "sentiment_score", "score", "num_comments"]
    for col in _numeric_sent:
        if col in sentiment.columns:
            sentiment[col] = pd.to_numeric(sentiment[col], errors="coerce").fillna(0.0)

    if indicators is None:
        logger.info("Cargando indicadores desde %s...", indicators_path)
        indicators = pd.read_parquet(indicators_path)

    required_ind = {"ticker", "date", "close", "rsi", "macd", "macd_signal",
                    "bb_upper", "bb_middle", "bb_lower"}
    missing = required_ind - set(indicators.columns)
    if missing:
        raise ValueError(f"Columnas faltantes en indicadores: {missing}")

    indicators = indicators.copy()
    indicators["date"] = pd.to_datetime(indicators["date"]).dt.date

    # Lista ordenada de días de trading (para mapear fechas no bursátiles)
    trading_days_sorted: list[date] = sorted(indicators["date"].unique().tolist())
    logger.info(
        "Universo de indicadores: %d tickers, %d días de trading (%s → %s).",
        indicators["ticker"].nunique(),
        len(trading_days_sorted),
        trading_days_sorted[0],
        trading_days_sorted[-1],
    )

    # --- Agregación de sentimiento ---
    sent_agg = aggregate_sentiment(sentiment, trading_days_sorted, min_mentions)

    # --- Features derivadas de indicadores ---
    indicators = _add_derived_indicators(indicators)

    # --- Inner join: solo filas donde existen AMBAS fuentes ---
    features = pd.merge(
        sent_agg,
        indicators,
        on=["ticker", "date"],
        how="inner",
    )

    n_sent_only = len(sent_agg) - len(features)
    n_ind_only = indicators[
        ~indicators.set_index(["ticker", "date"]).index.isin(
            sent_agg.set_index(["ticker", "date"]).index
        )
    ].shape[0] if n_sent_only > 0 else 0

    logger.info(
        "Join completado: %d filas finales. "
        "Descartados por falta de indicadores: %d; por falta de sentimiento: %d.",
        len(features), n_sent_only, n_ind_only,
    )

    # --- Ordenar y seleccionar columnas canónicas ---
    features = features.sort_values(["ticker", "date"]).reset_index(drop=True)
    available = [c for c in _OUTPUT_COLUMNS if c in features.columns]
    features = features[available]

    _log_feature_summary(features)

    # --- Persistencia ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)
    logger.info("Features guardadas en %s  (%d filas × %d cols).",
                output_path, len(features), len(features.columns))

    return features


# ---------------------------------------------------------------------------
# Diagnóstico
# ---------------------------------------------------------------------------

def _log_feature_summary(df: pd.DataFrame) -> None:
    """Registra estadísticas básicas del DataFrame de features para diagnóstico."""
    nan_pct = df.isnull().mean().mul(100).round(1)
    cols_with_nan = nan_pct[nan_pct > 0]
    if not cols_with_nan.empty:
        logger.warning(
            "Columnas con NaN: %s",
            ", ".join(f"{c}={v}%" for c, v in cols_with_nan.items()),
        )
    logger.info(
        "Resumen features: %d tickers | %d fechas | %.1f filas/ticker/día en promedio.",
        df["ticker"].nunique(),
        df["date"].nunique(),
        len(df) / max(df["ticker"].nunique(), 1),
    )
    logger.info(
        "Sentimiento: mention_count p50=%.0f | net_sentiment_mean p50=%.3f | "
        "positive_ratio p50=%.2f",
        df["mention_count"].median(),
        df["net_sentiment_mean"].median(),
        df["positive_ratio"].median(),
    )


def load_features(path: Path = _FEATURES_FILE) -> pd.DataFrame:
    """
    Carga el Parquet de features ya construido.

    Atajo para módulos downstream (xgboost_model.py, backtester.py) que no
    necesitan reconstruir las features desde cero.

    Args:
        path: Ruta al Parquet de features.

    Returns:
        DataFrame de features con tipos correctos.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de features en {path}. "
            "Ejecuta build_features() primero."
        )
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


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

    sent_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _SENTIMENT_FILE
    ind_path = Path(sys.argv[2]) if len(sys.argv) > 2 else _INDICATORS_FILE

    df = build_features(sentiment_path=sent_path, indicators_path=ind_path)
    print(df.head(8).to_string(index=False))
    print(f"\nShape: {df.shape}  |  Tickers: {df['ticker'].nunique()}  |  "
          f"Fechas: {df['date'].nunique()}")
