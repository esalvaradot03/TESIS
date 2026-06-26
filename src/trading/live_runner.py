"""
Orquestador del flujo de trading EN VIVO (usado por GitHub Actions).

Encadena, en un solo proceso:
  trending StockTwits → mensajes → preprocesado → FinBERT → indicadores →
  feature_engine → predicción XGBoost → ejecución en Alpaca paper.

Antes de operar valida que el mercado esté abierto (la API de Alpaca ya tiene en
cuenta fines de semana y feriados vía get_clock().is_open; además se consulta el
calendario del día para registrarlo).

Cada corrida persiste un "run log" en data/live/run_<ts>.parquet con las features
usadas + la predicción + el precio, para acumular un dataset de reentrenamiento.

Uso:
    python -m src.trading.live_runner                 # flujo completo
    python -m src.trading.live_runner --dry-run       # sin enviar órdenes
    python -m src.trading.live_runner --force          # ignora mercado cerrado
    python -m src.trading.live_runner --max-symbols 15 --max-messages 60
"""

import argparse
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from config.settings import (
    DATA_DIR,
    MIN_MENTIONS_PER_DAY,
    MODELS_DIR,
)

logger = logging.getLogger(__name__)

# Directorio donde se acumulan los run logs (se commitean desde el workflow)
LIVE_DIR = DATA_DIR / "live"

# Ventana de precios para tener warm-up de indicadores en cada corrida
_INDICATOR_LOOKBACK_DAYS = 120


# ---------------------------------------------------------------------------
# Validación de mercado / feriados
# ---------------------------------------------------------------------------

def check_market_open(force: bool = False) -> bool:
    """
    Comprueba contra la API de Alpaca si el mercado está abierto ahora.

    get_clock().is_open ya contempla fines de semana y feriados. Además se
    consulta el calendario del día para dejarlo registrado en el log.

    Args:
        force: Si True, devuelve True aunque el mercado esté cerrado (para pruebas
               manuales vía workflow_dispatch).

    Returns:
        True si se debe operar (mercado abierto o force).
    """
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetCalendarRequest

    from config.settings import ALPACA_API_KEY, ALPACA_API_SECRET

    client = TradingClient(
        api_key=ALPACA_API_KEY, secret_key=ALPACA_API_SECRET, paper=True
    )
    clock = client.get_clock()
    today = clock.timestamp.date()

    # Calendario del día: si la lista viene vacía, hoy NO es día hábil de mercado
    try:
        cal = client.get_calendar(
            GetCalendarRequest(start=today, end=today)
        )
        is_trading_day = len(cal) > 0
    except Exception as exc:  # noqa: BLE001
        logger.warning("No se pudo consultar el calendario: %s", exc)
        is_trading_day = clock.is_open

    logger.info(
        "Estado de mercado: is_open=%s | día_hábil=%s | next_open=%s | next_close=%s",
        clock.is_open, is_trading_day, clock.next_open, clock.next_close,
    )

    if force:
        logger.warning("--force activo: se continúa aunque el mercado esté cerrado.")
        return True

    if not is_trading_day:
        logger.info("Hoy NO es día hábil de mercado (feriado/fin de semana). Sin operar.")
        return False
    if not clock.is_open:
        logger.info("El mercado está cerrado en este momento. Sin operar.")
        return False
    return True


# ---------------------------------------------------------------------------
# Recolección de sentimiento en vivo
# ---------------------------------------------------------------------------

def collect_live_sentiment(
    max_symbols: int,
    max_messages: int,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Recolecta mensajes de los trending symbols, los puntúa con FinBERT y devuelve
    un DataFrame de sentimiento por (post, ticker) más la lista de tickers activos.

    Restringe a tickers del S&P 500 (universo del proyecto).

    Returns:
        (sentiment_df, target_tickers) donde sentiment_df tiene el esquema que
        consume feature_engine (post_id, timestamp, ticker, sentiment_label,
        probs, score, num_comments, stocktwits_sentiment).
    """
    import json

    from src.sentiment import scraper_stocktwits as st
    from src.sentiment.finbert_scorer import FinBERTScorer, _explode_by_ticker
    from src.sentiment.preprocessor import build_input_text, clean_text
    from src.universe.sp500_loader import get_tickers

    sp500 = set(get_tickers())

    trending = st.fetch_trending_symbols(limit=max_symbols * 2)
    target = [t for t in trending if t in sp500][:max_symbols]
    if not target:
        logger.warning("Ningún trending symbol pertenece al S&P 500.")
        return pd.DataFrame(), []
    logger.info("Trending ∩ S&P500 (%d): %s", len(target), target)

    # Recolectar y normalizar mensajes
    rows: list[dict] = []
    session = None
    for symbol in target:
        raw = st.fetch_symbol_messages(symbol, max_messages=max_messages)
        rows.extend(st.normalize_messages(raw))

    if not rows:
        logger.warning("No se recolectaron mensajes.")
        return pd.DataFrame(), target

    df = pd.DataFrame(rows, columns=st._OUTPUT_COLUMNS)
    df = df.drop_duplicates(subset=["post_id"]).reset_index(drop=True)

    # Preprocesar texto y puntuar con FinBERT
    df["clean_text"] = [
        clean_text(build_input_text(t, b)) for t, b in zip(df["title"], df["body"])
    ]
    scorer = FinBERTScorer()
    scores = scorer.score_texts(df["clean_text"].tolist())
    sentiment_df = _explode_by_ticker(df, scores)

    logger.info(
        "Sentimiento en vivo: %d filas (post×ticker) sobre %d tickers.",
        len(sentiment_df), sentiment_df["ticker"].nunique() if not sentiment_df.empty else 0,
    )
    return sentiment_df, target


# ---------------------------------------------------------------------------
# Construcción de features y predicción
# ---------------------------------------------------------------------------

def build_live_features(
    sentiment_df: pd.DataFrame,
    target_tickers: list[str],
    min_mentions: int,
) -> pd.DataFrame:
    """
    Calcula indicadores recientes para los tickers activos y combina con el
    sentimiento en vivo usando feature_engine.build_features.

    Returns:
        DataFrame de features (puede estar vacío si no hay solape sentimiento×precio).
    """
    from src.trading.feature_engine import build_features
    from src.trading.indicators import get_indicators

    end = date.today()
    start = end - timedelta(days=_INDICATOR_LOOKBACK_DAYS)
    indicators = get_indicators(
        tickers=target_tickers, start=start, end=end,
        use_price_cache=False, use_indicator_cache=False,
    )

    # Persistir el sentimiento de la corrida en un CSV temporal para build_features
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    tmp_sent = LIVE_DIR / "_sentiment_live_tmp.csv"
    sentiment_df.to_csv(tmp_sent, index=False, encoding="utf-8")

    features = build_features(
        sentiment_path=tmp_sent,
        indicators=indicators,
        min_mentions=min_mentions,
        output_path=LIVE_DIR / "_features_live_tmp.parquet",
    )
    return features


def predict_latest(features: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
    """
    Toma la fila más reciente por ticker, predice señal y confianza con el modelo.

    Returns:
        DataFrame con columnas: ticker, date, signal, confidence, close + features.
    """
    from src.trading.xgboost_model import load_model

    model, feature_cols = load_model(model_dir)

    # Fila más reciente por ticker (la señal opera sobre el último estado conocido)
    latest = (
        features.sort_values("date")
        .groupby("ticker", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    missing = [c for c in feature_cols if c not in latest.columns]
    if missing:
        raise ValueError(f"Faltan features que el modelo espera: {missing}")

    X = latest[feature_cols]
    proba_up = model.predict_proba(X)[:, 1]
    latest["confidence"] = proba_up
    latest["signal"] = (proba_up >= 0.5).astype(int)
    return latest


# ---------------------------------------------------------------------------
# Run log (para reentrenamiento)
# ---------------------------------------------------------------------------

def save_run_log(predictions: pd.DataFrame) -> Path:
    """
    Persiste la corrida (features + predicción + precio) en un Parquet con timestamp.

    Estos archivos se commitean desde el workflow para acumular datos de
    reentrenamiento del modelo.
    """
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = LIVE_DIR / f"run_{ts}.parquet"
    log = predictions.copy()
    log["run_timestamp"] = datetime.now(tz=timezone.utc).isoformat()
    log.to_parquet(out, index=False)
    logger.info("Run log guardado en %s (%d filas).", out, len(log))
    return out


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

def run(
    max_symbols: int = 15,
    max_messages: int = 60,
    min_mentions: int = MIN_MENTIONS_PER_DAY,
    model_dir: Path = MODELS_DIR,
    dry_run: bool = False,
    force: bool = False,
) -> int:
    """
    Ejecuta el flujo completo en vivo. Devuelve un exit code (0 = OK).
    """
    logger.info("═══ LIVE RUNNER [%s] ═══", datetime.now(tz=timezone.utc).isoformat())

    # 1) Validación de mercado / feriados
    if not check_market_open(force=force):
        logger.info("Mercado cerrado y sin --force. Terminando sin operar.")
        return 0

    # 2) Sentimiento en vivo
    sentiment_df, target = collect_live_sentiment(max_symbols, max_messages)
    if sentiment_df.empty:
        logger.warning("Sin sentimiento; no se generan señales.")
        return 0

    # 3) Features + 4) predicción
    features = build_live_features(sentiment_df, target, min_mentions)
    if features.empty:
        logger.warning("Sin solape sentimiento×indicadores; no se generan señales.")
        return 0

    predictions = predict_latest(features, model_dir)
    logger.info(
        "Predicciones: %d tickers | %d señales de compra (signal=1).",
        len(predictions), int(predictions["signal"].sum()),
    )
    for _, r in predictions.iterrows():
        logger.info(
            "  %-6s date=%s signal=%d confidence=%.3f close=%.2f",
            r["ticker"], r["date"], int(r["signal"]), r["confidence"], r["close"],
        )

    # 5) Run log para reentrenamiento (SIEMPRE, aunque sea dry-run)
    save_run_log(predictions)

    # 6) Ejecución en Alpaca
    if dry_run:
        logger.info("--dry-run activo: no se envían órdenes a Alpaca.")
        return 0

    from src.trading.alpaca_executor import AlpacaExecutor, signals_from_dataframe

    signals = signals_from_dataframe(predictions)
    executor = AlpacaExecutor()
    executor.execute(signals)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Flujo de trading en vivo (StockTwits → Alpaca).")
    parser.add_argument("--max-symbols", type=int, default=15, help="Máx. trending symbols a operar.")
    parser.add_argument("--max-messages", type=int, default=60, help="Máx. mensajes por símbolo.")
    parser.add_argument("--min-mentions", type=int, default=MIN_MENTIONS_PER_DAY)
    parser.add_argument("--model-dir", default=str(MODELS_DIR), help="Directorio del modelo entrenado.")
    parser.add_argument("--dry-run", action="store_true", help="No envía órdenes a Alpaca.")
    parser.add_argument("--force", action="store_true", help="Opera aunque el mercado esté cerrado.")
    args = parser.parse_args()

    exit_code = run(
        max_symbols=args.max_symbols,
        max_messages=args.max_messages,
        min_mentions=args.min_mentions,
        model_dir=Path(args.model_dir),
        dry_run=args.dry_run,
        force=args.force,
    )
    sys.exit(exit_code)
