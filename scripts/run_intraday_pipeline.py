"""
Pipeline intraday: orquesta el flujo completo cada 30 minutos.

Flujo:
  1. Verifica mercado abierto (sale silenciosamente si esta cerrado)
  2. Scraping de Reddit (solo posts nuevos gracias al cache de post_ids)
  3. Deteccion de tickers -> preprocesamiento -> scoring FinBERT
  4. Construccion de features de inferencia (sentimiento de hoy + indicadores T-1)
  5. Prediccion con XGBoost
  6. Ejecucion de ordenes en Alpaca paper trading
  7. Guardado de log estructurado en data/intraday_logs/YYYY-MM-DD/HH-MM.json

Uso:
    python scripts/run_intraday_pipeline.py
"""

import json
import logging
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from config.settings import MIN_MENTIONS_PER_DAY, MODELS_DIR, PROCESSED_DIR, RAW_DIR
from scripts.check_market_open import check_market_open

# Rutas de logs
_LOG_BASE = Path("data/intraday_logs")
_SENTIMENT_SCORES = PROCESSED_DIR / "sentiment_scores.csv"
_INDICATORS_CACHE = PROCESSED_DIR / "indicators.parquet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("intraday")


# ---------------------------------------------------------------------------
# Log estructurado
# ---------------------------------------------------------------------------

def _log_path() -> Path:
    """Ruta del JSON de log para esta ejecucion."""
    now = datetime.now(tz=timezone.utc)
    day_dir = _LOG_BASE / now.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir / f"{now.strftime('%H-%M')}.json"


def _save_log(record: dict, path: Path) -> None:
    """Guarda el registro de ejecucion como JSON."""
    path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    logger.info("Log guardado en %s", path)


# ---------------------------------------------------------------------------
# Paso 1: Scraping Reddit
# ---------------------------------------------------------------------------

def _step_scrape() -> tuple[Path | None, int]:
    """Ejecuta el scraper de Reddit. Devuelve (ruta_csv, n_posts)."""
    try:
        from src.sentiment.scraper_reddit import scrape
        out_path = scrape()
        df = pd.read_csv(out_path)
        logger.info("Scraping: %d posts descargados -> %s", len(df), out_path)
        return out_path, len(df)
    except Exception as exc:
        logger.error("Error en scraping: %s", exc)
        return None, 0


# ---------------------------------------------------------------------------
# Paso 2: Deteccion de tickers
# ---------------------------------------------------------------------------

def _step_detect(raw_csv: Path) -> Path | None:
    """Detecta tickers en el CSV crudo. Devuelve ruta del CSV con tickers."""
    try:
        from src.sentiment.ticker_detector import detect_tickers_in_csv
        out = detect_tickers_in_csv(raw_csv)
        logger.info("Tickers detectados -> %s", out)
        return out
    except Exception as exc:
        logger.error("Error en deteccion de tickers: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Paso 3: Preprocesamiento
# ---------------------------------------------------------------------------

def _step_preprocess(tickers_csv: Path) -> Path | None:
    """Limpia el texto para FinBERT."""
    try:
        from src.sentiment.preprocessor import preprocess_csv
        out = preprocess_csv(tickers_csv)
        logger.info("Preprocesamiento completado -> %s", out)
        return out
    except Exception as exc:
        logger.error("Error en preprocesamiento: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Paso 4: Scoring FinBERT (con cache por post_id)
# ---------------------------------------------------------------------------

def _step_score(clean_csv: Path) -> int:
    """Aplica FinBERT. Retorna numero de nuevas filas escritas."""
    try:
        from src.sentiment.finbert_scorer import score_csv
        score_csv(clean_csv)
        if _SENTIMENT_SCORES.exists():
            n = len(pd.read_csv(_SENTIMENT_SCORES))
        else:
            n = 0
        logger.info("Scoring completado. Total filas acumuladas: %d", n)
        return n
    except Exception as exc:
        logger.error("Error en scoring FinBERT: %s", exc)
        return 0


# ---------------------------------------------------------------------------
# Paso 5: Features de inferencia (sentimiento de hoy + indicadores T-1)
# ---------------------------------------------------------------------------

def _build_inference_features(
    today: date,
    feature_cols: list[str],
) -> pd.DataFrame | None:
    """
    Construye features para inferencia usando:
      - Sentimiento: posts de hoy en sentiment_scores.csv
      - Indicadores: el registro mas reciente por ticker en indicators.parquet

    No requiere que las fechas coincidan (T vs T-1 es aceptable para inferencia).
    """
    if not _SENTIMENT_SCORES.exists() or not _INDICATORS_CACHE.exists():
        logger.warning("Faltan archivos de sentimiento o indicadores para inferencia.")
        return None

    # Cargar sentimiento y convertir tipos
    sentiment = pd.read_csv(_SENTIMENT_SCORES, dtype=str)
    for col in ["prob_positive", "prob_negative", "prob_neutral", "score", "num_comments"]:
        if col in sentiment.columns:
            sentiment[col] = pd.to_numeric(sentiment[col], errors="coerce").fillna(0.0)

    # Filtrar solo posts de hoy
    sentiment["date_raw"] = pd.to_datetime(sentiment["timestamp"], utc=True).dt.date
    today_sent = sentiment[sentiment["date_raw"] == today].copy()

    if today_sent.empty:
        logger.warning("No hay posts de hoy (%s) en sentiment_scores.csv", today)
        return None

    # Agregar por ticker (min_mentions=1 para inferencia)
    from src.trading.feature_engine import _add_derived_indicators, aggregate_sentiment
    sent_agg = aggregate_sentiment(today_sent, [today], min_mentions=1)
    if sent_agg.empty:
        return None
    sent_agg = sent_agg.drop(columns=["date"], errors="ignore")

    # Indicadores mas recientes por ticker
    indicators = pd.read_parquet(_INDICATORS_CACHE)
    indicators["date"] = pd.to_datetime(indicators["date"]).dt.date
    latest_ind = indicators.sort_values("date").groupby("ticker").last().reset_index()
    latest_ind = _add_derived_indicators(latest_ind)

    # Join por ticker (sin restriccion de fecha)
    features = sent_agg.merge(latest_ind, on="ticker", how="inner")

    available_cols = [c for c in feature_cols if c in features.columns]
    missing = set(feature_cols) - set(available_cols)
    if missing:
        logger.warning("Features faltantes para inferencia: %s", missing)

    logger.info("Features de inferencia: %d tickers", len(features))
    return features[["ticker"] + available_cols] if available_cols else None


# ---------------------------------------------------------------------------
# Paso 6: Prediccion
# ---------------------------------------------------------------------------

def _step_predict(today: date) -> list[dict]:
    """Carga el modelo y genera predicciones para los tickers activos hoy."""
    try:
        from src.trading.xgboost_model import load_model

        model, feature_cols = load_model(MODELS_DIR)
        features_df = _build_inference_features(today, feature_cols)

        if features_df is None or features_df.empty:
            logger.warning("Sin features disponibles para prediccion.")
            return []

        X = features_df[feature_cols]
        signals = model.predict(X)
        confidences = model.predict_proba(X)[:, 1]

        predictions = [
            {
                "ticker": str(ticker),
                "signal": int(sig),
                "confidence": round(float(conf), 4),
            }
            for ticker, sig, conf in zip(features_df["ticker"], signals, confidences)
        ]
        buy_count = sum(1 for p in predictions if p["signal"] == 1)
        logger.info("Predicciones: %d tickers, %d senales BUY", len(predictions), buy_count)
        return predictions

    except FileNotFoundError:
        logger.warning("Modelo no encontrado en %s. Omitiendo prediccion.", MODELS_DIR)
        return []
    except Exception as exc:
        logger.error("Error en prediccion: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Paso 7: Ejecucion en Alpaca
# ---------------------------------------------------------------------------

def _step_execute(predictions: list[dict]) -> list[dict]:
    """Envia ordenes a Alpaca paper trading."""
    if not predictions:
        return []
    try:
        from src.trading.alpaca_executor import AlpacaExecutor, SignalRecord

        signal_records = [
            SignalRecord(
                ticker=p["ticker"],
                signal=p["signal"],
                confidence=p["confidence"],
            )
            for p in predictions
        ]
        executor = AlpacaExecutor()
        summary = executor.execute(signal_records)
        orders = [
            {
                "ticker": o.ticker,
                "action": o.action,
                "notional": o.notional,
                "status": o.status,
                "error": o.error,
            }
            for o in summary.orders
        ]
        logger.info(
            "Ejecucion: %d ordenes enviadas, %d fallidas",
            summary.positions_opened + summary.positions_closed,
            summary.orders_failed,
        )
        return orders
    except Exception as exc:
        logger.error("Error en ejecucion de ordenes: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Paso 8: Estado del portfolio
# ---------------------------------------------------------------------------

def _get_portfolio_snapshot() -> dict:
    """Obtiene equity y posiciones abiertas desde Alpaca."""
    try:
        from alpaca.trading.client import TradingClient
        from config.settings import ALPACA_API_KEY, ALPACA_API_SECRET

        client = TradingClient(
            api_key=ALPACA_API_KEY,
            secret_key=ALPACA_API_SECRET,
            paper=True,
        )
        acc = client.get_account()
        positions = client.get_all_positions()
        return {
            "equity": float(acc.equity),
            "cash": float(acc.cash),
            "portfolio_value": float(acc.portfolio_value),
            "daily_pnl": float(acc.equity) - float(acc.last_equity),
            "n_positions": len(positions),
        }
    except Exception as exc:
        logger.warning("No se pudo obtener snapshot del portfolio: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# Orquestador principal
# ---------------------------------------------------------------------------

def run() -> dict:
    """
    Ejecuta el pipeline intraday completo.
    Retorna el registro de ejecucion.
    """
    start_time = time.time()
    now_utc = datetime.now(tz=timezone.utc)
    today = now_utc.date()

    record: dict = {
        "run_id": now_utc.isoformat(),
        "timestamp": now_utc.isoformat(),
        "market_open": False,
        "posts_scraped": 0,
        "posts_new": 0,
        "tickers_detected": [],
        "predictions": [],
        "orders": [],
        "portfolio": {},
        "errors": [],
        "duration_seconds": 0.0,
    }

    logger.info("=== Pipeline intraday %s ===", now_utc.strftime("%Y-%m-%d %H:%M UTC"))

    # Verificar mercado
    if not check_market_open():
        logger.info("Mercado cerrado. Pipeline omitido.")
        record["duration_seconds"] = round(time.time() - start_time, 1)
        _save_log(record, _log_path())
        return record

    record["market_open"] = True

    # Paso 1: Scraping
    raw_csv, n_scraped = _step_scrape()
    record["posts_scraped"] = n_scraped

    if raw_csv is None:
        record["errors"].append("scraping_failed")
        record["duration_seconds"] = round(time.time() - start_time, 1)
        _save_log(record, _log_path())
        return record

    # Paso 2: Deteccion de tickers
    tickers_csv = _step_detect(raw_csv)
    if tickers_csv is not None:
        try:
            df_tickers = pd.read_csv(tickers_csv)
            import json as _json
            all_tickers: set[str] = set()
            for val in df_tickers.get("detected_tickers", []):
                try:
                    all_tickers.update(_json.loads(str(val)))
                except Exception:
                    pass
            record["tickers_detected"] = sorted(all_tickers)
        except Exception:
            pass
    else:
        record["errors"].append("ticker_detection_failed")

    # Paso 3: Preprocesamiento
    clean_csv = None
    if tickers_csv is not None:
        clean_csv = _step_preprocess(tickers_csv)
        if clean_csv is None:
            record["errors"].append("preprocessing_failed")

    # Paso 4: Scoring FinBERT
    if clean_csv is not None:
        n_total = _step_score(clean_csv)
        record["posts_new"] = n_total
    else:
        record["errors"].append("scoring_skipped")

    # Paso 5+6: Prediccion
    predictions = _step_predict(today)
    record["predictions"] = predictions

    # Paso 7: Ejecucion
    if predictions:
        orders = _step_execute(predictions)
        record["orders"] = orders
    else:
        logger.info("Sin predicciones activas. No se ejecutan ordenes.")

    # Paso 8: Snapshot del portfolio
    record["portfolio"] = _get_portfolio_snapshot()

    record["duration_seconds"] = round(time.time() - start_time, 1)
    logger.info("Pipeline completado en %.1fs", record["duration_seconds"])

    _save_log(record, _log_path())
    return record


if __name__ == "__main__":
    run()
