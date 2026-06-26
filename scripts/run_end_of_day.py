"""
Reporte de fin de dia: consolida los logs intraday en un resumen diario.

Guarda en data/daily_reports/YYYY-MM-DD.json:
  - Resumen de operaciones del dia
  - P&L realizado e irealizado
  - Metricas de sentimiento del dia
  - Snapshot final del portfolio
  - Indicadores tecnicos actualizados (re-descarga precios del dia)

Uso:
    python scripts/run_end_of_day.py
    python scripts/run_end_of_day.py 2024-01-15   # para un dia especifico
"""

import json
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from config.settings import PROCESSED_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("end_of_day")

_LOG_BASE = Path("data/intraday_logs")
_REPORTS_DIR = Path("data/daily_reports")
_SENTIMENT_SCORES = PROCESSED_DIR / "sentiment_scores.csv"


# ---------------------------------------------------------------------------
# Consolidacion de logs intraday
# ---------------------------------------------------------------------------

def _load_day_logs(target_date: date) -> list[dict]:
    """Carga todos los logs intraday del dia indicado."""
    day_dir = _LOG_BASE / target_date.strftime("%Y-%m-%d")
    if not day_dir.exists():
        logger.warning("No hay logs para %s en %s", target_date, day_dir)
        return []

    logs = []
    for log_file in sorted(day_dir.glob("*.json")):
        try:
            logs.append(json.loads(log_file.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("No se pudo leer %s: %s", log_file, exc)

    logger.info("Logs intraday cargados: %d ejecuciones", len(logs))
    return logs


def _summarize_orders(logs: list[dict]) -> dict:
    """Extrae y resume todas las ordenes del dia."""
    all_orders = []
    for log in logs:
        for order in log.get("orders", []):
            order["run_timestamp"] = log.get("timestamp", "")
            all_orders.append(order)

    buys = [o for o in all_orders if o.get("action") == "BUY"]
    closes = [o for o in all_orders if o.get("action") in ("CLOSE", "SELL")]
    failed = [o for o in all_orders if o.get("status") == "failed"]

    return {
        "total_orders": len(all_orders),
        "buy_orders": len(buys),
        "close_orders": len(closes),
        "failed_orders": len(failed),
        "total_notional_deployed": sum(o.get("notional", 0) for o in buys),
        "orders_detail": all_orders,
    }


def _summarize_sentiment(target_date: date) -> dict:
    """Resume el sentimiento del dia desde sentiment_scores.csv."""
    if not _SENTIMENT_SCORES.exists():
        return {}
    try:
        df = pd.read_csv(_SENTIMENT_SCORES, dtype=str)
        for col in ["prob_positive", "prob_negative", "prob_neutral", "score"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        df["date"] = pd.to_datetime(df["timestamp"], utc=True).dt.date
        today_df = df[df["date"] == target_date]

        if today_df.empty:
            return {"posts_today": 0}

        return {
            "posts_today": len(today_df),
            "tickers_mentioned": int(today_df["ticker"].nunique()),
            "top_tickers": today_df["ticker"].value_counts().head(10).to_dict(),
            "avg_prob_positive": round(float(today_df["prob_positive"].mean()), 4),
            "avg_prob_negative": round(float(today_df["prob_negative"].mean()), 4),
            "avg_prob_neutral": round(float(today_df["prob_neutral"].mean()), 4),
            "label_distribution": today_df["sentiment_label"].value_counts().to_dict()
            if "sentiment_label" in today_df.columns else {},
        }
    except Exception as exc:
        logger.warning("Error resumiendo sentimiento: %s", exc)
        return {}


def _get_final_portfolio() -> dict:
    """Obtiene el snapshot final del portfolio desde Alpaca."""
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
            "positions": [
                {
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "market_value": float(p.market_value),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc),
                }
                for p in positions
            ],
        }
    except Exception as exc:
        logger.warning("No se pudo obtener portfolio final: %s", exc)
        return {}


def _update_indicators_eod() -> bool:
    """Actualiza el cache de indicadores con precios hasta hoy."""
    try:
        from src.trading.indicators import get_indicators
        today = date.today()
        # Re-descarga los ultimos 6 meses de precios
        from datetime import timedelta
        start = today - timedelta(days=180)
        get_indicators(start=start, end=today, use_price_cache=False)
        logger.info("Indicadores actualizados hasta %s", today)
        return True
    except Exception as exc:
        logger.error("Error actualizando indicadores: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------

def run(target_date: date | None = None) -> Path:
    """
    Genera el reporte de fin de dia.

    Args:
        target_date: Fecha del reporte. Por defecto: hoy.

    Returns:
        Ruta al JSON del reporte generado.
    """
    if target_date is None:
        target_date = datetime.now(tz=timezone.utc).date()

    logger.info("=== Reporte EOD para %s ===", target_date)

    # 1. Cargar logs del dia
    day_logs = _load_day_logs(target_date)
    runs_with_market = [l for l in day_logs if l.get("market_open")]

    # 2. Actualizar indicadores tecnicos
    indicators_updated = _update_indicators_eod()

    # 3. Construir reporte
    report = {
        "date": target_date.isoformat(),
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "intraday_runs": len(day_logs),
        "runs_with_market_open": len(runs_with_market),
        "indicators_updated": indicators_updated,
        "orders": _summarize_orders(day_logs),
        "sentiment": _summarize_sentiment(target_date),
        "portfolio_eod": _get_final_portfolio(),
        "errors_total": sum(len(l.get("errors", [])) for l in day_logs),
    }

    # 4. Guardar
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = _REPORTS_DIR / f"{target_date.isoformat()}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    equity = report["portfolio_eod"].get("equity", "N/A")
    daily_pnl = report["portfolio_eod"].get("daily_pnl", "N/A")
    logger.info(
        "Reporte EOD guardado: equity=$%s | P&L dia=$%s | -> %s",
        equity, daily_pnl, report_path,
    )
    return report_path


if __name__ == "__main__":
    target = None
    if len(sys.argv) > 1:
        target = date.fromisoformat(sys.argv[1])
    out = run(target)
    print(f"Reporte guardado en: {out}")
