"""
Funciones de carga de datos para el dashboard, todas con cache de Streamlit.
"""

import json
import os
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
_INTRADAY_LOGS = _ROOT / "data" / "intraday_logs"
_DAILY_REPORTS = _ROOT / "data" / "daily_reports"
_SENTIMENT_SCORES = _ROOT / "data" / "processed" / "sentiment_scores.csv"
_INDICATORS = _ROOT / "data" / "processed" / "indicators.parquet"
_FEATURES = _ROOT / "data" / "processed" / "features.parquet"
_STRATEGY_RETURNS = _ROOT / "data" / "processed" / "strategy_returns.parquet"
_BACKTEST_RESULTS = _ROOT / "data" / "processed" / "backtest_results.json"
_IMPORTANCE = _ROOT / "models" / "feature_importance.csv"


# ---------------------------------------------------------------------------
# Alpaca
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def get_account() -> dict:
    """Devuelve datos de la cuenta Alpaca (equity, P&L, cash)."""
    try:
        from alpaca.trading.client import TradingClient
        key = os.getenv("ALPACA_API_KEY", "")
        secret = os.getenv("ALPACA_API_SECRET", "")
        if not key or not secret:
            return {}
        client = TradingClient(api_key=key, secret_key=secret, paper=True)
        acc = client.get_account()
        return {
            "equity": float(acc.equity),
            "last_equity": float(acc.last_equity),
            "cash": float(acc.cash),
            "portfolio_value": float(acc.portfolio_value),
            "daily_pnl": float(acc.equity) - float(acc.last_equity),
            "daily_pnl_pct": (float(acc.equity) - float(acc.last_equity)) / float(acc.last_equity) * 100
            if float(acc.last_equity) > 0 else 0.0,
        }
    except Exception:
        return {}


@st.cache_data(ttl=30)
def get_positions() -> list[dict]:
    """Devuelve las posiciones abiertas en Alpaca."""
    try:
        from alpaca.trading.client import TradingClient
        key = os.getenv("ALPACA_API_KEY", "")
        secret = os.getenv("ALPACA_API_SECRET", "")
        if not key or not secret:
            return []
        client = TradingClient(api_key=key, secret_key=secret, paper=True)
        positions = client.get_all_positions()
        return [
            {
                "ticker": p.symbol,
                "qty": float(p.qty),
                "avg_entry": float(p.avg_entry_price),
                "current_price": float(p.current_price) if p.current_price else 0.0,
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc) * 100,
            }
            for p in positions
        ]
    except Exception:
        return []


@st.cache_data(ttl=30)
def is_market_open() -> bool:
    """Verifica si el mercado NYSE esta abierto ahora."""
    try:
        from alpaca.trading.client import TradingClient
        key = os.getenv("ALPACA_API_KEY", "")
        secret = os.getenv("ALPACA_API_SECRET", "")
        if not key or not secret:
            return False
        client = TradingClient(api_key=key, secret_key=secret, paper=True)
        return bool(client.get_clock().is_open)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Logs intraday
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def get_today_logs() -> list[dict]:
    """Carga todos los logs intraday del dia de hoy."""
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    day_dir = _INTRADAY_LOGS / today
    if not day_dir.exists():
        return []
    logs = []
    for f in sorted(day_dir.glob("*.json"), reverse=True):
        try:
            logs.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    return logs


@st.cache_data(ttl=60)
def get_last_log() -> dict:
    """Devuelve el log intraday mas reciente disponible."""
    logs = get_today_logs()
    if logs:
        return logs[0]
    # Buscar en dias anteriores
    for day_dir in sorted(_INTRADAY_LOGS.glob("*"), reverse=True)[:5]:
        files = sorted(day_dir.glob("*.json"), reverse=True)
        if files:
            try:
                return json.loads(files[0].read_text(encoding="utf-8"))
            except Exception:
                pass
    return {}


# ---------------------------------------------------------------------------
# Reportes diarios (equity curve)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def get_equity_history() -> pd.DataFrame:
    """
    Construye la curva de equity desde los reportes EOD.
    Devuelve DataFrame con columnas: date, equity, daily_pnl.
    """
    rows = []
    for report_file in sorted(_DAILY_REPORTS.glob("*.json")):
        try:
            r = json.loads(report_file.read_text(encoding="utf-8"))
            portfolio = r.get("portfolio_eod", {})
            if portfolio.get("equity"):
                rows.append({
                    "date": r["date"],
                    "equity": float(portfolio["equity"]),
                    "daily_pnl": float(portfolio.get("daily_pnl", 0)),
                })
        except Exception:
            pass
    if not rows:
        return pd.DataFrame(columns=["date", "equity", "daily_pnl"])
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Comparacion de estrategias
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def get_strategy_returns() -> pd.DataFrame | None:
    """Carga los retornos diarios de las 3 estrategias del backtest."""
    if not _STRATEGY_RETURNS.exists():
        return None
    try:
        return pd.read_parquet(_STRATEGY_RETURNS)
    except Exception:
        return None


@st.cache_data(ttl=300)
def get_backtest_metrics() -> dict:
    """Carga las metricas del backtest comparativo."""
    if not _BACKTEST_RESULTS.exists():
        return {}
    try:
        return json.loads(_BACKTEST_RESULTS.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Sentimiento y predicciones
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def get_sentiment_today() -> pd.DataFrame:
    """Carga los scores de sentimiento del dia de hoy."""
    if not _SENTIMENT_SCORES.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(_SENTIMENT_SCORES, dtype=str)
        df["date"] = pd.to_datetime(df["timestamp"], utc=True).dt.date
        today = datetime.now(tz=timezone.utc).date()
        today_df = df[df["date"] == today].copy()
        for col in ["prob_positive", "prob_negative", "prob_neutral", "score"]:
            if col in today_df.columns:
                today_df[col] = pd.to_numeric(today_df[col], errors="coerce").fillna(0)
        return today_df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def get_latest_predictions() -> list[dict]:
    """Devuelve las predicciones del ultimo log intraday."""
    log = get_last_log()
    return log.get("predictions", [])


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_feature_importance() -> pd.DataFrame:
    """Carga el CSV de importancia de features."""
    if not _IMPORTANCE.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(_IMPORTANCE)
    except Exception:
        return pd.DataFrame()
