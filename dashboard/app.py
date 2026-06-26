"""
Dashboard de trading en tiempo real.

Ejecutar localmente:
    streamlit run dashboard/app.py

Deploy en Streamlit Cloud:
    Conectar el repo de GitHub y apuntar a dashboard/app.py
"""

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Agregar el directorio raiz al path para importar data_loader
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.data_loader import (
    get_account,
    get_backtest_metrics,
    get_equity_history,
    get_feature_importance,
    get_last_log,
    get_latest_predictions,
    get_positions,
    get_sentiment_today,
    get_strategy_returns,
    get_today_logs,
    is_market_open,
)

# ---------------------------------------------------------------------------
# Configuracion de la pagina
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Trading Sentiment Dashboard",
    page_icon="[chart]",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .metric-card { background: #1e1e2e; border-radius: 8px; padding: 16px; }
    .status-green { color: #50fa7b; font-weight: bold; }
    .status-yellow { color: #f1fa8c; font-weight: bold; }
    .status-red { color: #ff5555; font-weight: bold; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header + metricas clave
# ---------------------------------------------------------------------------

def render_header():
    """Muestra equity, P&L y estado del bot."""
    account = get_account()
    last_log = get_last_log()
    market_open = is_market_open()

    now_utc = datetime.now(tz=timezone.utc)
    last_run_str = last_log.get("timestamp", "")
    if last_run_str:
        try:
            last_run = datetime.fromisoformat(last_run_str)
            minutes_ago = (now_utc - last_run).total_seconds() / 60
        except Exception:
            minutes_ago = 9999
    else:
        minutes_ago = 9999

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        equity = account.get("equity", 0)
        st.metric("Equity total", f"${equity:,.2f}")

    with col2:
        daily_pnl = account.get("daily_pnl", 0)
        pnl_pct = account.get("daily_pnl_pct", 0)
        st.metric(
            "P&L del dia",
            f"${daily_pnl:,.2f}",
            delta=f"{pnl_pct:+.2f}%",
        )

    with col3:
        positions = get_positions()
        st.metric("Posiciones abiertas", len(positions))

    with col4:
        if market_open:
            st.markdown("**Estado mercado**")
            st.markdown('<span class="status-green">ABIERTO</span>', unsafe_allow_html=True)
        else:
            st.markdown("**Estado mercado**")
            st.markdown('<span class="status-red">CERRADO</span>', unsafe_allow_html=True)

    with col5:
        st.markdown("**Ultima ejecucion**")
        if minutes_ago < 35:
            color = "status-green"
            label = f"hace {int(minutes_ago)}m"
        elif minutes_ago < 120:
            color = "status-yellow"
            label = f"hace {int(minutes_ago)}m"
        else:
            color = "status-red"
            label = "hace >2h"
        st.markdown(f'<span class="{color}">{label}</span>', unsafe_allow_html=True)
        st.caption(f"Actualizado: {now_utc.strftime('%H:%M UTC')}")


# ---------------------------------------------------------------------------
# Curva de equity
# ---------------------------------------------------------------------------

def render_equity_curve():
    """Grafico de linea con la evolucion del portfolio."""
    st.subheader("Evolucion del portfolio")
    history = get_equity_history()
    account = get_account()

    if history.empty and not account:
        st.info("Sin datos historicos de equity. Los reportes EOD se acumulan con el tiempo.")
        return

    # Agregar el valor de hoy si no esta en el historico
    rows = history.to_dict("records")
    if account.get("equity"):
        today_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        if not rows or str(rows[-1]["date"])[:10] != today_str:
            rows.append({"date": pd.Timestamp.now(), "equity": account["equity"], "daily_pnl": account.get("daily_pnl", 0)})

    if not rows:
        st.info("Sin datos suficientes para la curva de equity.")
        return

    df = pd.DataFrame(rows)
    initial = float(df["equity"].iloc[0])
    df["cumulative_return_pct"] = (df["equity"] / initial - 1) * 100

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["equity"],
        mode="lines+markers",
        name="Equity",
        line=dict(color="#50fa7b", width=2),
        fill="tozeroy",
        fillcolor="rgba(80,250,123,0.1)",
    ))
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="USD", gridcolor="#333"),
        xaxis=dict(gridcolor="#333"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Comparacion de estrategias
# ---------------------------------------------------------------------------

def render_strategy_comparison():
    """Grafico comparativo de las 3 estrategias."""
    st.subheader("Comparacion de estrategias (backtest)")
    returns_df = get_strategy_returns()

    if returns_df is None or returns_df.empty:
        st.info("Ejecuta `python -m src.evaluation.backtester` para generar la comparacion.")
        return

    # Acumular retornos
    cumulative = (1 + returns_df).cumprod()

    fig = go.Figure()
    colors = {"hybrid": "#50fa7b", "technical_only": "#8be9fd", "buy_and_hold": "#ff79c6"}
    labels = {"hybrid": "Hibrida (sentiment+tecnico)", "technical_only": "Solo tecnico", "buy_and_hold": "Buy & Hold S&P 500"}

    for col in cumulative.columns:
        if col in returns_df.columns:
            fig.add_trace(go.Scatter(
                x=cumulative.index,
                y=cumulative[col],
                name=labels.get(col, col),
                line=dict(color=colors.get(col, "#888"), width=2),
            ))

    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="Retorno acumulado (x)", gridcolor="#333"),
        xaxis=dict(gridcolor="#333"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Metricas clave del backtest
    metrics = get_backtest_metrics()
    if metrics:
        m = metrics.get("metrics", {})
        cols = st.columns(4)
        hybrid = m.get("hybrid", {})
        cols[0].metric("Precision direccional", f"{hybrid.get('hybrid_directional_accuracy', 0):.1%}")
        cols[1].metric("Sharpe Ratio", f"{hybrid.get('hybrid_sharpe_ratio', 0):.2f}")
        cols[2].metric("Max Drawdown", f"{hybrid.get('hybrid_max_drawdown', 0):.1%}")
        ht = metrics.get("hypothesis_test", {})
        p_val = min(ht.get("p_value_ttest", 1), ht.get("p_value_wilcoxon", 1))
        cols[3].metric("p-valor vs solo tecnico", f"{p_val:.4f}")


# ---------------------------------------------------------------------------
# Posiciones actuales
# ---------------------------------------------------------------------------

def render_positions():
    """Tabla de posiciones abiertas en Alpaca."""
    st.subheader("Posiciones abiertas")
    positions = get_positions()

    if not positions:
        st.info("No hay posiciones abiertas en este momento.")
        return

    df = pd.DataFrame(positions)
    df["unrealized_pl_fmt"] = df["unrealized_pl"].apply(lambda x: f"${x:+,.2f}")
    df["unrealized_plpc_fmt"] = df["unrealized_plpc"].apply(lambda x: f"{x:+.2f}%")

    st.dataframe(
        df[["ticker", "qty", "avg_entry", "current_price", "market_value", "unrealized_pl_fmt", "unrealized_plpc_fmt"]].rename(columns={
            "ticker": "Ticker", "qty": "Cantidad", "avg_entry": "Precio entrada",
            "current_price": "Precio actual", "market_value": "Valor de mercado",
            "unrealized_pl_fmt": "P&L no realizado", "unrealized_plpc_fmt": "P&L %",
        }),
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------------------------
# Actividad intraday
# ---------------------------------------------------------------------------

def render_intraday_activity():
    """Timeline de ejecuciones del dia."""
    st.subheader("Actividad intraday (hoy)")
    logs = get_today_logs()

    if not logs:
        st.info("Sin ejecuciones registradas hoy.")
        return

    rows = []
    for log in logs[:13]:
        ts = log.get("timestamp", "")
        try:
            hour = datetime.fromisoformat(ts).strftime("%H:%M")
        except Exception:
            hour = ts[:5]
        rows.append({
            "Hora (UTC)": hour,
            "Mercado": "Abierto" if log.get("market_open") else "Cerrado",
            "Posts": log.get("posts_scraped", 0),
            "Tickers": len(log.get("tickers_detected", [])),
            "Predicciones": len(log.get("predictions", [])),
            "Ordenes": len(log.get("orders", [])),
            "Equity": f"${log.get('portfolio', {}).get('equity', 0):,.0f}",
            "Duracion (s)": log.get("duration_seconds", 0),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Ultimas predicciones
# ---------------------------------------------------------------------------

def render_predictions():
    """Tabla con las predicciones del ultimo ciclo."""
    st.subheader("Ultimas predicciones del modelo")
    predictions = get_latest_predictions()

    if not predictions:
        st.info("Sin predicciones recientes.")
        return

    df = pd.DataFrame(predictions)
    df["Senal"] = df["signal"].map({1: "BUY", 0: "FLAT"})
    df["Confianza"] = df["confidence"].apply(lambda x: f"{x:.1%}")
    df = df.sort_values("confidence", ascending=False)

    def color_signal(val):
        return "background-color: #1a3a1a" if val == "BUY" else "background-color: #3a1a1a"

    styled = df[["ticker", "Senal", "Confianza"]].rename(
        columns={"ticker": "Ticker"}
    ).style.applymap(color_signal, subset=["Senal"])

    st.dataframe(styled, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Analisis de sentimiento
# ---------------------------------------------------------------------------

def render_sentiment_analysis():
    """Top tickers y distribucion de sentimiento."""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top tickers por menciones (hoy)")
        today_sent = get_sentiment_today()

        if today_sent.empty:
            st.info("Sin datos de sentimiento de hoy.")
        else:
            ticker_counts = today_sent["ticker"].value_counts().head(15)
            fig = px.bar(
                x=ticker_counts.values,
                y=ticker_counts.index,
                orientation="h",
                color=ticker_counts.values,
                color_continuous_scale="Greens",
                labels={"x": "Menciones", "y": "Ticker"},
            )
            fig.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                coloraxis_showscale=False,
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Distribucion de sentimiento (hoy)")
        if today_sent.empty or "sentiment_label" not in today_sent.columns:
            st.info("Sin datos de sentimiento de hoy.")
        else:
            dist = today_sent["sentiment_label"].value_counts()
            fig = px.pie(
                values=dist.values,
                names=dist.index,
                color=dist.index,
                color_discrete_map={
                    "positive": "#50fa7b",
                    "negative": "#ff5555",
                    "neutral": "#8be9fd",
                },
                hole=0.4,
            )
            fig.update_layout(
                height=350,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Metricas de performance
# ---------------------------------------------------------------------------

def render_performance_metrics():
    """Sharpe, drawdown, precision direccional."""
    st.subheader("Metricas de performance (modelo hibrido)")
    metrics = get_backtest_metrics()

    if not metrics:
        st.info("Ejecuta el backtester para ver las metricas de performance.")
        return

    m = metrics.get("metrics", {}).get("hybrid", {})
    obj = metrics.get("thesis_objectives", {})

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sharpe Ratio", f"{m.get('hybrid_sharpe_ratio', 0):.3f}",
                delta="objetivo > 1.0", delta_color="normal" if m.get("hybrid_sharpe_ratio", 0) > 1 else "inverse")
    col2.metric("Max Drawdown", f"{m.get('hybrid_max_drawdown', 0):.1%}",
                delta="objetivo < 15%", delta_color="normal" if m.get("hybrid_max_drawdown", 0) < 0.15 else "inverse")
    col3.metric("Precision direccional", f"{m.get('hybrid_directional_accuracy', 0):.1%}",
                delta="objetivo > 55%", delta_color="normal" if m.get("hybrid_directional_accuracy", 0) > 0.55 else "inverse")
    ht = metrics.get("hypothesis_test", {})
    p_val = min(ht.get("p_value_ttest", 1.0), ht.get("p_value_wilcoxon", 1.0))
    col4.metric("p-valor", f"{p_val:.4f}",
                delta="objetivo < 0.05", delta_color="normal" if p_val < 0.05 else "inverse")

    # Checklist de objetivos de tesis
    st.markdown("**Objetivos de tesis:**")
    checks = {
        "Precision direccional > 55%": obj.get("directional_accuracy_gt_55"),
        "Sharpe Ratio > 1.0": obj.get("sharpe_ratio_gt_1"),
        "Max Drawdown < 15%": obj.get("max_drawdown_lt_15pct"),
        "p-valor < 0.05 vs solo tecnico": obj.get("p_value_lt_005"),
    }
    for label, passed in checks.items():
        icon = "OK" if passed else "PENDIENTE"
        color = "green" if passed else "orange"
        st.markdown(f":{color}[{icon}] {label}")


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def render_feature_importance():
    """Grafico de importancia de features del modelo XGBoost."""
    st.subheader("Importancia de features (XGBoost Gain)")
    df = get_feature_importance()

    if df.empty:
        st.info("Entrena el modelo para ver la importancia de features.")
        return

    top = df.nlargest(15, "gain")
    fig = px.bar(
        top, x="gain_normalized", y="feature",
        orientation="h",
        color="gain_normalized",
        color_continuous_scale="Blues",
        labels={"gain_normalized": "Importancia (gain normalizado)", "feature": "Feature"},
    )
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        coloraxis_showscale=False,
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Logs de ejecucion
# ---------------------------------------------------------------------------

def render_execution_logs():
    """Muestra las ultimas lineas del log mas reciente."""
    st.subheader("Log de ultima ejecucion")
    last_log = get_last_log()

    if not last_log:
        st.info("Sin logs disponibles.")
        return

    ts = last_log.get("timestamp", "desconocido")
    st.caption(f"Ejecucion: {ts} | Duracion: {last_log.get('duration_seconds', 0):.1f}s")

    with st.expander("Ver detalle del log", expanded=False):
        st.json(last_log)

    errors = last_log.get("errors", [])
    if errors:
        st.error(f"Errores en esta ejecucion: {', '.join(errors)}")


# ---------------------------------------------------------------------------
# Layout principal
# ---------------------------------------------------------------------------

def main():
    st.title("Trading Sentiment Dashboard")
    st.caption("Sistema de paper trading algoritmico con analisis de sentimiento de Reddit/WSB")

    # Boton de refresco manual
    col_btn, col_info = st.columns([1, 5])
    with col_btn:
        if st.button("Refrescar datos"):
            st.cache_data.clear()
            st.rerun()
    with col_info:
        st.caption("Los datos se actualizan automaticamente cada 60 segundos al navegar entre secciones.")

    st.divider()

    # Header con metricas clave
    render_header()

    st.divider()

    # Graficos principales
    col_left, col_right = st.columns([3, 2])
    with col_left:
        render_equity_curve()
    with col_right:
        render_positions()

    st.divider()
    render_strategy_comparison()

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        render_intraday_activity()
    with col_b:
        render_predictions()

    st.divider()
    render_sentiment_analysis()

    st.divider()
    render_performance_metrics()

    st.divider()

    col_feat, col_logs = st.columns(2)
    with col_feat:
        render_feature_importance()
    with col_logs:
        render_execution_logs()


if __name__ == "__main__":
    main()
