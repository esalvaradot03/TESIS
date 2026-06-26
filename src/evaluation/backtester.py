"""
Backtester: simula tres estrategias sobre el período de test y las compara.

Estrategias:
  1. Híbrida        — XGBoost entrenado con sentiment + indicadores técnicos.
  2. Solo técnicos  — XGBoost entrenado únicamente con indicadores técnicos
                      (mismo split temporal, misma arquitectura).
  3. Buy & Hold     — posición igual-ponderada en todos los tickers del universo
                      durante todo el período de test (benchmark pasivo).

Lógica de portfolio para estrategias 1 y 2 (long-only, igual ponderación):
  - En cada día T, se toman posiciones largas en los tickers con señal = 1.
  - Retorno del portfolio en T = media aritmética de retornos realizados
    (close[T+1] / close[T] − 1) de los tickers seleccionados.
  - Si ningún ticker tiene señal = 1, el retorno del día es 0 (cash).

La comparación es justa porque las tres estrategias operan sobre el mismo
universo de (ticker, fecha) — el subconjunto del test set del features DataFrame.

Salidas:
  data/processed/strategy_returns.parquet  — retornos diarios de las 3 estrategias
  data/processed/backtest_results.json     — métricas, hipótesis y metadatos
"""

import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import MODELS_DIR, PROCESSED_DIR, SEED
from src.evaluation.metrics import (
    HypothesisResult,
    compute_all_metrics,
    hypothesis_test,
)
from src.trading.feature_engine import (
    _TECHNICAL_FEATURES,
    load_features,
)
from src.trading.xgboost_model import (
    XGBHyperparams,
    _build_target,
    _internal_val_split,
    _TRAIN_RATIO,
    _temporal_split,
    load_model,
    train,
)

logger = logging.getLogger(__name__)

_FEATURES_FILE = PROCESSED_DIR / "features.parquet"
_RETURNS_FILE = PROCESSED_DIR / "strategy_returns.parquet"
_RESULTS_FILE = PROCESSED_DIR / "backtest_results.json"

_STRATEGY_HYBRID = "hybrid"
_STRATEGY_TECHNICAL = "technical_only"
_STRATEGY_BH = "buy_and_hold"


# ---------------------------------------------------------------------------
# Preparación del dataset para backtesting
# ---------------------------------------------------------------------------

def _add_daily_return(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade la columna 'daily_return' = retorno de close[T] a close[T+1] por ticker.

    La última fila de cada ticker queda con NaN (no hay T+1 disponible).
    Necesario para simular el P&L real de cada posición.
    """
    df = df.sort_values(["ticker", "date"]).copy()
    df["next_close"] = df.groupby("ticker")["close"].shift(-1)
    df["daily_return"] = (df["next_close"] - df["close"]) / df["close"]
    return df


def _rebuild_splits(
    features_df: pd.DataFrame,
    feature_cols_hybrid: list[str],
    feature_cols_technical: list[str],
) -> dict:
    """
    Reconstruye los conjuntos de entrenamiento y test con el mismo split temporal
    que se usó al entrenar el modelo híbrido.

    Devuelve un dict con los DataFrames y arrays necesarios para entrenar el
    baseline técnico y simular el backtest.
    """
    df = _build_target(features_df)
    df = _add_daily_return(df)

    train_df, test_df, train_cutoff, test_start = _temporal_split(df, _TRAIN_RATIO)
    core_df, val_df = _internal_val_split(train_df)

    return {
        "test_df": test_df,
        "test_start": test_start,
        "train_cutoff": train_cutoff,
        # Para entrenar el baseline técnico
        "X_train_tech": core_df[feature_cols_technical],
        "y_train_tech": core_df["target"],
        "X_val_tech": val_df[feature_cols_technical],
        "y_val_tech": val_df["target"],
        # Para predecir con el modelo híbrido ya cargado
        "X_test_hybrid": test_df[feature_cols_hybrid],
        "X_test_tech": test_df[feature_cols_technical],
        "y_test": test_df["target"],
    }


# ---------------------------------------------------------------------------
# Simulación de portfolio
# ---------------------------------------------------------------------------

def _simulate_long_only(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    strategy_name: str,
) -> pd.Series:
    """
    Simula un portfolio long-only de igual ponderación.

    En cada día T:
      - Si señal[T] = 1 para el ticker i → posición larga (1 unidad de peso).
      - Retorno del día = media de retornos de las posiciones largas abiertas.
      - Si ninguna señal es 1, el retorno del día es 0 (permanece en efectivo).

    Los retornos diarios se indexan por fecha para alinearlos entre estrategias.

    Args:
        test_df: Rows del test set con columnas 'date', 'daily_return', 'target'.
        predictions: Array de señales (0/1) con la misma longitud y orden que test_df.
        strategy_name: Nombre para el log.

    Returns:
        Serie de retornos diarios indexada por date, sin NaN.
    """
    df = test_df.copy().reset_index(drop=True)
    df = df.dropna(subset=["daily_return"])
    preds_aligned = predictions[df.index]

    df["signal"] = preds_aligned
    df["position_return"] = df["daily_return"] * df["signal"]

    # Retorno del portfolio = media de posiciones largas abiertas cada día
    # Si signal=0 para todos los tickers ese día → retorno = 0
    daily = (
        df.groupby("date")
        .apply(
            lambda g: g.loc[g["signal"] == 1, "daily_return"].mean()
            if (g["signal"] == 1).any()
            else 0.0,
            include_groups=False,
        )
        .rename(strategy_name)
    )
    daily = daily.fillna(0.0)

    pos_days = (daily != 0).sum()
    logger.info(
        "[%s] %d días en el test. Días con posición larga: %d (%.1f %%).",
        strategy_name, len(daily), pos_days, pos_days / len(daily) * 100,
    )
    return daily


def _simulate_buy_and_hold(
    test_df: pd.DataFrame,
) -> pd.Series:
    """
    Simula buy & hold igual-ponderado sobre todos los tickers del universo.

    El retorno de cada día es la media aritmética de los retornos diarios de
    todos los tickers disponibles ese día (equivalente a rebalanceo diario).

    Args:
        test_df: Rows del test set con columnas 'date' y 'daily_return'.

    Returns:
        Serie de retornos diarios indexada por date.
    """
    df = test_df.dropna(subset=["daily_return"])
    daily = (
        df.groupby("date")["daily_return"]
        .mean()
        .rename(_STRATEGY_BH)
        .fillna(0.0)
    )
    logger.info(
        "[%s] Retorno medio diario: %.4f | Días: %d.",
        _STRATEGY_BH, daily.mean(), len(daily),
    )
    return daily


# ---------------------------------------------------------------------------
# Reporte de resultados
# ---------------------------------------------------------------------------

def _build_comparison_table(metrics_list: list[dict]) -> pd.DataFrame:
    """
    Construye la tabla comparativa de métricas de las tres estrategias.

    Cada dict en metrics_list debe tener claves con prefijo igual al nombre
    de la estrategia (ej. 'hybrid_sharpe_ratio').
    """
    rows = []
    metric_keys = [
        "cumulative_return",
        "annualized_return",
        "sharpe_ratio",
        "max_drawdown",
        "mean_daily_return",
        "std_daily_return",
        "directional_accuracy",
        "n_trading_days",
    ]

    _suffix = "_sharpe_ratio"
    for m in metrics_list:
        # La etiqueta de estrategia puede tener '_' (technical_only, buy_and_hold),
        # así que se obtiene quitando el sufijo de métrica, no con split('_')[0].
        strategy = next(
            (k[: -len(_suffix)] for k in m if k.endswith(_suffix)), "?"
        )
        row = {"strategy": strategy}
        for key in metric_keys:
            full_key = f"{strategy}_{key}"
            row[key] = m.get(full_key, float("nan"))
        rows.append(row)

    return pd.DataFrame(rows).set_index("strategy")


def _log_comparison(table: pd.DataFrame, hyp: HypothesisResult) -> None:
    """Imprime la tabla comparativa y los resultados de la hipótesis al log."""
    logger.info("\n%s", "═" * 70)
    logger.info("COMPARACIÓN DE ESTRATEGIAS — PERÍODO DE TEST")
    logger.info("%s", "─" * 70)

    fmt = {
        "cumulative_return": "{:.2%}",
        "annualized_return": "{:.2%}",
        "sharpe_ratio": "{:.3f}",
        "max_drawdown": "{:.2%}",
        "mean_daily_return": "{:.4%}",
        "std_daily_return": "{:.4%}",
        "directional_accuracy": "{:.3f}",
        "n_trading_days": "{:.0f}",
    }

    header = f"{'Métrica':<26}" + "".join(f"{s:>18}" for s in table.index)
    logger.info(header)
    logger.info("─" * 70)
    for col, f in fmt.items():
        if col not in table.columns:
            continue
        row_str = f"  {col:<24}"
        for s in table.index:
            v = table.loc[s, col]
            try:
                row_str += f"{f.format(v):>18}"
            except (ValueError, TypeError):
                row_str += f"{'N/A':>18}"
        logger.info(row_str)

    logger.info("%s", "─" * 70)
    logger.info("PRUEBA DE HIPÓTESIS  (H1: híbrida > solo técnicos)")
    logger.info("  Diferencia media diaria  : %+.6f", hyp.mean_diff)
    logger.info("  t-test (pareado)         : t=%.4f  p=%.4f", hyp.t_statistic, hyp.p_value_ttest)
    logger.info("  Wilcoxon signed-rank     : W=%.1f  p=%.4f", hyp.w_statistic, hyp.p_value_wilcoxon)
    logger.info(
        "  Significativa al 5 %%     : %s  (objetivo tesis: p < 0.05)",
        "✓ SÍ" if hyp.significant_005 else "✗ NO",
    )
    logger.info(
        "  Objetivo Sharpe > 1.0    : %s",
        "✓ SÍ" if not np.isnan(table.loc[_STRATEGY_HYBRID, "sharpe_ratio"])
               and table.loc[_STRATEGY_HYBRID, "sharpe_ratio"] > 1.0 else "✗ NO",
    )
    logger.info(
        "  Objetivo MDD < 15%%       : %s",
        "✓ SÍ" if not np.isnan(table.loc[_STRATEGY_HYBRID, "max_drawdown"])
               and table.loc[_STRATEGY_HYBRID, "max_drawdown"] < 0.15 else "✗ NO",
    )
    logger.info("%s", "═" * 70)


# ---------------------------------------------------------------------------
# Punto de entrada principal
# ---------------------------------------------------------------------------

def run_comparison(
    features_path: Path = _FEATURES_FILE,
    model_dir: Path = MODELS_DIR,
    returns_path: Path = _RETURNS_FILE,
    results_path: Path = _RESULTS_FILE,
) -> tuple[pd.DataFrame, dict]:
    """
    Ejecuta el backtest completo y genera el reporte comparativo.

    Flujo:
      1. Carga features y el modelo híbrido entrenado.
      2. Reconstruye el mismo split temporal.
      3. Entrena el baseline técnico (mismo split, sin features de sentimiento).
      4. Genera predicciones de ambos modelos sobre el test set.
      5. Simula portfolios y buy & hold.
      6. Calcula métricas completas para cada estrategia.
      7. Ejecuta prueba de hipótesis (híbrida vs técnico).
      8. Persiste retornos diarios y métricas en disco.

    Args:
        features_path: Parquet de features combinadas.
        model_dir: Directorio con el modelo híbrido entrenado.
        returns_path: Parquet de salida con retornos diarios de las 3 estrategias.
        results_path: JSON de salida con todas las métricas y metadatos.

    Returns:
        (comparison_table, results_dict)
    """
    logger.info("═══ Iniciando backtest ═════════════════════════════════════")

    # --- 1. Carga ---
    features_df = load_features(features_path)
    model_hybrid, feature_cols_hybrid = load_model(model_dir)

    # Features técnicas: subconjunto de las del modelo híbrido
    feature_cols_tech = [c for c in _TECHNICAL_FEATURES if c in features_df.columns]
    logger.info(
        "Modelo híbrido: %d features | Modelo técnico: %d features.",
        len(feature_cols_hybrid), len(feature_cols_tech),
    )

    # --- 2. Splits ---
    splits = _rebuild_splits(features_df, feature_cols_hybrid, feature_cols_tech)
    test_df = splits["test_df"]
    logger.info(
        "Período de test: %s → %s | %d filas | %d tickers.",
        splits["test_start"], test_df["date"].max(),
        len(test_df), test_df["ticker"].nunique(),
    )

    # --- 3. Entrenar baseline técnico ---
    logger.info("Entrenando baseline (solo indicadores técnicos)...")
    model_tech = train(
        splits["X_train_tech"],
        splits["y_train_tech"],
        splits["X_val_tech"],
        splits["y_val_tech"],
        hyperparams=XGBHyperparams(random_state=SEED),
    )

    # --- 4. Predicciones ---
    # Predicciones se generan sobre el test set completo (antes del dropna de daily_return)
    test_full = test_df.copy().reset_index(drop=True)

    preds_hybrid = model_hybrid.predict(splits["X_test_hybrid"])
    preds_tech = model_tech.predict(splits["X_test_tech"])

    # --- 5. Simular portfolios ---
    returns_hybrid = _simulate_long_only(test_full, preds_hybrid, _STRATEGY_HYBRID)
    returns_tech = _simulate_long_only(test_full, preds_tech, _STRATEGY_TECHNICAL)
    returns_bh = _simulate_buy_and_hold(test_full)

    # Alinear los tres en el mismo índice de fechas
    all_returns = pd.concat(
        [returns_hybrid, returns_tech, returns_bh], axis=1, join="inner"
    ).fillna(0.0)
    all_returns.index = pd.to_datetime(all_returns.index)
    all_returns.index.name = "date"

    # --- 6. Métricas ---
    # Señales de dirección alineadas con el test_df (para accuracy)
    test_valid = test_full.dropna(subset=["daily_return"]).reset_index(drop=True)
    y_true = splits["y_test"].iloc[: len(test_valid)].reset_index(drop=True)

    metrics_hybrid = compute_all_metrics(
        all_returns[_STRATEGY_HYBRID],
        y_true=y_true,
        y_pred=preds_hybrid[: len(test_valid)],
        label=_STRATEGY_HYBRID,
    )
    metrics_tech = compute_all_metrics(
        all_returns[_STRATEGY_TECHNICAL],
        y_true=y_true,
        y_pred=preds_tech[: len(test_valid)],
        label=_STRATEGY_TECHNICAL,
    )
    metrics_bh = compute_all_metrics(
        all_returns[_STRATEGY_BH],
        label=_STRATEGY_BH,
    )

    # --- 7. Prueba de hipótesis ---
    hyp = hypothesis_test(
        all_returns[_STRATEGY_HYBRID],
        all_returns[_STRATEGY_TECHNICAL],
    )

    # --- 8. Tabla comparativa y log ---
    table = _build_comparison_table([metrics_hybrid, metrics_tech, metrics_bh])
    _log_comparison(table, hyp)

    # --- Persistencia ---
    results_path.parent.mkdir(parents=True, exist_ok=True)
    all_returns.to_parquet(returns_path, index=True)
    logger.info("Retornos diarios guardados en %s", returns_path)

    results_dict = {
        "metrics": {
            _STRATEGY_HYBRID: metrics_hybrid,
            _STRATEGY_TECHNICAL: metrics_tech,
            _STRATEGY_BH: metrics_bh,
        },
        "hypothesis_test": {
            "h0": "E[R_hybrid - R_technical] = 0",
            "h1": "E[R_hybrid - R_technical] > 0 (una cola)",
            "mean_diff": hyp.mean_diff,
            "t_statistic": hyp.t_statistic,
            "p_value_ttest": hyp.p_value_ttest,
            "w_statistic": hyp.w_statistic,
            "p_value_wilcoxon": hyp.p_value_wilcoxon,
            "significant_at_005": hyp.significant_005,
            "n_paired_observations": hyp.n_paired_obs,
        },
        "thesis_objectives": {
            "directional_accuracy_gt_55": (
                metrics_hybrid.get(f"{_STRATEGY_HYBRID}_directional_accuracy", 0) > 0.55
            ),
            "sharpe_ratio_gt_1": (
                metrics_hybrid.get(f"{_STRATEGY_HYBRID}_sharpe_ratio", 0) > 1.0
            ),
            "max_drawdown_lt_15pct": (
                metrics_hybrid.get(f"{_STRATEGY_HYBRID}_max_drawdown", 1.0) < 0.15
            ),
            "p_value_lt_005": hyp.significant_005,
        },
        "split_info": {
            "test_start": str(splits["test_start"]),
            "test_end": str(test_df["date"].max()),
            "n_tickers": int(test_df["ticker"].nunique()),
            "n_test_rows": int(len(test_df)),
        },
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, default=str)
    logger.info("Resultados guardados en %s", results_path)

    return table, results_dict


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

    feat_path = Path(sys.argv[1]) if len(sys.argv) > 1 else _FEATURES_FILE
    mod_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else MODELS_DIR

    table, results = run_comparison(features_path=feat_path, model_dir=mod_dir)

    print("\n" + table.to_string(float_format="{:.4f}".format))
    ht = results["hypothesis_test"]
    print(
        f"\np-valor t-test    : {ht['p_value_ttest']:.4f}"
        f"\np-valor Wilcoxon  : {ht['p_value_wilcoxon']:.4f}"
        f"\nSignificativa 5 % : {ht['significant_at_005']}"
    )
    print("\nObjetivos de tesis:")
    for k, v in results["thesis_objectives"].items():
        print(f"  {k:<40} {'✓' if v else '✗'}")
