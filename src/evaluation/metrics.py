"""
Métricas de evaluación de estrategias de trading.

Módulo de funciones puras (sin I/O): recibe Series de pandas o arrays de numpy
y devuelve escalares o dicts. Sin efectos secundarios, sin logging de alto nivel.

Métricas implementadas:
  - Precisión direccional
  - Sharpe Ratio anualizado
  - Máximo Drawdown
  - Retorno acumulado
  - Retorno anualizado (CAGR)
  - Prueba de hipótesis entre dos series de retornos (t-test pareado + Wilcoxon)
"""

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

logger = logging.getLogger(__name__)

# Días de trading al año (NYSE convención estándar)
_TRADING_DAYS_PER_YEAR: int = 252


# ---------------------------------------------------------------------------
# Métricas individuales
# ---------------------------------------------------------------------------

def directional_accuracy(y_true: pd.Series | np.ndarray,
                         y_pred: pd.Series | np.ndarray) -> float:
    """
    Fracción de predicciones en las que la dirección es correcta.

    Args:
        y_true: Dirección real (1=subió, 0=bajó).
        y_pred: Dirección predicha (1 o 0).

    Returns:
        Valor en [0, 1]. 0.5 equivale a un clasificador aleatorio.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(y_true == y_pred))


def sharpe_ratio(returns: pd.Series,
                 risk_free_rate: float = 0.0,
                 periods_per_year: int = _TRADING_DAYS_PER_YEAR) -> float:
    """
    Sharpe Ratio anualizado.

    Sharpe = (R_media − R_libre) / σ  × √(períodos/año)

    Args:
        returns: Serie de retornos diarios del portfolio.
        risk_free_rate: Tasa libre de riesgo diaria (por defecto 0).
        periods_per_year: Factores de anualización (252 para retornos diarios).

    Returns:
        Sharpe Ratio anualizado. NaN si la desviación típica es 0.
    """
    excess = returns - risk_free_rate
    std = excess.std(ddof=1)
    if std == 0 or np.isnan(std):
        return float("nan")
    return float(excess.mean() / std * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """
    Máximo Drawdown como fracción positiva (e.g. 0.15 = caída del 15 %).

    Se calcula sobre la curva de riqueza acumulada; la caída se mide desde
    el máximo rodante previo hasta el mínimo posterior.

    Args:
        returns: Serie de retornos diarios del portfolio.

    Returns:
        Valor ≥ 0. 0 significa que nunca hubo retroceso.
    """
    if returns.empty:
        return float("nan")
    wealth = (1 + returns).cumprod()
    rolling_peak = wealth.cummax()
    drawdown_series = (wealth - rolling_peak) / rolling_peak
    return float(-drawdown_series.min())


def cumulative_return(returns: pd.Series) -> float:
    """
    Retorno total acumulado del período.

    Args:
        returns: Serie de retornos diarios del portfolio.

    Returns:
        Retorno total como fracción (e.g. 0.23 = +23 %).
    """
    if returns.empty:
        return float("nan")
    return float((1 + returns).prod() - 1)


def annualized_return(returns: pd.Series,
                      periods_per_year: int = _TRADING_DAYS_PER_YEAR) -> float:
    """
    Retorno anualizado (CAGR) calculado a partir de los retornos diarios.

    Args:
        returns: Serie de retornos diarios del portfolio.
        periods_per_year: Días de trading al año para anualizar.

    Returns:
        CAGR como fracción. NaN si la serie está vacía.
    """
    if returns.empty or len(returns) == 0:
        return float("nan")
    total = (1 + returns).prod()
    if total <= 0:
        return float("-inf")
    return float(total ** (periods_per_year / len(returns)) - 1)


def compute_all_metrics(
    returns: pd.Series,
    y_true: pd.Series | None = None,
    y_pred: pd.Series | None = None,
    label: str = "",
) -> dict:
    """
    Calcula el conjunto completo de métricas para una estrategia.

    Args:
        returns: Retornos diarios del portfolio (una observación por día de trading).
        y_true: Dirección real por observación (para accuracy). Opcional.
        y_pred: Dirección predicha por observación (para accuracy). Opcional.
        label: Prefijo añadido a todas las claves del resultado.

    Returns:
        Dict con las métricas nombradas (con prefijo si se indicó label).
    """
    prefix = f"{label}_" if label else ""

    metrics: dict = {
        f"{prefix}cumulative_return": cumulative_return(returns),
        f"{prefix}annualized_return": annualized_return(returns),
        f"{prefix}sharpe_ratio": sharpe_ratio(returns),
        f"{prefix}max_drawdown": max_drawdown(returns),
        f"{prefix}n_trading_days": len(returns),
        f"{prefix}mean_daily_return": float(returns.mean()),
        f"{prefix}std_daily_return": float(returns.std(ddof=1)),
    }

    if y_true is not None and y_pred is not None:
        metrics[f"{prefix}directional_accuracy"] = directional_accuracy(y_true, y_pred)

    return metrics


# ---------------------------------------------------------------------------
# Prueba de hipótesis entre dos estrategias
# ---------------------------------------------------------------------------

class HypothesisResult(NamedTuple):
    """Resultado completo de la prueba de hipótesis entre dos estrategias."""
    mean_diff: float            # E[R_hibrida] − E[R_tecnica]
    t_statistic: float          # estadístico t del test pareado
    p_value_ttest: float        # p-valor (cola derecha, H1: hibrida > tecnica)
    w_statistic: float          # estadístico W de Wilcoxon
    p_value_wilcoxon: float     # p-valor Wilcoxon (cola derecha)
    significant_005: bool       # True si p < 0.05 en al menos un test
    n_paired_obs: int           # número de observaciones pareadas


def hypothesis_test(
    returns_strategy1: pd.Series,
    returns_strategy2: pd.Series,
) -> HypothesisResult:
    """
    Prueba de hipótesis pareada: H1 = estrategia 1 tiene mayores retornos.

    Aplica dos tests complementarios:
      1. t-test pareado (Welch): asume distribución aproximadamente normal.
      2. Wilcoxon signed-rank: no paramétrico, robusto a colas pesadas.
         Los retornos financieros rara vez son gaussianos, por lo que
         Wilcoxon es más adecuado para la tesis.

    H0: E[R1 − R2] = 0   (sin diferencia)
    H1: E[R1 − R2] > 0   (estrategia 1 supera a estrategia 2), test de una cola

    Args:
        returns_strategy1: Retornos diarios de la estrategia a evaluar (híbrida).
        returns_strategy2: Retornos diarios del baseline (solo técnicos).

    Returns:
        HypothesisResult con estadísticos, p-valores e indicador de significancia.
    """
    # Alinear por índice (fechas comunes)
    aligned = pd.concat(
        [returns_strategy1.rename("s1"), returns_strategy2.rename("s2")],
        axis=1,
        join="inner",
    ).dropna()

    n = len(aligned)
    diff = aligned["s1"] - aligned["s2"]
    mean_diff = float(diff.mean())

    # t-test pareado, una cola
    if n < 2:
        logger.warning("Solo %d observaciones pareadas; p-valores no son confiables.", n)
        return HypothesisResult(mean_diff, float("nan"), float("nan"),
                                float("nan"), float("nan"), False, n)

    t_stat, p_t = ttest_rel(aligned["s1"], aligned["s2"], alternative="greater")

    # Wilcoxon signed-rank, una cola
    # Requiere al menos una diferencia no nula; si todas son 0, lanza ValueError
    nonzero_diff = diff[diff != 0]
    if len(nonzero_diff) < 2:
        w_stat, p_w = float("nan"), float("nan")
        logger.warning("Diferencias de retornos casi idénticas; Wilcoxon no aplica.")
    else:
        w_stat, p_w = wilcoxon(nonzero_diff, alternative="greater")

    significant = any(
        p < 0.05 for p in [p_t, p_w] if not np.isnan(p)
    )

    return HypothesisResult(
        mean_diff=mean_diff,
        t_statistic=float(t_stat),
        p_value_ttest=float(p_t),
        w_statistic=float(w_stat) if not isinstance(w_stat, float) else w_stat,
        p_value_wilcoxon=float(p_w) if not isinstance(p_w, float) else p_w,
        significant_005=significant,
        n_paired_obs=n,
    )
