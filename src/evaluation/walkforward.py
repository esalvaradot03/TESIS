"""
Evaluación walk-forward del modelo ESTÁTICO.

A diferencia de un único split test, aquí el modelo se entrena UNA sola vez sobre
la ventana inicial y luego predice **día a día** sobre todo el horizonte forward,
sin reentrenar. Es el baseline justo contra el cual se comparará después el
sistema adaptativo (que sí reentrenará a medida que avanza).

El bucle día-a-día es trivialmente equivalente a una predicción batch para un
modelo estático, pero se implementa explícitamente para que el mismo armazón
sirva luego a la variante adaptativa (misma cronología, mismas métricas).

NO entrena nada nuevo en import; todo se dispara desde las funciones.
"""

import logging
from dataclasses import asdict
from datetime import date

import pandas as pd

from src.trading.feature_engine import _SENTIMENT_FEATURES, _TECHNICAL_FEATURES
from src.trading.xgboost_model import (
    XGBHyperparams,
    _build_target,
    _internal_val_split,
    train,
)

logger = logging.getLogger(__name__)


def _resolve_feature_cols(df: pd.DataFrame, feature_cols: list[str] | None) -> list[str]:
    if feature_cols is not None:
        return feature_cols
    return [c for c in _SENTIMENT_FEATURES + _TECHNICAL_FEATURES if c in df.columns]


def walk_forward_static(
    features: pd.DataFrame,
    initial_train_ratio: float = 0.5,
    hyperparams: XGBHyperparams | None = None,
    feature_cols: list[str] | None = None,
    early_stopping_rounds: int = 50,
    balance: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Entrena el modelo una vez sobre la ventana inicial y predice día a día el resto.

    Args:
        features: DataFrame de features (salida de feature_engine.build_features).
        initial_train_ratio: Fracción de fechas únicas para la ventana de
            entrenamiento inicial; el resto es el horizonte forward.
        hyperparams: Hiperparámetros. Si None se usan los defaults; con balance=True
            se fija scale_pos_weight = neg/pos calculado en el core del train inicial.
        feature_cols: Features a usar. None → todas las disponibles del pipeline.
        early_stopping_rounds: Paciencia para early stopping (val interna del train).
        balance: Si True y no se pasó scale_pos_weight, lo ajusta al ratio del train.

    Returns:
        (pred_df, info) donde pred_df tiene columnas
        [ticker, date, target, proba, pred] por cada (ticker, día) del forward,
        e info incluye la fecha origen, tamaños y el scale_pos_weight usado.
    """
    df = _build_target(features)
    feature_cols = _resolve_feature_cols(df, feature_cols)

    dates: list[date] = sorted(df["date"].unique().tolist())
    n0 = int(len(dates) * initial_train_ratio)
    origin = dates[n0 - 1]                 # último día de la ventana de entrenamiento
    forward_dates = dates[n0:]             # horizonte de predicción

    train_block = df[df["date"] <= origin].copy()
    core_df, val_df = _internal_val_split(train_block)

    # scale_pos_weight ajustado al ratio del core (si procede)
    hp = hyperparams
    if hp is None:
        hp = XGBHyperparams()
        if balance:
            pos = int(core_df["target"].sum())
            neg = len(core_df) - pos
            if pos > 0:
                hp = XGBHyperparams(**{**asdict(hp), "scale_pos_weight": neg / pos})

    logger.info(
        "Walk-forward estático: origen=%s | train_inicial=%d filas | "
        "forward=%d días (%s → %s) | scale_pos_weight=%.4f",
        origin, len(train_block), len(forward_dates),
        forward_dates[0], forward_dates[-1], hp.scale_pos_weight,
    )

    model = train(
        core_df[feature_cols], core_df["target"],
        val_df[feature_cols], val_df["target"],
        hyperparams=hp, early_stopping_rounds=early_stopping_rounds,
    )

    # Predicción día a día sobre el horizonte forward (modelo fijo)
    forward = df[df["date"] > origin]
    chunks: list[pd.DataFrame] = []
    for d in forward_dates:
        day = forward[forward["date"] == d]
        if day.empty:
            continue
        proba = model.predict_proba(day[feature_cols])[:, 1]
        sub = day[["ticker", "date", "target"]].copy()
        sub["proba"] = proba
        sub["pred"] = (proba >= 0.5).astype(int)
        chunks.append(sub)

    pred_df = pd.concat(chunks, ignore_index=True)
    info = {
        "origin": origin,
        "n_train_initial": len(train_block),
        "n_core": len(core_df),
        "n_val": len(val_df),
        "n_forward": len(pred_df),
        "forward_start": forward_dates[0],
        "forward_end": forward_dates[-1],
        "scale_pos_weight": hp.scale_pos_weight,
        "feature_cols": feature_cols,
        "best_iteration": int(model.best_iteration),
    }
    return pred_df, info
