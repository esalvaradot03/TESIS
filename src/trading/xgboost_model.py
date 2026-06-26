"""
Modelo XGBoost para predicción de dirección de precio al día siguiente.

Target binario:
  1 = el precio de cierre sube mañana respecto a hoy
  0 = el precio de cierre baja o no cambia

Split temporal estricto (sin shuffle):
  - Entrenamiento : primero 80 % de las fechas únicas del dataset
  - Validación    : último 20 % del bloque de entrenamiento (early stopping)
  - Test          : último 20 % de todas las fechas (nunca visto durante el entrenamiento)

Artefactos persistidos en models/:
  xgboost_model.json        — modelo entrenado (formato nativo XGBoost)
  feature_names.json        — lista de features en el orden de entrenamiento
  split_info.json           — fechas de corte del split para reproducibilidad
  feature_importance.csv    — importancia (gain, weight, cover) de cada feature
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from xgboost import XGBClassifier

from config.settings import MODELS_DIR, PROCESSED_DIR, SEED
from src.trading.feature_engine import (
    _SENTIMENT_FEATURES,
    _TECHNICAL_FEATURES,
    load_features,
)

logger = logging.getLogger(__name__)

_FEATURES_FILE = PROCESSED_DIR / "features.parquet"
_MODEL_FILE = MODELS_DIR / "xgboost_model.json"
_FEATURE_NAMES_FILE = MODELS_DIR / "feature_names.json"
_SPLIT_INFO_FILE = MODELS_DIR / "split_info.json"
_IMPORTANCE_FILE = MODELS_DIR / "feature_importance.csv"

# Fracción de fechas únicas que va a entrenamiento (el resto es test)
_TRAIN_RATIO: float = 0.80
# Fracción interna del bloque de entrenamiento usada como validación (early stopping)
_INTERNAL_VAL_RATIO: float = 0.20


# ---------------------------------------------------------------------------
# Configuración de hiperparámetros
# ---------------------------------------------------------------------------

@dataclass
class XGBHyperparams:
    """
    Hiperparámetros por defecto del clasificador XGBoost.

    Justificación de los valores:
      n_estimators=500    — upper bound; early stopping detiene antes si procede.
      max_depth=6         — árbol de profundidad media; equilibra varianza y sesgo.
      learning_rate=0.05  — tasa conservadora para generalizar mejor con early stopping.
      subsample=0.8       — 80 % de filas por árbol: reduce sobreajuste y acelera.
      colsample_bytree=0.8— 80 % de features por árbol: similar a Random Forest.
      min_child_weight=5  — nodo mínimo de 5 muestras; evita splits en ruido.
      reg_alpha=0.1       — regularización L1 leve: promueve sparsity en features.
      reg_lambda=1.0      — regularización L2: valor por defecto de XGBoost.
      scale_pos_weight=1  — ajustado automáticamente si hay desequilibrio de clases.
    """
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 5
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    scale_pos_weight: float = 1.0
    tree_method: str = "hist"       # más rápido que "exact" para datasets medianos
    eval_metric: str = "logloss"
    random_state: int = SEED
    n_jobs: int = -1


# ---------------------------------------------------------------------------
# Preparación del dataset
# ---------------------------------------------------------------------------

def _build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade la columna 'target' = 1 si close[t+1] > close[t], 0 en caso contrario.

    Procesa por ticker de forma independiente para evitar contaminar el
    último día de un ticker con el primer día del siguiente.
    Descarta la última fila de cada ticker (no tiene close futuro).
    """
    df = df.sort_values(["ticker", "date"]).copy()
    df["next_close"] = df.groupby("ticker")["close"].shift(-1)
    df["target"] = (df["next_close"] > df["close"]).astype(int)
    df = df.dropna(subset=["next_close"]).drop(columns=["next_close"])
    return df.reset_index(drop=True)


def _temporal_split(
    df: pd.DataFrame,
    train_ratio: float = _TRAIN_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, date, date]:
    """
    Divide el DataFrame por fecha usando un corte temporal estricto.

    No mezcla fechas entre train y test; garantiza que el modelo nunca
    aprende sobre el futuro durante el entrenamiento.

    Args:
        df: DataFrame con columna 'date'.
        train_ratio: Fracción de fechas únicas para el bloque de entrenamiento.

    Returns:
        (train_df, test_df, train_cutoff_date, test_start_date)
    """
    all_dates: list[date] = sorted(df["date"].unique().tolist())
    n_train = int(len(all_dates) * train_ratio)

    train_cutoff = all_dates[n_train - 1]   # último día incluido en train
    test_start = all_dates[n_train]         # primer día de test

    train_df = df[df["date"] <= train_cutoff].copy()
    test_df = df[df["date"] >= test_start].copy()

    logger.info(
        "Split temporal: train %s → %s (%d filas, %d fechas) | "
        "test %s → %s (%d filas, %d fechas).",
        all_dates[0], train_cutoff, len(train_df), n_train,
        test_start, all_dates[-1], len(test_df), len(all_dates) - n_train,
    )
    return train_df, test_df, train_cutoff, test_start


def _internal_val_split(
    train_df: pd.DataFrame,
    val_ratio: float = _INTERNAL_VAL_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide el bloque de entrenamiento en (train_real, val_early_stop).

    La validación interna solo se usa para early stopping; no se incluye
    en el reporte de métricas finales.
    """
    train_dates = sorted(train_df["date"].unique().tolist())
    n_core = int(len(train_dates) * (1.0 - val_ratio))
    val_cutoff = train_dates[n_core - 1]

    core = train_df[train_df["date"] <= val_cutoff]
    val = train_df[train_df["date"] > val_cutoff]
    logger.info(
        "Validación interna (early stopping): core=%d filas | val=%d filas.",
        len(core), len(val),
    )
    return core, val


def prepare_dataset(
    features_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.DataFrame, pd.DataFrame, list[str], dict]:
    """
    Construye el target, realiza el split temporal y devuelve los conjuntos.

    Args:
        features_df: DataFrame de features (salida de feature_engine.build_features).
        feature_cols: Columnas a usar como features. Si es None, se usan todas
                      las columnas de _SENTIMENT_FEATURES + _TECHNICAL_FEATURES
                      presentes en el DataFrame.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, split_info
    """
    df = _build_target(features_df)

    if feature_cols is None:
        all_candidates = _SENTIMENT_FEATURES + _TECHNICAL_FEATURES
        feature_cols = [c for c in all_candidates if c in df.columns]

    logger.info("Features seleccionadas (%d): %s", len(feature_cols), feature_cols)

    # Distribución del target
    pos_rate = df["target"].mean()
    logger.info(
        "Target: %.1f %% positivos (sube) | %.1f %% negativos (baja). "
        "Total filas: %d.",
        pos_rate * 100, (1 - pos_rate) * 100, len(df),
    )

    train_df, test_df, train_cutoff, test_start = _temporal_split(df)
    core_df, val_df = _internal_val_split(train_df)

    X_train = core_df[feature_cols]
    y_train = core_df["target"]
    X_val = val_df[feature_cols]
    y_val = val_df["target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    split_info = {
        "train_start": str(df["date"].min()),
        "train_cutoff": str(train_cutoff),
        "test_start": str(test_start),
        "test_end": str(df["date"].max()),
        "n_train": len(core_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "n_features": len(feature_cols),
        "target_positive_rate": round(float(pos_rate), 4),
    }

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, split_info


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    hyperparams: XGBHyperparams | None = None,
    early_stopping_rounds: int = 50,
) -> XGBClassifier:
    """
    Entrena el clasificador XGBoost con early stopping sobre la validación interna.

    Args:
        X_train: Features de entrenamiento (core).
        y_train: Target de entrenamiento.
        X_val: Features de validación para early stopping.
        y_val: Target de validación.
        hyperparams: Objeto de hiperparámetros. Si None se usan los defaults.
        early_stopping_rounds: Detiene el entrenamiento si la métrica de val
                               no mejora en este número de rondas.

    Returns:
        Clasificador XGBoost entrenado.
    """
    hp = hyperparams or XGBHyperparams()

    # Ajustar scale_pos_weight si el dataset está desequilibrado
    pos_count = int(y_train.sum())
    neg_count = int(len(y_train) - pos_count)
    if pos_count > 0 and abs(pos_count - neg_count) / len(y_train) > 0.05:
        hp = XGBHyperparams(**{**asdict(hp), "scale_pos_weight": neg_count / pos_count})
        logger.info(
            "Clase desbalanceada (%.1f%% positivos): scale_pos_weight ajustado a %.3f.",
            pos_count / len(y_train) * 100,
            hp.scale_pos_weight,
        )

    model = XGBClassifier(**asdict(hp))

    logger.info(
        "Entrenando XGBoost: %d muestras train | %d val | "
        "max_estimators=%d | lr=%.3f | early_stop=%d rondas.",
        len(X_train), len(X_val), hp.n_estimators,
        hp.learning_rate, early_stopping_rounds,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=early_stopping_rounds,
        verbose=False,
    )

    best = model.best_iteration
    logger.info(
        "Entrenamiento completado. Mejor iteración: %d (de %d).",
        best, hp.n_estimators,
    )
    return model


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------

def evaluate(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    split_label: str = "test",
) -> dict:
    """
    Calcula métricas sobre el conjunto indicado y las registra en el log.

    Args:
        model: Modelo entrenado.
        X_test: Features del conjunto a evaluar.
        y_test: Target real.
        split_label: Etiqueta para el log ('val' o 'test').

    Returns:
        Dict con accuracy, roc_auc, precision_0/1, recall_0/1, f1_0/1.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["baja(0)", "sube(1)"],
                                   output_dict=True)

    logger.info("─── Métricas en %s ───────────────────────────────", split_label.upper())
    logger.info("Precisión direccional : %.4f  (objetivo > 0.55)", acc)
    logger.info("AUC-ROC               : %.4f", auc)
    logger.info(
        "Matriz de confusión:\n  TN=%d  FP=%d\n  FN=%d  TP=%d",
        cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1],
    )
    logger.info("\n%s", classification_report(
        y_test, y_pred, target_names=["baja(0)", "sube(1)"]
    ))

    metrics = {
        f"{split_label}_accuracy": round(acc, 4),
        f"{split_label}_roc_auc": round(auc, 4),
        f"{split_label}_precision_0": round(report["baja(0)"]["precision"], 4),
        f"{split_label}_precision_1": round(report["sube(1)"]["precision"], 4),
        f"{split_label}_recall_0": round(report["baja(0)"]["recall"], 4),
        f"{split_label}_recall_1": round(report["sube(1)"]["recall"], 4),
        f"{split_label}_f1_0": round(report["baja(0)"]["f1-score"], 4),
        f"{split_label}_f1_1": round(report["sube(1)"]["f1-score"], 4),
    }
    return metrics


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def _save_feature_importance(
    model: XGBClassifier,
    feature_cols: list[str],
    output_path: Path = _IMPORTANCE_FILE,
) -> pd.DataFrame:
    """
    Calcula e imprime la importancia de features (gain, weight, cover) y la persiste.

    Args:
        model: Modelo entrenado.
        feature_cols: Nombres de las features en el orden de entrenamiento.
        output_path: Ruta CSV de salida.

    Returns:
        DataFrame de importancia ordenado por gain descendente.
    """
    booster = model.get_booster()

    def _scores(importance_type: str) -> dict[str, float]:
        raw = booster.get_score(importance_type=importance_type)
        # get_score solo incluye features usadas; completar con 0 las demás
        return {f: raw.get(f, 0.0) for f in feature_cols}

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "gain": [_scores("gain")[f] for f in feature_cols],
        "weight": [_scores("weight")[f] for f in feature_cols],
        "cover": [_scores("cover")[f] for f in feature_cols],
    })

    # Normalizar gain a [0, 1] para facilitar interpretación
    total_gain = importance_df["gain"].sum()
    importance_df["gain_normalized"] = (
        importance_df["gain"] / total_gain if total_gain > 0 else 0.0
    )
    importance_df = importance_df.sort_values("gain", ascending=False).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(output_path, index=False)

    top_n = min(15, len(importance_df))
    logger.info("─── Top %d features por Gain ─────────────────────────", top_n)
    for _, row in importance_df.head(top_n).iterrows():
        bar = "█" * int(row["gain_normalized"] * 40)
        logger.info("  %-28s %.4f  %s", row["feature"], row["gain_normalized"], bar)

    logger.info("Feature importance guardada en %s", output_path)
    return importance_df


# ---------------------------------------------------------------------------
# Persistencia del modelo
# ---------------------------------------------------------------------------

def save_model(
    model: XGBClassifier,
    feature_cols: list[str],
    split_info: dict,
    model_dir: Path = MODELS_DIR,
) -> None:
    """
    Persiste el modelo y los metadatos asociados en model_dir.

    Archivos generados:
      xgboost_model.json  — pesos del modelo en formato nativo XGBoost.
      feature_names.json  — lista de features en el orden exacto de entrenamiento.
      split_info.json     — fechas del split y estadísticas del dataset.
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    model.save_model(str(model_dir / "xgboost_model.json"))

    (model_dir / "feature_names.json").write_text(
        json.dumps(feature_cols, indent=2), encoding="utf-8"
    )
    (model_dir / "split_info.json").write_text(
        json.dumps(split_info, indent=2, default=str), encoding="utf-8"
    )

    logger.info("Modelo guardado en %s", model_dir)


def load_model(model_dir: Path = MODELS_DIR) -> tuple[XGBClassifier, list[str]]:
    """
    Carga el modelo y la lista de features desde model_dir.

    Returns:
        (modelo, lista_de_features)
    """
    model_path = model_dir / "xgboost_model.json"
    names_path = model_dir / "feature_names.json"

    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en {model_path}.")

    model = XGBClassifier()
    model.load_model(str(model_path))

    feature_cols: list[str] = json.loads(names_path.read_text(encoding="utf-8"))
    logger.info("Modelo cargado desde %s (%d features).", model_path, len(feature_cols))
    return model, feature_cols


# ---------------------------------------------------------------------------
# Punto de entrada principal
# ---------------------------------------------------------------------------

def train_and_evaluate(
    features_path: Path = _FEATURES_FILE,
    model_dir: Path = MODELS_DIR,
    hyperparams: XGBHyperparams | None = None,
    feature_cols: list[str] | None = None,
    early_stopping_rounds: int = 50,
) -> dict:
    """
    Pipeline completo: carga features → prepara datos → entrena → evalúa → persiste.

    Args:
        features_path: Parquet de features (salida de feature_engine.build_features).
        model_dir: Directorio donde se guardan el modelo y los artefactos.
        hyperparams: Hiperparámetros del modelo. None usa los valores por defecto.
        feature_cols: Subconjunto de features a usar. None usa todos los disponibles.
        early_stopping_rounds: Rondas de paciencia para early stopping.

    Returns:
        Dict con todas las métricas de train/val/test y metadatos del experimento.
    """
    logger.info("═══ Iniciando pipeline XGBoost ═════════════════════════════")

    # --- Carga ---
    features_df = load_features(features_path)
    logger.info("Features cargadas: %d filas × %d columnas.", *features_df.shape)

    # --- Preparación ---
    (X_train, y_train, X_val, y_val,
     X_test, y_test, feat_cols, split_info) = prepare_dataset(features_df, feature_cols)

    # --- Entrenamiento ---
    model = train(X_train, y_train, X_val, y_val, hyperparams, early_stopping_rounds)

    # --- Evaluación ---
    val_metrics = evaluate(model, X_val, y_val, split_label="val")
    test_metrics = evaluate(model, X_test, y_test, split_label="test")

    # --- Feature importance ---
    importance_df = _save_feature_importance(model, feat_cols, model_dir / "feature_importance.csv")

    # --- Persistencia ---
    save_model(model, feat_cols, split_info, model_dir)

    # --- Resumen final ---
    results = {
        "split_info": split_info,
        "best_iteration": model.best_iteration,
        "hyperparams": asdict(hyperparams or XGBHyperparams()),
        **val_metrics,
        **test_metrics,
        "top_features_by_gain": importance_df.head(10)["feature"].tolist(),
    }

    logger.info("═══ Resumen final ══════════════════════════════════════════")
    logger.info("  Precisión direccional (test) : %.4f", results["test_accuracy"])
    logger.info("  AUC-ROC (test)               : %.4f", results["test_roc_auc"])
    logger.info("  Best iteration               : %d", results["best_iteration"])
    logger.info("  Objetivo tesis (acc > 0.55)  : %s",
                "✓ CUMPLIDO" if results["test_accuracy"] > 0.55 else "✗ no alcanzado")

    return results


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
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else MODELS_DIR

    results = train_and_evaluate(features_path=feat_path, model_dir=out_dir)

    print("\n── Métricas finales ──────────────────────────────────────────")
    for k, v in results.items():
        if isinstance(v, (int, float)):
            print(f"  {k:<35} {v}")
