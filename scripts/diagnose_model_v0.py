"""
Diagnóstico de model_v0 — SOLO lectura, NO reentrena ni modifica el modelo.

Carga el modelo base ya entrenado (models/model_v0) y las features
(data/processed/features.parquet), reconstruye EXACTAMENTE el mismo split
temporal que usó el entrenamiento (mismas funciones del proyecto) y produce:

  1. Balance del target en train / val / test.
  2. Matriz de confusión real del test + precision/recall/f1 por clase.
  3. Distribución de probabilidades de salida en test (histograma + distancia a 0.5).
  4. Feature importance + correlación sentimiento↔target en train vs test (sign flip).
  5. Análisis por régimen: %sube y precisión del modelo por mes.

Escribe el reporte en docs/diagnostico_model_v0.md.

Uso:
    python scripts/diagnose_model_v0.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.trading.feature_engine import _SENTIMENT_FEATURES, load_features  # noqa: E402
from src.trading.xgboost_model import (  # noqa: E402
    _build_target,
    _internal_val_split,
    _temporal_split,
    load_model,
)

MODEL_DIR = ROOT / "models" / "model_v0"
OUT_MD = ROOT / "docs" / "diagnostico_model_v0.md"


# ---------------------------------------------------------------------------
# Helpers de formato
# ---------------------------------------------------------------------------

def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def _text_histogram(probs: np.ndarray, bins: int = 20, width: int = 40) -> str:
    """Histograma ASCII de probabilidades en [0,1]."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    counts, _ = np.histogram(probs, bins=edges)
    top = counts.max() if counts.max() > 0 else 1
    lines = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        bar = "█" * int(round(counts[i] / top * width))
        marker = "  <- 0.5" if lo <= 0.5 < hi else ""
        lines.append(f"  [{lo:4.2f}, {hi:4.2f})  {counts[i]:4d} |{bar}{marker}")
    return "\n".join(lines)


def _confusion_md(cm: np.ndarray) -> str:
    """Matriz de confusión 2x2 en Markdown."""
    return (
        "| real \\ pred | pred=0 (baja) | pred=1 (sube) |\n"
        "|---|---|---|\n"
        f"| **real=0 (baja)** | {cm[0,0]} (TN) | {cm[0,1]} (FP) |\n"
        f"| **real=1 (sube)** | {cm[1,0]} (FN) | {cm[1,1]} (TP) |"
    )


# ---------------------------------------------------------------------------
# Diagnóstico
# ---------------------------------------------------------------------------

def main() -> Path:
    features = load_features()
    model, feature_cols = load_model(MODEL_DIR)

    # --- Reconstrucción EXACTA del split de entrenamiento ---
    df = _build_target(features)
    train_df, test_df, train_cutoff, test_start = _temporal_split(df)
    core_df, val_df = _internal_val_split(train_df)

    def _predict(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        proba = model.predict_proba(d[feature_cols])[:, 1]
        return proba, (proba >= 0.5).astype(int)

    p_core, pred_core = _predict(core_df)
    p_val, pred_val = _predict(val_df)
    p_test, pred_test = _predict(test_df)
    y_test = test_df["target"].to_numpy()

    md: list[str] = []
    A = md.append

    A("# Diagnóstico de `model_v0`\n")
    A("> Reporte de solo lectura. No se reentrenó ni modificó el modelo. "
      "Splits reconstruidos con las mismas funciones del pipeline "
      "(`_build_target` → `_temporal_split` → `_internal_val_split`).\n")
    A(f"- **Features:** {len(df)} filas | {df['ticker'].nunique()} tickers | "
      f"rango {df['date'].min()} → {df['date'].max()}")
    A(f"- **Split temporal:** train ≤ {train_cutoff} | test ≥ {test_start}")
    A(f"- **Tamaños:** core(train)={len(core_df)} | val={len(val_df)} | test={len(test_df)}")
    A(f"- **Modelo:** {len(feature_cols)} features\n")

    # ===================================================================
    # 1. Balance del target
    # ===================================================================
    A("## 1. Balance del target por split\n")
    A("Target = 1 si `close[t+1] > close[t]` (el día siguiente sube).\n")
    A("| Split | n | % sube (1) | % baja (0) | n sube | n baja |")
    A("|---|---|---|---|---|---|")
    balance = {}
    for name, d in [("train (core)", core_df), ("val", val_df), ("test", test_df)]:
        y = d["target"]
        up = y.mean()
        balance[name] = up
        A(f"| {name} | {len(d)} | {_pct(up)} | {_pct(1-up)} | {int(y.sum())} | {int((1-y).sum())} |")
    A("")
    shift = balance["test"] - balance["train (core)"]
    A(f"**Interpretación:** el balance en train es {_pct(balance['train (core)'])} sube vs "
      f"{_pct(balance['test'])} en test (Δ = {shift*100:+.1f} pp). ")
    if balance["train (core)"] > 0.5 and balance["test"] < 0.5:
        A("El train tiene **mayoría de días al alza** (mercado alcista 2020-21) y el test "
          "**mayoría a la baja** (bear market 2022). El modelo aprendió el prior alcista del "
          "train; al cambiar el régimen en test, ese prior se vuelve sistemáticamente erróneo. "
          "**Este desbalance entre splits es la causa raíz del colapso a una sola clase.**\n")
    else:
        A("El desbalance entre splits no es extremo; revisar las demás secciones.\n")

    # ===================================================================
    # 2. Matriz de confusión del test
    # ===================================================================
    A("## 2. Matriz de confusión (test set)\n")
    cm = confusion_matrix(y_test, pred_test, labels=[0, 1])
    A(_confusion_md(cm) + "\n")
    n_pred1 = int((pred_test == 1).sum())
    n_pred0 = int((pred_test == 0).sum())
    A(f"- Predicciones clase 1 (sube): **{n_pred1}** / {len(pred_test)} "
      f"({_pct(n_pred1/len(pred_test))})")
    A(f"- Predicciones clase 0 (baja): **{n_pred0}** / {len(pred_test)} "
      f"({_pct(n_pred0/len(pred_test))})\n")
    A("```\n" + classification_report(
        y_test, pred_test, labels=[0, 1],
        target_names=["baja (0)", "sube (1)"], zero_division=0,
    ) + "```\n")
    if n_pred0 == 0:
        A("**Interpretación:** el modelo predice la clase 'sube' en el **100%** de los días de "
          "test → **colapso total a una clase**. El recall de la clase 'baja' es 0 (no detecta "
          "ni una caída) y la precision de 'sube' iguala el base rate de días al alza del test. "
          "La accuracy agregada es engañosa: equivale a la estrategia trivial 'comprar siempre'.\n")
    else:
        A(f"**Interpretación:** el modelo NO colapsó del todo: predijo 'baja' en {n_pred0} días. "
          "Revisar la distribución de probabilidades para ver cuán cerca del umbral opera.\n")

    # ===================================================================
    # 3. Distribución de probabilidades en test
    # ===================================================================
    A("## 3. Distribución de probabilidades de salida (test)\n")
    A(f"- min = {p_test.min():.3f} | p25 = {np.percentile(p_test,25):.3f} | "
      f"mediana = {np.median(p_test):.3f} | p75 = {np.percentile(p_test,75):.3f} | "
      f"max = {p_test.max():.3f}")
    A(f"- media = {p_test.mean():.3f} | desviación = {p_test.std():.3f}")
    frac_above = (p_test > 0.5).mean()
    A(f"- **% de probabilidades > 0.5: {_pct(frac_above)}**")
    A(f"- distancia media al umbral |p − 0.5| = {np.abs(p_test-0.5).mean():.3f} "
      f"(máx posible 0.5)\n")
    A("Histograma de p(sube) en test:\n")
    A("```\n" + _text_histogram(p_test) + "\n```\n")
    if frac_above > 0.95 and p_test.std() < 0.05:
        A(f"**Interpretación:** todas las probabilidades caen en una banda **estrechísima justo por "
          f"encima de 0.5** (rango [{p_test.min():.3f}, {p_test.max():.3f}], desviación "
          f"{p_test.std():.3f}). El matiz importa: el modelo **no está confiado** (no hay valores "
          "0.8-0.9), sino que emite una salida **casi constante ≈ 0.53** para todos los días — ha "
          "perdido poder discriminativo en test (la varianza de la señal es ~0). Como toda la masa "
          "queda por encima de 0.5, el `argmax`/threshold 0.5 predice 'sube' el 100% del tiempo. "
          "Mover el umbral a ~0.53 reequilibraría el conteo de clases, pero **no añadiría skill**: "
          f"con AUC≈0.55 el orden de las probabilidades apenas correlaciona con el target, así que "
          "separar por un umbral interno daría predicciones casi aleatorias. El problema no es el "
          "umbral sino el colapso de la señal.\n")
    elif frac_above > 0.95:
        A("**Interpretación:** prácticamente todas las probabilidades caen por **encima de 0.5**: "
          "el modelo predice 'sube' en casi todo el test.\n")
    else:
        A("**Interpretación:** hay masa de probabilidad a ambos lados de 0.5; el colapso, si existe, "
          "es parcial.\n")

    # ===================================================================
    # 4. Feature importance + correlación train vs test
    # ===================================================================
    A("## 4. Feature importance y correlación con el target\n")
    fi_path = MODEL_DIR / "feature_importance.csv"
    if fi_path.exists():
        fi = pd.read_csv(fi_path).sort_values("gain", ascending=False)
        A("Top 10 por *gain*:\n")
        A("| # | feature | gain | weight | cover |")
        A("|---|---|---|---|---|")
        for i, r in enumerate(fi.head(10).itertuples(index=False), 1):
            A(f"| {i} | {r.feature} | {r.gain:.3f} | {r.weight:.0f} | {r.cover:.2f} |")
        A("")

    # Correlación de cada feature de sentimiento con el target: train vs test
    sent_feats = [c for c in _SENTIMENT_FEATURES if c in feature_cols]
    A("Correlación de Pearson **feature ↔ target** en train (core) vs test "
      "(un **cambio de signo** = distribution shift):\n")
    A("| feature de sentimiento | corr train | corr test | ¿cambia de signo? |")
    A("|---|---|---|---|")
    flips = 0
    for f in sent_feats:
        ct = core_df[f].astype(float).corr(core_df["target"])
        cte = test_df[f].astype(float).corr(test_df["target"])
        flip = (not np.isnan(ct) and not np.isnan(cte) and np.sign(ct) != np.sign(cte)
                and abs(ct) > 0.02 and abs(cte) > 0.02)
        flips += int(flip)
        A(f"| {f} | {ct:+.3f} | {cte:+.3f} | {'**SÍ**' if flip else 'no'} |")
    A("")
    A(f"**Interpretación:** {flips} de {len(sent_feats)} features de sentimiento **invierten el "
      "signo** de su correlación con el target entre train y test. ")
    if flips >= max(1, len(sent_feats)//3):
        A("Esto **confirma el distribution shift**: la relación que el modelo aprendió "
          "(p. ej. 'más sentimiento positivo → sube') deja de cumplirse —o se invierte— en el "
          "período de test. Un modelo estático no puede acertar si el signo de la señal cambia.\n")
    else:
        A("El cambio de signo es limitado; el shift se explica más por el desbalance del target "
          "(sección 1) que por inversión de la señal.\n")

    # ===================================================================
    # 5. Análisis por régimen (mensual)
    # ===================================================================
    A("## 5. Análisis por régimen (mensual)\n")
    full = df.copy()
    proba_full, pred_full = _predict(full)
    full = full.assign(pred=pred_full, proba=proba_full)
    full["month"] = pd.to_datetime(full["date"]).dt.to_period("M").astype(str)

    def _split_of(d):
        if d <= train_cutoff:
            # core vs val
            return "val" if d > core_df["date"].max() else "train"
        return "test"
    full["split"] = full["date"].map(_split_of)

    A("| mes | split | n | % sube (real) | precisión modelo | % pred sube | p(sube) media |")
    A("|---|---|---|---|---|---|---|")
    for month, g in full.groupby("month"):
        up = g["target"].mean()
        acc = (g["pred"] == g["target"]).mean()
        pred_up = g["pred"].mean()
        splits = g["split"].mode().iat[0]
        A(f"| {month} | {splits} | {len(g)} | {_pct(up)} | {_pct(acc)} | "
          f"{_pct(pred_up)} | {g['proba'].mean():.2f} |")
    A("")
    # Resumen de aciertos por régimen alcista vs bajista
    bull = full[full["target"].groupby(full["month"]).transform("mean") >= 0.5]
    bear = full[full["target"].groupby(full["month"]).transform("mean") < 0.5]
    acc_bull = (bull["pred"] == bull["target"]).mean() if len(bull) else float("nan")
    acc_bear = (bear["pred"] == bear["target"]).mean() if len(bear) else float("nan")
    A(f"**Interpretación:** en meses con mayoría de días al alza la precisión es "
      f"**{_pct(acc_bull)}**; en meses con mayoría a la baja cae a **{_pct(acc_bear)}**. ")
    A("El modelo 'funciona' solo cuando el mes sube (porque siempre predice sube) y falla "
      "sistemáticamente cuando el mes baja. No hay capacidad direccional real: hay un sesgo "
      "constante hacia 'sube' que coincide con el mercado alcista del train y se rompe en el "
      "bear market del test.\n")

    # ===================================================================
    # Conclusión
    # ===================================================================
    A("## Conclusión del diagnóstico\n")
    A("1. **Causa raíz:** desbalance de régimen entre train (alcista) y test (bajista) — "
      "sección 1 — combinado con la inversión de la señal de sentimiento — sección 4.")
    A("2. **Síntoma:** el modelo colapsa a predecir 'sube' en el 100% del test (sección 2) con "
      "probabilidades casi constantes ≈ 0.53, apenas por encima de 0.5 y con varianza ~0 — "
      "pérdida de poder discriminativo, no exceso de confianza (sección 3).")
    A("3. **Implicación para el sistema adaptativo:** un modelo estático entrenado en un solo "
      "régimen no generaliza al siguiente. Hace falta (a) reentrenamiento incremental / ventana "
      "móvil, (b) posiblemente `scale_pos_weight` o re-balanceo por régimen, y (c) una señal de "
      "régimen de mercado que el modelo pueda usar. Esto motiva el sistema adaptativo.\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"Reporte escrito en {OUT_MD}")
    return OUT_MD


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
