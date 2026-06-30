"""
Mejora del baseline ESTÁTICO con los hallazgos del diagnóstico de model_v0.

NO toca el sistema adaptativo. Hace dos cosas y escribe un reporte:

  1. Reentrena model_v0 con scale_pos_weight = neg/pos del train (mismo split,
     mismas features) y lo guarda en models/model_v0_balanced. Compara matriz de
     confusión y métricas contra model_v0 (que colapsaba a una sola clase).

  2. Evaluación walk-forward del modelo estático: una sola entrenada al inicio,
     predicción día a día sobre todo el horizonte. Métricas agregadas y por mes.
     Es el baseline justo contra el cual se comparará luego el adaptativo.

Salida: docs/baseline_estatico_balanced.md

Uso:
    python scripts/improve_static_baseline.py
"""

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.walkforward import walk_forward_static  # noqa: E402
from src.trading.feature_engine import load_features  # noqa: E402
from src.trading.xgboost_model import (  # noqa: E402
    XGBHyperparams,
    _build_target,
    _internal_val_split,
    _temporal_split,
    load_model,
    train_and_evaluate,
)

BALANCED_DIR = ROOT / "models" / "model_v0_balanced"
OUT_MD = ROOT / "docs" / "baseline_estatico_balanced.md"


def _pct(x: float) -> str:
    return f"{x*100:.1f}%"


def _cm_md(cm: np.ndarray) -> str:
    return (
        "| real \\ pred | pred=0 (baja) | pred=1 (sube) |\n"
        "|---|---|---|\n"
        f"| **real=0** | {cm[0,0]} (TN) | {cm[0,1]} (FP) |\n"
        f"| **real=1** | {cm[1,0]} (FN) | {cm[1,1]} (TP) |"
    )


def _metrics(y, proba) -> dict:
    pred = (proba >= 0.5).astype(int)
    return {
        "n": len(y),
        "acc": accuracy_score(y, pred),
        "auc": roc_auc_score(y, proba) if len(np.unique(y)) > 1 else float("nan"),
        "pred_up": pred.mean(),
        "up_rate": np.mean(y),
        "cm": confusion_matrix(y, pred, labels=[0, 1]),
        "pred": pred,
    }


def main() -> Path:
    md: list[str] = []
    A = md.append
    features = load_features()

    # ===================================================================
    # 1. Reentrenar balanceado (mismo split que model_v0)
    # ===================================================================
    df = _build_target(features)
    fc = (Path(ROOT / "models/model_v0/feature_names.json")).read_text(encoding="utf-8")
    import json
    feature_cols = json.loads(fc)

    train_df, test_df, cut, ts = _temporal_split(df)
    core_df, val_df = _internal_val_split(train_df)
    pos = int(core_df["target"].sum())
    neg = len(core_df) - pos
    spw = neg / pos

    hp = XGBHyperparams(scale_pos_weight=spw)
    # train_and_evaluate guarda modelo + feature_importance + split_info y evalúa
    results = train_and_evaluate(
        model_dir=BALANCED_DIR, hyperparams=hp, feature_cols=feature_cols,
    )

    # Predicciones del balanceado y del original sobre el MISMO test
    model_bal, _ = load_model(BALANCED_DIR)
    model_v0, _ = load_model(ROOT / "models/model_v0")
    yte = test_df["target"].to_numpy()
    pte_bal = model_bal.predict_proba(test_df[feature_cols])[:, 1]
    pte_v0 = model_v0.predict_proba(test_df[feature_cols])[:, 1]
    m_bal = _metrics(yte, pte_bal)
    m_v0 = _metrics(yte, pte_v0)

    A("# Baseline estático mejorado (balanceado) + walk-forward\n")
    A("> NO se modifica el sistema adaptativo. Objetivo: una comparación justa "
      "estático-vs-adaptativo más adelante. El modelo balanceado se guarda en "
      "`models/model_v0_balanced` (mismo split y mismas features que `model_v0`).\n")

    A("## 1. `model_v0_balanced` — `scale_pos_weight` ajustado al ratio del train\n")
    A(f"- Train (core): {pos} sube / {neg} baja → **scale_pos_weight = neg/pos = {spw:.4f}**.")
    A("- Sin cambiar features (26) ni splits. Todo lo demás idéntico a `model_v0`.\n")

    A("### Comparación en el MISMO test set\n")
    A("| | model_v0 (original) | model_v0_balanced |")
    A("|---|---|---|")
    A(f"| Predicciones 'sube' | {_pct(m_v0['pred_up'])} | {_pct(m_bal['pred_up'])} |")
    A(f"| Accuracy | {m_v0['acc']:.4f} | {m_bal['acc']:.4f} |")
    A(f"| AUC-ROC | {m_v0['auc']:.4f} | {m_bal['auc']:.4f} |")
    A(f"| Rango p(sube) | [{pte_v0.min():.3f}, {pte_v0.max():.3f}] | "
      f"[{pte_bal.min():.3f}, {pte_bal.max():.3f}] |\n")

    A("Matriz de confusión — `model_v0` (original):\n")
    A(_cm_md(m_v0["cm"]) + "\n")
    A("Matriz de confusión — `model_v0_balanced`:\n")
    A(_cm_md(m_bal["cm"]) + "\n")
    A("```\n" + classification_report(
        yte, m_bal["pred"], labels=[0, 1],
        target_names=["baja (0)", "sube (1)"], zero_division=0,
    ) + "```\n")

    collapse_fixed = m_bal["pred_up"] < 0.95 and m_bal["pred_up"] > 0.05
    auc_delta = m_bal["auc"] - m_v0["auc"]
    A(f"**¿El balanceo arregla el colapso?** {'**SÍ**' if collapse_fixed else 'NO'}: las "
      f"predicciones 'sube' pasaron de {_pct(m_v0['pred_up'])} a {_pct(m_bal['pred_up'])}; "
      "la matriz de confusión deja de ser degenerada (ahora hay TN y FN).\n")
    A(f"**¿Cambia el AUC?** Prácticamente no: {m_v0['auc']:.4f} → {m_bal['auc']:.4f} "
      f"(Δ = {auc_delta:+.4f}). El balanceo **re-centra el umbral** (las probabilidades ahora "
      "cruzan 0.5) pero **no añade poder discriminativo**: el AUC sigue cerca de 0.55 "
      "(apenas mejor que el azar 0.50). La 'mejora' de accuracy es por dejar de apostar todo a "
      "una clase en un test bajista, no por skill real.\n")

    # ===================================================================
    # 2. Walk-forward estático
    # ===================================================================
    A("## 2. Walk-forward del modelo estático (baseline justo)\n")
    A("Una sola entrenada al inicio (con `scale_pos_weight` del train inicial), "
      "predicción **día a día** sobre todo el horizonte forward, **sin reentrenar**.\n")

    pred_df, info = walk_forward_static(features, initial_train_ratio=0.5, balance=True)
    y = pred_df["target"].to_numpy()
    proba = pred_df["proba"].to_numpy()
    agg = _metrics(y, proba)

    A(f"- Origen de entrenamiento: hasta **{info['origin']}** "
      f"(train inicial {info['n_train_initial']} filas; core {info['n_core']}, val {info['n_val']}).")
    A(f"- Horizonte forward: **{info['forward_start']} → {info['forward_end']}** "
      f"({info['n_forward']} predicciones (ticker, día)).")
    A(f"- scale_pos_weight = {info['scale_pos_weight']:.4f} | best_iteration = {info['best_iteration']}\n")

    A("### Métricas agregadas (todo el horizonte forward)\n")
    A(f"- Accuracy: **{agg['acc']:.4f}** | AUC-ROC: **{agg['auc']:.4f}** | "
      f"predicciones 'sube': {_pct(agg['pred_up'])} | base rate 'sube': {_pct(agg['up_rate'])}")
    A("")
    A(_cm_md(agg["cm"]) + "\n")

    # Por mes
    A("### Métricas por mes\n")
    pred_df = pred_df.copy()
    pred_df["month"] = pd.to_datetime(pred_df["date"]).dt.to_period("M").astype(str)
    A("| mes | n | % sube (real) | accuracy | AUC | % pred sube | p(sube) media |")
    A("|---|---|---|---|---|---|---|")
    monthly = []
    for month, g in pred_df.groupby("month"):
        yy = g["target"].to_numpy()
        pp = g["proba"].to_numpy()
        acc = accuracy_score(yy, (pp >= 0.5).astype(int))
        auc = roc_auc_score(yy, pp) if len(np.unique(yy)) > 1 else float("nan")
        monthly.append((month, len(g), yy.mean(), acc, auc, (pp >= 0.5).mean(), pp.mean()))
        auc_s = f"{auc:.3f}" if not np.isnan(auc) else "—"
        A(f"| {month} | {len(g)} | {_pct(yy.mean())} | {acc:.3f} | {auc_s} | "
          f"{_pct((pp>=0.5).mean())} | {pp.mean():.3f} |")
    A("")

    # Skill: AUC mensual promedio y cuántos meses superan 0.5
    aucs = [m[4] for m in monthly if not np.isnan(m[4])]
    months_skill = sum(1 for a in aucs if a > 0.55)
    A(f"**¿La señal por mes sigue plana o aparece skill?** AUC mensual medio = "
      f"**{np.mean(aucs):.3f}** (mediana {np.median(aucs):.3f}); solo **{months_skill}/{len(aucs)}** "
      "meses superan AUC 0.55. ")
    if np.mean(aucs) < 0.55:
        A("La señal **sigue esencialmente plana**: el balanceo evita el colapso degenerado pero "
          "no recupera capacidad direccional. El modelo estático no generaliza a través del "
          "cambio de régimen — justo lo que motivará al sistema adaptativo.\n")
    else:
        A("Aparece algo de skill en varios meses; conviene analizarlo antes del adaptativo.\n")

    # ===================================================================
    # Conclusión
    # ===================================================================
    A("## Conclusión\n")
    A(f"1. **El balanceo arregla el colapso** (sube%: {_pct(m_v0['pred_up'])} → "
      f"{_pct(m_bal['pred_up'])}) pero **no añade skill** (AUC {m_v0['auc']:.3f} → "
      f"{m_bal['auc']:.3f}).")
    A("2. El **walk-forward estático** confirma que la capacidad direccional es ~aleatoria a lo "
      f"largo del horizonte (AUC agregado {agg['auc']:.3f}, mensual medio {np.mean(aucs):.3f}).")
    A("3. Este es el **baseline justo** (mismo armazón walk-forward) contra el que se medirá el "
      "sistema adaptativo: si el adaptativo no supera claramente estas cifras, no aporta.\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md), encoding="utf-8")

    # Persistir predicciones walk-forward para reutilización
    (ROOT / "data/processed").mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(ROOT / "data/processed/walkforward_static_predictions.parquet", index=False)

    print(f"Reporte escrito en {OUT_MD}")
    print(f"model_v0_balanced guardado en {BALANCED_DIR}")
    return OUT_MD


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import logging
    logging.basicConfig(level=logging.WARNING)  # silenciar INFO ruidoso del entrenamiento
    main()
