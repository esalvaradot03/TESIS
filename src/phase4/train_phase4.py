"""
Fase 4 — Tarea 2, entrenamiento y evaluación (10 modelos + 10 placebos).

Por cada activo (TSLA, AMD, DIS, BA, GILD) entrena dos XGBoost independientes:
  - Modelo A: solo features StockTwits (st_*)
  - Modelo B: solo features Noticias FNSPID (nw_*)
Split temporal estricto (2015-2020 train / 2021-2022 test), validación interna
temporal (último 15% del train) para early stopping. Control placebo obligatorio
(features permutadas temporalmente). Criterio pre-registrado: AUC test > 0.55 Y
AUC test > AUC placebo. Hiperparámetros y semilla fijos (ver REGISTRY.md).
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import SEED  # noqa: E402

_OUT = ROOT / "experiments" / "phase4"
_DS = _OUT / "dataset_phase4"
_REPORT = _OUT / "results_phase4.md"

TICKERS = ["TSLA", "AMD", "DIS", "BA", "GILD"]
_SOURCES = {
    "A_StockTwits": ["st_net", "st_net_3d", "st_vol", "st_mom", "st_has"],
    "B_Noticias":   ["nw_net", "nw_net_3d", "nw_vol", "nw_mom", "nw_has"],
}
_VAL_FRAC = 0.15

_PARAMS = dict(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8,
               colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss",
               early_stopping_rounds=40, random_state=SEED, n_jobs=4,
               tree_method="hist")


def _train_eval(Xtr, ytr, Xte, yte) -> dict:
    """Entrena con early stopping (val temporal = último 15% del train) y evalúa en test."""
    n_val = max(20, int(len(Xtr) * _VAL_FRAC))
    Xfit, yfit = Xtr[:-n_val], ytr[:-n_val]
    Xval, yval = Xtr[-n_val:], ytr[-n_val:]
    model = XGBClassifier(**_PARAMS)
    model.fit(Xfit, yfit, eval_set=[(Xval, yval)], verbose=False)
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(yte, proba) if len(np.unique(yte)) > 1 else float("nan")
    return {
        "auc": auc,
        "acc": accuracy_score(yte, pred),
        "prec_up": precision_score(yte, pred, pos_label=1, zero_division=0),
        "prec_down": precision_score(yte, pred, pos_label=0, zero_division=0),
        "_model": model,
    }


def _permute_rows(X: np.ndarray, seed: int) -> np.ndarray:
    """Permuta temporalmente las filas del bloque de features (placebo)."""
    rng = np.random.default_rng(seed)
    return X[rng.permutation(len(X))]


def _top_features(model, feats: list[str], k: int = 3) -> str:
    booster = model.get_booster()
    score = booster.get_score(importance_type="gain")
    # nombres f0,f1,... → feats
    named = {feats[int(k_[1:])]: v for k_, v in score.items()}
    top = sorted(named.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return ", ".join(f"{n}={v:.1f}" for n, v in top) if top else "—"


def run() -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("phase4_train")

    rows = []
    for t in TICKERS:
        df = pd.read_parquet(_DS / f"{t}.parquet").sort_values("date").reset_index(drop=True)
        tr = df[df["split"] == "train"]
        te = df[df["split"] == "test"]
        ytr = tr["y"].astype(int).to_numpy()
        yte = te["y"].astype(int).to_numpy()
        for src, feats in _SOURCES.items():
            Xtr = tr[feats].to_numpy(dtype=float)
            Xte = te[feats].to_numpy(dtype=float)

            real = _train_eval(Xtr, ytr, Xte, yte)
            # placebo: mismas features permutadas temporalmente (train y test por separado)
            plc = _train_eval(_permute_rows(Xtr, SEED), ytr,
                              _permute_rows(Xte, SEED), yte)

            success = (not np.isnan(real["auc"]) and real["auc"] > 0.55
                       and real["auc"] > plc["auc"])
            rows.append({
                "activo": t, "fuente": src, "n_test": len(yte),
                "auc": round(real["auc"], 4), "acc": round(real["acc"], 4),
                "prec_up": round(real["prec_up"], 4), "prec_down": round(real["prec_down"], 4),
                "auc_placebo": round(plc["auc"], 4),
                "top_feats": _top_features(real["_model"], feats),
                "EXITO": bool(success),
            })
            log.info("  %s / %s: AUC=%.4f (placebo %.4f) -> %s",
                     t, src, real["auc"], plc["auc"], "EXITO" if success else "neg")

    res = pd.DataFrame(rows)
    res.to_parquet(_OUT / "results_phase4.parquet", index=False)
    _write_report(res, log)
    return res


def _write_report(res: pd.DataFrame, log):
    n_ok = int(res["EXITO"].sum())
    a = res[res["fuente"] == "A_StockTwits"]
    b = res[res["fuente"] == "B_Noticias"]
    L = ["# Fase 4 — Resultados (10 modelos + 10 placebos)\n",
         "> **Criterio pre-registrado (REGISTRY.md):** ÉXITO ⇔ AUC test > 0.55 Y "
         "AUC test > AUC placebo. Sin ajuste post-hoc. Modelo A = StockTwits (label nativo), "
         "Modelo B = Noticias FNSPID (FinBERT sobre titulares). Solo features de sentimiento.\n",
         f"**Experimentos ÉXITO: {n_ok} / 10.** "
         f"AUC media — A (StockTwits): {a['auc'].mean():.4f} | B (Noticias): {b['auc'].mean():.4f}. "
         f"Placebo medio — A: {a['auc_placebo'].mean():.4f} | B: {b['auc_placebo'].mean():.4f}.\n",
         "## Tabla resumen\n",
         "| activo | fuente | n test | AUC | acc | prec sube | prec baja | AUC placebo | "
         "top features (gain) | veredicto |",
         "|--------|--------|--------|-----|-----|-----------|-----------|-------------|"
         "---------------------|-----------|"]
    for _, r in res.iterrows():
        L.append(
            f"| {r['activo']} | {r['fuente']} | {r['n_test']} | {r['auc']:.4f} | {r['acc']:.4f} | "
            f"{r['prec_up']:.4f} | {r['prec_down']:.4f} | {r['auc_placebo']:.4f} | "
            f"{r['top_feats']} | {'**ÉXITO**' if r['EXITO'] else 'neg'} |")
    L += ["", "## Veredicto agregado\n"]
    if n_ok == 0:
        L.append("**NEGATIVO en los 10 experimentos.** Ni el sentimiento de StockTwits ni el de "
                 "noticias FNSPID predice la dirección del retorno del día siguiente por encima "
                 "del azar (AUC>0.55) y del placebo, en ninguno de los 5 activos. Consistente con "
                 "el resultado negativo de las Fases 1-3 sobre el universo amplio: concentrarse "
                 "en 5 activos de alta actividad y añadir noticias no revierte la conclusión.")
    else:
        L.append(f"**{n_ok}/10 cumplen el criterio pre-registrado.** Ver detalle por activo/fuente "
                 "en la tabla. Se reporta tal cual, sin reajustar umbrales.")
    L.append("")
    _REPORT.write_text("\n".join(L), encoding="utf-8")
    log.info("Reporte -> %s", _REPORT)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import json
    print(json.dumps(run().to_dict(orient="records"), indent=2, ensure_ascii=False, default=str))
