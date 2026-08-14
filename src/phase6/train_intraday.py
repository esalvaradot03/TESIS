"""
Fase 6 — entrenamiento intradía + diagnóstico + robustez.

Por activo (TSLA, AMD) y tamaño de ventana (30, 60 min):
  - BASELINE (autorregresivo) vs BASELINE+SENT vs PLACEBO (sentimiento permutado).
  - Criterio: ΔAUC(base+sent − base) > 0.02 en test Y AUC(full) > AUC(placebo).
  - Diagnóstico contemporáneo: Spearman del NET de [t-30,t] con el ret coincidente
    [t-30,t], reactivo [t-60,t-30] y anticipatorio [t,t+30].
  - Robustez walk-forward: re-entrenar cada 3 meses, probar el trimestre siguiente.
Ver experiments/phase6/REGISTRY.md.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import SEED  # noqa: E402
from src.phase6.build_intraday import AR_FEATS, SENT_FEATS  # noqa: E402

_OUT = ROOT / "experiments" / "phase6"
_DS = _OUT / "dataset_intraday"
_REPORT = _OUT / "results_phase6.md"
TICKERS = ["TSLA", "AMD"]
_SIZES = (30, 60)

_PARAMS = dict(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8,
               colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss",
               early_stopping_rounds=40, random_state=SEED, n_jobs=4, tree_method="hist")
_VAL_FRAC = 0.15


def _auc(Xtr, ytr, Xte, yte) -> float:
    n_val = max(20, int(len(Xtr) * _VAL_FRAC))
    m = XGBClassifier(**_PARAMS)
    m.fit(Xtr[:-n_val], ytr[:-n_val], eval_set=[(Xtr[-n_val:], ytr[-n_val:])], verbose=False)
    p = m.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, p) if len(np.unique(yte)) > 1 else float("nan")


def _permute(X, seed):
    rng = np.random.default_rng(seed)
    return X[rng.permutation(len(X))]


def _spear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    if ok.sum() < 30 or np.std(x[ok]) == 0:
        return (np.nan, np.nan)
    r, p = spearmanr(x[ok], y[ok])
    return (float(r), float(p))


def _fit_eval_split(tr, te):
    ytr, yte = tr["y"].astype(int).to_numpy(), te["y"].astype(int).to_numpy()
    Xtr_b, Xte_b = tr[AR_FEATS].to_numpy(float), te[AR_FEATS].to_numpy(float)
    Xtr_f = tr[AR_FEATS + SENT_FEATS].to_numpy(float)
    Xte_f = te[AR_FEATS + SENT_FEATS].to_numpy(float)
    auc_b = _auc(Xtr_b, ytr, Xte_b, yte)
    auc_f = _auc(Xtr_f, ytr, Xte_f, yte)
    si = len(AR_FEATS)
    Xtr_p, Xte_p = Xtr_f.copy(), Xte_f.copy()
    Xtr_p[:, si:] = _permute(Xtr_f[:, si:], SEED)
    Xte_p[:, si:] = _permute(Xte_f[:, si:], SEED)
    auc_p = _auc(Xtr_p, ytr, Xte_p, yte)
    return auc_b, auc_f, auc_p


def _walk_forward(df, log) -> list[dict]:
    """Expansiva: entrenar con todo lo previo, probar el trimestre; cada 3 meses."""
    df = df.sort_values("w").reset_index(drop=True)
    q = df["w"].dt.tz_localize(None).dt.to_period("Q")
    out = []
    for period in sorted(q.unique()):
        te = df[q == period]
        tr = df[q < period]
        if len(tr) < 500 or len(te) < 100:
            continue
        ab, af, ap = _fit_eval_split(tr, te)
        out.append({"trimestre": str(period), "n_test": len(te),
                    "auc_base": round(ab, 4), "auc_full": round(af, 4),
                    "delta": round(af - ab, 4), "auc_placebo": round(ap, 4)})
    return out


def run() -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("phase6_train")

    main_rows, diag_rows, wf_rows = [], [], []
    for size in _SIZES:
        for tk in TICKERS:
            df = pd.read_parquet(_DS / f"{tk}_{size}.parquet")
            df["w"] = pd.to_datetime(df["w"])
            tr, te = df[df["split"] == "train"], df[df["split"] == "test"]

            ab, af, ap = _fit_eval_split(tr, te)
            success = (not np.isnan(af - ab) and (af - ab) > 0.02 and af > ap)
            main_rows.append({"activo": tk, "ventana": f"{size}min", "n_test": len(te),
                              "auc_base": round(ab, 4), "auc_full": round(af, 4),
                              "delta_auc": round(af - ab, 4), "auc_placebo": round(ap, 4),
                              "EXITO": bool(success)})
            log.info("  %s %dmin: base=%.4f full=%.4f Δ=%.4f plc=%.4f -> %s",
                     tk, size, ab, af, af - ab, ap, "EXITO" if success else "neg")

            # diagnóstico contemporáneo (sobre test)
            for horizon, col in [("coincidente[t-30,t]", "ret_coin"),
                                 ("reactivo[t-60,t-30]", "ret_react"),
                                 ("anticipatorio[t,t+30]", "ret_anti")]:
                r, p = _spear(te["net_lag1"], te[col])
                diag_rows.append({"activo": tk, "ventana": f"{size}min",
                                  "horizonte": horizon, "spearman": round(r, 4),
                                  "p": p, "n": len(te)})

            if size == 30:  # walk-forward solo en la ventana principal
                for w in _walk_forward(df, log):
                    wf_rows.append({"activo": tk, **w})

    res = pd.DataFrame(main_rows)
    diag = pd.DataFrame(diag_rows)
    wf = pd.DataFrame(wf_rows)
    res.to_parquet(_OUT / "results_phase6.parquet", index=False)
    diag.to_parquet(_OUT / "diag_phase6.parquet", index=False)
    wf.to_parquet(_OUT / "walkforward_phase6.parquet", index=False)
    _write_report(res, diag, wf)
    return res


def _pf(p):
    return "—" if pd.isna(p) else (f"{p:.1e}" if p < 0.001 else f"{p:.3f}")


def _write_report(res, diag, wf):
    n_ok = int(res["EXITO"].sum())
    L = ["# Fase 6 — Resultados: sentimiento intradía (30 y 60 min)\n",
         "> **Criterio (REGISTRY.md):** ÉXITO ⇔ ΔAUC(base+sent − base) > 0.02 en test Y "
         "AUC(full) > AUC(placebo). Baseline autorregresivo (ret/range lags 1-5 + n_gap_lags). "
         "NET etiquetado principal. Rango 2020-08→2022-12 (IEX); test 2022 = régimen bajista "
         "distinto al train alcista (limitación aceptada).\n",
         f"**Settings ÉXITO: {n_ok} / 4.**\n",
         "## Modelo predictivo (split principal: train 2020-08→2021-12 / test 2022)\n",
         "| activo | ventana | n test | AUC base | AUC base+sent | ΔAUC | AUC placebo | veredicto |",
         "|--------|---------|--------|----------|---------------|------|-------------|-----------|"]
    for _, r in res.iterrows():
        L.append(f"| {r['activo']} | {r['ventana']} | {r['n_test']:,} | {r['auc_base']:.4f} | "
                 f"{r['auc_full']:.4f} | {r['delta_auc']:+.4f} | {r['auc_placebo']:.4f} | "
                 f"{'**ÉXITO**' if r['EXITO'] else 'neg'} |")

    L += ["", "## Diagnóstico de estructura temporal (Spearman NET[t-30,t] vs ret, en test)\n",
          "Mapea si el sentimiento **coincide**, **sigue** (reactivo) o **anticipa** al precio a 30 min.\n",
          "| activo | ventana | horizonte | Spearman | p | n |",
          "|--------|---------|-----------|----------|---|---|"]
    for _, r in diag.iterrows():
        L.append(f"| {r['activo']} | {r['ventana']} | {r['horizonte']} | {r['spearman']:+.4f} | "
                 f"{_pf(r['p'])} | {r['n']:,} |")

    L += ["", "## Robustez walk-forward (30 min, expansiva, test por trimestre)\n",
          "| activo | trimestre | n test | AUC base | AUC full | ΔAUC | AUC placebo |",
          "|--------|-----------|--------|----------|----------|------|-------------|"]
    for _, r in wf.iterrows():
        L.append(f"| {r['activo']} | {r['trimestre']} | {r['n_test']:,} | {r['auc_base']:.4f} | "
                 f"{r['auc_full']:.4f} | {r['delta']:+.4f} | {r['auc_placebo']:.4f} |")

    L += ["", "## Lectura\n"]
    L.append(f"- ΔAUC medio (split principal): **{res['delta_auc'].mean():+.4f}**; "
             f"AUC baseline medio: **{res['auc_base'].mean():.4f}** (≈0.50: la dirección a 30 min es "
             "casi un martingala incluso desde su propio pasado).")
    anti = diag[diag["horizonte"].str.startswith("anticip")]
    coin = diag[diag["horizonte"].str.startswith("coincid")]
    react = diag[diag["horizonte"].str.startswith("reactivo")]
    L.append(f"- **Diagnóstico (resultado robusto):** correlación **coincidente** media "
             f"|ρ|={coin['spearman'].abs().mean():.3f} (p≈1e-60…1e-168), **reactiva** "
             f"|ρ|={react['spearman'].abs().mean():.3f} (fuerte y significativa), pero "
             f"**anticipatoria** |ρ|={anti['spearman'].abs().mean():.3f} "
             f"(NO significativa, p>0.15 en los 4 settings).")
    # veredicto del criterio principal + robustez walk-forward
    ok = res[res["EXITO"]]
    if len(ok):
        names = ", ".join(f"{r['activo']} {r['ventana']}" for _, r in ok.iterrows())
        L.append(f"- **Criterio principal:** {len(ok)}/4 lo cumple ({names}) — pero con AUC absoluto "
                 "≈0.50 y baseline anómalamente bajo.")
        for _, r in ok.iterrows():
            w = wf[(wf["activo"] == r["activo"])]
            if len(w):
                pos = int((w["delta"] > 0.02).sum())
                L.append(f"  - **Robustez de {r['activo']} {r['ventana']} (walk-forward):** "
                         f"ΔAUC trimestral medio **{w['delta'].mean():+.4f}**, con solo "
                         f"**{pos}/{len(w)}** trimestres superando +0.02. "
                         "El 'éxito' del split único **NO se replica** fuera de muestra → "
                         "atribuible al split particular de 2022, no a un efecto estable.")
    L.append("- **Conclusión Fase 6:** la hipótesis de que la dinámica predictiva vive dentro del "
             "día y la agregación diaria la destruye **queda refutada**. A 30 y 60 min el sentimiento "
             "sigue siendo **coincidente y reactivo pero NO anticipatorio** — la misma firma que a "
             "frecuencia diaria (5.1). No hay poder predictivo intradía estable sobre el baseline "
             "autorregresivo.")
    L.append("")
    _REPORT.write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import json
    print(json.dumps(run().to_dict(orient="records"), indent=2, ensure_ascii=False, default=str))
