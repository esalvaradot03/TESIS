"""
FASE 9 — Entrenamiento controlado de modelos (anti-p-hacking).

Walk-forward expanding por año (7 folds, test 2017..2023). Entrena EXACTAMENTE
las configuraciones definidas (M1-M6 sobre dataset_v1; M7-M8 si existe dataset_v2).
Sin búsqueda de hiperparámetros: XGBClassifier por defecto + scale_pos_weight.
Reporta TODOS los modelos. Backtest solo del mejor.

Criterio de "señal real" (definido ANTES de ver resultados):
  AUC walk-forward agregado > 0.55 Y ≥70% de folds con AUC > 0.53
  Y al menos una feature de sentimiento con |ρ| > 0.05 y p < 0.001 (de Fase 8).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from xgboost import XGBClassifier

from src.data.longrun.common import MASTER_DIR, setup_logger

JOB = "phase9_models"
ROOT = Path(__file__).resolve().parents[3]
_V1 = MASTER_DIR / "dataset_v1.parquet"
_V2 = MASTER_DIR / "dataset_v2.parquet"
_DONE = MASTER_DIR.parent / "logs" / "phase9.done"

_TECH = ["rsi", "macd_hist", "bb_pct", "bb_width", "volatility_20d", "volume_ratio"]
_STOCKTWITS = ["bullish_count", "bearish_count", "total_count", "unique_users", "bullish_ratio"]
_WSB = ["mention_count", "avg_score", "total_comments"]
_FNSPID = ["news_count", "news_sentiment_mean"]
_FINBERT = ["finbert_net_sentiment_mean", "finbert_prob_neg_mean", "finbert_msg_count"]

_FOLDS = [  # (train_start, train_end, val_year, test_year)
    (2010, 2015, 2016, 2017),
    (2010, 2016, 2017, 2018),
    (2010, 2017, 2018, 2019),
    (2010, 2018, 2019, 2020),
    (2010, 2019, 2020, 2021),
    (2010, 2020, 2021, 2022),
    (2010, 2021, 2022, 2023),
]


def _model_specs(has_v2: bool) -> dict[str, list[str]]:
    specs = {
        "M1_tecnico": _TECH,
        "M2_stocktwits": _STOCKTWITS,
        "M3_wsb": _WSB,
        "M4_fnspid": _FNSPID,
        "M5_todas": _TECH + _STOCKTWITS + _WSB + _FNSPID,
        "M6_sin_stocktwits": _TECH + _WSB + _FNSPID,
    }
    if has_v2:
        specs["M7_todas_finbert"] = _TECH + _STOCKTWITS + _WSB + _FNSPID + _FINBERT
        specs["M8_solo_finbert"] = _FINBERT
    return specs


def _fit_eval_fold(df, feats, fold, smoke=False):
    tr0, tr1, vy, ty = fold
    yr = df["date"].dt.year
    tr = df[(yr >= tr0) & (yr <= tr1)]
    va = df[yr == vy]
    te = df[yr == ty]
    feats = [c for c in feats if c in df.columns]
    if len(tr) < 50 or len(te) < 20 or tr["target"].nunique() < 2:
        return None
    pos = int(tr["target"].sum()); neg = len(tr) - pos
    spw = neg / pos if pos else 1.0
    n_est = 30 if smoke else 300
    model = XGBClassifier(n_estimators=n_est, scale_pos_weight=spw, random_state=42,
                          eval_metric="logloss", n_jobs=-1)
    Xtr, ytr = tr[feats].astype(float), tr["target"]
    if len(va) >= 20 and va["target"].nunique() == 2:
        try:
            model.fit(Xtr, ytr, eval_set=[(va[feats].astype(float), va["target"])],
                      early_stopping_rounds=30, verbose=False)
        except TypeError:
            model.fit(Xtr, ytr)  # versiones que no aceptan early_stopping_rounds en fit
    else:
        model.fit(Xtr, ytr)
    yte = te["target"].to_numpy()
    proba = model.predict_proba(te[feats].astype(float))[:, 1]
    pred = (proba >= 0.5).astype(int)
    auc = roc_auc_score(yte, proba) if len(np.unique(yte)) > 1 else float("nan")
    return {
        "test_year": ty, "n_test": len(te),
        "auc": auc, "acc": accuracy_score(yte, pred),
        "precision": precision_score(yte, pred, zero_division=0),
        "recall": recall_score(yte, pred, zero_division=0),
        "model": model, "feats": feats, "test_df": te, "proba": proba,
    }


def _train_model(df, name, feats, smoke, log):
    folds = _FOLDS[:1] if smoke else _FOLDS
    results = [r for f in folds if (r := _fit_eval_fold(df, feats, f, smoke)) is not None]
    if not results:
        return None
    aucs = [r["auc"] for r in results if not np.isnan(r["auc"])]
    agg = {
        "name": name, "n_folds": len(results),
        "auc_mean": float(np.nanmean([r["auc"] for r in results])),
        "acc_mean": float(np.mean([r["acc"] for r in results])),
        "prec_mean": float(np.mean([r["precision"] for r in results])),
        "rec_mean": float(np.mean([r["recall"] for r in results])),
        "folds_auc_gt_053": sum(1 for a in aucs if a > 0.53),
        "n_aucs": len(aucs),
        "per_fold": results,
    }
    # Feature importance del último fold (modelo entrenado con más datos)
    last = results[-1]["model"]
    imp = sorted(zip(results[-1]["feats"], last.feature_importances_),
                 key=lambda t: t[1], reverse=True)[:10]
    agg["top_importance"] = imp
    log.info("  %s: AUC medio %.4f (%d/%d folds >0.53)", name, agg["auc_mean"],
             agg["folds_auc_gt_053"], agg["n_aucs"])
    return agg


def _sentiment_signal_exists(df) -> tuple[bool, str]:
    """¿Alguna feature de sentimiento con |ρ|>0.05 y p<0.001 vs target?"""
    best = (None, 0.0, 1.0)
    for c in _STOCKTWITS + _WSB + _FNSPID + _FINBERT:
        if c not in df.columns:
            continue
        x = pd.to_numeric(df[c], errors="coerce")
        m = x.notna() & df["target"].notna()
        if m.sum() < 50 or x[m].nunique() < 3:
            continue
        rho, p = spearmanr(x[m], df["target"][m])
        if abs(rho) > abs(best[1]):
            best = (c, rho, p)
    ok = best[0] is not None and abs(best[1]) > 0.05 and best[2] < 0.001
    return ok, f"{best[0]}: ρ={best[1]:+.4f} p={best[2]:.2e}"


def _backtest_best(best, df, log) -> dict:
    """Long top-decil por proba, rebalanceo no solapado cada 5 días; vs SPY."""
    rows = []
    for r in best["per_fold"]:
        t = r["test_df"][["ticker", "date", "return_5d_forward"]].copy()
        t["proba"] = r["proba"]
        rows.append(t)
    if not rows:
        return {}
    bt = pd.concat(rows, ignore_index=True)
    bt["date"] = pd.to_datetime(bt["date"])

    # SPY benchmark (de indicators_daily)
    spy_ret = {}
    ind = MASTER_DIR / "indicators_daily.parquet"
    if ind.exists():
        s = pd.read_parquet(ind, columns=["ticker", "date", "return_5d_forward"])
        s = s[s["ticker"] == "SPY"]
        s["date"] = pd.to_datetime(s["date"])
        spy_ret = dict(zip(s["date"], s["return_5d_forward"]))

    dates = sorted(bt["date"].unique())
    rebal = dates[::5]  # no solapado: cada 5 días de trading
    strat, bench = [], []
    for d in rebal:
        day = bt[bt["date"] == d]
        if len(day) < 10:
            continue
        thr = day["proba"].quantile(0.9)
        picks = day[day["proba"] >= thr]
        strat.append(picks["return_5d_forward"].mean())
        bench.append(spy_ret.get(d, np.nan))
    strat = pd.Series(strat).dropna()
    bench = pd.Series(bench).dropna()
    if strat.empty:
        return {}

    def _metrics(s):
        eq = (1 + s).cumprod()
        sharpe = (s.mean() / s.std() * np.sqrt(52)) if s.std() > 0 else float("nan")
        dd = (eq / eq.cummax() - 1).min()
        return {"cum_return": float(eq.iloc[-1] - 1), "sharpe": float(sharpe),
                "max_drawdown": float(dd), "n_periods": len(s)}

    res = {"strategy": _metrics(strat), "spy_buyhold": _metrics(bench)}
    log.info("  Backtest %s: strat cumret %.2f%% sharpe %.2f | SPY cumret %.2f%%",
             best["name"], res["strategy"]["cum_return"]*100, res["strategy"]["sharpe"],
             res["spy_buyhold"]["cum_return"]*100)
    return res


def _write_report(path, dataset_label, models, criterion, best_name, backtest, signal_str):
    L = [f"# Modelos — {dataset_label}\n",
         "> Anti-p-hacking: configuraciones fijas, sin tuning, todas reportadas.\n",
         f"**Criterio de señal real (definido a priori):** AUC agregado > 0.55 Y "
         f"≥70% folds > 0.53 Y feature de sentimiento |ρ|>0.05 p<0.001.\n",
         f"- Señal de sentimiento (Fase 8): {signal_str}\n",
         "## Resumen de modelos\n",
         "| modelo | folds | AUC medio | acc | prec | recall | folds>0.53 | ¿señal? |",
         "|---|---|---|---|---|---|---|---|"]
    for m in models:
        passes = criterion[m["name"]]
        L.append(f"| {m['name']} | {m['n_folds']} | {m['auc_mean']:.4f} | {m['acc_mean']:.3f} | "
                 f"{m['prec_mean']:.3f} | {m['rec_mean']:.3f} | "
                 f"{m['folds_auc_gt_053']}/{m['n_aucs']} | {'**SÍ**' if passes else 'no'} |")
    L.append("")
    for m in models:
        L.append(f"### {m['name']} — AUC por fold\n")
        L.append("| test year | n | AUC | acc | prec | recall |")
        L.append("|---|---|---|---|---|---|")
        for r in m["per_fold"]:
            auc = f"{r['auc']:.3f}" if not np.isnan(r["auc"]) else "—"
            L.append(f"| {r['test_year']} | {r['n_test']} | {auc} | {r['acc']:.3f} | "
                     f"{r['precision']:.3f} | {r['recall']:.3f} |")
        L.append("\nTop features (importance): " +
                 ", ".join(f"{f}({v:.3f})" for f, v in m["top_importance"]) + "\n")
    L.append(f"## Mejor modelo: **{best_name}**\n")
    if backtest:
        s, b = backtest["strategy"], backtest["spy_buyhold"]
        L.append("Backtest (long top-decil, rebalanceo 5d no solapado):\n")
        L.append("| | retorno acum. | Sharpe | max drawdown | periodos |")
        L.append("|---|---|---|---|---|")
        L.append(f"| estrategia | {s['cum_return']:.2%} | {s['sharpe']:.2f} | {s['max_drawdown']:.2%} | {s['n_periods']} |")
        L.append(f"| SPY buy&hold | {b['cum_return']:.2%} | {b['sharpe']:.2f} | {b['max_drawdown']:.2%} | {b['n_periods']} |")
    L.append("")
    Path(path).write_text("\n".join(L), encoding="utf-8")


def run(smoke: bool = False, **kw) -> dict:
    log = setup_logger(JOB)
    if not _V1.exists():
        log.error("Fase 9: no existe dataset_v1.")
        return {"phase": JOB, "error": "dataset_v1 missing"}

    out = {}
    for label, path, doc in [("dataset_v1", _V1, ROOT/"docs"/"modelos_dataset_v1.md"),
                             ("dataset_v2", _V2, ROOT/"docs"/"modelos_dataset_v2.md")]:
        if not path.exists():
            if label == "dataset_v2":
                log.info("dataset_v2 no existe aún; se omite (M7-M8). Fase 7 puede completarlo después.")
            continue
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        has_v2 = (label == "dataset_v2")
        specs = _model_specs(has_v2)
        if smoke:  # solo un modelo en smoke
            specs = {"M1_tecnico": _TECH}
        log.info("Fase 9 sobre %s: %d filas, entrenando %d modelos.", label, len(df), len(specs))

        models = [m for n, f in specs.items() if (m := _train_model(df, n, f, smoke, log))]
        if not models:
            continue
        signal_ok, signal_str = _sentiment_signal_exists(df)
        criterion = {m["name"]: (m["auc_mean"] > 0.55 and
                                 m["folds_auc_gt_053"] >= 0.7 * m["n_aucs"] and signal_ok)
                     for m in models}
        best = max(models, key=lambda m: m["auc_mean"])
        backtest = _backtest_best(best, df, log) if not smoke else {}
        _write_report(doc, label, models, criterion, best["name"], backtest, signal_str)
        out[label] = {"models": {m["name"]: round(m["auc_mean"], 4) for m in models},
                      "best": best["name"], "any_signal": any(criterion.values()), "doc": str(doc)}
        log.info("Fase 9 %s: mejor=%s AUC=%.4f | señal real en algún modelo: %s",
                 label, best["name"], best["auc_mean"], any(criterion.values()))

    if not smoke:
        _DONE.write_text(pd.Timestamp.now().isoformat(), encoding="utf-8")
    return {"phase": JOB, **out}


if __name__ == "__main__":
    run(smoke="--smoke" in sys.argv)
