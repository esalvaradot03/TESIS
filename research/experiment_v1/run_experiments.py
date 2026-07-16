"""
MATRIZ DE EXPERIMENTOS v1 (exactamente 8, protocolo pre-registrado).

Todos: XGBoost con los MISMOS hiperparámetros que model_v0 (XGBHyperparams) +
scale_pos_weight por fold, walk-forward expanding por año idéntico al protocolo
existente (test 2017-2023), features híbridas salvo controles.

  E1: h=1,  universo completo, híbrido
  E2: h=1,  ex-top50, híbrido
  E3: h=5,  ex-top50, híbrido
  E4: h=10, ex-top50, híbrido
  E5: h=1,  ex-top50, solo días evento, híbrido
  E6: h=5,  ex-top50, solo días evento, híbrido
  E7: h=5,  ex-top50, SOLO técnico (control)
  E8: h=5,  ex-top50, híbrido con sentimiento+aceleración PERMUTADOS (placebo)

Métricas (todas walk-forward): (a) AUC agregado + %meses>0.55, (b) precisión del
decil superior, (c) backtest long top-10% (10 bps/lado, exceso vs SPY, Sharpe
anualizado), (d) nº de trades (<100 => no concluyente).

Éxito (pre-registrado): precisión decil >= 55% Y Sharpe > 0.5 Y el placebo E8
NO cumple lo mismo.
"""

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.longrun.common import MASTER_DIR  # noqa: E402
from src.trading.xgboost_model import XGBHyperparams  # noqa: E402

_ENRICHED = MASTER_DIR / "dataset_v1_enriched.parquet"
_MODELS_DIR = ROOT / "research" / "experiment_v1" / "models"
_REPORT = ROOT / "research" / "experiment_v1" / "results_experiment_v1.md"

_TECH = ["rsi", "macd_hist", "bb_pct", "bb_width", "volatility_20d", "volume_ratio"]
_SENT = ["bullish_count", "bearish_count", "total_count", "unique_users", "bullish_ratio",
         "mention_count", "avg_score", "total_comments", "news_count", "news_sentiment_mean"]
_ACCEL = ["sent_delta_ma7", "mention_zscore_30d", "bull_bear_change"]
_HYBRID = _TECH + _SENT + _ACCEL

_FOLDS = [(2010, y - 1, y, y + 1) for y in range(2016, 2023)]  # (tr0,tr1,val,test): test 2017..2023

_COST_RT = 0.002   # 10 bps por lado, round-trip
_SEED = 42

# (nombre, horizonte, ex_top50, event_only, feature_set, placebo)
_EXPERIMENTS = [
    ("E1", 1, False, False, _HYBRID, False),
    ("E2", 1, True, False, _HYBRID, False),
    ("E3", 5, True, False, _HYBRID, False),
    ("E4", 10, True, False, _HYBRID, False),
    ("E5", 1, True, True, _HYBRID, False),
    ("E6", 5, True, True, _HYBRID, False),
    ("E7", 5, True, False, _TECH, False),          # control: solo técnico
    ("E8", 5, True, False, _HYBRID, True),         # control: placebo (sent+accel permutados)
]


def _fit(Xtr, ytr, Xval, yval, spw):
    hp = asdict(XGBHyperparams(scale_pos_weight=spw))
    m = XGBClassifier(**hp)
    if len(Xval) > 20 and yval.nunique() == 2:
        try:
            m.fit(Xtr, ytr, eval_set=[(Xval, yval)], early_stopping_rounds=50, verbose=False)
        except TypeError:
            m.fit(Xtr, ytr)
    else:
        m.fit(Xtr, ytr)
    return m


def _run_one(df, horizon, feats, placebo, log):
    """Corre el walk-forward de un experimento. Devuelve (pred_df, last_model)."""
    tgt = f"target_{horizon}d"
    exc = f"excess_{horizon}d"
    d = df.dropna(subset=[tgt, exc]).copy()

    if placebo:  # permutar sentimiento+aceleración (romper relación con el target)
        rng = np.random.default_rng(_SEED)
        perm_cols = [c for c in (_SENT + _ACCEL) if c in d.columns]
        idx = rng.permutation(len(d))
        for c in perm_cols:
            d[c] = d[c].to_numpy()[idx]

    d["year"] = d["date"].dt.year
    preds, last_model = [], None
    for tr0, tr1, vy, ty in _FOLDS:
        tr = d[(d["year"] >= tr0) & (d["year"] <= tr1)]
        va = d[d["year"] == vy]
        te = d[d["year"] == ty]
        if len(tr) < 100 or len(te) < 20 or tr[tgt].nunique() < 2:
            continue
        pos = int(tr[tgt].sum()); neg = len(tr) - pos
        spw = neg / pos if pos else 1.0
        m = _fit(tr[feats].astype(float), tr[tgt], va[feats].astype(float), va[tgt], spw)
        proba = m.predict_proba(te[feats].astype(float))[:, 1]
        sub = te[["date", exc]].copy()
        sub["proba"] = proba
        sub["y"] = te[tgt].to_numpy()
        sub = sub.rename(columns={exc: "excess"})
        preds.append(sub)
        last_model = m
    if not preds:
        return pd.DataFrame(), None
    return pd.concat(preds, ignore_index=True), last_model


def _metrics(pred, horizon):
    y = pred["y"].to_numpy()
    p = pred["proba"].to_numpy()
    # (a) AUC agregado + % meses > 0.55
    auc_agg = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    pm = pred.copy()
    pm["ym"] = pm["date"].dt.to_period("M")
    month_aucs = []
    for _, g in pm.groupby("ym"):
        yy = g["y"].to_numpy()
        if len(np.unique(yy)) > 1 and len(g) >= 20:
            month_aucs.append(roc_auc_score(yy, g["proba"].to_numpy()))
    pct_months = 100.0 * np.mean([a > 0.55 for a in month_aucs]) if month_aucs else float("nan")

    # (b) precisión decil superior
    thr = np.quantile(p, 0.9)
    top = pred[pred["proba"] >= thr]
    prec_dec = float(top["y"].mean()) if len(top) else float("nan")

    # (c)(d) backtest long top-10%, no solapado cada h, 10 bps/lado, exceso vs SPY
    dates = sorted(pred["date"].unique())
    rebal = dates[::horizon]
    rets, n_trades = [], 0
    for dt in rebal:
        day = pred[pred["date"] == dt]
        if len(day) < 10:
            continue
        q = day["proba"].quantile(0.9)
        picks = day[day["proba"] >= q]
        if len(picks) == 0:
            continue
        rets.append(float(picks["excess"].mean()) - _COST_RT)
        n_trades += len(picks)
    rets = pd.Series(rets).dropna()
    if len(rets) > 1 and rets.std() > 0:
        sharpe = rets.mean() / rets.std() * np.sqrt(252.0 / horizon)
        cum = float((1 + rets).prod() - 1)
    else:
        sharpe, cum = float("nan"), float("nan")
    return {
        "auc_agg": auc_agg, "pct_months_gt055": pct_months, "prec_decile": prec_dec,
        "sharpe": float(sharpe), "cum_excess": cum, "n_trades": int(n_trades),
        "n_rebal": len(rets),
    }


def run() -> dict:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("experiments_v1")
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(_ENRICHED)
    df["date"] = pd.to_datetime(df["date"])
    log.info("Enriched: %d filas, %d tickers.", len(df), df["ticker"].nunique())

    results = {}
    for name, h, ex_top, event_only, feats, placebo in _EXPERIMENTS:
        d = df
        if ex_top:
            d = d[d["is_top50"] == 0]
        if event_only:
            d = d[d["event_day"] == 1]
        log.info("--- %s (h=%dd, ex_top50=%s, event=%s, nfeat=%d, placebo=%s) sobre %d filas ---",
                 name, h, ex_top, event_only, len(feats), placebo, len(d))
        pred, model = _run_one(d, h, feats, placebo, log)
        if pred.empty:
            results[name] = {"error": "sin predicciones"}
            continue
        m = _metrics(pred, h)
        results[name] = {"h": h, "ex_top50": ex_top, "event_only": event_only,
                         "nfeat": len(feats), "placebo": placebo, "n_pred": len(pred), **m}
        if model is not None:
            model.save_model(str(_MODELS_DIR / f"{name}.json"))
        log.info("  %s: AUC=%.4f %%meses>0.55=%.0f prec_dec=%.3f sharpe=%.2f trades=%d",
                 name, m["auc_agg"], m["pct_months_gt055"], m["prec_decile"], m["sharpe"], m["n_trades"])

    _write_report(results, log)
    return results


def _verdict(r, placebo_ok):
    """Éxito pre-registrado: prec_decile>=0.55 Y sharpe>0.5 Y placebo NO cumple."""
    if "prec_decile" not in r:
        return "—"
    meets = (r["prec_decile"] >= 0.55) and (r["sharpe"] > 0.5) and (r["n_trades"] >= 100)
    if meets and not placebo_ok:
        return "**SEÑAL**"
    if meets and placebo_ok:
        return "no (placebo tambien cumple)"
    return "no"


def _write_report(results, log):
    e8 = results.get("E8", {})
    placebo_ok = ("prec_decile" in e8 and e8["prec_decile"] >= 0.55 and
                  e8.get("sharpe", 0) > 0.5 and e8.get("n_trades", 0) >= 100)
    L = ["# Resultados experimentos v1 (protocolo pre-registrado)\n",
         "Criterio de éxito (fijo): precisión decil superior >= 55% Y Sharpe backtest > 0.5 "
         "Y >= 100 trades Y el placebo E8 NO cumple lo mismo.\n",
         f"- E8 (placebo) cumpliría el umbral por sí solo: **{'SÍ' if placebo_ok else 'no'}** "
         "(si SÍ, invalida cualquier 'señal' que dependa del sentimiento).\n",
         "## Tabla de métricas (walk-forward, test 2017-2023)\n",
         "| Exp | config | AUC agg | %meses>0.55 | prec. decil | Sharpe (bt) | cum exceso | trades | veredicto |",
         "|-----|--------|--------|-------------|-------------|-------------|-----------|--------|-----------|"]
    labels = {
        "E1": "1d, universo completo, híbrido", "E2": "1d, ex-top50, híbrido",
        "E3": "5d, ex-top50, híbrido", "E4": "10d, ex-top50, híbrido",
        "E5": "1d, ex-top50, evento, híbrido", "E6": "5d, ex-top50, evento, híbrido",
        "E7": "5d, ex-top50, SOLO técnico (control)", "E8": "5d, ex-top50, placebo (sent permutado)",
    }
    for name in ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"]:
        r = results.get(name, {})
        if "prec_decile" not in r:
            L.append(f"| {name} | {labels[name]} | ERROR | | | | | | |")
            continue
        nc = "  ⚠<100" if r["n_trades"] < 100 else ""
        L.append(f"| {name} | {labels[name]} | {r['auc_agg']:.4f} | {r['pct_months_gt055']:.0f}% | "
                 f"{r['prec_decile']:.3f} | {r['sharpe']:.2f} | {r['cum_excess']:.2%} | "
                 f"{r['n_trades']}{nc} | {_verdict(r, placebo_ok)} |")
    L.append("")
    # E7 vs E3/E5/E6: aporte del sentimiento sobre lo técnico
    L.append("## E7 (solo técnico) vs experimentos con sentimiento\n")
    e7 = results.get("E7", {})
    if "prec_decile" in e7:
        L.append(f"- E7 (solo técnico, 5d): AUC {e7['auc_agg']:.4f}, prec.decil {e7['prec_decile']:.3f}, "
                 f"Sharpe {e7['sharpe']:.2f}.")
        for n in ["E3", "E6"]:
            r = results.get(n, {})
            if "prec_decile" in r:
                d_auc = r["auc_agg"] - e7["auc_agg"]
                L.append(f"- {n} (con sentimiento): AUC {r['auc_agg']:.4f} (Δ vs E7 = {d_auc:+.4f}), "
                         f"prec.decil {r['prec_decile']:.3f}, Sharpe {r['sharpe']:.2f}.")
        L.append("\nSi los experimentos con sentimiento NO superan a E7, el sentimiento no aporta "
                 "sobre lo técnico (respuesta a la pregunta de la tesis).\n")
    _REPORT.parent.mkdir(parents=True, exist_ok=True)
    _REPORT.write_text("\n".join(L), encoding="utf-8")
    log.info("Reporte escrito en %s", _REPORT)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run()
