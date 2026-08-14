"""
Fase 5 — Experimento 5.2: ¿el sentimiento predice ACTIVIDAD (volumen/rango anormal)
por encima de un baseline autorregresivo?

Por activo y target (volumen anormal / rango anormal de mañana) entrena:
  (i) BASELINE: solo features autorregresivas (relvol/range rezagados 1-5).
  (ii) BASELINE+SENT: + las 10 features de sentimiento de Fase 4.
  + placebo (sentimiento permutado, autorregresivas intactas).
Criterio: ΔAUC(sent) > 0.02 en test Y modelo completo > placebo. Ver REGISTRY.md.
OHLCV completo de yfinance (el dataset de Fase 4 solo tenía close).
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import SEED  # noqa: E402

_DS = ROOT / "experiments" / "phase4" / "dataset_phase4"
_OUT = ROOT / "experiments" / "phase5"
_CACHE = _OUT / "cache"
_REPORT = _OUT / "results_5_2.md"

TICKERS = ["TSLA", "AMD", "DIS", "BA", "GILD"]
_SENT_FEATS = ["st_net", "st_net_3d", "st_vol", "st_mom", "st_has",
               "nw_net", "nw_net_3d", "nw_vol", "nw_mom", "nw_has"]
_AR_FEATS = [f"relvol_l{k}" for k in range(1, 6)] + [f"range_l{k}" for k in range(1, 6)]

_PARAMS = dict(n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8,
               colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss",
               early_stopping_rounds=40, random_state=SEED, n_jobs=4, tree_method="hist")
_VAL_FRAC = 0.15


def load_ohlcv(log) -> dict[str, pd.DataFrame]:
    cache = _CACHE / "ohlcv_5.parquet"
    if cache.exists():
        log.info("OHLCV yfinance: cache %s", cache.name)
        allp = pd.read_parquet(cache)
    else:
        import yfinance as yf
        rows = []
        for t in TICKERS:
            h = yf.Ticker(t).history(start="2015-01-01", end="2023-01-01", auto_adjust=True)
            h = h.reset_index()[["Date", "High", "Low", "Close", "Volume"]]
            h.columns = ["date", "high", "low", "close", "volume"]
            h["date"] = pd.to_datetime(h["date"]).dt.tz_localize(None).dt.normalize()
            h["ticker"] = t
            rows.append(h)
            log.info("  OHLCV %s: %d barras", t, len(h))
            time.sleep(0.4)
        allp = pd.concat(rows, ignore_index=True)
        _CACHE.mkdir(parents=True, exist_ok=True)
        allp.to_parquet(cache, index=False)
    return {t: g.sort_values("date").reset_index(drop=True) for t, g in allp.groupby("ticker")}


def _train_auc(Xtr, ytr, Xte, yte) -> float:
    n_val = max(20, int(len(Xtr) * _VAL_FRAC))
    m = XGBClassifier(**_PARAMS)
    m.fit(Xtr[:-n_val], ytr[:-n_val], eval_set=[(Xtr[-n_val:], ytr[-n_val:])], verbose=False)
    proba = m.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, proba) if len(np.unique(yte)) > 1 else float("nan"), m


def _permute(X, seed):
    rng = np.random.default_rng(seed)
    return X[rng.permutation(len(X))]


def _top_feats(model, feats, k=4):
    sc = model.get_booster().get_score(importance_type="gain")
    named = {feats[int(f[1:])]: v for f, v in sc.items()}
    top = sorted(named.items(), key=lambda kv: kv[1], reverse=True)[:k]
    return ", ".join(f"{n}={v:.1f}" for n, v in top) if top else "—"


def build_asset(ticker: str, ohlcv: pd.DataFrame, log) -> pd.DataFrame:
    df = ohlcv.copy()
    df["relvol"] = df["volume"] / df["volume"].rolling(20, min_periods=20).mean()
    df["range"] = (df["high"] - df["low"]) / df["close"]
    for k in range(1, 6):
        df[f"relvol_l{k}"] = df["relvol"].shift(k - 1)
        df[f"range_l{k}"] = df["range"].shift(k - 1)
    # valores continuos de mañana (para target)
    df["relvol_next"] = df["relvol"].shift(-1)
    df["range_next"] = df["range"].shift(-1)
    # merge features de sentimiento + split de Fase 4
    s4 = pd.read_parquet(_DS / f"{ticker}.parquet")[["date"] + _SENT_FEATS + ["split"]]
    df = df.merge(s4, on="date", how="inner")
    return df.sort_values("date").reset_index(drop=True)


def run() -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("phase5_5_2")
    ohlcv = load_ohlcv(log)

    targets = {"volumen": "relvol_next", "rango": "range_next"}
    rows = []
    for t in TICKERS:
        df = build_asset(t, ohlcv[t], log)
        for tname, contcol in targets.items():
            d = df.dropna(subset=_AR_FEATS + [contcol]).copy()
            tr = d[d["split"] == "train"]
            te = d[d["split"] == "test"]
            thr = tr[contcol].median()  # umbral solo con train
            ytr = (tr[contcol] > thr).astype(int).to_numpy()
            yte = (te[contcol] > thr).astype(int).to_numpy()

            Xtr_ar = tr[_AR_FEATS].to_numpy(float)
            Xte_ar = te[_AR_FEATS].to_numpy(float)
            Xtr_full = tr[_AR_FEATS + _SENT_FEATS].to_numpy(float)
            Xte_full = te[_AR_FEATS + _SENT_FEATS].to_numpy(float)

            auc_base, _ = _train_auc(Xtr_ar, ytr, Xte_ar, yte)
            auc_full, m_full = _train_auc(Xtr_full, ytr, Xte_full, yte)

            # placebo: permutar SOLO el bloque de sentimiento, AR intacto
            si = len(_AR_FEATS)
            Xtr_p = Xtr_full.copy(); Xtr_p[:, si:] = _permute(Xtr_full[:, si:], SEED)
            Xte_p = Xte_full.copy(); Xte_p[:, si:] = _permute(Xte_full[:, si:], SEED)
            auc_plc, _ = _train_auc(Xtr_p, ytr, Xte_p, yte)

            delta = auc_full - auc_base
            success = (not np.isnan(delta) and delta > 0.02 and auc_full > auc_plc)
            rows.append({
                "activo": t, "target": tname, "n_test": len(yte),
                "auc_base": round(auc_base, 4), "auc_full": round(auc_full, 4),
                "delta_auc": round(delta, 4), "auc_placebo": round(auc_plc, 4),
                "top_feats": _top_feats(m_full, _AR_FEATS + _SENT_FEATS),
                "EXITO": bool(success),
            })
            log.info("  %s / %s: base=%.4f full=%.4f Δ=%.4f plc=%.4f -> %s",
                     t, tname, auc_base, auc_full, delta, auc_plc,
                     "EXITO" if success else "neg")

    res = pd.DataFrame(rows)
    res.to_parquet(_OUT / "results_5_2.parquet", index=False)
    _write_report(res, log)
    return res


def _write_report(res: pd.DataFrame, log):
    n_ok = int(res["EXITO"].sum())
    L = ["# Fase 5.2 — Sentimiento → actividad (volumen/rango anormal de mañana)\n",
         "> **Criterio (REGISTRY.md):** ÉXITO ⇔ AUC(base+sent) − AUC(base) > 0.02 en test Y "
         "AUC(base+sent) > AUC(placebo). Baseline = autorregresivo (relvol/range rezagados 1-5). "
         "Sentimiento (10 features StockTwits+noticias) debe agregar valor SOBRE ese baseline. "
         "Placebo = sentimiento permutado, autorregresivas intactas.\n",
         f"**Settings ÉXITO: {n_ok} / 10** (5 activos × 2 targets).\n",
         "| activo | target | n test | AUC base | AUC base+sent | ΔAUC | AUC placebo | "
         "top features (gain) | veredicto |",
         "|--------|--------|--------|----------|---------------|------|-------------|"
         "---------------------|-----------|"]
    for _, r in res.iterrows():
        L.append(
            f"| {r['activo']} | {r['target']} | {r['n_test']} | {r['auc_base']:.4f} | "
            f"{r['auc_full']:.4f} | {r['delta_auc']:+.4f} | {r['auc_placebo']:.4f} | "
            f"{r['top_feats']} | {'**ÉXITO**' if r['EXITO'] else 'neg'} |")
    L += ["", "## Lectura\n"]
    L.append(f"- AUC baseline autorregresivo medio: **{res['auc_base'].mean():.4f}** "
             "(volumen y rango son fuertemente autocorrelacionados, como se esperaba).")
    L.append(f"- Mejora media al añadir sentimiento (ΔAUC): **{res['delta_auc'].mean():+.4f}**.")
    if n_ok == 0:
        L.append("- **Ningún setting supera el umbral incremental de +0.02 sobre el baseline "
                 "batiendo al placebo.** El sentimiento no aporta valor predictivo sobre la "
                 "actividad de mañana más allá del propio pasado autorregresivo del activo.")
    else:
        L.append(f"- **{n_ok}/10 settings** superan el criterio incremental. Ver detalle en la tabla.")
    L.append("")
    _REPORT.write_text("\n".join(L), encoding="utf-8")
    log.info("Reporte -> %s", _REPORT)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import json
    print(json.dumps(run().to_dict(orient="records"), indent=2, ensure_ascii=False, default=str))
