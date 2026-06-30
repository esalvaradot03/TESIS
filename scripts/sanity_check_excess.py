"""
Variante metodológica única: target de EXCESO sobre el mercado (SPY) a 5 días
+ features ESTACIONARIAS solamente. Reporta números crudos, sin interpretación.

Target:  target[T] = 1  si  ret_ticker[T..T+5] > ret_SPY[T..T+5]   (mismo calendario)
         excess[T]   = ret_ticker - ret_SPY     (para Spearman)
Features estacionarias: rsi, macd_hist, bb_pct, bb_width + todas las de sentimiento.
Se eliminan: close, bb_upper, bb_middle, bb_lower, volume, macd, macd_signal.

Salida: docs/sanity_check_excess.md
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.trading.feature_engine import _SENTIMENT_FEATURES, load_features  # noqa: E402
from src.trading.indicators import download_prices  # noqa: E402
from src.trading.xgboost_model import (  # noqa: E402
    XGBHyperparams,
    _internal_val_split,
    _temporal_split,
    train,
)

HORIZON = 5
OUT_MD = ROOT / "docs" / "sanity_check_excess.md"

# Features estacionarias técnicas a conservar (volume_ratio no existe en el dataset)
TECH_STATIONARY = ["rsi", "macd_hist", "bb_pct", "bb_width"]


def _pct(x: float) -> str:
    return f"{x*100:.1f}%"


def _auc(y, proba) -> float:
    return roc_auc_score(y, proba) if len(np.unique(y)) > 1 else float("nan")


def _get_spy_closes(start: date, end: date) -> dict:
    prices = download_prices(["SPY"], start, end)
    prices = prices[prices["ticker"] == "SPY"]
    return dict(zip(prices["date"], prices["close"].astype(float)))


def _build_excess_target(features: pd.DataFrame, spy: dict, h: int) -> pd.DataFrame:
    df = features.sort_values(["ticker", "date"]).copy()
    df["fwd_close"] = df.groupby("ticker")["close"].shift(-h)
    df["fwd_date"] = df.groupby("ticker")["date"].shift(-h)
    df = df.dropna(subset=["fwd_close", "fwd_date"]).copy()
    df["ret_ticker"] = df["fwd_close"] / df["close"] - 1.0
    df["spy_now"] = df["date"].map(spy)
    df["spy_fwd"] = df["fwd_date"].map(spy)
    df = df.dropna(subset=["spy_now", "spy_fwd"]).copy()
    df["ret_spy"] = df["spy_fwd"] / df["spy_now"] - 1.0
    df["excess"] = df["ret_ticker"] - df["ret_spy"]
    df["target"] = (df["ret_ticker"] > df["ret_spy"]).astype(int)
    return df.reset_index(drop=True)


def _fit_predict_test(df: pd.DataFrame, feature_cols: list[str]):
    train_df, test_df, cut, ts = _temporal_split(df)
    core_df, val_df = _internal_val_split(train_df)
    pos = int(core_df["target"].sum()); neg = len(core_df) - pos
    hp = XGBHyperparams(scale_pos_weight=(neg / pos if pos else 1.0))
    model = train(core_df[feature_cols], core_df["target"],
                  val_df[feature_cols], val_df["target"], hyperparams=hp)
    proba = model.predict_proba(test_df[feature_cols])[:, 1]
    y = test_df["target"].to_numpy()
    return {
        "auc": _auc(y, proba), "acc": accuracy_score(y, (proba >= 0.5).astype(int)),
        "pred_up": (proba >= 0.5).mean(),
        "cm": confusion_matrix(y, (proba >= 0.5).astype(int), labels=[0, 1]),
        "core": core_df, "val": val_df, "test": test_df, "cut": cut, "ts": ts,
    }


def _walk_forward(df: pd.DataFrame, feature_cols: list[str], ratio: float = 0.5):
    dates = sorted(df["date"].unique().tolist())
    n0 = int(len(dates) * ratio)
    origin = dates[n0 - 1]
    train_block = df[df["date"] <= origin]
    core_df, val_df = _internal_val_split(train_block)
    pos = int(core_df["target"].sum()); neg = len(core_df) - pos
    hp = XGBHyperparams(scale_pos_weight=(neg / pos if pos else 1.0))
    model = train(core_df[feature_cols], core_df["target"],
                  val_df[feature_cols], val_df["target"], hyperparams=hp)
    fwd = df[df["date"] > origin].copy()
    fwd["proba"] = model.predict_proba(fwd[feature_cols])[:, 1]
    fwd["pred"] = (fwd["proba"] >= 0.5).astype(int)
    return fwd, origin, len(train_block)


def main() -> Path:
    md, A = [], lambda s: md.append(s)
    features = load_features()
    d0, d1 = features["date"].min(), features["date"].max()
    spy = _get_spy_closes(date(d0.year, d0.month, 1), date(2022, 3, 31))

    df = _build_excess_target(features, spy, HORIZON)

    tech = [c for c in TECH_STATIONARY if c in df.columns]
    sent = [c for c in _SENTIMENT_FEATURES if c in df.columns]
    hybrid = tech + sent

    A("# Sanity check — target de EXCESO sobre SPY (H=5) + features estacionarias\n")
    A("> Números crudos, sin interpretación.\n")
    A(f"- Filas tras shift+merge SPY: **{len(df)}** | rango {df['date'].min()} → {df['date'].max()}")
    A(f"- SPY: {len(spy)} cierres cargados.")
    A(f"- Features técnicas estacionarias ({len(tech)}): {tech}")
    A(f"- Features de sentimiento ({len(sent)}): {sent}")
    A(f"- Híbrido = {len(hybrid)} features. Eliminadas de nivel: close, volume, macd, "
      "macd_signal, bb_upper, bb_middle, bb_lower.\n")

    # --- 1. Balance ---
    res_h = _fit_predict_test(df, hybrid)
    res_t = _fit_predict_test(df, tech)
    A("## 1. Balance del target (excess > 0) por split\n")
    A("| split | n | % target=1 (bate al mercado) | % target=0 |")
    A("|---|---|---|---|")
    for name, dd in [("train (core)", res_h["core"]), ("val", res_h["val"]), ("test", res_h["test"])]:
        up = dd["target"].mean()
        A(f"| {name} | {len(dd)} | {_pct(up)} | {_pct(1-up)} |")
    A(f"\n_split temporal: train ≤ {res_h['cut']}, test ≥ {res_h['ts']}._\n")

    # --- 2. AUC test ---
    A("## 2. AUC test — técnico-puro vs híbrido\n")
    A("| modelo | features | AUC test | accuracy | % pred=1 | CM [TN,FP,FN,TP] |")
    A("|---|---|---|---|---|---|")
    A(f"| técnico-puro | {len(tech)} | {res_t['auc']:.4f} | {res_t['acc']:.4f} | "
      f"{_pct(res_t['pred_up'])} | {res_t['cm'].ravel().tolist()} |")
    A(f"| híbrido | {len(hybrid)} | {res_h['auc']:.4f} | {res_h['acc']:.4f} | "
      f"{_pct(res_h['pred_up'])} | {res_h['cm'].ravel().tolist()} |")
    A("")

    # --- 3. Spearman feature[T] vs excess[T+5] ---
    A("## 3. Top 5 Spearman feature[T] vs EXCESO[T+5]\n")
    corrs = []
    for c in hybrid:
        x = pd.to_numeric(df[c], errors="coerce")
        mask = x.notna() & df["excess"].notna()
        if mask.sum() < 30 or x[mask].nunique() < 3:
            continue
        rho, p = spearmanr(x[mask], df["excess"][mask])
        corrs.append((c, rho, p))
    corrs.sort(key=lambda t: abs(t[1]), reverse=True)
    A("| # | feature | ρ | ρ² | p | tipo |")
    A("|---|---|---|---|---|---|")
    for i, (c, rho, p) in enumerate(corrs[:5], 1):
        tipo = "sentimiento" if c in _SENTIMENT_FEATURES else "técnica"
        A(f"| {i} | {c} | {rho:+.4f} | {_pct(rho**2)} | {p:.3f} | {tipo} |")
    A("")

    # --- 4. Walk-forward ---
    A("## 4. Walk-forward (entrena una vez al 50%, predice día a día)\n")
    for label, fcols in [("híbrido", hybrid), ("técnico-puro", tech)]:
        fwd, origin, ntr = _walk_forward(df, fcols)
        y = fwd["target"].to_numpy(); proba = fwd["proba"].to_numpy()
        agg_auc = _auc(y, proba); agg_acc = accuracy_score(y, (proba >= 0.5).astype(int))
        A(f"### {label}\n")
        A(f"- origen entrenamiento: ≤ {origin} ({ntr} filas) | forward: {len(fwd)} filas")
        A(f"- **AUC agregado: {agg_auc:.4f}** | accuracy: {agg_acc:.4f} | "
          f"% pred=1: {_pct((proba>=0.5).mean())} | base rate target=1: {_pct(y.mean())}\n")
        fwd = fwd.copy()
        fwd["month"] = pd.to_datetime(fwd["date"]).dt.to_period("M").astype(str)
        A("| mes | n | % target=1 | accuracy | AUC | % pred=1 |")
        A("|---|---|---|---|---|---|")
        for m, g in fwd.groupby("month"):
            yy = g["target"].to_numpy(); pp = g["proba"].to_numpy()
            auc = _auc(yy, pp)
            auc_s = f"{auc:.3f}" if not np.isnan(auc) else "—"
            A(f"| {m} | {len(g)} | {_pct(yy.mean())} | "
              f"{accuracy_score(yy,(pp>=0.5).astype(int)):.3f} | {auc_s} | {_pct((pp>=0.5).mean())} |")
        A("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\n(Reporte en {OUT_MD})")
    return OUT_MD


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import logging
    logging.basicConfig(level=logging.WARNING)
    main()
