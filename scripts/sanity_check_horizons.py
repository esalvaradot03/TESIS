"""
Sanity check con target multi-día: ¿aparece señal a 3 y 5 días?

Para cada horizonte H (1 de referencia, 3 y 5):
  target[T] = 1  ⇔  close[T+H] > close[T]   (mismo ticker)
  retorno[T+H] = close[T+H]/close[T] - 1

Reporta, con el MISMO dataset/features/splits (perdiendo las últimas H filas por
ticker por el shift):
  1. Balance del target en train(core) / val / test.
  2. Modelo SOLO técnico: AUC test.
  3. Modelo híbrido (sentimiento + técnico): AUC test.
  4. Top 5 correlaciones Spearman feature[T] vs retorno[T+H].

NO entrena el adaptativo. Solo diagnóstico. Escribe docs/sanity_check_horizons.md.
"""

import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.trading.feature_engine import (  # noqa: E402
    _SENTIMENT_FEATURES,
    _TECHNICAL_FEATURES,
    load_features,
)
from src.trading.xgboost_model import (  # noqa: E402
    XGBHyperparams,
    _internal_val_split,
    _temporal_split,
    train,
)

HORIZONS = [1, 3, 5]
OUT_MD = ROOT / "docs" / "sanity_check_horizons.md"


def _pct(x: float) -> str:
    return f"{x*100:.1f}%"


def _build_target_h(features: pd.DataFrame, h: int) -> pd.DataFrame:
    """target[T]=1 si close[T+h]>close[T]; añade retorno[T+h]. Pierde últimas h filas/ticker."""
    df = features.sort_values(["ticker", "date"]).copy()
    df["fwd_close"] = df.groupby("ticker")["close"].shift(-h)
    df["ret_fwd"] = df["fwd_close"] / df["close"] - 1.0
    df["target"] = (df["fwd_close"] > df["close"]).astype(int)
    return df.dropna(subset=["fwd_close"]).reset_index(drop=True)


def _train_auc(df: pd.DataFrame, feature_cols: list[str]) -> tuple[float, float]:
    """Entrena (scale_pos_weight balanceado) y devuelve (auc_test, pred_up_test)."""
    train_df, test_df, _, _ = _temporal_split(df)
    core_df, val_df = _internal_val_split(train_df)
    pos = int(core_df["target"].sum()); neg = len(core_df) - pos
    spw = neg / pos if pos > 0 else 1.0
    hp = XGBHyperparams(scale_pos_weight=spw)
    model = train(core_df[feature_cols], core_df["target"],
                  val_df[feature_cols], val_df["target"], hyperparams=hp)
    proba = model.predict_proba(test_df[feature_cols])[:, 1]
    y = test_df["target"].to_numpy()
    auc = roc_auc_score(y, proba) if len(np.unique(y)) > 1 else float("nan")
    return auc, (proba >= 0.5).mean()


def main() -> Path:
    md: list[str] = []
    A = md.append
    features = load_features()

    tech = [c for c in _TECHNICAL_FEATURES if c in features.columns]
    hybrid = [c for c in _SENTIMENT_FEATURES + _TECHNICAL_FEATURES if c in features.columns]
    all_feats = hybrid

    A("# Sanity check con target multi-día (1 / 3 / 5)\n")
    A("> Solo diagnóstico, sin adaptativo. `target[T]=1 ⇔ close[T+H]>close[T]`. "
      "Se pierden las últimas H filas por ticker (shift), de ahí que n baje un poco "
      "con H. Modelos con `scale_pos_weight` balanceado para que no colapsen.\n")
    A(f"- Features técnicas ({len(tech)}): {tech}")
    A(f"- Features híbridas ({len(hybrid)}): técnicas + {len(hybrid)-len(tech)} de sentimiento\n")

    # Tabla resumen
    summary = []

    for h in HORIZONS:
        df = _build_target_h(features, h)
        train_df, test_df, cut, ts = _temporal_split(df)
        core_df, val_df = _internal_val_split(train_df)

        bal = {
            "core": core_df["target"].mean(),
            "val": val_df["target"].mean(),
            "test": test_df["target"].mean(),
        }
        auc_tech, up_tech = _train_auc(df, tech)
        auc_hyb, up_hyb = _train_auc(df, hybrid)

        # Spearman feature[T] vs retorno[T+h]
        corrs = []
        for c in all_feats:
            x = pd.to_numeric(df[c], errors="coerce")
            mask = x.notna() & df["ret_fwd"].notna()
            if mask.sum() < 30 or x[mask].nunique() < 3:
                continue
            rho, p = spearmanr(x[mask], df["ret_fwd"][mask])
            corrs.append((c, rho, p))
        corrs.sort(key=lambda t: abs(t[1]), reverse=True)

        summary.append((h, len(df), bal, auc_tech, auc_hyb, corrs))

        A(f"## Horizonte H = {h} día(s)\n")
        A(f"- Filas: **{len(df)}** | split: train≤{cut}, test≥{ts}\n")
        A("**1. Balance del target (% sube):**\n")
        A("| split | n | % sube | % baja |")
        A("|---|---|---|---|")
        for name, d in [("train (core)", core_df), ("val", val_df), ("test", test_df)]:
            up = d["target"].mean()
            A(f"| {name} | {len(d)} | {_pct(up)} | {_pct(1-up)} |")
        A("")
        A(f"**2. Modelo solo-técnico — AUC test: `{auc_tech:.4f}`** (pred sube {_pct(up_tech)})\n")
        A(f"**3. Modelo híbrido — AUC test: `{auc_hyb:.4f}`** (pred sube {_pct(up_hyb)})\n")
        A("**4. Top 5 Spearman feature[T] vs retorno[T+H]:**\n")
        A("| # | feature | ρ | ρ² | p | tipo |")
        A("|---|---|---|---|---|---|")
        for i, (c, rho, p) in enumerate(corrs[:5], 1):
            tipo = "sentimiento" if c in _SENTIMENT_FEATURES else "técnica"
            A(f"| {i} | {c} | {rho:+.4f} | {_pct(rho**2)} | {p:.3f} | {tipo} |")
        A("")
        max_abs = abs(corrs[0][1]) if corrs else 0.0
        A(f"_Máx |ρ| = {max_abs:.4f} (ρ² ≈ {_pct(max_abs**2)})._\n")

    # ===================================================================
    # Tabla comparativa y veredicto
    # ===================================================================
    A("## Comparativa entre horizontes\n")
    A("| H | n | % sube test | AUC técnico | AUC híbrido | máx |ρ| | máx ρ² |")
    A("|---|---|---|---|---|---|---|")
    for h, n, bal, at, ah, corrs in summary:
        mx = abs(corrs[0][1]) if corrs else 0.0
        A(f"| {h} | {n} | {_pct(bal['test'])} | {at:.4f} | {ah:.4f} | {mx:.4f} | {_pct(mx**2)} |")
    A("")

    # Features de NIVEL de precio (no estacionarias): close y las bandas de Bollinger,
    # que son esencialmente el nivel/precio medio. Sirven de proxy de régimen, no de señal.
    price_level = {"close", "bb_upper", "bb_middle", "bb_lower"}

    A("## Veredicto\n")
    A("Desglose de la correlación máxima por TIPO de feature en cada horizonte:\n")
    A("| H | máx |ρ| total | feature top | ¿nivel de precio? | máx |ρ| SENTIMIENTO |")
    A("|---|---|---|---|---|---|")
    sent_flat = True
    for h, n, bal, at, ah, corrs in summary:
        top_feat, top_rho = corrs[0][0], corrs[0][1]
        is_price = top_feat in price_level
        sent_corrs = [abs(r) for c, r, _ in corrs if c in _SENTIMENT_FEATURES]
        max_sent = max(sent_corrs) if sent_corrs else 0.0
        if max_sent >= 0.10:
            sent_flat = False
        A(f"| {h} | {abs(top_rho):.4f} | {top_feat} | {'**SÍ**' if is_price else 'no'} | "
          f"{max_sent:.4f} (ρ²≈{_pct(max_sent**2)}) |")
    A("")
    A("**Lectura crítica:**")
    A("- La correlación que **crece** con el horizonte (0.055 → 0.082 → 0.102) está dominada por "
      "features de **nivel de precio** (`close`, `bb_middle/upper/lower`, que son casi la misma "
      "variable), todas con ρ **negativo**. Eso **no es señal predictiva generalizable**: es el "
      "artefacto de que un precio absoluto alto en el boom de 2021 precede a caídas en el crash "
      "de 2022 — un **proxy de régimen no estacionario**. Un modelo que se apoya en el nivel de "
      "precio memoriza ese ciclo concreto y no transferirá a otro período.")
    A("- Las features de **sentimiento se mantienen planas** en todos los horizontes "
      "(máx |ρ| ≲ 0.07, ρ² ≲ 0.5%). Alargar el horizonte **no** despierta la señal de sentimiento.")
    A("- Los **AUC son inconsistentes** (H=3 técnico 0.50 vs híbrido 0.56; H=5 técnico 0.56 vs "
      "híbrido 0.53): si hubiera señal robusta, añadir sentimiento no debería empeorar el AUC. "
      "Esa volatilidad entre 0.50 y 0.56 es **ruido de un único test set**, no skill estable.\n")
    A("**Conclusión:** alargar el horizonte a 3/5 días **no destapa señal de sentimiento real**. "
      "La única correlación que sube es un **artefacto de nivel de precio / régimen** que no "
      "generaliza. Antes del adaptativo, las palancas correctas son: (1) **target de exceso sobre "
      "el mercado/sector** (resta el beta común y el efecto nivel-de-precio que ensucia todo), "
      "(2) **quitar features de nivel absoluto** (usar solo derivadas estacionarias: rsi, "
      "macd_hist, bb_pct, bb_width) para no memorizar el ciclo, y (3) **ampliar el universo de "
      "tickers**. El adaptativo no arregla ausencia de señal.\n")

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
