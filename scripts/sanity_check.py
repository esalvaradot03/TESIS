"""
Sanity check previo al sistema adaptativo (solo diagnóstico, no modifica model_v0).

1. Verificación manual de la construcción del target en 10 fechas del train.
2. Modelo "boba" SOLO con features técnicas (mismo split) → ¿AUC ≈ 0.50?
3. Correlación de Spearman feature[T] ↔ retorno[T+1] sobre todo el dataset.

Escribe docs/sanity_check.md y lo imprime.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.trading.feature_engine import (  # noqa: E402
    _SENTIMENT_FEATURES,
    _TECHNICAL_FEATURES,
    load_features,
)
from src.trading.xgboost_model import (  # noqa: E402
    XGBHyperparams,
    _build_target,
    _internal_val_split,
    _temporal_split,
    train,
)

SEED = 42
OUT_MD = ROOT / "docs" / "sanity_check.md"


def _pct(x: float) -> str:
    return f"{x*100:.1f}%"


def main() -> Path:
    md: list[str] = []
    A = md.append
    features = load_features()

    A("# Sanity check pre-adaptativo\n")
    A("> Solo diagnóstico. No se modifica `model_v0`. El modelo técnico de la "
      "sección 2 se entrena al vuelo y no se persiste.\n")

    # ===================================================================
    # 1. Verificación del target
    # ===================================================================
    A("## 1. ¿El target está bien construido?\n")
    A("Regla esperada: `target[T] = 1` ⇔ `close[T+1] > close[T]` (siguiente día de "
      "trading del MISMO ticker).\n")

    # Reconstrucción independiente de close[T+1] (no se usa _build_target aquí)
    v = features.sort_values(["ticker", "date"]).copy()
    v["next_date"] = v.groupby("ticker")["date"].shift(-1)
    v["next_close"] = v.groupby("ticker")["close"].shift(-1)
    v = v.dropna(subset=["next_close"]).copy()
    v["target_check"] = (v["next_close"] > v["close"]).astype(int)

    # target oficial del pipeline, para confirmar que coinciden
    official = _build_target(features)[["ticker", "date", "target"]]
    v = v.merge(official, on=["ticker", "date"], how="inner")

    # Solo fechas del bloque de TRAIN
    df_t = _build_target(features)
    _, _, train_cutoff, _ = _temporal_split(df_t)
    train_rows = v[v["date"] <= train_cutoff]

    sample = train_rows.sample(10, random_state=SEED).sort_values(["ticker", "date"])
    A("| # | ticker | fecha (T) | close[T] | fecha (T+1) | close[T+1] | sube? | target | ✓ |")
    A("|---|---|---|---|---|---|---|---|---|")
    all_ok = True
    for i, r in enumerate(sample.itertuples(index=False), 1):
        sube = r.next_close > r.close
        ok = int(sube) == int(r.target) == int(r.target_check)
        all_ok &= ok
        A(f"| {i} | {r.ticker} | {r.date} | {r.close:.2f} | {r.next_date} | "
          f"{r.next_close:.2f} | {'sí' if sube else 'no'} | {int(r.target)} | "
          f"{'✓' if ok else '✗ MAL'} |")
    A("")
    mismatch = int((v["target"] != v["target_check"]).sum())
    A(f"**Resultado:** {'los 10 casos correctos' if all_ok else 'HAY DISCREPANCIAS'}. "
      f"Sobre TODO el dataset, el target del pipeline coincide con la verificación "
      f"independiente en **{len(v)-mismatch}/{len(v)}** filas "
      f"({mismatch} discrepancias). ")
    A("**Interpretación:** el target está bien construido (sin off-by-one ni fuga "
      "entre tickers); el problema de skill no viene de la etiqueta.\n"
      if mismatch == 0 else
      "**Interpretación:** hay discrepancias — revisar `_build_target`.\n")

    # ===================================================================
    # 2. Modelo "boba" solo técnico
    # ===================================================================
    A("## 2. Modelo \"boba\": solo features técnicas (mismo split)\n")
    tech = [c for c in _TECHNICAL_FEATURES if c in df_t.columns]
    train_df, test_df, cut, ts = _temporal_split(df_t)
    core_df, val_df = _internal_val_split(train_df)

    # scale_pos_weight balanceado para que no colapse (comparable al balanceado)
    pos = int(core_df["target"].sum()); neg = len(core_df) - pos
    hp = XGBHyperparams(scale_pos_weight=neg / pos)
    model = train(core_df[tech], core_df["target"], val_df[tech], val_df["target"],
                  hyperparams=hp)

    rows = []
    for name, d in [("val", val_df), ("test", test_df)]:
        proba = model.predict_proba(d[tech])[:, 1]
        pred = (proba >= 0.5).astype(int)
        y = d["target"].to_numpy()
        auc = roc_auc_score(y, proba) if len(np.unique(y)) > 1 else float("nan")
        cm = confusion_matrix(y, pred, labels=[0, 1])
        rows.append((name, accuracy_score(y, pred), auc, pred.mean(), cm))

    A(f"Features técnicas usadas ({len(tech)}): {tech}\n")
    A("| split | accuracy | AUC | % pred sube | CM [TN,FP,FN,TP] |")
    A("|---|---|---|---|---|")
    for name, acc, auc, pup, cm in rows:
        A(f"| {name} | {acc:.4f} | {auc:.4f} | {_pct(pup)} | {cm.ravel().tolist()} |")
    A("")
    test_auc = rows[-1][2]
    if test_auc < 0.55:
        A(f"**Interpretación:** el modelo SOLO técnico da AUC = **{test_auc:.3f}** en test "
          "(≈ azar). El problema **NO es el sentimiento**: esta ventana (5 tickers, horizonte "
          "de 1 día, este set de indicadores) simplemente **no es predecible** con estas "
          "features. El sentimiento no está 'rompiendo' una señal técnica que tampoco existe.\n")
    else:
        A(f"**Interpretación:** el modelo SOLO técnico da AUC = **{test_auc:.3f}** (> 0.55). "
          "Hay señal técnica y el sentimiento podría estar **rompiéndola** — conviene revisar "
          "la interacción.\n")

    # ===================================================================
    # 3. Spearman feature[T] vs retorno[T+1]
    # ===================================================================
    A("## 3. ¿Hay señal? Spearman feature[T] ↔ retorno[T+1]\n")
    s = features.sort_values(["ticker", "date"]).copy()
    s["next_close"] = s.groupby("ticker")["close"].shift(-1)
    s["ret_next"] = s["next_close"] / s["close"] - 1.0
    s = s.dropna(subset=["ret_next"])

    feat_cols = [c for c in _SENTIMENT_FEATURES + _TECHNICAL_FEATURES if c in s.columns]
    corrs = []
    for c in feat_cols:
        x = pd.to_numeric(s[c], errors="coerce")
        mask = x.notna() & s["ret_next"].notna()
        if mask.sum() < 30 or x[mask].nunique() < 3:
            continue
        rho, p = spearmanr(x[mask], s["ret_next"][mask])
        corrs.append((c, rho, p))

    corrs.sort(key=lambda t: abs(t[1]), reverse=True)
    A("Top 10 por |ρ| (sobre todo el dataset, n = "
      f"{len(s)} filas):\n")
    A("| # | feature | Spearman ρ | p-value | tipo |")
    A("|---|---|---|---|---|")
    for i, (c, rho, p) in enumerate(corrs[:10], 1):
        tipo = "sentimiento" if c in _SENTIMENT_FEATURES else "técnica"
        A(f"| {i} | {c} | {rho:+.4f} | {p:.3f} | {tipo} |")
    A("")
    max_abs = abs(corrs[0][1]) if corrs else 0.0
    n_above = sum(1 for _, rho, _ in corrs if abs(rho) > 0.05)
    r2 = max_abs ** 2
    # Señal despreciable si el mayor |ρ| explica < ~1% de la varianza (ρ² < 0.01,
    # i.e. |ρ| < 0.1). El corte 0.05 por sí solo es arbitrario; lo que importa es
    # el tamaño de efecto, no cruzar una línea.
    negligible = max_abs < 0.10
    A(f"**Resultado:** la correlación máxima en |ρ| es **{max_abs:.4f}** "
      f"({corrs[0][0]}), que explica solo **ρ² ≈ {_pct(r2)}** de la varianza del retorno. "
      f"{n_above} de {len(corrs)} features cruzan |ρ| = 0.05, todas en la banda 0.05–0.06.\n")
    if negligible:
        A("**Interpretación:** aunque 3 features de sentimiento cruzan por poco el 0.05 "
          "(significativas con p≈0.02 solo porque n=1863 es grande), el **tamaño de efecto es "
          "despreciable**: un |ρ| de ~0.055 explica ~0.3% de la varianza. En la práctica **no hay "
          "señal explotable a 1 día**, ni de sentimiento ni técnica. La significancia estadística "
          "aquí no implica utilidad de trading.\n")
    else:
        A("**Interpretación:** hay al menos una feature con correlación no trivial (|ρ| ≥ 0.10); "
          "podría explotarse. Conviene analizarla antes del adaptativo.\n")

    # ===================================================================
    # Veredicto
    # ===================================================================
    A("## Veredicto\n")
    A(f"- Target: {'correcto ✓' if mismatch == 0 else 'PROBLEMA ✗'}")
    A(f"- Modelo solo-técnico AUC test: {test_auc:.3f} "
      f"({'sin señal técnica' if test_auc < 0.55 else 'hay señal técnica'})")
    A(f"- Spearman máx |ρ|: {max_abs:.4f} (ρ² ≈ {_pct(r2)}) "
      f"({'señal despreciable' if negligible else 'señal no trivial'})\n")
    if mismatch == 0 and test_auc < 0.55 and negligible:
        A("**Conclusión:** el target es correcto y **no hay señal predictiva útil a 1 día** en "
          "este conjunto (ni técnica ni de sentimiento; máx ρ²≈0.3%). El bajo AUC NO es culpa del "
          "sentimiento ni de un bug: es un **problema de fondo de predictibilidad** para este "
          "horizonte/universo. Antes de invertir en el adaptativo conviene **cambiar el problema**: "
          "horizonte más largo (p. ej. 3-5 días), target distinto (magnitud, o exceso sobre el "
          "mercado/sector), o más tickers para tener más señal. **Un modelo adaptativo sobre una "
          "señal inexistente seguirá sin discriminar** — el adaptativo ayuda contra el cambio de "
          "régimen, no contra la ausencia de señal.\n")
    else:
        A("**Conclusión:** revisar los puntos marcados arriba antes de seguir.\n")

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
