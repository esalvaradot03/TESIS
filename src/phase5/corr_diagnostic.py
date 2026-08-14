"""
Fase 5 — Experimento 5.1: diagnóstico de correlación contemporánea.

¿El sentimiento diario (StockTwits y noticias, por separado) se mueve junto con el
retorno del MISMO día? Diagnóstico, no predicción. Reutiliza el dataset de Fase 4.
Reporta Spearman y Pearson (completo/train/test), reactividad (net_t vs ret_{t-1}) y
placebo por permutación. Criterio en experiments/phase5/REGISTRY.md.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import SEED  # noqa: E402

_DS = ROOT / "experiments" / "phase4" / "dataset_phase4"
_OUT = ROOT / "experiments" / "phase5"
_REPORT = _OUT / "results_5_1.md"

TICKERS = ["TSLA", "AMD", "DIS", "BA", "GILD"]
_SOURCES = {"A_StockTwits": ("st_net", "st_has"), "B_Noticias": ("nw_net", "nw_has")}
_N_PERM = 200


def _spear(x, y):
    if len(x) < 10 or np.std(x) == 0 or np.std(y) == 0:
        return (np.nan, np.nan)
    rho, p = spearmanr(x, y)
    return (float(rho), float(p))


def _pears(x, y):
    if len(x) < 10 or np.std(x) == 0 or np.std(y) == 0:
        return (np.nan, np.nan)
    r, p = pearsonr(x, y)
    return (float(r), float(p))


def _placebo_p(x, y, rho_real, seed=SEED, n=_N_PERM):
    """Fracción de permutaciones con |rho| >= |rho_real| (p empírico)."""
    if np.isnan(rho_real) or len(x) < 10:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    xa = np.asarray(x)
    ge = 0
    perm_rhos = []
    for _ in range(n):
        rp, _p = spearmanr(rng.permutation(xa), y)
        perm_rhos.append(rp)
        if abs(rp) >= abs(rho_real):
            ge += 1
    return ((ge + 1) / (n + 1), float(np.mean(perm_rhos)))


def run() -> pd.DataFrame:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("phase5_5_1")

    rows = []
    for t in TICKERS:
        df = pd.read_parquet(_DS / f"{t}.parquet").sort_values("date").reset_index(drop=True)
        df["ret"] = df["close"].pct_change()
        df["ret_prev"] = df["ret"].shift(1)
        for src, (netcol, hascol) in _SOURCES.items():
            act = df[(df[hascol] == 1)].dropna(subset=["ret"])
            full = act
            tr = act[act["split"] == "train"]
            te = act[act["split"] == "test"]

            sp_full, sp_full_p = _spear(full[netcol], full["ret"])
            pe_full, pe_full_p = _pears(full[netcol], full["ret"])
            sp_tr, sp_tr_p = _spear(tr[netcol], tr["ret"])
            sp_te, sp_te_p = _spear(te[netcol], te["ret"])
            react = act.dropna(subset=["ret_prev"])
            sp_react, sp_react_p = _spear(react[netcol], react["ret_prev"])
            plc_p, plc_mean = _placebo_p(full[netcol].to_numpy(), full["ret"].to_numpy(), sp_full)

            sign_stable = (not np.isnan(sp_tr) and not np.isnan(sp_te)
                           and np.sign(sp_tr) == np.sign(sp_te) and sp_tr > 0 and sp_te > 0
                           and sp_tr_p < 0.01 and sp_te_p < 0.01)
            confirm = (not np.isnan(sp_full) and sp_full_p < 0.01 and sp_full > 0
                       and sign_stable and not np.isnan(plc_p) and plc_p < 0.01)

            rows.append({
                "activo": t, "fuente": src, "n_full": len(full), "n_test": len(te),
                "spearman_full": round(sp_full, 4), "sp_full_p": sp_full_p,
                "pearson_full": round(pe_full, 4), "pe_full_p": pe_full_p,
                "spearman_train": round(sp_tr, 4), "sp_tr_p": sp_tr_p,
                "spearman_test": round(sp_te, 4), "sp_te_p": sp_te_p,
                "react_prev_spearman": round(sp_react, 4), "react_p": sp_react_p,
                "placebo_p_emp": round(plc_p, 4), "placebo_rho_mean": round(plc_mean, 4),
                "CONFIRMA": bool(confirm),
            })
            log.info("  %s / %s: ρ_full=%.4f (p=%.1e) ρ_tr=%.4f ρ_te=%.4f react=%.4f -> %s",
                     t, src, sp_full, sp_full_p, sp_tr, sp_te, sp_react,
                     "CONFIRMA" if confirm else "no")

    res = pd.DataFrame(rows)
    res.to_parquet(_OUT / "results_5_1.parquet", index=False)
    _write_report(res, log)
    return res


def _pfmt(p):
    return "—" if pd.isna(p) else (f"{p:.1e}" if p < 0.001 else f"{p:.3f}")


def _write_report(res: pd.DataFrame, log):
    n_ok = int(res["CONFIRMA"].sum())
    L = ["# Fase 5.1 — Correlación contemporánea sentimiento ↔ retorno mismo día\n",
         "> **Criterio (REGISTRY.md):** CONFIRMA ⇔ Spearman p<0.01 completo, ρ>0, signo estable "
         "(train y test ambos ρ>0 con p<0.01), y supera placebo permutado (p emp<0.01). "
         "Diagnóstico de contenido, NO predicción. Correlaciones sobre días con actividad real "
         "de la fuente (`*_has==1`).\n",
         f"**Pares que CONFIRMAN H5.1: {n_ok} / 10.**\n",
         "## Correlación mismo día (Spearman ρ / p)\n",
         "| activo | fuente | n | ρ completo | p | Pearson r | ρ train | ρ test | placebo p | "
         "react ρ(net,ret_ayer) | veredicto |",
         "|--------|--------|---|-----------|---|-----------|---------|--------|-----------|"
         "-----------------------|-----------|"]
    for _, r in res.iterrows():
        L.append(
            f"| {r['activo']} | {r['fuente']} | {r['n_full']} | {r['spearman_full']:.4f} | "
            f"{_pfmt(r['sp_full_p'])} | {r['pearson_full']:.4f} | {r['spearman_train']:.4f} | "
            f"{r['spearman_test']:.4f} | {_pfmt(r['placebo_p_emp'])} | "
            f"{r['react_prev_spearman']:.4f} ({_pfmt(r['react_p'])}) | "
            f"{'**CONFIRMA**' if r['CONFIRMA'] else 'no'} |")
    L += ["", "## Interpretación\n"]
    strong = res[(res["sp_full_p"] < 0.01) & (res["spearman_full"] > 0)]
    L.append(f"- Correlación contemporánea positiva y significativa (p<0.01, completo) en "
             f"**{len(strong)}/10** pares.")
    L.append("- La columna `react ρ(net,ret_ayer)` mide si el sentimiento de hoy correlaciona "
             "con el retorno de AYER (reactividad al precio). Si es del mismo orden que la "
             "correlación contemporánea, el sentimiento refleja el movimiento ya ocurrido más "
             "que aportar información nueva.")
    if n_ok == 0:
        L.append("- **Ningún par cumple el criterio completo de H5.1.**")
    L.append("")
    _REPORT.write_text("\n".join(L), encoding="utf-8")
    log.info("Reporte -> %s", _REPORT)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import json
    print(json.dumps(run().to_dict(orient="records"), indent=2, ensure_ascii=False, default=str))
