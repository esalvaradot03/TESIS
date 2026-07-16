"""
FASE 2 — ETAPA B: 4 experimentos survivors-only sobre dataset_v2.

Reutiliza EXACTAMENTE la lógica de Fase 1 (mismo _run_one, _metrics, folds,
hiperparámetros, backtest) para garantizar protocolo idéntico. Solo cambia el
dataset (small/mid-cap survivors) y se corren 4 experimentos:

  F1: 5d, universo survivors completo, híbrido
  F2: 5d, solo días evento, híbrido
  F3: 5d, solo técnico (control)
  F4: 5d, placebo (sentimiento+aceleración permutados)

Criterio de éxito (pre-registrado): prec. decil >= 55% Y Sharpe > 0.5 Y F4 no cumple.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.data.longrun.common import MASTER_DIR  # noqa: E402
from research.experiment_v1.run_experiments import (  # noqa: E402
    _HYBRID, _TECH, _metrics, _run_one,
)

_DATASET = MASTER_DIR / "dataset_v2_enriched.parquet"
_MODELS = ROOT / "research" / "experiment_v2" / "models"
_REPORT = ROOT / "research" / "experiment_v2" / "results_experiment_v2.md"

_EXPERIMENTS = [
    ("F1", 5, False, _HYBRID, False),   # universo completo
    ("F2", 5, True, _HYBRID, False),    # solo evento
    ("F3", 5, False, _TECH, False),     # control técnico
    ("F4", 5, False, _HYBRID, True),    # control placebo
]
_LABELS = {
    "F1": "5d, survivors completo, híbrido", "F2": "5d, survivors, solo evento, híbrido",
    "F3": "5d, survivors, SOLO técnico (control)", "F4": "5d, survivors, placebo (sent permutado)",
}


def run() -> dict:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("experiments_v2")
    _MODELS.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(_DATASET)
    df["date"] = pd.to_datetime(df["date"])
    log.info("dataset_v2: %d filas, %d tickers, target5d %.1f%% sube.",
             len(df), df["ticker"].nunique(), df["target_5d"].mean() * 100)

    results = {}
    for name, h, event_only, feats, placebo in _EXPERIMENTS:
        d = df[df["event_day"] == 1] if event_only else df
        log.info("--- %s (event=%s, nfeat=%d, placebo=%s) sobre %d filas ---",
                 name, event_only, len(feats), placebo, len(d))
        pred, model = _run_one(d, h, feats, placebo, log)
        if pred.empty:
            results[name] = {"error": "sin predicciones"}
            continue
        m = _metrics(pred, h)
        results[name] = {"h": h, "event_only": event_only, "placebo": placebo,
                         "n_pred": len(pred), **m}
        if model is not None:
            model.save_model(str(_MODELS / f"{name}.json"))
        log.info("  %s: AUC=%.4f prec_dec=%.3f sharpe=%.2f trades=%d",
                 name, m["auc_agg"], m["prec_decile"], m["sharpe"], m["n_trades"])

    _write_report(results, log)
    import json
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))
    return results


def _write_report(results, log):
    f4 = results.get("F4", {})
    f4_ok = ("prec_decile" in f4 and f4["prec_decile"] >= 0.55 and
             f4.get("sharpe", 0) > 0.5 and f4.get("n_trades", 0) >= 100)

    def verdict(r):
        if "prec_decile" not in r:
            return "—"
        meets = r["prec_decile"] >= 0.55 and r["sharpe"] > 0.5 and r["n_trades"] >= 100
        if meets and not f4_ok:
            return "**SEÑAL**"
        if meets and f4_ok:
            return "no (placebo tambien cumple)"
        return "no"

    any_signal = any(verdict(results.get(n, {})) == "**SEÑAL**" for n in ["F1", "F2", "F3"])
    L = ["# Resultados Fase 2 — Etapa B (survivors-only small/mid-cap)\n",
         "> **Pre-registro (ver README):** test survivors-only, sesgado A FAVOR de encontrar "
         "señal. (a) Si NO se cumple el criterio → negativo fuerte, hipótesis small-cap "
         "rechazada incluso en el mejor caso. (b) Si SÍ se cumple → NO confiable, atribuible a "
         "survivorship, no promocionable.\n",
         "Criterio: prec. decil >= 55% Y Sharpe > 0.5 (10 bps/lado) Y F4 (placebo) no cumple.\n",
         f"- F4 (placebo) cumpliría por sí solo: **{'SÍ' if f4_ok else 'no'}**\n",
         "## Tabla de métricas (walk-forward, test 2017-2022)\n",
         "| Exp | config | AUC agg | %meses>0.55 | prec. decil | Sharpe (bt) | cum exceso | trades | veredicto |",
         "|-----|--------|--------|-------------|-------------|-------------|-----------|--------|-----------|"]
    for name in ["F1", "F2", "F3", "F4"]:
        r = results.get(name, {})
        if "prec_decile" not in r:
            L.append(f"| {name} | {_LABELS[name]} | ERROR | | | | | | |")
            continue
        nc = " ⚠<100" if r["n_trades"] < 100 else ""
        L.append(f"| {name} | {_LABELS[name]} | {r['auc_agg']:.4f} | {r['pct_months_gt055']:.0f}% | "
                 f"{r['prec_decile']:.3f} | {r['sharpe']:.2f} | {r['cum_excess']:.2%} | "
                 f"{r['n_trades']}{nc} | {verdict(r)} |")
    L += ["", "## Conexión con la interpretación pre-registrada\n"]
    if any_signal:
        L.append("Algún experimento CUMPLE el criterio → por el pre-registro (b): **resultado NO "
                 "confiable, atribuible a survivorship bias, NO promocionable a model_v1.**")
    else:
        L.append("**Ningún experimento cumple el criterio** → por el pre-registro (a): **negativo "
                 "fuerte. La hipótesis small-cap se rechaza incluso en el mejor caso sesgado a "
                 "favor (survivors-only). El sentimiento retail no predice de forma explotable ni "
                 "en small/mid-caps survivors.**")
    L.append("")
    _REPORT.write_text("\n".join(L), encoding="utf-8")
    log.info("Reporte en %s", _REPORT)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run()
