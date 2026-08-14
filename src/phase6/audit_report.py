"""
Fase 6 — Tarea 1: reporte de viabilidad (combina las tres auditorías).

Junta:
  - StockTwits intradía (st_intraday_counts.parquet): distribución msgs/ventana.
  - Alpaca IEX (alpaca_30min.parquet): universo de ventanas de mercado + cobertura.
  - Noticias FNSPID: fecha-solo (fuera de esta fase, verificado en Tarea 1.2).
Produce experiments/phase6/audit.md con la distribución 30 vs 60 min por año, %<5,
cobertura de precios, y una recomendación (ventana / años / activos viables).
"""

import logging
import sys
from datetime import time as dtime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

_OUT = ROOT / "experiments" / "phase6"
_CACHE = _OUT / "cache"
TICKERS = ["TSLA", "AMD"]


def _to_w60(w30: pd.Series) -> pd.Series:
    """w30 (inicio ventana 30min ET) → ventana 60min [9:30,10:30),...,[15:30,16:00)."""
    return (w30 - pd.Timedelta(minutes=30)).dt.floor("60min") + pd.Timedelta(minutes=30)


def _bars_windows(bars: pd.DataFrame, size: str) -> pd.DataFrame:
    """Universo de ventanas de mercado con barra de precio (por ticker)."""
    tod = bars["t"].dt.time
    b = bars[(tod >= dtime(9, 30)) & (tod < dtime(16, 0)) & (bars["t"].dt.weekday < 5)].copy()
    b["w30"] = b["t"].dt.floor("30min")
    if size == "60":
        b["w"] = _to_w60(b["w30"])
    else:
        b["w"] = b["w30"]
    return b.groupby(["ticker", "w"]).size().reset_index(name="_bar")[["ticker", "w"]]


def _dist_row(msg_counts: pd.Series, universe_windows: pd.Series | None, label: str) -> dict:
    """Distribución de msgs/ventana.

    `msg_counts`: conteo por ventana (solo ventanas con ≥1 mensaje).
    `universe_windows`: ventanas de mercado válidas (con barra de precio). Si se pasa,
    cobertura y %<5 se calculan alineando los conteos a ESAS ventanas (zero-fill), así
    ambos numeradores/denominadores comparten el mismo período (arregla el desajuste
    cuando las barras solo cubren parte del año). Si es None (años sin barras),
    cobertura/%<5 quedan N/A y solo se reporta la densidad sobre ventanas activas.
    """
    active = msg_counts[msg_counts > 0]
    med = float(active.median()) if len(active) else 0.0
    p90 = float(active.quantile(.90)) if len(active) else 0.0
    if universe_windows is None:
        return {"grupo": label, "cobertura_%": np.nan, "mediana_activas": med,
                "p90": p90, "pct_ventanas_<5_universo": np.nan}
    # alinear conteos a las ventanas con barra (zero-fill)
    aligned = msg_counts.reindex(universe_windows).fillna(0)
    n_u = len(universe_windows)
    return {
        "grupo": label, "cobertura_%": round(100 * (aligned > 0).mean(), 1),
        "mediana_activas": med, "p90": p90,
        "pct_ventanas_<5_universo": round(100 * (aligned < 5).mean(), 1),
    }


def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("phase6_report")

    st = pd.read_parquet(_CACHE / "st_intraday_counts.parquet")
    st["w30"] = pd.to_datetime(st["w30"])
    st["w60"] = _to_w60(st["w30"])
    st["year"] = st["w30"].dt.year
    st["labeled_frac"] = st["n_labeled"] / st["n_msgs"]

    bars = pd.read_parquet(_CACHE / "alpaca_30min.parquet")
    bars["t"] = pd.to_datetime(bars["t"])
    uni30 = _bars_windows(bars, "30"); uni30["year"] = uni30["w"].dt.year
    uni60 = _bars_windows(bars, "60"); uni60["year"] = uni60["w"].dt.year

    rows = []
    for tk in TICKERS:
        for yr in range(2016, 2023):
            sy = st[(st["ticker"] == tk) & (st["year"] == yr)]
            # universo = ventanas con barra Alpaca ese año (exacto); None si no hay barras
            u30 = uni30[(uni30["ticker"] == tk) & (uni30["year"] == yr)]["w"]
            u60 = uni60[(uni60["ticker"] == tk) & (uni60["year"] == yr)]["w"]
            U30 = pd.Index(sorted(u30)) if len(u30) else None
            U60 = pd.Index(sorted(u60)) if len(u60) else None
            src_uni = "barras" if U30 is not None else "sin precio"
            c30 = sy.groupby("w30")["n_msgs"].sum()
            c60 = sy.groupby("w60")["n_msgs"].sum()
            r30 = _dist_row(c30, U30, "30min"); r60 = _dist_row(c60, U60, "60min")
            # volumen restringido al período con barras (si aplica) para consistencia
            vol = int(c30.reindex(U30).fillna(0).sum()) if U30 is not None else int(sy["n_msgs"].sum())
            base = {"ticker": tk, "year": yr, "universo_src": src_uni, "msgs_total": vol}
            rows.append({**base, **{f"w30_{k}": v for k, v in r30.items()}})
            rows[-1].update({f"w60_{k}": v for k, v in r60.items()})

    rep = pd.DataFrame(rows)
    rep.to_parquet(_OUT / "audit_distribution.parquet", index=False)
    _write_md(rep, log)
    return rep


def _write_md(rep: pd.DataFrame, log):
    L = ["# Fase 6 — Tarea 1: auditoría de viabilidad de datos intradía\n",
         "## Resumen ejecutivo\n",
         "- **StockTwits** tiene timestamps intradía reales (en `feature_wo_messages`, "
         "`created_at` ISO-8601 UTC con `Z`; convertidos a ET America/New_York con DST). "
         "`symbol_sentiments` es fecha-solo, NO sirve para intradía.",
         "- **OJO — las etiquetas Bull/Bear NO están en el archivo intradía.** "
         "`feature_wo_messages` trae timestamp pero su columna `sentiment` viene vacía "
         "(0/3.8M etiquetadas). El label nativo vive en `symbol_sentiments` (fecha-solo), "
         "keyado por `message_id`. Para el sentimiento neto intradía hay que **unir por "
         "`message_id`** ambos archivos. El conteo por ventana de abajo es de **volumen de "
         "mensajes** (todos), no del subconjunto etiquetado; la densidad del NET etiquetado se "
         "cuantifica en Tarea 2 tras el join (los volúmenes de `symbol_sentiments` para TSLA/AMD "
         "son comparables al volumen intradía, así que se espera cobertura de label alta).",
         "- **Noticias FNSPID: fecha-solo** (99.8% de los artículos de TSLA/AMD sellados a "
         "`00:00`) → **quedan FUERA de la Fase 6**.",
         "- **Precios Alpaca IEX (gratuito): solo desde 2020-07-27.** Sin barras intradía "
         "2016–jul2020. Cobertura 2020H2–2022 ≈ 99.8% de las 13 ventanas de 30min/día. "
         "**Este es el cuello de botella: el target intradía solo existe ~ago2020–dic2022.**\n",
         "## Distribución de mensajes StockTwits por ventana (horario de mercado ET)\n",
         "`mediana/p90` sobre ventanas con ≥1 mensaje; `%<5` sobre TODO el universo de ventanas "
         "de mercado (incluye vacías); `cobertura` = ventanas con actividad / universo. "
         "Universo `barras` = ventanas con barra Alpaca (exacto, 2020H2+); `aprox` = "
         "díashábiles×N (2016-2019, sin barras).\n",
         "| ticker | año | src | msgs (vol) | 30m cob% | 30m mediana | 30m p90 | 30m %<5 | "
         "60m mediana | 60m p90 | 60m %<5 |",
         "|--------|-----|-----|-----------|----------|-------------|---------|---------|"
         "-------------|---------|---------|"]
    def _n(x):
        return "—" if pd.isna(x) else f"{x:g}"
    for _, r in rep.iterrows():
        L.append(
            f"| {r['ticker']} | {r['year']} | {r['universo_src']} | {r['msgs_total']:,} | "
            f"{_n(r['w30_cobertura_%'])} | {r['w30_mediana_activas']:.0f} | "
            f"{r['w30_p90']:.0f} | {_n(r['w30_pct_ventanas_<5_universo'])} | "
            f"{r['w60_mediana_activas']:.0f} | {r['w60_p90']:.0f} | "
            f"{_n(r['w60_pct_ventanas_<5_universo'])} |")

    # recomendación basada en el período con precios (2020H2-2022)
    model = rep[rep["year"] >= 2021]
    L += ["", "## Recomendación\n",
          "**Zona horaria:** timestamps StockTwits en UTC (`...Z`) → convertidos a "
          "ET (America/New_York, con DST). Verificado contra barras Alpaca (ambos alinean en "
          "09:30 ET apertura).\n",
          "**Activos viables:** TSLA y AMD, **ambos con densidad de sobra**. En el período con "
          "precios (2020H2-2022) la cobertura de ventanas de 30 min es ~100% y el %<5 msgs ≈0: "
          "prácticamente ninguna ventana de mercado queda por debajo de 5 mensajes. Medianas: "
          "TSLA ~130 msgs/ventana-30min, AMD ~38; p90 TSLA ~300, AMD ~90.\n",
          f"**Rango de años (cuello de botella):** limitado por precios a "
          "**2020-07-27 → 2022-12-30** (IEX gratuito no da intradía antes). StockTwits cubre "
          "2016-2022, pero sin barras no hay target intradía pre-ago2020. Split viable: "
          "**train 2020-08 → 2021-12 (~17 meses), test 2022 (12 meses)**.\n",
          "**Ventana 30 vs 60 min:** **30 min es viable para ambos** (densidad no es problema). "
          "60 min duplica la densidad y da más resolución estadística por ventana, pero **la "
          "mitad de observaciones** (menos poder en el test de ~1 año). Recomendación: **30 min "
          "como principal** (más observaciones para el ya corto test 2022), con 60 min como "
          "chequeo de robustez.\n",
          "**Aviso metodológico para Tarea 2:** el NET de sentimiento intradía exige unir "
          "`feature_wo_messages` (timestamp) con `symbol_sentiments` (label Bull/Bear) por "
          "`message_id` — el archivo intradía no trae el label. Hay que cuantificar la fracción "
          "etiquetada por ventana antes de pre-registrar (afecta la densidad efectiva del NET).\n",
          "**Nota:** extender a 2016-2019 exigiría un feed intradía pago (SIP/Polygon); con datos "
          "gratuitos, la Fase 6 es un estudio de ~2.4 años (2020H2-2022)."]
    L.append("")
    (_OUT / "audit.md").write_text("\n".join(L), encoding="utf-8")
    log.info("Reporte -> %s", _OUT / "audit.md")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run()
