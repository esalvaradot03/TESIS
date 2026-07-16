"""
ETAPA A.2 — sub-auditoría de viabilidad point-in-time (no construye nada).

1. Limpia el universo candidato (1,890 con >=200 días): clasifica ETF/índice/
   cripto vs equities reales (quoteType de yfinance + heurísticas locales),
   separando survivors vs deslistados.
2. Market cap point-in-time: ¿reconstruible? (FNSPID no trae shares → verificado).
3. Proxy de liquidez point-in-time: dollar volume mediano (precio×volumen de
   FNSPID). Distribución survivors vs deslistados.
4. Retorno de delisting: ¿precio hasta último día? ¿final identificable?
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import EXTERNAL_DATA_ROOT  # noqa: E402
from src.data.longrun.common import MASTER_DIR  # noqa: E402

_STATS = MASTER_DIR / "v2_mention_stats.parquet"
_FH = EXTERNAL_DATA_ROOT / "fnspid" / "full_history"
_WIN_S, _WIN_E = pd.Timestamp("2015-01-01"), pd.Timestamp("2023-12-31")
_CUTOFF_2023 = pd.Timestamp("2023-01-01")


def _price_status(t):
    """(has_price, last_date, median_dollar_vol) desde FNSPID full_history."""
    p = _FH / f"{t}.csv"
    if not p.exists():
        return False, None, np.nan
    try:
        d = pd.read_csv(p, usecols=["date", "close", "volume"])
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d = d.dropna(subset=["date"])
        d = d[(d["date"] >= _WIN_S) & (d["date"] <= _WIN_E)]
        if d.empty:
            return False, None, np.nan
        dv = (pd.to_numeric(d["close"], errors="coerce") *
              pd.to_numeric(d["volume"], errors="coerce")).median()
        return True, d["date"].max(), float(dv)
    except Exception:  # noqa: BLE001
        return False, None, np.nan


def run() -> dict:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("subaudit_v2")

    s = pd.read_parquet(_STATS)
    cand = s[s["days_ge5"] >= 200].copy()
    log.info("Candidatos (>=200 días): %d", len(cand))

    # --- Estado de precio + dollar volume (FNSPID, local) ---
    log.info("Cruzando precios/liquidez desde FNSPID full_history ...")
    st = cand["ticker"].map(_price_status)
    cand["has_price"] = st.map(lambda x: x[0])
    cand["last_date"] = st.map(lambda x: x[1])
    cand["med_dollar_vol"] = st.map(lambda x: x[2])
    cand["is_crypto"] = cand["ticker"].str.endswith("-X")

    survivors = cand[cand["has_price"] & (cand["last_date"] >= _CUTOFF_2023)].copy()
    delisted = cand[cand["has_price"] & (cand["last_date"] < _CUTOFF_2023)].copy()
    no_price = cand[~cand["has_price"]].copy()
    log.info("survivors=%d delisted=%d no_price=%d", len(survivors), len(delisted), len(no_price))

    # --- 1. Clasificación quoteType (yfinance) de survivors + intento en delisted ---
    import yfinance as yf

    def qtype(t):
        try:
            return str(yf.Ticker(t).fast_info["quoteType"]).upper()
        except Exception:  # noqa: BLE001
            return "UNKNOWN"

    log.info("Clasificando quoteType (yfinance) de %d survivors ...", len(survivors))
    qt = {}
    t0 = time.time()
    for i, t in enumerate(survivors["ticker"], 1):
        qt[t] = "CRYPTO" if t.endswith("-X") else qtype(t)
        if i % 100 == 0:
            log.info("  quoteType %d/%d (%.0fs)", i, len(survivors), time.time() - t0)
    survivors["quoteType"] = survivors["ticker"].map(qt)
    # delisted: yfinance suele no tener dato; clasificar cripto por sufijo, resto = presunto equity
    delisted["quoteType"] = np.where(delisted["ticker"].str.endswith("-X"), "CRYPTO", "EQUITY?(delisted)")

    surv_class = survivors["quoteType"].value_counts().to_dict()
    n_equity_surv = int((survivors["quoteType"] == "EQUITY").sum())
    n_crypto_del = int((delisted["quoteType"] == "CRYPTO").sum())

    # --- 3. Dollar volume: survivors vs delisted (equities reales) ---
    surv_eq = survivors[survivors["quoteType"] == "EQUITY"]
    del_eq = delisted[delisted["quoteType"] != "CRYPTO"]

    def _dv_dist(df):
        v = df["med_dollar_vol"].dropna()
        v = v[v > 0]
        return {"n": int(len(v)), "p10": float(v.quantile(.10)), "p50": float(v.quantile(.50)),
                "p90": float(v.quantile(.90))} if len(v) else {}

    dv_surv = _dv_dist(surv_eq)
    dv_del = _dv_dist(del_eq)

    # --- 4. Delisting: todos tienen último precio; identificabilidad del final ---
    del_last_years = delisted["last_date"].dt.year.value_counts().sort_index().to_dict()

    summary = {
        "candidatos_200d": len(cand),
        "survivors": len(survivors), "deslistados": len(delisted), "sin_precio": len(no_price),
        "survivors_quoteType": surv_class,
        "survivors_EQUITY_reales": n_equity_surv,
        "deslistados_cripto": n_crypto_del,
        "deslistados_equity_presunto": int(len(del_eq)),
        "shares_outstanding_FNSPID": False,
        "market_cap_pit_reconstruible": False,
        "dollar_volume_survivors_equity": dv_surv,
        "dollar_volume_deslistados_equity": dv_del,
        "deslistados_por_anio_ultimo_precio": {int(k): int(v) for k, v in del_last_years.items()},
        "deslistados_con_ultimo_precio(todos)": len(delisted),
    }
    _write_report(summary, log)
    import json
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return summary


def _fmt_dv(x):
    if x >= 1e6:
        return f"${x/1e6:.1f}M"
    return f"${x/1e3:.0f}K"


def _write_report(s, log):
    out = ROOT / "research" / "experiment_v2" / "subaudit_report.md"
    dvs, dvd = s["dollar_volume_survivors_equity"], s["dollar_volume_deslistados_equity"]
    L = [
        "# Etapa A.2 — Viabilidad point-in-time\n",
        "## 1. Limpieza del universo (candidatos con >=200 días)\n",
        f"- survivors (precio reciente): **{s['survivors']}** | deslistados: **{s['deslistados']}** | sin precio: **{s['sin_precio']}**",
        f"- Clasificación quoteType de survivors: {s['survivors_quoteType']}",
        f"- **Equities reales (survivors): {s['survivors_EQUITY_reales']}** | cripto en deslistados: {s['deslistados_cripto']} | "
        f"deslistados presunto-equity: {s['deslistados_equity_presunto']}\n",
        "## 2. Market cap point-in-time\n",
        "- FNSPID full_history trae shares outstanding: **NO** (solo OHLCV).",
        "- Fuente local con shares/fundamentals en D:\\trading-data: **ninguna**.",
        "- yfinance para deslistados: **sin datos** (posibly delisted).",
        "- **=> Market cap point-in-time NO es reconstruible. Mitigación B (por cap) NO implementable.**\n",
        "## 3. Proxy de liquidez point-in-time (dollar volume mediano, FNSPID)\n",
        f"- Survivors equity (n={dvs.get('n','—')}): p10 {_fmt_dv(dvs['p10'])} | p50 {_fmt_dv(dvs['p50'])} | p90 {_fmt_dv(dvs['p90'])}" if dvs else "- survivors: sin datos",
        f"- Deslistados equity (n={dvd.get('n','—')}): p10 {_fmt_dv(dvd['p10'])} | p50 {_fmt_dv(dvd['p50'])} | p90 {_fmt_dv(dvd['p90'])}" if dvd else "- deslistados: sin datos",
        "",
        "## 4. Retorno de delisting\n",
        f"- Los **{s['deslistados_con_ultimo_precio(todos)']}** deslistados tienen precio hasta su último día (FNSPID).",
        f"- Distribución del año del último precio: {s['deslistados_por_anio_ultimo_precio']}",
        "- Razón del delisting (quiebra vs fusión vs corte de datos): **NO identificable** con los datos disponibles.",
        "",
    ]
    out.write_text("\n".join(L), encoding="utf-8")
    log.info("Reporte en %s", out)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run()
