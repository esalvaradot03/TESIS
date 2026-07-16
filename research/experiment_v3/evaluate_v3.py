"""
FASE 3 — Evaluación asset-pricing de la hipótesis contrarian.

Carteras mensuales por decil de bullishness (quintiles si hay pocos nombres/mes).
Long-short = LONG decil bajo − SHORT decil alto (contrarian → positivo esperado).

(a) retorno medio por decil + long-short con t-stat
(b) alfa del long-short sobre Fama-French 3 (Mkt-RF, SMB, HML) + t-stat
(c) exposición tamaño/vol por decil
(d) placebo: permutar sentimiento dentro de cada mes ×100 → distribución del alfa

Criterio (pre-registrado): alfa real t>2 con signo predicho (LS positivo) Y alfa real
fuera de la distribución placebo (p<0.05).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import EXTERNAL_DATA_ROOT  # noqa: E402
from src.data.longrun.common import MASTER_DIR  # noqa: E402

_PANEL = MASTER_DIR / "panel_v3_monthly.parquet"
_FH = EXTERNAL_DATA_ROOT / "fnspid" / "full_history"
_REPORT = ROOT / "research" / "experiment_v3" / "results_experiment_v3.md"
_SEED = 42
_N_PLACEBO = 100
_RET = "ret_fwd_1m"   # primario: 1m forward, rebalanceo mensual
_MIN_PRICE = 1.0      # excluir penny stocks (< $1)
_WIN_LO, _WIN_HI = 0.01, 0.99   # winsorización de retornos por mes


def _month_end_close(tickers, log):
    """Precio de cierre (nominal) a fin de mes por (ticker, mes) desde FNSPID."""
    rows = []
    for t in tickers:
        p = _FH / f"{t}.csv"
        if not p.exists():
            continue
        try:
            d = pd.read_csv(p, usecols=["date", "close"])
            d["date"] = pd.to_datetime(d["date"], errors="coerce")
            d = d.dropna(subset=["date"]).set_index("date").sort_index()
            me = pd.to_numeric(d["close"], errors="coerce").resample("ME").last()
            df = me.rename("price").reset_index()
            df["month"] = df["date"].dt.to_period("M").astype(str)
            df["ticker"] = t
            rows.append(df[["ticker", "month", "price"]])
        except Exception:  # noqa: BLE001
            continue
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["ticker", "month", "price"])


def _winsorize_by_month(panel, col):
    def _w(s):
        return s.clip(s.quantile(_WIN_LO), s.quantile(_WIN_HI))
    panel[col] = panel.groupby("month")[col].transform(_w)
    return panel


def _ff3(months_period):
    import pandas_datareader.data as web
    ff = web.DataReader("F-F_Research_Data_Factors", "famafrench",
                        start="2014-06-01", end="2023-12-31")[0] / 100.0
    ff.index = ff.index.astype(str)   # 'YYYY-MM'
    return ff


def _groups(panel, n):
    """Asigna cada (ticker,mes) a un grupo por bullishness dentro del mes."""
    def _assign(g):
        if g["bullishness"].nunique() < n:
            return pd.Series(np.nan, index=g.index)
        return pd.qcut(g["bullishness"].rank(method="first"), n, labels=False)
    return panel.groupby("month", group_keys=False).apply(_assign)


def _ls_series(panel, grp_col, n, ret=_RET):
    """Serie mensual del long-short (grupo 0 = bajo sent, grupo n-1 = alto).
    LS = ret(bajo) - ret(alto). Indexada por el MES REALIZADO (formación + 1)."""
    d = panel.dropna(subset=[grp_col, ret]).copy()
    dec = d.groupby(["month", grp_col])[ret].mean().unstack()
    ls = dec[0] - dec[n - 1]  # low - high (contrarian: positivo si high pierde)
    # el retorno fwd_1m del mes de formación m se realiza en m+1
    ls.index = (pd.PeriodIndex(ls.index, freq="M") + 1).astype(str)
    return ls.dropna(), dec


def _alpha_ff3(ls, ff):
    import statsmodels.api as sm
    df = pd.concat([ls.rename("ls"), ff[["Mkt-RF", "SMB", "HML"]]], axis=1).dropna()
    if len(df) < 12:
        return np.nan, np.nan, len(df)
    X = sm.add_constant(df[["Mkt-RF", "SMB", "HML"]])
    res = sm.OLS(df["ls"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    return float(res.params["const"]), float(res.tvalues["const"]), len(df)


def _tstat(x):
    x = pd.Series(x).dropna()
    return float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))) if len(x) > 1 and x.std() > 0 else np.nan


def run() -> dict:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("eval_v3")

    panel = pd.read_parquet(_PANEL)
    n_raw = len(panel)

    # --- HIGIENE DE DATOS (winsorización estándar) ---
    # 1. Excluir penny stocks (precio de cierre < $1) al momento de formación.
    prices = _month_end_close(panel["ticker"].unique().tolist(), log)
    panel = panel.merge(prices, on=["ticker", "month"], how="left")
    n_penny = int((panel["price"] < _MIN_PRICE).sum())
    panel = panel[panel["price"] >= _MIN_PRICE].copy()
    # 3. Sanity de splits: cuántos ticker-mes con |retorno| > 200% ANTES de winsorizar.
    n_extreme = int((panel[_RET].abs() > 2.0).sum())
    log.info("Higiene: %d penny-stock ticker-mes excluidos; %d con |ret|>200%% antes de winsorizar.",
             n_penny, n_extreme)
    # 2. Winsorizar retorno mensual al 1%/99% dentro de cada mes.
    panel = _winsorize_by_month(panel, _RET)

    hygiene = {"panel_crudo": n_raw, "penny_excluidos(<$1)": n_penny,
               "extremos_|ret|>200pct_antes_winsor": n_extreme,
               "panel_limpio": len(panel)}
    per_month = panel.groupby("month").size()
    n = 10 if per_month.median() >= 50 else 5
    gname = "decil" if n == 10 else "quintil"
    log.info("Panel: %d filas, %d meses, mediana %d nombres/mes → %s (%d grupos).",
             len(panel), panel["month"].nunique(), int(per_month.median()), gname, n)

    panel["grp"] = _groups(panel, n)
    ls, dec = _ls_series(panel, "grp", n)
    ff = _ff3(panel["month"])

    # (a) retornos por grupo + long-short
    dec_means = dec.mean() * 100
    dec_t = {g: _tstat(dec[g]) for g in dec.columns}
    ls_mean = ls.mean() * 100
    ls_t = _tstat(ls)

    # (b) alfa FF3
    alpha, alpha_t, n_obs = _alpha_ff3(ls, ff)

    # (c) exposición tamaño/vol por grupo
    exp = panel.dropna(subset=["grp"]).groupby("grp").agg(
        size_med=("size", "median"), vol_med=("vol", "median"), n=("ticker", "size"))

    # (d) placebo ×100
    rng = np.random.default_rng(_SEED)
    placebo_alphas = []
    for _ in range(_N_PLACEBO):
        p = panel.copy()
        p["bullishness"] = p.groupby("month")["bullishness"].transform(
            lambda s: rng.permutation(s.to_numpy()))
        p["grp"] = _groups(p, n)
        ls_p, _ = _ls_series(p, "grp", n)
        a, _, _ = _alpha_ff3(ls_p, ff)
        if not np.isnan(a):
            placebo_alphas.append(a)
    placebo_alphas = np.array(placebo_alphas)
    # p-value one-sided en la dirección predicha (LS positivo)
    p_val = float(np.mean(placebo_alphas >= alpha)) if not np.isnan(alpha) else np.nan

    success = (not np.isnan(alpha_t) and alpha > 0 and alpha_t > 2 and p_val < 0.05)
    summary = {
        "higiene": hygiene,
        "grupos": n, "meses_LS": int(len(ls)), "obs_FF3": n_obs,
        "ls_mean_pct": round(ls_mean, 3), "ls_t": round(ls_t, 2),
        "alpha_ff3_pct_mensual": round(alpha * 100, 4), "alpha_t": round(alpha_t, 2),
        "placebo_alpha_mean_pct": round(float(placebo_alphas.mean()) * 100, 4),
        "placebo_alpha_std_pct": round(float(placebo_alphas.std()) * 100, 4),
        "p_value_placebo": round(p_val, 3), "n_placebo": len(placebo_alphas),
        "EXITO": bool(success),
    }
    _write_report(summary, dec_means, dec_t, exp, alpha, alpha_t, placebo_alphas, n, gname, log)
    import json
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    return summary


def _hist(vals, real):
    lo, hi = min(vals.min(), real), max(vals.max(), real)
    edges = np.linspace(lo, hi, 21)
    cnt, _ = np.histogram(vals, edges)
    top = max(cnt.max(), 1)
    out = []
    for i in range(20):
        bar = "#" * int(round(cnt[i] / top * 40))
        mark = "  <== ALFA REAL" if edges[i] <= real < edges[i + 1] else ""
        out.append(f"  [{edges[i]*100:+.3f}%] {bar}{mark}")
    return "\n".join(out)


def _write_report(s, dec_means, dec_t, exp, alpha, alpha_t, placebo, n, gname, log):
    h = s["higiene"]
    L = ["# Resultados Fase 3 — hipótesis contrarian (asset pricing) — VERSIÓN WINSORIZADA\n",
         "> **Pre-registro (README):** high-sentiment rinde por debajo (contrarian). Éxito: alfa "
         "FF3 del long-short (long bajo − short alto) t>2 con signo positivo Y fuera del placebo "
         "(p<0.05). Survivors-only sesga EN CONTRA del short → positivo creíble, negativo débil.\n",
         "> **Higiene de datos (estándar Baker-Wurgler / Stambaugh-Yu-Yuan), NO cambia hipótesis "
         "ni criterio:** excluir precio<$1, winsorizar retornos 1%/99% por mes.\n",
         f"Universo survivors-only. Grupos: **{gname}es ({n})**. Métrica: bullishness nativa "
         "(bull/(bull+bear)), >=20 menciones/mes.\n",
         "## Higiene de datos aplicada\n",
         f"- Panel crudo: {h['panel_crudo']:,} → penny stocks (<$1) excluidos: **{h['penny_excluidos(<$1)']:,}** "
         f"→ panel limpio: **{h['panel_limpio']:,}**",
         f"- Ticker-mes con **|retorno| > 200% ANTES de winsorizar: {h['extremos_|ret|>200pct_antes_winsor']:,}** "
         "(ruido de datos / posibles splits sin ajustar)\n",
         "## (a) Retorno mensual medio por grupo (0=sent bajo … {}=sent alto)\n".format(n - 1),
         "| grupo | ret medio (%/mes) | t-stat |", "|---|---|---|"]
    for g in sorted(dec_means.index):
        L.append(f"| {int(g)}{' (bajo)' if g==0 else ' (alto)' if g==n-1 else ''} | "
                 f"{dec_means[g]:+.3f} | {dec_t[g]:+.2f} |")
    L += ["", f"**Long-short (bajo − alto): {s['ls_mean_pct']:+.3f} %/mes, t={s['ls_t']:+.2f}** "
          "(positivo = contrarian se cumple)\n",
          "## (b) Alfa Fama-French 3 factores (HAC t-stats)\n",
          f"- **Alfa long-short: {s['alpha_ff3_pct_mensual']:+.4f} %/mes | t = {s['alpha_t']:+.2f}** "
          f"(obs={s['obs_FF3']} meses)\n",
          "## (c) Exposición tamaño/vol por grupo\n",
          "| grupo | dollar vol mediano | vol diaria mediana | n |", "|---|---|---|---|"]
    for g in sorted(exp.index):
        L.append(f"| {int(g)} | ${exp.loc[g,'size_med']/1e6:.1f}M | {exp.loc[g,'vol_med']:.4f} | {int(exp.loc[g,'n'])} |")
    L += ["", "## (d) Alfa real vs distribución placebo (100 permutaciones)\n",
          f"- Placebo alfa: media {s['placebo_alpha_mean_pct']:+.4f}% ± {s['placebo_alpha_std_pct']:.4f}% "
          f"| **p-value (real fuera del placebo): {s['p_value_placebo']}**\n",
          "```", _hist(placebo, alpha), "```", "",
          "## Veredicto (criterio pre-registrado)\n"]
    if s["EXITO"]:
        L.append("**ÉXITO:** alfa contrarian t>2 con signo predicho y fuera del placebo (p<0.05). "
                 "Por el caveat: survivors-only sesga EN CONTRA del short, así que el positivo es "
                 "**creíble**. (Aun así es investigación, no despliegue.)")
    else:
        reason = []
        if np.isnan(alpha_t) or alpha_t <= 2:
            reason.append(f"alfa t={s['alpha_t']} no supera 2")
        if alpha <= 0:
            reason.append("signo no es el predicho")
        if s["p_value_placebo"] >= 0.05:
            reason.append(f"placebo lo replica (p={s['p_value_placebo']})")
        L.append(f"**NEGATIVO:** {'; '.join(reason)}. Por el caveat pre-registrado, un negativo "
                 "aquí es **débil** (survivorship sesga en contra del efecto short) — pero no hay "
                 "evidencia de alfa contrarian explotable con estos datos.")
    L += ["", "## Winsorizado vs crudo\n",
          "- **Crudo:** alfa +48.26%/mes, t=1.01, placebo p=0.30 → NEGATIVO (magnitudes ininterpretables).",
          f"- **Winsorizado:** alfa {s['alpha_ff3_pct_mensual']:+.4f}%/mes, t={s['alpha_t']}, "
          f"placebo p={s['p_value_placebo']} → {'ÉXITO' if s['EXITO'] else 'NEGATIVO'} "
          "(magnitudes creíbles).",
          f"- **Veredicto coincide: {'sí' if not s['EXITO'] else 'NO (cambió!)'}** — la higiene de "
          "datos solo corrige las magnitudes, no altera la conclusión."]
    L.append("")
    _REPORT.write_text("\n".join(L), encoding="utf-8")
    log.info("Reporte en %s", _REPORT)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run()
