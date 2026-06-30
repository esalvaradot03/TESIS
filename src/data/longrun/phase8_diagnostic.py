"""
FASE 8 — Diagnóstico estructural profundo de dataset_v1 (NO entrena modelos).

Produce docs/diagnostico_dataset_v1.md y docs/outliers_dataset_v1.md.

8.1 Integridad: verificación del target, outliers (retornos extremos, sentimientos
    fuera de rango, fechas fuera de 2010-2023, duplicados).
8.2 Cobertura por ticker (histograma, top/bottom 20, bins).
8.3 Cobertura temporal (por año/trimestre, gaps).
8.4 Correlaciones Spearman feature↔target (global y por sub-grupo: década,
    cuartil de liquidez). Sign flips entre sub-grupos.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data.longrun.common import MASTER_DIR, setup_logger

JOB = "phase8_diagnostic"
ROOT = Path(__file__).resolve().parents[3]
_DATASET = MASTER_DIR / "dataset_v1.parquet"
_DOC_DIAG = ROOT / "docs" / "diagnostico_dataset_v1.md"
_DOC_OUT = ROOT / "docs" / "outliers_dataset_v1.md"
_DONE = MASTER_DIR.parent / "logs" / "phase8.done"

_TECH = ["rsi", "macd_hist", "bb_pct", "bb_width", "volatility_20d", "volume_ratio"]
_STOCKTWITS = ["bullish_count", "bearish_count", "total_count", "unique_users", "bullish_ratio"]
_WSB = ["mention_count", "avg_score", "total_comments"]
_FNSPID = ["news_count", "news_sentiment_mean"]
_ALL_FEATS = _TECH + _STOCKTWITS + _WSB + _FNSPID


def _pct(x: float) -> str:
    return f"{x*100:.1f}%"


def run(smoke: bool = False, **kw) -> dict:
    log = setup_logger(JOB)
    if not _DATASET.exists():
        log.error("Fase 8: no existe %s", _DATASET)
        return {"phase": JOB, "error": "dataset_v1 missing"}

    df = pd.read_parquet(_DATASET)
    df["date"] = pd.to_datetime(df["date"])
    log.info("Fase 8: dataset_v1 cargado %d filas × %d cols.", *df.shape)

    out_lines: list[str] = ["# Outliers — dataset_v1\n"]
    diag: list[str] = ["# Diagnóstico estructural — dataset_v1\n"]
    A = diag.append
    O = out_lines.append

    # ===================================================================
    # 8.1 Integridad
    # ===================================================================
    A("## 8.1 Integridad\n")
    # Target check
    excess = df["return_excess_over_spy_5d_forward"]
    expected = (excess > 0).astype(int)
    mismatch = int((expected != df["target"]).sum())
    A(f"- **Verificación del target:** target == (excess_return_5d_forward > 0) en "
      f"**{len(df)-mismatch}/{len(df)}** filas ({mismatch} discrepancias).\n")
    sample = df.sample(min(20, len(df)), random_state=42)[
        ["ticker", "date", "return_excess_over_spy_5d_forward", "target"]]
    A("20 (ticker, fecha) aleatorios:\n")
    A("| ticker | fecha | excess_5d | target | ok |")
    A("|---|---|---|---|---|")
    for r in sample.itertuples(index=False):
        ok = int(r.return_excess_over_spy_5d_forward > 0) == int(r.target)
        A(f"| {r.ticker} | {r.date.date()} | {r.return_excess_over_spy_5d_forward:+.4f} | "
          f"{int(r.target)} | {'✓' if ok else '✗'} |")
    A("")

    # Outliers
    O("## Retornos extremos (|return_5d_forward| > 50%)\n")
    extreme = df[df["return_5d_forward"].abs() > 0.50]
    O(f"- {len(extreme)} filas ({_pct(len(extreme)/len(df))}). Top 10 por |retorno|:\n")
    if len(extreme):
        top = extreme.reindex(extreme["return_5d_forward"].abs().sort_values(ascending=False).index).head(10)
        O("| ticker | fecha | return_5d | total_count | mention_count |")
        O("|---|---|---|---|---|")
        for r in top.itertuples(index=False):
            O(f"| {r.ticker} | {pd.to_datetime(r.date).date()} | {r.return_5d_forward:+.3f} | "
              f"{getattr(r,'total_count',0):.0f} | {getattr(r,'mention_count',0):.0f} |")
    O("")

    O("## Sentimientos fuera de rango\n")
    bad_ratio = df[(df["bullish_ratio"] < 0) | (df["bullish_ratio"] > 1)]
    bad_news = df[df["news_sentiment_mean"].notna() &
                  ((df["news_sentiment_mean"] < -1) | (df["news_sentiment_mean"] > 1))]
    O(f"- bullish_ratio fuera de [0,1]: **{len(bad_ratio)}**")
    O(f"- news_sentiment_mean fuera de [-1,1]: **{len(bad_news)}**\n")

    O("## Fechas fuera de 2010-2023\n")
    bad_dates = df[(df["date"] < "2010-01-01") | (df["date"] > "2023-12-31")]
    O(f"- {len(bad_dates)} filas fuera de rango.\n")

    O("## Duplicados por (ticker, fecha)\n")
    dups = int(df.duplicated(subset=["ticker", "date"]).sum())
    O(f"- {dups} duplicados.\n")

    O("## Nota: precios\n- dataset_v1 no contiene columna de precio absoluto "
      "(solo retornos y features estacionarias), así que no aplica el chequeo "
      "'precio < 0'.\n")

    A(f"- Outliers: retornos extremos={len(extreme)}, sentimiento fuera de rango="
      f"{len(bad_ratio)+len(bad_news)}, fechas malas={len(bad_dates)}, duplicados={dups}. "
      f"Detalle en `docs/outliers_dataset_v1.md`.\n")

    # ===================================================================
    # 8.2 Cobertura por ticker
    # ===================================================================
    A("## 8.2 Cobertura por ticker\n")
    per_ticker = df.groupby("ticker").size().sort_values(ascending=False)
    b1 = int((per_ticker < 100).sum())
    b2 = int(((per_ticker >= 100) & (per_ticker <= 1000)).sum())
    b3 = int((per_ticker > 1000).sum())
    A(f"- Tickers totales: **{len(per_ticker)}** | <100 filas: {b1} | 100-1000: {b2} | >1000: {b3}")
    A(f"- Mediana filas/ticker: {int(per_ticker.median())} | máx: {int(per_ticker.max())} "
      f"({per_ticker.index[0]}) | mín: {int(per_ticker.min())}\n")
    A("Top 20 por filas:\n")
    A("| ticker | filas |  | ticker | filas |")
    A("|---|---|---|---|---|")
    top20 = per_ticker.head(20).reset_index().values
    bot20 = per_ticker.tail(20).reset_index().values
    for i in range(20):
        t1, n1 = top20[i]
        t2, n2 = bot20[i] if i < len(bot20) else ("", "")
        A(f"| {t1} | {n1} |  | {t2} | {n2} |")
    A("")
    share_top20 = per_ticker.head(20).sum() / len(df)
    A(f"**Concentración:** los 20 tickers con más datos acumulan {_pct(share_top20)} de las "
      f"filas → la señal está {'concentrada en mega-caps' if share_top20 > 0.4 else 'razonablemente distribuida'}.\n")

    # ===================================================================
    # 8.3 Cobertura temporal
    # ===================================================================
    A("## 8.3 Cobertura temporal\n")
    by_year = df.groupby(df["date"].dt.year).size()
    A("| año | filas |")
    A("|---|---|")
    for y, n in by_year.items():
        A(f"| {y} | {n:,} |")
    A("")
    months = pd.period_range(df["date"].min().to_period("M"), df["date"].max().to_period("M"), freq="M")
    present = set(df["date"].dt.to_period("M").astype(str))
    gaps = [str(m) for m in months if str(m) not in present]
    A(f"- Meses sin datos (gaps): {len(gaps)}" + (f" → {gaps[:12]}..." if gaps else " (ninguno)") + "\n")

    # ===================================================================
    # 8.4 Correlaciones Spearman feature ↔ target
    # ===================================================================
    A("## 8.4 Correlaciones Spearman feature ↔ target\n")
    A(f"Con n={len(df):,} se exige **p < 0.001** para considerar señal.\n")

    def _spearman_table(d: pd.DataFrame, label: str) -> dict[str, float]:
        rows = []
        for c in _ALL_FEATS:
            if c not in d.columns:
                continue
            x = pd.to_numeric(d[c], errors="coerce")
            m = x.notna() & d["target"].notna()
            if m.sum() < 50 or x[m].nunique() < 3:
                continue
            rho, p = spearmanr(x[m], d["target"][m])
            rows.append((c, rho, p))
        rows.sort(key=lambda t: abs(t[1]), reverse=True)
        A(f"### {label}  (n={len(d):,})\n")
        A("| # | feature | ρ | p | señal (p<0.001) |")
        A("|---|---|---|---|---|")
        for i, (c, rho, p) in enumerate(rows[:15], 1):
            sig = "**sí**" if p < 0.001 and abs(rho) > 0.05 else "no"
            A(f"| {i} | {c} | {rho:+.4f} | {p:.2e} | {sig} |")
        A("")
        return {c: rho for c, rho, _ in rows}

    glob = _spearman_table(df, "Global")

    # Sub-grupos por década
    decades = {"2010-2015": (2010, 2015), "2016-2020": (2016, 2020), "2021-2023": (2021, 2023)}
    sub_corrs = {}
    for name, (a, b) in decades.items():
        d = df[(df["date"].dt.year >= a) & (df["date"].dt.year <= b)]
        if len(d) > 100:
            sub_corrs[name] = _spearman_table(d, f"Década {name}")

    # Sub-grupos por liquidez (cuartiles de mention_count combinado)
    A("### Por liquidez (cuartiles de combined_mentions)\n")
    try:
        df["_liq"] = pd.qcut(df["combined_mentions"], 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
        for q in df["_liq"].cat.categories:
            d = df[df["_liq"] == q]
            sub_corrs[f"liq_{q}"] = _spearman_table(d, f"Liquidez {q}")
    except Exception as exc:  # noqa: BLE001
        A(f"_No se pudo cuartilizar liquidez: {exc}_\n")

    # Sign flips
    A("## Sign flips entre sub-grupos\n")
    sent_feats = [c for c in _STOCKTWITS + _WSB + _FNSPID if c in glob]
    A("| feature | global | " + " | ".join(sub_corrs.keys()) + " | ¿flip? |")
    A("|---|---|" + "|".join(["---"] * len(sub_corrs)) + "|---|")
    n_flips = 0
    for f in sent_feats:
        vals = [glob.get(f, float("nan"))] + [sub_corrs[k].get(f, float("nan")) for k in sub_corrs]
        signs = [np.sign(v) for v in vals if not np.isnan(v) and abs(v) > 0.02]
        flip = len(set(signs)) > 1
        n_flips += int(flip)
        cells = " | ".join(f"{v:+.3f}" if not np.isnan(v) else "—" for v in vals)
        A(f"| {f} | {cells} | {'**SÍ**' if flip else 'no'} |")
    A(f"\n**{n_flips}/{len(sent_feats)} features de sentimiento cambian de signo entre sub-grupos.**\n")

    # --- Persistencia ---
    _DOC_DIAG.parent.mkdir(parents=True, exist_ok=True)
    _DOC_DIAG.write_text("\n".join(diag), encoding="utf-8")
    _DOC_OUT.write_text("\n".join(out_lines), encoding="utf-8")
    if not smoke:
        _DONE.write_text(pd.Timestamp.now().isoformat(), encoding="utf-8")
    log.info("Fase 8 fin: %s y %s escritos; DONE.", _DOC_DIAG, _DOC_OUT)
    return {"phase": JOB, "rows": len(df), "tickers": int(per_ticker.shape[0]),
            "target_mismatch": mismatch, "extreme_returns": len(extreme),
            "sentiment_sign_flips": n_flips, "docs": [str(_DOC_DIAG), str(_DOC_OUT)]}


if __name__ == "__main__":
    run(smoke="--smoke" in sys.argv)
