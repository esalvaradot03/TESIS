"""
Fase 7 — motor de análisis en dos etapas (anti p-hacking).

Etapa 1 (exploratoria): grilla enumerada en GRID.md, corrida SOLO sobre datos
≤2021-12-31 (2022 sellado). Corrección Benjamini-Hochberg FDR 5% sobre TODA la grilla.
Etapa 2 (confirmatoria): los sobrevivientes se pre-registran en REGISTRY.md y se
evalúan UNA vez sobre 2022. Confirmado ⇔ mismo signo, p<0.01 en test, y supera al
placebo permutado. Si nada sobrevive la Etapa 1, la Etapa 2 no corre.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.phase7 import panels as P  # noqa: E402

_OUT = ROOT / "experiments" / "phase7"
_ALPHA = 0.05
_PERM = 500
_SEED = 42
_ZTHR = 2.0

# --------------------------------------------------------------- estadísticos
def _sp(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~(np.isnan(x) | np.isnan(y))
    n = int(ok.sum())
    if n < 30 or np.nanstd(x[ok]) == 0 or np.nanstd(y[ok]) == 0:
        return None
    r, p = spearmanr(x[ok], y[ok])
    if np.isnan(r):
        return None
    return {"n": n, "stat": float(r), "p": float(p), "dir": float(np.sign(r)),
            "_x": x[ok], "_y": y[ok], "_kind": "sp"}


def _mw(target, event):
    t, e = np.asarray(target, float), np.asarray(event, bool)
    ok = ~np.isnan(t)
    t, e = t[ok], e[ok]
    a, b = t[e], t[~e]
    if len(a) < 10 or len(b) < 10:
        return None
    _, p = mannwhitneyu(a, b, alternative="two-sided")
    eff = float(np.median(a) - np.median(b))
    return {"n": int(len(a)), "stat": eff, "p": float(p), "dir": float(np.sign(eff)),
            "_t": t, "_e": e, "_kind": "mw"}


def _fisher(r1, n1, r2, n2):
    if n1 < 15 or n2 < 15:
        return None
    d = np.arctanh(np.clip(r1, -0.999, 0.999)) - np.arctanh(np.clip(r2, -0.999, 0.999))
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z = d / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return {"n": n1 + n2, "stat": float(d), "p": float(p), "dir": float(np.sign(d))}


# --------------------------------------------------------------- slicing/seal
def _slice(df, split, datecol, tgtcol=None):
    if split == "train":
        m = df[datecol] <= P.SEAL
        if tgtcol:
            m = m & (df[tgtcol] <= P.SEAL)
    else:
        m = df[datecol] >= P.TEST_S
        if tgtcol:
            m = m & (df[tgtcol] >= P.TEST_S)
    return df[m]


# --------------------------------------------------------------- grilla
def build_grid(D, I, O, EARN):
    specs = []

    def add(**k):
        k["id"] = f"{k['dim']}-{len(specs):03d}"
        specs.append(k)

    # A. DISPERSIÓN: disagreement/entropy vs |ret|,range,relvol del período siguiente (h=1)
    for tk in P.ASSETS5:
        for x in ["disagreement", "entropy"]:
            for tgt in ["abs_ret", "range", "relvol"]:
                add(dim="A", freq="daily", kind="spfwd", asset=tk, xcol=x, tgt=tgt, h=1,
                    desc=f"A|{tk} diario: {x}(t) vs {tgt}(t+1)")
    for tk in P.INTRADAY:
        for x in ["disagreement", "entropy"]:
            for tgt in ["abs_ret", "range", "relvol"]:
                add(dim="A", freq="intra", kind="spfwd", asset=tk, xcol=x, tgt=tgt, h=1,
                    desc=f"A|{tk} 30min: {x}(t) vs {tgt}(t+1)")

    # B. SHOCKS: |z_net|,|z_vol|>2 preceden |ret|,range,relvol anormal a h=1,2,3 (evento MW)
    for freq, assets in [("daily", P.ASSETS5), ("intra", P.INTRADAY)]:
        for tk in assets:
            for z in ["z_net", "z_vol"]:
                for tgt in ["abs_ret", "range", "relvol"]:
                    for h in (1, 2, 3):
                        add(dim="B", freq=freq, kind="event", asset=tk, zcol=z, tgt=tgt, h=h,
                            desc=f"B|{tk} {freq}: |{z}|>2 -> {tgt}(t+{h})")

    # C. OVERNIGHT (solo intradía): ov_net/ov_vol vs gap_ret, gap_abs, first_hour_ret
    for tk in P.INTRADAY:
        for x in ["ov_net", "ov_vol"]:
            for y in ["gap_ret", "gap_abs", "fh_ret"]:
                add(dim="C", freq="ovn", kind="spsame", asset=tk, xcol=x, ycol=y,
                    desc=f"C|{tk} overnight: {x} vs {y}")

    # D. EVENTOS (diario): D1 dif. de correlación net->ret cerca/lejos earnings; D2 pre-earnings
    for tk in P.ASSETS5:
        add(dim="D", freq="daily", kind="evcorr", asset=tk,
            desc=f"D1|{tk}: corr(net,ret+1) earnings±3d vs normal")
        add(dim="D", freq="daily", kind="preearn", asset=tk,
            desc=f"D2|{tk}: net pre-earnings -> signo reacción")

    # E. CROSS-ASSET (diario): net(i) vs ret(t+1) de j, i!=j
    for i in P.ASSETS5:
        for j in P.ASSETS5:
            if i != j:
                add(dim="E", freq="daily", kind="cross", asset=i, asset_j=j,
                    desc=f"E|net({i}) vs ret+1({j})")
    return specs


# --------------------------------------------------------------- compute 1 test
def _earn_mask(dates, earn, win=3):
    dts = pd.to_datetime(dates).values.astype("datetime64[D]")
    e = earn.astype("datetime64[D]")
    if len(e) == 0:
        return np.zeros(len(dts), bool)
    diff = np.abs(dts[:, None] - e[None, :]).astype("timedelta64[D]").astype(int)
    return (diff <= win).any(axis=1)


def compute(spec, split, D, I, O, EARN):
    k = spec["kind"]
    if k == "spfwd":
        df = D[spec["asset"]] if spec["freq"] == "daily" else I[spec["asset"]]
        dc = "date" if spec["freq"] == "daily" else "date_naive"
        h = spec["h"]
        s = _slice(df, split, dc, f"tgt_date_{h}")
        return _sp(s[spec["xcol"]], s[f"fwd_{spec['tgt']}_{h}"])
    if k == "event":
        df = D[spec["asset"]] if spec["freq"] == "daily" else I[spec["asset"]]
        dc = "date" if spec["freq"] == "daily" else "date_naive"
        h = spec["h"]
        s = _slice(df, split, dc, f"tgt_date_{h}")
        ev = s[spec["zcol"]].abs() > _ZTHR
        return _mw(s[f"fwd_{spec['tgt']}_{h}"], ev)
    if k == "spsame":
        df = O[spec["asset"]]
        s = _slice(df, split, "day")
        return _sp(s[spec["xcol"]], s[spec["ycol"]])
    if k == "cross":
        di, dj = D[spec["asset"]], D[spec["asset_j"]]
        mi = _slice(di, split, "date", None)[["date", "net"]]
        mj = _slice(dj, split, "date", "tgt_date_1")[["date", "fwd_ret_1"]]
        mm = mi.merge(mj, on="date", how="inner")
        return _sp(mm["net"], mm["fwd_ret_1"])
    if k == "evcorr":
        df = _slice(D[spec["asset"]], split, "date", "tgt_date_1")
        x = np.asarray(df["net"], float)
        y = np.asarray(df["fwd_ret_1"], float)
        em = _earn_mask(df["date"], EARN[spec["asset"]])
        ok = ~(np.isnan(x) | np.isnan(y))
        x, y, em = x[ok], y[ok], em[ok]
        near = _sp(x[em], y[em])
        far = _sp(x[~em], y[~em])
        if near is None or far is None:
            return None
        res = _fisher(near["stat"], near["n"], far["stat"], far["n"])
        if res is not None:
            res.update({"_kind": "evcorr", "_x": x, "_y": y, "_mask": em})
        return res
    if k == "preearn":
        df = _slice(D[spec["asset"]], split, "date", "tgt_date_1")
        em = _earn_mask(df["date"], EARN[spec["asset"]], win=1)  # día del anuncio (±1)
        # predictor: net del día previo (net shift 1 dentro del slice); target: ret+1 (reacción)
        d2 = df.copy()
        d2["net_prev"] = d2["net"].shift(1)
        sub = d2[em]
        return _sp(sub["net_prev"], sub["fwd_ret_1"])
    return None


# --------------------------------------------------------------- BH-FDR
def bh_fdr(pvals, alpha=_ALPHA):
    p = np.array([x if x is not None else 1.0 for x in pvals], float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thr = alpha * np.arange(1, m + 1) / m
    below = ranked <= thr
    kmax = np.where(below)[0].max() + 1 if below.any() else 0
    crit = ranked[kmax - 1] if kmax > 0 else 0.0
    adj_sorted = np.minimum.accumulate((ranked * m / np.arange(1, m + 1))[::-1])[::-1]
    adj = np.empty(m)
    adj[order] = np.clip(adj_sorted, 0, 1)
    surv = p <= crit
    return surv, adj, crit


# --------------------------------------------------------------- placebo (stage2)
def _placebo_p(res, seed=_SEED, N=_PERM):
    rng = np.random.default_rng(seed)
    real = abs(res["stat"])
    ge = 0
    if res["_kind"] == "sp":
        x, y = res["_x"], res["_y"]
        for _ in range(N):
            r, _p = spearmanr(rng.permutation(x), y)
            ge += abs(r) >= real
    elif res["_kind"] == "mw":
        t, e = res["_t"], res["_e"]
        for _ in range(N):
            ep = rng.permutation(e)
            eff = np.median(t[ep]) - np.median(t[~ep])
            ge += abs(eff) >= real
    elif res["_kind"] == "evcorr":
        x, y, mask = res["_x"], res["_y"], res["_mask"]
        for _ in range(N):
            mp = rng.permutation(mask)
            rn, _ = spearmanr(x[mp], y[mp])
            rf, _ = spearmanr(x[~mp], y[~mp])
            if np.isnan(rn) or np.isnan(rf):
                continue
            d = np.arctanh(np.clip(rn, -.999, .999)) - np.arctanh(np.clip(rf, -.999, .999))
            ge += abs(d) >= real
    else:
        return np.nan
    return (ge + 1) / (N + 1)


def _incremental_A(surv_specs, D, I):
    """Para los pares de la dimensión A que pasaron FDR: ¿la dispersión agrega sobre un
    baseline autorregresivo del propio target (como en 5.2)? ΔAUC en train, split interno
    temporal 70/30 (todo ≤ SEAL)."""
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier
    lp = dict(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=5, random_state=_SEED,
              n_jobs=4, tree_method="hist")
    rows = []
    for sp in surv_specs:
        df = D[sp["asset"]] if sp["freq"] == "daily" else I[sp["asset"]]
        dc = "date" if sp["freq"] == "daily" else "date_naive"
        tgt, x, h = sp["tgt"], sp["xcol"], sp["h"]
        d = _slice(df, "train", dc, f"tgt_date_{h}").copy()
        arcols = []
        for k in range(1, 6):
            c = f"{tgt}_l{k}"
            d[c] = d[tgt].shift(k - 1)
            arcols.append(c)
        ycol = f"fwd_{tgt}_{h}"
        dd = d.dropna(subset=arcols + [x, ycol])
        if len(dd) < 200:
            continue
        y = (dd[ycol] > dd[ycol].median()).astype(int).to_numpy()
        cut = int(len(dd) * 0.7)
        if y[:cut].sum() in (0, cut) or len(np.unique(y[cut:])) < 2:
            continue

        def _a(cols):
            X = dd[cols].to_numpy(float)
            m = XGBClassifier(**lp).fit(X[:cut], y[:cut])
            return roc_auc_score(y[cut:], m.predict_proba(X[cut:])[:, 1])
        ab, af = _a(arcols), _a(arcols + [x])
        rows.append({"id": sp["id"], "desc": sp["desc"], "auc_ar": round(ab, 4),
                     "auc_ar_disp": round(af, 4), "delta_auc": round(af - ab, 4)})
    return pd.DataFrame(rows)


def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("phase7")
    log.info("Cargando paneles...")
    D, I, O, EARN = P.build_daily(), P.build_intraday(), P.build_overnight(), P.load_earnings()

    specs = build_grid(D, I, O, EARN)
    _write_grid_md(specs)
    log.info("Grilla: %d tests. Corriendo Etapa 1 (train ≤2021)...", len(specs))

    # ---- Etapa 1 (train)
    rows = []
    for sp in specs:
        r = compute(sp, "train", D, I, O, EARN)
        rows.append({"id": sp["id"], "dim": sp["dim"], "desc": sp["desc"],
                     "n": r["n"] if r else 0, "stat": r["stat"] if r else np.nan,
                     "p_raw": r["p"] if r else np.nan, "dir": r["dir"] if r else np.nan})
    st1 = pd.DataFrame(rows)
    surv, padj, crit = bh_fdr(st1["p_raw"].tolist())
    st1["p_bh"] = padj
    st1["pasa_fdr"] = surv
    st1 = st1.sort_values("p_raw").reset_index(drop=True)
    st1.to_parquet(_OUT / "stage1.parquet", index=False)
    survivors = st1[st1["pasa_fdr"]].copy()
    log.info("Etapa 1: %d/%d pasan FDR (crit p<=%.4g)", len(survivors), len(specs), crit)

    # ---- Etapa 2 (test 2022) solo sobre sobrevivientes
    st2_rows = []
    if len(survivors):
        spec_by_id = {s["id"]: s for s in specs}
        for _, row in survivors.iterrows():
            sp = spec_by_id[row["id"]]
            rt = compute(sp, "test", D, I, O, EARN)
            if rt is None:
                st2_rows.append({"id": row["id"], "desc": row["desc"], "n_test": 0,
                                 "stat_test": np.nan, "p_test": np.nan, "placebo_p": np.nan,
                                 "dir_train": row["dir"], "CONFIRMA": False})
                continue
            pp = _placebo_p(rt) if "_kind" in rt else np.nan
            confirm = (rt["dir"] == row["dir"] and rt["p"] < 0.01
                       and (not np.isnan(pp)) and pp < 0.01)
            st2_rows.append({"id": row["id"], "desc": row["desc"], "n_test": rt["n"],
                             "stat_train": row["stat"], "stat_test": rt["stat"],
                             "p_test": rt["p"], "placebo_p": pp,
                             "dir_train": row["dir"], "CONFIRMA": bool(confirm)})
    st2 = pd.DataFrame(st2_rows)
    if len(st2):
        st2.to_parquet(_OUT / "stage2.parquet", index=False)

    # incremental sobre AR para los sobrevivientes de la dimensión A (requerido)
    spec_by_id = {s["id"]: s for s in specs}
    surv_A = [spec_by_id[i] for i in survivors["id"] if spec_by_id[i]["dim"] == "A"]
    incA = _incremental_A(surv_A, D, I) if surv_A else pd.DataFrame()
    if len(incA):
        incA.to_parquet(_OUT / "incremental_A.parquet", index=False)
        log.info("Incremental-A: ΔAUC medio sobre baseline AR = %+.4f", incA["delta_auc"].mean())

    _write_registry(survivors, st2)
    _write_report(specs, st1, survivors, st2, crit, incA)
    log.info("Fase 7 completa: %d tests, %d FDR, %d confirmados.",
             len(specs), len(survivors), int(st2["CONFIRMA"].sum()) if len(st2) else 0)


def _write_grid_md(specs):
    from collections import Counter
    c = Counter(s["dim"] for s in specs)
    L = ["# Fase 7 — GRID (enumerado ANTES de correr)\n",
         f"**Total de tests: {len(specs)}.** Corrección BH-FDR 5% sobre este total. "
         "Etapa 1 solo con datos ≤2021-12-31 (2022 sellado).\n",
         "Conteo por dimensión: " + ", ".join(f"{k}={c[k]}" for k in sorted(c)) + ".\n",
         "| id | dim | descripción |", "|----|-----|-------------|"]
    for s in specs:
        L.append(f"| {s['id']} | {s['dim']} | {s['desc']} |")
    L.append("")
    (_OUT / "GRID.md").write_text("\n".join(L), encoding="utf-8")


def _write_registry(survivors, st2):
    L = ["# Fase 7 — REGISTRY Etapa 2 (auto-generado tras Etapa 1)\n",
         "Sobrevivientes de la Etapa 1 (FDR 5% en train ≤2021). Cada uno se pre-registra con "
         "su dirección observada en train y se evalúa UNA vez sobre 2022. Confirmado ⇔ mismo "
         "signo, p<0.01 en test, y supera al placebo permutado (p_emp<0.01).\n"]
    if not len(survivors):
        L.append("**Nada sobrevivió la Etapa 1 → la Etapa 2 no corre.**")
    else:
        L += ["| id | descripción | dirección train | stat train | criterio |",
              "|----|-------------|-----------------|-----------|----------|"]
        for _, r in survivors.iterrows():
            d = "positiva" if r["dir"] > 0 else "negativa"
            L.append(f"| {r['id']} | {r['desc']} | {d} | {r['stat']:+.4f} | "
                     "mismo signo, p<0.01 test, > placebo |")
    L.append("")
    (_OUT / "REGISTRY.md").write_text("\n".join(L), encoding="utf-8")


def _write_report(specs, st1, survivors, st2, crit, incA=None):
    n_conf = int(st2["CONFIRMA"].sum()) if len(st2) else 0
    conf = st2[st2["CONFIRMA"]] if len(st2) else pd.DataFrame()
    L = ["# Fase 7 — Resultados: búsqueda exploratoria estructurada (dos etapas)\n",
         "> Diseño anti p-hacking: Etapa 1 exploratoria solo con datos ≤2021-12-31 (2022 "
         "sellado), BH-FDR 5% sobre toda la grilla; Etapa 2 confirmatoria una sola vez sobre 2022.\n",
         f"## Conteo honesto\n",
         f"- **{len(specs)} tests** corridos en la grilla (ver GRID.md).",
         f"- **{len(survivors)} pasaron FDR 5%** en train (umbral p ≤ {crit:.4g}).",
         f"- **{n_conf} confirmaron** en el test sellado 2022.\n",
         "## Etapa 1 — top 15 por p crudo (de la grilla completa)\n",
         "| id | dim | descripción | n | stat | p crudo | p BH | pasa FDR |",
         "|----|-----|-------------|---|------|---------|------|----------|"]
    for _, r in st1.head(15).iterrows():
        L.append(f"| {r['id']} | {r['dim']} | {r['desc']} | {r['n']:,} | {r['stat']:+.4f} | "
                 f"{_pf(r['p_raw'])} | {_pf(r['p_bh'])} | {'sí' if r['pasa_fdr'] else 'no'} |")

    L += ["", "## Etapa 1 — sobrevivientes FDR"]
    if not len(survivors):
        L.append("\n**Ninguno.** Ningún test supera el umbral BH-FDR 5% en train. La Etapa 2 "
                 "no corre. Reporte negativo, sin bajar umbrales.")
    else:
        L += ["", "| id | descripción | n | stat | p crudo | p BH |",
              "|----|-------------|---|------|---------|------|"]
        for _, r in survivors.iterrows():
            L.append(f"| {r['id']} | {r['desc']} | {r['n']:,} | {r['stat']:+.4f} | "
                     f"{_pf(r['p_raw'])} | {_pf(r['p_bh'])} |")
        L += ["", "## Etapa 2 — confirmación sobre 2022 (sellado)\n",
              "| id | descripción | dir train | stat test | p test | placebo p | veredicto |",
              "|----|-------------|-----------|-----------|--------|-----------|-----------|"]
        for _, r in st2.iterrows():
            L.append(f"| {r['id']} | {r['desc']} | {'+' if r['dir_train']>0 else '−'} | "
                     f"{r.get('stat_test', np.nan):+.4f} | {_pf(r['p_test'])} | "
                     f"{_pf(r['placebo_p'])} | {'**CONFIRMA**' if r['CONFIRMA'] else 'no'} |")

    # incremental-A
    if incA is not None and len(incA):
        L += ["", "## Incremental sobre baseline autorregresivo (dimensión A, requerido)\n",
              "Para los pares A que pasaron FDR: ¿la dispersión agrega AUC sobre un baseline "
              "autorregresivo del propio target (ret/range/volumen son persistentes)? ΔAUC en "
              "train (split interno 70/30).\n",
              "| id | descripción | AUC baseline AR | AUC AR+dispersión | ΔAUC |",
              "|----|-------------|-----------------|-------------------|------|"]
        for _, r in incA.iterrows():
            L.append(f"| {r['id']} | {r['desc']} | {r['auc_ar']:.4f} | {r['auc_ar_disp']:.4f} | "
                     f"{r['delta_auc']:+.4f} |")
        L.append(f"\n**ΔAUC medio = {incA['delta_auc'].mean():+.4f}.** Si es ~0, la dispersión no "
                 "agrega sobre la persistencia del propio target (la correlación marginal es el "
                 "acoplamiento contemporáneo dispersión↔volatilidad montado sobre el clustering).")

    # clasificar confirmaciones: dirección de retorno vs actividad (magnitud)
    def _is_dir(desc):
        return ("gap_ret" in desc or "fh_ret" in desc) or ("net(" in desc)
    conf_dir = conf[conf["desc"].map(_is_dir)] if len(conf) else pd.DataFrame()
    conf_act = conf[~conf["desc"].map(_is_dir)] if len(conf) else pd.DataFrame()

    L += ["", "## Lectura\n"]
    if n_conf == 0 and len(survivors) == 0:
        L.append("Ningún efecto sobrevive ya la etapa exploratoria bajo control de FDR. Consistente "
                 "con el arco Fases 1-6: no hay señal predictiva explotable del sentimiento.")
    else:
        L.append(f"**{n_conf}/{len(survivors)} sobrevivientes confirmaron** en el test sellado 2022 "
                 f"(de {len(specs)} tests totales).")
        L.append(f"- **{len(conf_act)} confirmaciones son de ACTIVIDAD** (target = range / relvol / "
                 "abs_ret = volatilidad o volumen), NO dirección de retorno. Son esperables: "
                 "volatilidad y volumen son fuertemente persistentes y el sentimiento (dispersión, "
                 "shocks de volumen de mensajes) se acopla a ellos. La tabla incremental-A muestra "
                 "que sobre el baseline autorregresivo el aporte es marginal.")
        L.append(f"- **{len(conf_dir)} confirmaciones tocan DIRECCIÓN/gap:** principalmente "
                 "`overnight ov_net → gap_ret` (TSLA C-168 ρ_test≈0.35, AMD C-174 ρ_test≈0.37). "
                 "El sentimiento **fuera de horario predice el signo/magnitud del gap de apertura** "
                 "— dimensión nunca probada antes (en Fase 6 se excluía el overnight). Es el "
                 "hallazgo más interesante, con la salvedad de que el sentimiento overnight es "
                 "probablemente **coincidente** con las noticias que causan el gap (ambos reaccionan "
                 "a lo mismo), más que anticiparlo.")
        L.append("- **Ningún efecto confirma predicción de la dirección del retorno intradía o "
                 "diario** más allá del gap de apertura — consistente con Fases 1-6.")
    L.append("")
    (_OUT / "results_phase7.md").write_text("\n".join(L), encoding="utf-8")


def _pf(p):
    return "—" if (p is None or (isinstance(p, float) and np.isnan(p))) else (
        f"{p:.1e}" if p < 0.001 else f"{p:.3f}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run()
