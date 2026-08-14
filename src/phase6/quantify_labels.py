"""
Fase 6 — paso previo al pre-registro: cuantificación del join de labels intradía.

Une feature_wo_messages (message_id → timestamp intradía) con symbol_sentiments
(message_id → label Bull/Bear) para TSLA y AMD, 2020-08 → 2022-12, y mide cuántos
mensajes ETIQUETADOS hay por ventana de 30 min. Regla (fijada antes de ver el
resultado): si la mediana de mensajes etiquetados por ventana es >=10 en AMBOS
activos → el NET etiquetado es la feature principal; si <10 en alguno → ese activo
usa volumen + aceleración como principales y el NET como secundaria.
"""

import logging
import re
import sys
import time
from datetime import time as dtime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config.settings import (  # noqa: E402
    EXTERNAL_DATA_ROOT,
    STOCKTWITS_NYU_SYMBOL_SENTIMENTS,
)

_FWM = EXTERNAL_DATA_ROOT / "stocktwits_nyu" / "feature_wo_messages"
_OUT = ROOT / "experiments" / "phase6"
_CACHE = _OUT / "cache"
TICKERS = ["TSLA", "AMD"]
_WIN_S = pd.Timestamp("2020-08-01", tz="America/New_York")
_WIN_E = pd.Timestamp("2022-12-31 23:59:59", tz="America/New_York")
_S_DATE, _E_DATE = pd.Timestamp("2020-08-01"), pd.Timestamp("2022-12-31")
_MKT_OPEN, _MKT_CLOSE = dtime(9, 30), dtime(16, 0)
_CHUNK = 1_000_000
_TICK_RE = re.compile(r"'(TSLA|AMD)'")
_HAS_RE = re.compile(r"'(?:TSLA|AMD)'")


def scan_intraday_msgs(log) -> pd.DataFrame:
    """feature_wo_messages → (message_id, ticker, w30) intradía en rango."""
    cache = _CACHE / "intraday_msgids.parquet"
    if cache.exists():
        log.info("intraday msgids: cache %s", cache.name)
        return pd.read_parquet(cache)
    files = sorted(_FWM.glob("*.csv"))
    parts, t0 = [], time.time()
    for fi, f in enumerate(files, 1):
        try:
            reader = pd.read_csv(f, usecols=["message_id", "created_at", "symbol_list"],
                                 dtype="string", chunksize=_CHUNK, on_bad_lines="skip",
                                 encoding="utf-8", encoding_errors="replace")
        except Exception:  # noqa: BLE001
            continue
        for chunk in reader:
            hit = chunk["symbol_list"].str.contains(_HAS_RE, na=False)
            if not hit.any():
                continue
            sub = chunk[hit]
            t = pd.to_datetime(sub["created_at"], errors="coerce", utc=True)
            d = pd.DataFrame({
                "message_id": sub["message_id"],
                "et": t.dt.tz_convert("America/New_York"),
                "ticker": sub["symbol_list"].map(_TICK_RE.findall),
            }).explode("ticker").dropna(subset=["et", "ticker"])
            d = d[(d["et"] >= _WIN_S) & (d["et"] <= _WIN_E)]
            tod = d["et"].dt.time
            d = d[(tod >= _MKT_OPEN) & (tod < _MKT_CLOSE) & (d["et"].dt.weekday < 5)]
            if d.empty:
                continue
            d["w30"] = d["et"].dt.floor("30min")
            parts.append(d[["message_id", "ticker", "w30"]])
        if fi % 40 == 0:
            log.info("  intraday %d/%d (%.0fs, %d msgs)", fi, len(files),
                     time.time() - t0, sum(len(p) for p in parts))
    res = pd.concat(parts, ignore_index=True)
    _CACHE.mkdir(parents=True, exist_ok=True)
    res.to_parquet(cache, index=False)
    log.info("intraday msgids: %d filas -> %s", len(res), cache.name)
    return res


def scan_labeled(log) -> pd.DataFrame:
    """symbol_sentiments → (message_id, sentiment) etiquetados de TSLA/AMD en rango."""
    cache = _CACHE / "labeled_msgids.parquet"
    if cache.exists():
        log.info("labeled msgids: cache %s", cache.name)
        return pd.read_parquet(cache)
    files = sorted(STOCKTWITS_NYU_SYMBOL_SENTIMENTS.glob("*.csv"))
    parts, t0 = [], time.time()
    for fi, f in enumerate(files, 1):
        reader = pd.read_csv(f, usecols=["message_id", "created_at", "sentiment", "symbol_list"],
                             dtype="string", chunksize=_CHUNK, on_bad_lines="skip",
                             encoding="utf-8", encoding_errors="replace")
        for chunk in reader:
            hit = chunk["symbol_list"].str.contains(_HAS_RE, na=False)
            if not hit.any():
                continue
            sub = chunk[hit]
            dt = pd.to_datetime(sub["created_at"], errors="coerce")
            sent = pd.to_numeric(sub["sentiment"], errors="coerce")
            keep = dt.notna() & (dt >= _S_DATE) & (dt <= _E_DATE) & sent.isin([-1.0, 1.0])
            if not keep.any():
                continue
            parts.append(pd.DataFrame({"message_id": sub["message_id"][keep],
                                       "sentiment": sent[keep]}))
        if fi % 10 == 0:
            log.info("  labeled %d/%d (%.0fs, %d etiquetados)", fi, len(files),
                     time.time() - t0, sum(len(p) for p in parts))
    res = pd.concat(parts, ignore_index=True).drop_duplicates("message_id")
    res.to_parquet(cache, index=False)
    log.info("labeled msgids: %d etiquetados únicos -> %s", len(res), cache.name)
    return res


def run() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    log = logging.getLogger("phase6_join")

    intr = scan_intraday_msgs(log)
    lab = scan_labeled(log)
    lab_set = set(lab["message_id"].tolist())
    intr["labeled"] = intr["message_id"].isin(lab_set)

    # por ventana de 30 min y ticker
    per_win = intr.groupby(["ticker", "w30"]).agg(
        n_msgs=("message_id", "size"), n_labeled=("labeled", "sum")).reset_index()
    per_win["frac"] = per_win["n_labeled"] / per_win["n_msgs"]

    stats, decision = {}, {}
    for tk in TICKERS:
        s = per_win[per_win["ticker"] == tk]
        med_lab = float(s["n_labeled"].median())
        stats[tk] = {
            "ventanas": int(len(s)),
            "msgs_totales": int(s["n_msgs"].sum()),
            "etiquetados_totales": int(s["n_labeled"].sum()),
            "frac_label_mediana": round(float(s["frac"].median()), 3),
            "frac_label_p10": round(float(s["frac"].quantile(.10)), 3),
            "n_labeled_mediana": round(med_lab, 1),
            "n_labeled_p10": round(float(s["n_labeled"].quantile(.10)), 1),
        }
        decision[tk] = "NET_principal" if med_lab >= 10 else "volumen_principal_NET_secundaria"

    both_ok = all(v == "NET_principal" for v in decision.values())
    out = {"stats": stats, "decision_por_activo": decision,
           "regla": "NET principal si mediana etiquetados/ventana >=10 en ambos",
           "NET_es_principal_global": both_ok}
    per_win.to_parquet(_OUT / "label_join_perwindow.parquet", index=False)
    _write_md(stats, decision, both_ok)
    log.info("Decisión: %s", out)
    return out


def _write_md(stats, decision, both_ok):
    L = ["# Fase 6 — cuantificación del join de labels intradía (30 min)\n",
         "Join `feature_wo_messages` (timestamp) × `symbol_sentiments` (label) por "
         "`message_id`, TSLA/AMD, 2020-08→2022-12, horario de mercado ET.\n",
         "**Regla fijada ANTES de ver el resultado:** mediana de mensajes etiquetados por "
         "ventana >=10 en AMBOS → NET etiquetado es feature principal; <10 en alguno → ese "
         "activo usa volumen+aceleración como principales y NET como secundaria.\n",
         "| activo | ventanas | msgs | etiquetados | frac label (mediana / p10) | "
         "etiquetados/ventana (mediana / p10) | decisión |",
         "|--------|----------|------|-------------|----------------------------|"
         "-------------------------------------|----------|"]
    for tk in TICKERS:
        s = stats[tk]
        L.append(f"| {tk} | {s['ventanas']:,} | {s['msgs_totales']:,} | "
                 f"{s['etiquetados_totales']:,} | {s['frac_label_mediana']} / {s['frac_label_p10']} | "
                 f"**{s['n_labeled_mediana']}** / {s['n_labeled_p10']} | {decision[tk]} |")
    L += ["", f"**Resultado de la regla:** el NET etiquetado "
          f"{'ES la feature principal en ambos activos' if both_ok else 'NO es principal en algún activo'} "
          "(ver columna decisión). Esta elección queda fijada para el pre-registro de H6.", ""]
    (_OUT / "label_join.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    import json
    print(json.dumps(run(), indent=2, ensure_ascii=False))
