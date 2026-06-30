"""
FASE 7 — FinBERT estratégico sobre messages/ del NYU → dataset_v2.

messages/ son ~51 GB / cientos de millones de mensajes: procesar todo con FinBERT
en CPU es inviable. Estrategia:

  7.1 SELECCIÓN: stream de feature_wo_messages/ (28.7 GB), filtrar a mensajes
      SIN label nativo (sentiment NULL), con ticker en S&P 500 y fecha 2010-2024,
      explotar por ticker, y muestrear como máximo CAP por (ticker, día).
      Dado el volumen (~430M mensajes sin label), se usa CAP=15 desde el inicio
      (regla 7.2 para >15M esperados). Se reporta el conteo real.
  7.1b JOIN TEXTO: stream de messages/ uniendo por message_id para traer el texto.
  7.3 FINBERT: ProsusAI/finbert, max_length=64, batch 32, torch threads al máximo.
      Checkpoint por bloque de 10k (escribe scores parciales y marca hecho).
      Límite duro de 70 h de reloj desde el primer arranque: para y guarda.
  7.4 CONSOLIDA: master/finbert_scores.parquet.
  7.5 dataset_v2 = dataset_v1 + features FinBERT diarias por ticker.

Reanudable: salta archivos/bloques ya hechos (checkpoints). Atómico (temp+rename).
"""

import ast
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import EXTERNAL_DATA_ROOT
from src.data.longrun.common import (
    MASTER_DIR,
    STAGING_DIR,
    Checkpoint,
    normalize_ticker,
    safe_to_parquet,
    setup_logger,
    sp500_set,
)

JOB = "phase7_finbert"
_FEAT_DIR = EXTERNAL_DATA_ROOT / "stocktwits_nyu" / "feature_wo_messages"
_MSG_DIR = EXTERNAL_DATA_ROOT / "stocktwits_nyu" / "messages"
_SEL_DIR = STAGING_DIR / "phase7_selection"
_TEXT_DIR = STAGING_DIR / "phase7_text"
_SCORES_DIR = STAGING_DIR / "phase7_scores"
_SCORES_OUT = MASTER_DIR / "finbert_scores.parquet"
_DATASET_V2 = MASTER_DIR / "dataset_v2.parquet"
_DONE = MASTER_DIR.parent / "logs" / "phase7.done"

_DATE_MIN, _DATE_MAX = pd.Timestamp("2010-01-01"), pd.Timestamp("2024-12-31")
_CAP_PER_DAY = 15
_MAX_LENGTH = 64
_BATCH = 32
_BLOCK = 10_000
_TIME_LIMIT_S = 70 * 3600
_MSG_CHUNK = 500_000


# ---------------------------------------------------------------------------
# 7.1 Selección
# ---------------------------------------------------------------------------

def _parse_symbols(cell: object) -> list[str]:
    if not isinstance(cell, str) or not cell.strip() or cell.strip() == "[]":
        return []
    try:
        v = ast.literal_eval(cell)
    except (ValueError, SyntaxError, TypeError, MemoryError):
        return []
    return [normalize_ticker(x) for x in v] if isinstance(v, (list, tuple)) else []


def _iter_csv(path: Path, usecols, dtype):
    """Itera un CSV en chunks tolerando líneas malas. Motor C (rápido); ante
    ParserError de buffer (archivos NYU corruptos) reintenta con motor python."""
    base = dict(usecols=usecols, dtype=dtype, chunksize=_MSG_CHUNK,
                encoding="utf-8", encoding_errors="replace", on_bad_lines="skip")
    try:
        for ch in pd.read_csv(path, **base):
            yield ch
    except Exception:  # noqa: BLE001 — fallback motor python (más tolerante)
        for ch in pd.read_csv(path, engine="python", **base):
            yield ch


def _rebuild_counter(log) -> dict:
    """Reconstruye el contador (ticker,fecha)->n desde la selección ya escrita."""
    parts = sorted(_SEL_DIR.glob("*.parquet"))
    if not parts:
        return {}
    log.info("Reconstruyendo contador de muestreo desde %d archivos de selección ...", len(parts))
    cnt: dict = {}
    for p in parts:
        d = pd.read_parquet(p, columns=["ticker", "date"])
        for (t, dd), n in d.groupby(["ticker", "date"]).size().items():
            cnt[(t, str(dd))] = cnt.get((t, str(dd)), 0) + int(n)
    return cnt


def select_messages(cap: int, sp500: set[str], ckpt: Checkpoint, log, smoke_limit=None) -> int:
    _SEL_DIR.mkdir(parents=True, exist_ok=True)
    counter = _rebuild_counter(log)
    files = sorted(_FEAT_DIR.glob("*.csv"))
    if smoke_limit:
        files = files[:2]
    total_sel = sum(v for v in counter.values())

    for f in files:
        if ckpt.is_done(f"sel:{f.name}"):
            continue
        kept_rows: list = []
        stop = False
        try:
            for chunk in _iter_csv(
                f, ["message_id", "created_at", "sentiment", "symbol_list"], "string",
            ):
                # Leer como string y coercionar (robusto ante valores no numéricos)
                sent = pd.to_numeric(chunk["sentiment"], errors="coerce")
                chunk["message_id"] = pd.to_numeric(chunk["message_id"], errors="coerce")
                chunk = chunk[sent.isna()]                          # SIN label nativo
                chunk = chunk.dropna(subset=["message_id"])
                d = pd.to_datetime(chunk["created_at"], errors="coerce", utc=True).dt.tz_localize(None)
                chunk = chunk[(d >= _DATE_MIN) & (d <= _DATE_MAX)]
                if chunk.empty:
                    continue
                chunk = chunk.assign(date=d[chunk.index].dt.date)
                chunk["symbols"] = chunk["symbol_list"].map(_parse_symbols)
                chunk = chunk[chunk["symbols"].map(len) > 0].explode("symbols")
                chunk = chunk[chunk["symbols"].isin(sp500)]
                for mid, tk, dt in zip(chunk["message_id"], chunk["symbols"], chunk["date"]):
                    key = (tk, str(dt))
                    if counter.get(key, 0) < cap:
                        counter[key] = counter.get(key, 0) + 1
                        kept_rows.append((int(mid), tk, dt))
                        total_sel += 1
                        if smoke_limit and total_sel >= smoke_limit:
                            stop = True
                            break
                if stop:
                    break
        except Exception as exc:  # noqa: BLE001 — archivo corrupto: salvar parcial y seguir
            log.error("  selección parcial %s: %s", f.name, exc)
        # Salvar lo conseguido y marcar hecho (no reprocesar un archivo problemático)
        if kept_rows:
            out = pd.DataFrame(kept_rows, columns=["message_id", "ticker", "date"])
            safe_to_parquet(out, _SEL_DIR / f"{f.stem}.parquet")
        ckpt.mark(f"sel:{f.name}")
        log.info("  selección %s: total acumulado %d", f.name, total_sel)
        if stop:
            break
    log.info("7.1 selección: %d mensajes seleccionados (cap=%d/día).", total_sel, cap)
    return total_sel


# ---------------------------------------------------------------------------
# 7.1b Join de texto
# ---------------------------------------------------------------------------

def join_text(ckpt: Checkpoint, log, smoke_limit=None) -> int:
    _TEXT_DIR.mkdir(parents=True, exist_ok=True)
    sel_parts = sorted(_SEL_DIR.glob("*.parquet"))
    if not sel_parts:
        log.warning("7.1b: sin selección.")
        return 0
    sel = pd.concat([pd.read_parquet(p) for p in sel_parts], ignore_index=True)
    sel = sel.drop_duplicates(subset=["message_id"]).set_index("message_id")
    log.info("7.1b: %d message_id seleccionados; uniendo texto de messages/.", len(sel))

    msg_files = sorted(_MSG_DIR.glob("*.csv"))
    if smoke_limit:               # smoke: pocos archivos (solapan por rango de id)
        msg_files = msg_files[:6]
    total = 0
    for f in msg_files:
        if ckpt.is_done(f"text:{f.name}"):
            continue
        matched = []
        try:
            for chunk in _iter_csv(f, ["message_id", "message_body"], "string"):
                chunk["mid"] = pd.to_numeric(chunk["message_id"], errors="coerce").astype("Int64")
                chunk = chunk.dropna(subset=["mid"])
                j = chunk.join(sel, on="mid", how="inner")
                if not j.empty:
                    matched.append(j[["mid", "ticker", "date", "message_body"]].rename(
                        columns={"mid": "message_id", "message_body": "text"}))
        except Exception as exc:  # noqa: BLE001 — corrupto: salvar parcial y seguir
            log.error("  texto parcial %s: %s", f.name, exc)
        if matched:
            out = pd.concat(matched, ignore_index=True).drop_duplicates(subset=["message_id"])
            safe_to_parquet(out, _TEXT_DIR / f"{f.stem}.parquet")
            total += len(out)
        ckpt.mark(f"text:{f.name}")
        if smoke_limit and total >= smoke_limit:
            log.info("  texto (smoke): %d acumulados, suficiente.", total)
            break
        if total and total % 50000 < 5000:
            log.info("  texto %s: %d acumulados", f.name, total)
    log.info("7.1b: %d mensajes con texto.", total)
    return total


# ---------------------------------------------------------------------------
# 7.3 FinBERT
# ---------------------------------------------------------------------------

def _load_finbert(log):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    torch.set_num_threads(os.cpu_count() or 4)
    log.info("Cargando FinBERT (torch threads=%d, max_length=%d).", torch.get_num_threads(), _MAX_LENGTH)
    tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    model.eval()
    return tok, model


def _score_block(texts: list[str], tok, model):
    import torch
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=_MAX_LENGTH)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    # ProsusAI/finbert: 0=positive, 1=negative, 2=neutral
    return probs


def run_finbert(ckpt: Checkpoint, log, smoke=False) -> int:
    _SCORES_DIR.mkdir(parents=True, exist_ok=True)
    text_files = sorted(_TEXT_DIR.glob("*.parquet"))
    if not text_files:
        log.warning("7.3: sin texto que puntuar.")
        return 0

    start_ts = float(ckpt.meta.get("finbert_start_ts", 0) or 0)
    if start_ts == 0:
        start_ts = time.time()
        ckpt.set_meta(finbert_start_ts=start_ts)

    tok, model = _load_finbert(log)
    total = 0
    for tf in text_files:
        df = pd.read_parquet(tf)
        n_blocks = (len(df) + _BLOCK - 1) // _BLOCK
        for bi in range(n_blocks):
            key = f"score:{tf.stem}:{bi}"
            if ckpt.is_done(key):
                continue
            if not smoke and (time.time() - start_ts) > _TIME_LIMIT_S:
                log.warning("Límite de 70 h alcanzado. Parando y guardando lo procesado.")
                _consolidate_scores(log)
                return total
            block = df.iloc[bi * _BLOCK:(bi + 1) * _BLOCK]
            texts = block["text"].astype(str).fillna("").tolist()
            try:
                probs = _score_block(texts, tok, model)
            except Exception as exc:  # noqa: BLE001 — bloque problemático: marcar y seguir
                log.error("  bloque %s falló: %s", key, exc)
                ckpt.mark(key)
                continue
            res = block[["message_id", "ticker", "date"]].reset_index(drop=True).copy()
            res["prob_positive"] = probs[:, 0]
            res["prob_negative"] = probs[:, 1]
            res["prob_neutral"] = probs[:, 2]
            res["net_sentiment"] = probs[:, 0] - probs[:, 1]
            safe_to_parquet(res, _SCORES_DIR / f"{tf.stem}_{bi:04d}.parquet")
            ckpt.mark(key)
            total += len(res)
            if total % 100_000 < _BLOCK:
                log.info("  FinBERT: %d mensajes puntuados (%.1f h transcurridas).",
                         total, (time.time() - start_ts) / 3600)
            if smoke and total >= 10_000:
                _consolidate_scores(log)
                return total
    _consolidate_scores(log)
    return total


def _consolidate_scores(log) -> None:
    parts = sorted(_SCORES_DIR.glob("*.parquet"))
    if not parts:
        return
    frames = []
    for p in parts:
        try:
            frames.append(pd.read_parquet(p))
        except Exception:  # noqa: BLE001
            pass
    if frames:
        big = pd.concat(frames, ignore_index=True)
        safe_to_parquet(big, _SCORES_OUT)
        log.info("7.4 finbert_scores: %s | %d filas | %d tickers.",
                 _SCORES_OUT, len(big), big["ticker"].nunique())


# ---------------------------------------------------------------------------
# 7.5 dataset_v2
# ---------------------------------------------------------------------------

def build_dataset_v2(log) -> dict:
    v1 = MASTER_DIR / "dataset_v1.parquet"
    if not v1.exists() or not _SCORES_OUT.exists():
        log.warning("7.5: falta dataset_v1 o finbert_scores; no se construye v2.")
        return {}
    scores = pd.read_parquet(_SCORES_OUT)
    # Dedup defensivo: el fallback de lectura C→python puede duplicar message_id
    scores = scores.drop_duplicates(subset=["message_id"])
    scores["date"] = pd.to_datetime(scores["date"]).dt.date
    agg = scores.groupby(["ticker", "date"]).agg(
        finbert_net_sentiment_mean=("net_sentiment", "mean"),
        finbert_prob_neg_mean=("prob_negative", "mean"),
        finbert_msg_count=("net_sentiment", "count"),
    ).reset_index()

    df = pd.read_parquet(v1)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    merged = df.merge(agg, on=["ticker", "date"], how="outer")
    for c in ["finbert_msg_count"]:
        merged[c] = merged[c].fillna(0)
    # Recalcular target solo donde hay excess (igual que v1); filas nuevas sin v1
    # quedan con target NaN → se descartan para entrenamiento.
    merged = merged.dropna(subset=["return_excess_over_spy_5d_forward"])
    merged["target"] = (merged["return_excess_over_spy_5d_forward"] > 0).astype(int)
    safe_to_parquet(merged, _DATASET_V2)

    nonnull = merged["finbert_net_sentiment_mean"].notna().mean()
    log.info("7.5 dataset_v2: %s | %d filas | %% con FinBERT: %.1f%%",
             _DATASET_V2, len(merged), nonnull * 100)
    return {"rows": len(merged), "finbert_coverage_pct": round(nonnull * 100, 1),
            "output": str(_DATASET_V2)}


# ---------------------------------------------------------------------------
# Orquestación de la fase
# ---------------------------------------------------------------------------

def run(smoke: bool = False, resume: bool = True, **kw) -> dict:
    log = setup_logger(JOB)
    if _DONE.exists() and not smoke:
        log.info("Fase 7 ya marcada DONE. Saltando.")
        return {"phase": JOB, "status": "already_done"}
    ckpt = Checkpoint(JOB if not smoke else JOB + "_smoke")
    sp500 = sp500_set()
    smoke_limit = 10_000 if smoke else None

    n_sel = select_messages(_CAP_PER_DAY, sp500, ckpt, log, smoke_limit)
    n_txt = join_text(ckpt, log, smoke_limit)
    n_scored = run_finbert(ckpt, log, smoke)
    v2 = build_dataset_v2(log) if not smoke else {}

    # cobertura por (ticker, año)
    cov = {}
    if _SCORES_OUT.exists():
        s = pd.read_parquet(_SCORES_OUT, columns=["ticker", "date"])
        s["year"] = pd.to_datetime(s["date"]).dt.year
        cov = {int(k): int(v) for k, v in s.groupby("year").size().items()}

    summary = {"phase": JOB, "selected": n_sel, "with_text": n_txt, "scored": n_scored,
               "coverage_by_year": cov, "dataset_v2": v2}
    if not smoke and n_scored > 0:
        _DONE.write_text(pd.Timestamp.now().isoformat(), encoding="utf-8")
    log.info("Fase 7 fin: %s", summary)
    return summary


if __name__ == "__main__":
    run(smoke="--smoke" in sys.argv)
