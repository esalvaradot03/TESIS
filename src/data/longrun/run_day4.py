"""
ORQUESTADOR de las Fases 7, 8, 9 (corrida de 4 días).

Orden (decisión de robustez): 8 (diagnóstico) → 9 sobre dataset_v1 (M1-M6) →
7 (FinBERT, ~70h) → 9 sobre dataset_v2 (M7-M8 si Fase 7 produjo v2).
Se front-loadean las fases rápidas y robustas para que los entregables clave
existan aunque la máquina muera durante la FinBERT.

- Marcador DONE por fase en logs/. Al re-correr (--resume), salta las hechas.
- Aislamiento de fallos: si una fase falla, las demás siguen.
- Lock por PID (no dos instancias).
- REPORT_DAY4.md al final.
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from src.data.longrun.common import LOGS_DIR, MASTER_DIR, ensure_dirs, setup_logger

_LOCK = LOGS_DIR / "day4.lock"
_MARK = {  # marcadores de orquestación
    "phase8": LOGS_DIR / "od_phase8.done",
    "phase9v1": LOGS_DIR / "od_phase9v1.done",
    "phase7": LOGS_DIR / "od_phase7.done",
    "phase9v2": LOGS_DIR / "od_phase9v2.done",
}


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError as exc:
        import errno
        return getattr(exc, "errno", None) == errno.EPERM
    return True


def _acquire_lock(log) -> bool:
    if _LOCK.exists():
        try:
            other = int(json.loads(_LOCK.read_text())["pid"])
            if other != os.getpid() and _pid_alive(other):
                log.warning("Otra instancia day4 (PID %d) corriendo. Saliendo.", other)
                return False
        except Exception:  # noqa: BLE001
            pass
    _LOCK.write_text(json.dumps({"pid": os.getpid(), "ts": datetime.now().isoformat()}))
    return True


def _run_phase(name: str, fn, smoke: bool, log, results: dict) -> None:
    if _MARK[name].exists():
        log.info("Fase %s ya hecha (marcador). Saltando.", name)
        results[name] = {"status": "skipped (done)"}
        return
    log.info("─── Iniciando %s ───", name)
    t0 = time.time()
    try:
        res = fn(smoke=smoke)
        if not smoke:   # no marcar en smoke (no debe bloquear el run real)
            _MARK[name].write_text(datetime.now().isoformat(), encoding="utf-8")
        results[name] = {"status": "DONE", "duration_s": round(time.time() - t0, 1), "result": res}
        log.info("✓ %s DONE en %.0fs", name, time.time() - t0)
    except Exception as exc:  # noqa: BLE001 — una fase no tumba las demás
        results[name] = {"status": "FAILED", "duration_s": round(time.time() - t0, 1),
                         "error": repr(exc)}
        log.error("✗ %s FALLÓ: %s", name, exc)


def _human(n: float) -> str:
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def _write_report(results: dict, log) -> None:
    outputs = {
        "dataset_v1.parquet": MASTER_DIR / "dataset_v1.parquet",
        "dataset_v2.parquet": MASTER_DIR / "dataset_v2.parquet",
        "finbert_scores.parquet": MASTER_DIR / "finbert_scores.parquet",
    }
    docs = ["diagnostico_dataset_v1", "outliers_dataset_v1", "modelos_dataset_v1", "modelos_dataset_v2"]
    L = [f"# REPORT_DAY4 ({datetime.now():%Y-%m-%d %H:%M})\n",
         "## Estado de cada fase", "", "| fase | estado | duración |", "|---|---|---|"]
    for ph in ["phase8", "phase9v1", "phase7", "phase9v2"]:
        r = results.get(ph, {"status": "no ejecutada"})
        L.append(f"| {ph} | {r['status']} | {r.get('duration_s', 0):.0f}s |")
    L += ["", "## Salidas", "", "| archivo | tamaño |", "|---|---|"]
    for name, p in outputs.items():
        L.append(f"| {name} | {_human(p.stat().st_size) if p.exists() else 'no existe'} |")

    # Resumen de modelos (señal real?)
    L += ["", "## Resumen ejecutivo de modelos", ""]
    for ph in ["phase9v1", "phase9v2"]:
        res = results.get(ph, {}).get("result", {})
        for label, info in (res.items() if isinstance(res, dict) else []):
            if isinstance(info, dict) and "best" in info:
                L.append(f"- **{label}**: mejor={info['best']} | "
                         f"¿señal real en algún modelo? **{'SÍ' if info.get('any_signal') else 'NO'}** | "
                         f"AUCs: {info.get('models')}")
    p7 = results.get("phase7", {}).get("result", {})
    if isinstance(p7, dict):
        L.append(f"- **Fase 7 FinBERT**: seleccionados={p7.get('selected')}, "
                 f"puntuados={p7.get('scored')}, dataset_v2={p7.get('dataset_v2')}")

    L += ["", "## Archivos producidos", "",
          "- D:\\trading-data\\master\\: dataset_v1, dataset_v2, finbert_scores, *_daily.parquet",
          "- docs/: " + ", ".join(f"{d}.md" for d in docs),
          "- D:\\trading-data\\logs\\: *.log, REPORT_DAY4.md, marcadores od_*.done",
          "", "## Para revisar al regresar", ""]
    for ph, r in results.items():
        if r.get("status") == "FAILED":
            L.append(f"- ⚠ {ph} FALLÓ: {r.get('error')}")
    if results.get("phase7", {}).get("status") not in ("DONE", "skipped (done)"):
        L.append("- ⚠ Fase 7 (FinBERT) no completó del todo: revisar finbert_scores parcial "
                 "y cuántos mensajes quedaron sin procesar (límite 70h o crash).")
    L.append("")
    (LOGS_DIR / "REPORT_DAY4.md").write_text("\n".join(L), encoding="utf-8")
    log.info("REPORT_DAY4.md escrito.")


def main(smoke: bool = False, resume: bool = False) -> None:
    ensure_dirs()
    log = setup_logger("orchestrator_day4")
    if not _acquire_lock(log):
        return
    try:
        from src.data.longrun import phase7_finbert, phase8_diagnostic, phase9_models
        results: dict = {}
        log.info("═══ DAY4 inicio (smoke=%s, PID=%d) ═══", smoke, os.getpid())

        _run_phase("phase8", phase8_diagnostic.run, smoke, log, results)
        _run_phase("phase9v1", phase9_models.run, smoke, log, results)
        _run_phase("phase7", phase7_finbert.run, smoke, log, results)
        # Fase 9 v2 solo si Fase 7 produjo dataset_v2
        if (MASTER_DIR / "dataset_v2.parquet").exists():
            _run_phase("phase9v2", phase9_models.run, smoke, log, results)
        else:
            log.info("dataset_v2 no existe; se omite phase9v2.")
            results["phase9v2"] = {"status": "skipped (sin dataset_v2)"}

        _write_report(results, log)
        log.info("═══ DAY4 fin ═══")
    finally:
        try:
            _LOCK.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv, resume="--resume" in sys.argv)
