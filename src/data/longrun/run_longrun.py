"""
ORQUESTADOR del pipeline de larga duración.

Flujo:
  - Jobs 1, 2, 3, 4 EN PARALELO (ProcessPoolExecutor, 4 workers).
  - Cuando termina el 4, corre el Job 5 (indicadores, depende de precios).
  - Cuando terminan 1, 2, 3, 5, corre el Job 6 (dataset maestro) con lo disponible.
  - Aislamiento de fallos: si un job crashea, los demás siguen; el Job 6 reporta
    qué fuente faltó.
  - Log maestro: D:\\trading-data\\logs\\orchestrator.log
  - Al final genera D:\\trading-data\\logs\\REPORT.md

Uso:
    python -m src.data.longrun.run_longrun            # corrida real
    python -m src.data.longrun.run_longrun --smoke    # smoke test (subconjuntos)
"""

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from src.data.longrun.common import LOGS_DIR, MASTER_DIR, ensure_dirs, setup_logger

_LOCK = LOGS_DIR / "orchestrator.lock"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError as exc:
        import errno
        # ESRCH = no existe; EPERM = existe pero sin permiso (vivo)
        return getattr(exc, "errno", None) == errno.EPERM
    return True


def _acquire_lock(log) -> bool:
    """Lock por PID. Si hay un lock vivo, NO arranca (evita doble corrida). Roba locks stale."""
    if _LOCK.exists():
        try:
            data = json.loads(_LOCK.read_text(encoding="utf-8"))
            other = int(data.get("pid", -1))
            if other != os.getpid() and _pid_alive(other):
                log.warning("Otra instancia (PID %d) ya está corriendo. Saliendo.", other)
                return False
            log.info("Lock stale (PID %d muerto). Se reclama.", other)
        except Exception:  # noqa: BLE001
            log.info("Lock ilegible; se reclama.")
    _LOCK.write_text(json.dumps({"pid": os.getpid(), "ts": datetime.now().isoformat()}),
                     encoding="utf-8")
    return True


def _release_lock() -> None:
    try:
        if _LOCK.exists():
            _LOCK.unlink()
    except OSError:
        pass

# job_name -> módulo
MODULES = {
    "job1": "process_stocktwits_nyu",
    "job2": "process_wsb",
    "job3": "process_fnspid",
    "job4": "download_prices_yahoo",
    "job5": "compute_indicators_long",
    "job6": "build_master_dataset",
}


def _dispatch(args: tuple) -> dict:
    """Ejecuta un job en un proceso aislado. Nunca propaga excepción."""
    import importlib
    job_name, smoke = args
    mod = importlib.import_module(f"src.data.longrun.{MODULES[job_name]}")
    t0 = time.time()
    try:
        res = mod.run(smoke=smoke)
        status = "success"
    except Exception as exc:  # noqa: BLE001 — captura total: el job no debe tumbar el pool
        res = {"error": repr(exc)}
        status = "fail"
    return {"job": job_name, "status": status,
            "duration_s": round(time.time() - t0, 1), "result": res}


def _human(n: float) -> str:
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def _path_size(p: Path) -> int:
    if not p.exists():
        return 0
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def generate_report(results: list[dict], total_dur: float, log) -> Path:
    lines = [
        f"# REPORT — pipeline dataset maestro ({datetime.now():%Y-%m-%d %H:%M})",
        "",
        f"Duración total: **{total_dur/60:.1f} min**.",
        "",
        "## Estado de cada job",
        "| job | estado | duración | salida |",
        "|---|---|---|---|",
    ]
    outputs = {
        "job1": MASTER_DIR / "stocktwits_sentiment_daily.parquet",
        "job2": MASTER_DIR / "wsb_mentions_daily.parquet",
        "job3": MASTER_DIR / "fnspid_news_daily.parquet",
        "job4": MASTER_DIR / "prices_2008_2024.parquet",
        "job5": MASTER_DIR / "indicators_daily.parquet",
        "job6": MASTER_DIR / "dataset_v1.parquet",
    }
    by_job = {r["job"]: r for r in results}
    for j in ["job1", "job2", "job3", "job4", "job5", "job6"]:
        r = by_job.get(j, {"status": "no ejecutado", "duration_s": 0})
        sz = _human(_path_size(outputs[j]))
        emoji = "✅" if r["status"] == "success" else "❌"
        lines.append(f"| {j} | {emoji} {r['status']} | {r['duration_s']:.0f}s | "
                     f"{outputs[j].name} ({sz}) |")

    # Estadísticas del dataset_v1 (resultado del job6)
    j6 = by_job.get("job6", {}).get("result", {})
    lines += ["", "## dataset_v1", ""]
    if j6 and "rows" in j6:
        lines += [
            f"- Filas: **{j6['rows']:,}** | Tickers: {j6.get('tickers')} | "
            f"Rango: {j6.get('date_min')} → {j6.get('date_max')}",
            f"- Balance target (sube): {j6.get('target_pos_rate')}",
            f"- Fuentes faltantes: {j6.get('missing_sources') or 'ninguna'}",
            "",
            "Cobertura por año:",
            "",
            "| año | filas |", "|---|---|",
        ]
        for y, n in sorted((j6.get("coverage_by_year") or {}).items()):
            lines.append(f"| {y} | {n:,} |")
        lines += ["", "NaN ratio por columna (>0):", ""]
        for c, v in (j6.get("nan_ratio_by_col") or {}).items():
            lines.append(f"- {c}: {v:.2%}")
    else:
        lines.append("_dataset_v1 no se generó (ver estado del job6)._")

    # Errores agrupados
    lines += ["", "## Errores / warnings por job", ""]
    for r in results:
        if r["status"] != "success":
            lines.append(f"- **{r['job']}**: {r['result'].get('error', 'desconocido')}")
        else:
            res = r["result"]
            errs = {k: v for k, v in res.items() if "err" in k and v}
            if errs:
                lines.append(f"- {r['job']}: {errs}")
    lines.append("")

    out = LOGS_DIR / "REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info("REPORT.md escrito en %s", out)
    return out


_DONE_MARKER = LOGS_DIR / "ORCHESTRATOR_DONE"


def main(smoke: bool = False, force: bool = False) -> None:
    ensure_dirs()
    log = setup_logger("orchestrator")
    # Si ya completó una corrida real, no re-ejecutar (la tarea programada hace no-op).
    if _DONE_MARKER.exists() and not smoke and not force:
        log.info("Marcador DONE presente (%s). Pipeline ya completado; nada que hacer. "
                 "Borralo o usá --force para re-correr.", _DONE_MARKER)
        return
    if not _acquire_lock(log):
        return
    try:
        _run_all(smoke, log)
        if not smoke:
            _DONE_MARKER.write_text(datetime.now().isoformat(), encoding="utf-8")
            log.info("Marcador DONE escrito en %s", _DONE_MARKER)
    finally:
        _release_lock()


def _run_all(smoke: bool, log) -> None:
    t_start = time.time()
    log.info("═══ ORQUESTADOR inicio (smoke=%s, PID=%d) ═══", smoke, os.getpid())
    results: list[dict] = []

    # --- Fase 1: Jobs 1-4 en paralelo ---
    log.info("Fase 1: jobs 1,2,3,4 en paralelo (4 workers).")
    with ProcessPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_dispatch, (j, smoke)): j for j in ["job1", "job2", "job3", "job4"]}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            log.info("  [%s] %s en %.0fs → %s", r["job"], r["status"],
                     r["duration_s"], r["result"])

    # --- Fase 2: Job 5 (depende del 4) ---
    log.info("Fase 2: job5 (indicadores).")
    r5 = _dispatch(("job5", smoke))
    results.append(r5)
    log.info("  [job5] %s en %.0fs → %s", r5["status"], r5["duration_s"], r5["result"])

    # --- Fase 3: Job 6 (depende de 1,2,3,5) ---
    log.info("Fase 3: job6 (dataset maestro).")
    r6 = _dispatch(("job6", smoke))
    results.append(r6)
    log.info("  [job6] %s en %.0fs → %s", r6["status"], r6["duration_s"], r6["result"])

    total = time.time() - t_start
    generate_report(results, total, log)
    log.info("═══ ORQUESTADOR fin en %.1f min ═══", total / 60)


if __name__ == "__main__":
    main(smoke="--smoke" in sys.argv, force="--force" in sys.argv)
