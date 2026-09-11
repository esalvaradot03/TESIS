"""
Runner unificado: BAJA datos de Drive → CORRE un comando → SUBE resultados a Drive.

Drive no es un disco montable: no se puede ejecutar código "sobre Drive"
directamente. Este script implementa el ciclo real en UN solo comando, de modo
que da igual correrlo en VS Code (local) o en Railway:

    1. drive_sync   → baja las secciones que necesites (processed, models, ...)
    2. <tu comando> → corre tu pipeline/script sobre los datos ya locales
    3. drive_upload → sube de vuelta las secciones que hayas generado

Ejemplos (desde la raíz del repo, con el venv activo):

    # Entrenar el modelo: baja features, entrena, sube el modelo resultante
    python -m src.data.run_on_drive \
        --pull processed --run "python -m src.trading.xgboost_model" --push models

    # Recalcular features: baja lo necesario, corre feature_engine, sube processed
    python -m src.data.run_on_drive \
        --pull processed models --run "python -m src.trading.feature_engine" --push processed

    # Backtest: baja todo, corre el backtester, sube los resultados
    python -m src.data.run_on_drive \
        --pull processed models --run "python -m src.evaluation.backtester" --push processed

    # Solo bajar (sin correr ni subir)
    python -m src.data.run_on_drive --pull processed models

    # Correr algo sin tocar Drive (útil para probar)
    python -m src.data.run_on_drive --run "python -m src.trading.xgboost_model"

Flags:
    --pull SECCIONES   secciones a bajar antes de correr (raw processed models external)
    --run "COMANDO"    comando shell a ejecutar (entre comillas)
    --push SECCIONES   secciones a subir después de correr
    --dry-run          muestra el plan sin bajar/correr/subir
    --skip-pull-errors continúa aunque falle una descarga

Requisitos de autenticación (una sola vez):
    credentials.json + token.json en la raíz (ver docs/GUIA_GOOGLE_DRIVE.md).
    El token debe tener scope de escritura si vas a usar --push.

Variables de entorno (compartidas con drive_sync/drive_upload):
    DRIVE_FOLDER_NAME=TESIS
    DRIVE_EXTERNAL_SUBDIR=trading-data
    TESIS_DATA_ROOT=...   destino de la sección 'external'
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time

logger = logging.getLogger("run_on_drive")

VALID_SECTIONS = ["raw", "processed", "models", "external"]


def _run_sync(sections: list[str], dry_run: bool, skip_errors: bool) -> None:
    """Baja las secciones indicadas usando drive_sync."""
    for sec in sections:
        cmd = [sys.executable, "-m", "src.data.drive_sync", "--only", sec, "--yes"]
        if dry_run:
            cmd.append("--dry-run")
        logger.info("⬇  PULL '%s' desde Drive ...", sec)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            msg = f"drive_sync --only {sec} devolvió código {rc}"
            if dry_run or skip_errors:
                # En dry-run no importan los fallos (no hay auth real); solo es un plan.
                logger.warning("%s (continuo).", msg)
            else:
                logger.error("%s. Abortando (usá --skip-pull-errors para continuar).", msg)
                sys.exit(rc)


def _run_upload(sections: list[str], dry_run: bool) -> int:
    """Sube las secciones indicadas usando drive_upload. Devuelve el peor rc."""
    worst = 0
    for sec in sections:
        cmd = [sys.executable, "-m", "src.data.drive_upload", "--only", sec, "--yes"]
        if dry_run:
            cmd.append("--dry-run")
        logger.info("⬆  PUSH '%s' a Drive ...", sec)
        rc = subprocess.run(cmd).returncode
        if rc != 0:
            logger.error("drive_upload --only %s devolvió código %d.", sec, rc)
            worst = worst or rc
    return worst


def _run_command(command: str, dry_run: bool) -> int:
    """Ejecuta el comando de usuario. Devuelve su returncode."""
    if dry_run:
        logger.info("[dry-run] CORRERÍA: %s", command)
        return 0
    logger.info("▶  RUN: %s", command)
    start = time.time()
    rc = subprocess.run(command, shell=True).returncode
    elapsed = time.time() - start
    if rc == 0:
        logger.info("✓  Comando terminó en %.1fs.", elapsed)
    else:
        logger.error("✗  El comando falló (código %d) tras %.1fs.", rc, elapsed)
    return rc


def _validate(sections: list[str], flag: str) -> None:
    bad = [s for s in sections if s not in VALID_SECTIONS]
    if bad:
        logger.error("Secciones inválidas en %s: %s. Válidas: %s",
                     flag, ", ".join(bad), ", ".join(VALID_SECTIONS))
        sys.exit(2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Baja de Drive → corre un comando → sube resultados a Drive.",
    )
    parser.add_argument("--pull", nargs="*", default=[], metavar="SECCION",
                        help=f"Secciones a bajar antes de correr ({'/'.join(VALID_SECTIONS)}).")
    parser.add_argument("--run", default=None, metavar="COMANDO",
                        help="Comando shell a ejecutar (entre comillas).")
    parser.add_argument("--push", nargs="*", default=[], metavar="SECCION",
                        help=f"Secciones a subir después de correr ({'/'.join(VALID_SECTIONS)}).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Muestra el plan sin bajar/correr/subir.")
    parser.add_argument("--skip-pull-errors", action="store_true",
                        help="Continúa aunque falle una descarga.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    _validate(args.pull, "--pull")
    _validate(args.push, "--push")

    if not args.pull and not args.run and not args.push:
        parser.error("Nada que hacer: especificá al menos --pull, --run o --push.")

    # Plan
    print("\n═══ PLAN run_on_drive ═══")
    print(f"  1. PULL : {args.pull or '(nada)'}")
    print(f"  2. RUN  : {args.run or '(nada)'}")
    print(f"  3. PUSH : {args.push or '(nada)'}")
    if args.dry_run:
        print("  [dry-run] No se ejecutará nada.\n")
    print()

    # 1. PULL
    if args.pull:
        _run_sync(args.pull, args.dry_run, args.skip_pull_errors)

    # 2. RUN
    if args.run:
        rc = _run_command(args.run, args.dry_run)
        if rc != 0:
            logger.error("El comando falló; NO se subirá nada a Drive.")
            return rc

    # 3. PUSH
    if args.push:
        rc = _run_upload(args.push, args.dry_run)
        if rc != 0:
            return rc

    logger.info("✓ run_on_drive completado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
