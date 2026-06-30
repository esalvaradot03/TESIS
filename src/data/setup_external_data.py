"""
Setup de datasets externos de la tesis en un disco aparte (D:\\trading-data\\).

NADA de estos datos vive dentro del repo. El script:
  - Crea la estructura de carpetas en D:\\trading-data\\.
  - Reporta espacio libre y el tamaño de cada carpeta del bucket S3 a descargar.
  - Pide confirmación interactiva (y/n) antes de descargar (salvo --yes).
  - Instala AWS CLI v2 si no está (Windows MSI, silencioso).
  - Descarga el bucket público s3://stocktwits-nyu (--no-sign-request) con
    `aws s3 sync` (reanudable e idempotente).
  - Descarga dos datasets de Kaggle vía kagglehub y los COPIA al destino
    (dejando el caché intacto).
  - Genera D:\\trading-data\\README.md con el inventario.

FNSPID NO se toca: solo se crea su carpeta vacía; el usuario mueve sus archivos.

Uso:
    python -m src.data.setup_external_data                 # interactivo
    python -m src.data.setup_external_data --dry-run       # solo plan + tamaños
    python -m src.data.setup_external_data --yes           # sin preguntar
    python -m src.data.setup_external_data --only s3        # solo S3
    python -m src.data.setup_external_data --only kaggle    # solo Kaggle
    python -m src.data.setup_external_data --only readme    # regenerar README
"""

import argparse
import csv
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from datetime import date
from pathlib import Path

logger = logging.getLogger("setup_external_data")

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

DATA_ROOT = Path(os.getenv("TESIS_DATA_ROOT", "D:/trading-data"))

# Subcarpetas a crear (vacías si no se descarga nada)
SUBDIRS = [
    Path("stocktwits_nyu") / "symbol_sentiments",
    Path("stocktwits_nyu") / "feature_wo_messages",
    Path("stocktwits_nyu") / "messages",
    Path("wsb_kaggle") / "kevinwang313_wallstreetbets",
    Path("wsb_kaggle") / "unanimad_reddit_rwallstreetbets",
    Path("fnspid"),
]

# Bucket público S3 (no requiere credenciales)
_S3_BASE = "s3://stocktwits-nyu/dataset/v1/data/csv"
S3_FOLDERS = {
    # nombre_local: (uri_origen, destino_relativo)
    "symbol_sentiments": (
        f"{_S3_BASE}/symbol_sentiments/",
        Path("stocktwits_nyu") / "symbol_sentiments",
    ),
    "feature_wo_messages": (
        f"{_S3_BASE}/feature_wo_messages/",
        Path("stocktwits_nyu") / "feature_wo_messages",
    ),
    "messages": (
        f"{_S3_BASE}/messages/",
        Path("stocktwits_nyu") / "messages",
    ),
}

# Datasets de Kaggle (slug: destino_relativo)
KAGGLE_DATASETS = {
    "kevinwang313/wallstreetbets-dataset": Path("wsb_kaggle") / "kevinwang313_wallstreetbets",
    "unanimad/reddit-rwallstreetbets": Path("wsb_kaggle") / "unanimad_reddit_rwallstreetbets",
}

_AWS_MSI_URL = "https://awscli.amazonaws.com/AWSCLIV2.msi"


# ---------------------------------------------------------------------------
# Utilidades de formato / espacio
# ---------------------------------------------------------------------------

def _human(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def free_space(path: Path) -> int:
    """Bytes libres en el disco que contiene `path` (o su raíz si no existe aún)."""
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def dir_stats(path: Path) -> tuple[int, int]:
    """(número de archivos, bytes totales) recursivo. (0,0) si no existe."""
    if not path.exists():
        return 0, 0
    n, size = 0, 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                size += (Path(root) / f).stat().st_size
                n += 1
            except OSError:
                pass
    return n, size


# ---------------------------------------------------------------------------
# a) Estructura de carpetas
# ---------------------------------------------------------------------------

def ensure_structure() -> None:
    """Crea D:\\trading-data\\ y todas las subcarpetas (idempotente)."""
    logger.info("Asegurando estructura en %s ...", DATA_ROOT)
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    for sub in SUBDIRS:
        (DATA_ROOT / sub).mkdir(parents=True, exist_ok=True)
        logger.info("  ✓ %s", DATA_ROOT / sub)


# ---------------------------------------------------------------------------
# c) AWS CLI
# ---------------------------------------------------------------------------

def find_aws() -> list[str] | None:
    """
    Devuelve el prefijo argv para invocar aws, o None.

    Puede ser ['C:/.../aws.exe'] (v2 / PATH) o [python, '.venv/Scripts/aws']
    (awscli v1 instalado por pip, que en Windows no genera aws.exe ejecutable).
    """
    exe = shutil.which("aws")
    if exe:
        return [exe]
    for c in [
        Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Amazon" / "AWSCLIV2" / "aws.exe",
        Path("C:/Program Files/Amazon/AWSCLIV2/aws.exe"),
    ]:
        if c.exists():
            return [str(c)]
    # awscli v1 vía pip: el script 'aws' se ejecuta con el python del venv
    venv_scripts = Path(sys.executable).parent
    for c in [venv_scripts / "aws", venv_scripts / "Scripts" / "aws"]:
        if c.exists():
            return [sys.executable, str(c)]
    return None


def _aws_works(aws: list[str]) -> bool:
    try:
        out = subprocess.run(aws + ["--version"], capture_output=True, text=True, timeout=60)
        if out.returncode == 0:
            logger.info("AWS CLI disponible: %s", (out.stdout or out.stderr).strip())
            return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("aws --version falló: %s", exc)
    return False


def ensure_aws_cli(skip_install: bool = False) -> list[str] | None:
    """
    Garantiza AWS CLI v2. Si no está, descarga e instala el MSI en silencio.

    Returns:
        Ruta a aws.exe si quedó disponible, None si hay que intervenir a mano.
    """
    aws = find_aws()
    if aws and _aws_works(aws):
        return aws

    if skip_install:
        logger.warning("AWS CLI no encontrado y --skip-aws-install activo.")
        return None

    logger.info("AWS CLI no encontrado. Descargando instalador v2 ...")
    msi = Path(tempfile.gettempdir()) / "AWSCLIV2.msi"
    try:
        urllib.request.urlretrieve(_AWS_MSI_URL, msi)
        logger.info("MSI descargado (%s) en %s", _human(msi.stat().st_size), msi)
    except Exception as exc:  # noqa: BLE001
        _aws_manual_instructions(f"no se pudo descargar el MSI: {exc}")
        return None

    logger.info("Instalando AWS CLI en silencio (msiexec /quiet) ...")
    try:
        proc = subprocess.run(
            ["msiexec", "/i", str(msi), "/quiet", "/norestart"],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            logger.warning("msiexec devolvió código %d.", proc.returncode)
    except subprocess.TimeoutExpired:
        logger.warning("La instalación de AWS CLI superó el timeout.")
    except Exception as exc:  # noqa: BLE001
        _aws_manual_instructions(f"falló msiexec: {exc}")
        return None

    aws = find_aws()
    if aws and _aws_works(aws):
        logger.info("AWS CLI v2 instalado correctamente.")
        return aws

    # Fallback SIN admin: el MSI v2 necesita elevación (msiexec 1603). awscli v1
    # vía pip se instala en el venv sin permisos de administrador y soporta
    # `s3 sync --no-sign-request` igual que v2.
    logger.warning(
        "El MSI v2 requirió permisos de administrador (no disponibles). "
        "Fallback: instalando awscli v1 vía pip en el entorno actual ..."
    )
    aws = _pip_install_awscli()
    if aws and _aws_works(aws):
        logger.info("AWS CLI v1 (pip) listo: %s", aws)
        return aws

    _aws_manual_instructions(
        "ni el MSI v2 (requiere admin) ni el fallback pip funcionaron."
    )
    return None


def _pip_install_awscli() -> list[str] | None:
    """Instala awscli v1 vía pip en el entorno actual (sin admin) y devuelve su prefijo argv."""
    try:
        rc = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "awscli"],
            timeout=600,
        ).returncode
        if rc != 0:
            logger.warning("pip install awscli devolvió código %d.", rc)
            return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("Fallo pip install awscli: %s", exc)
        return None
    return find_aws()


def _aws_manual_instructions(reason: str) -> None:
    logger.error(
        "No se pudo dejar AWS CLI listo automáticamente (%s).\n"
        "PASOS MANUALES:\n"
        "  1. Descargá e instalá: %s\n"
        "  2. CERRÁ esta terminal y abrí una NUEVA (para refrescar el PATH).\n"
        "  3. Verificá con:  aws --version\n"
        "  4. Volvé a correr:  python -m src.data.setup_external_data\n"
        "  (El script es reanudable: las descargas ya hechas no se rebajan.)",
        reason, _AWS_MSI_URL,
    )


# ---------------------------------------------------------------------------
# b) Tamaño del bucket S3
# ---------------------------------------------------------------------------

def s3_folder_size(aws: list[str], uri: str) -> tuple[int, int]:
    """(número de objetos, bytes) de un prefijo S3 vía `aws s3 ls --summarize`."""
    try:
        out = subprocess.run(
            aws + ["s3", "ls", "--no-sign-request", "--summarize", "--recursive", uri],
            capture_output=True, text=True, timeout=1800,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("No se pudo listar %s: %s", uri, exc)
        return 0, 0
    n_obj, total = 0, 0
    for line in out.stdout.splitlines():
        s = line.strip()
        if s.startswith("Total Objects:"):
            n_obj = int(s.split(":")[1].strip())
        elif s.startswith("Total Size:"):
            total = int(s.split(":")[1].strip())
    return n_obj, total


def report_s3_sizes(aws: list[str]) -> int:
    """Imprime el tamaño de cada carpeta S3 y devuelve el total en bytes."""
    logger.info("Midiendo tamaño del bucket S3 (esto puede tardar) ...")
    grand = 0
    print("\n  Carpeta S3                | objetos | tamaño")
    print("  " + "-" * 52)
    for name, (uri, _dest) in S3_FOLDERS.items():
        n, size = s3_folder_size(aws, uri)
        grand += size
        print(f"  {name:<25} | {n:>7} | {_human(size)}")
    print("  " + "-" * 52)
    print(f"  {'TOTAL a descargar (S3)':<25} | {'':>7} | {_human(grand)}\n")
    return grand


# ---------------------------------------------------------------------------
# Ejecución con heartbeat (para descargas largas)
# ---------------------------------------------------------------------------

def run_with_heartbeat(cmd: list[str], label: str, heartbeat: int = 30) -> int:
    """
    Ejecuta un comando mostrando su salida + un latido cada `heartbeat` segundos.

    Devuelve el returncode. El heartbeat confirma que el proceso sigue vivo aun
    cuando AWS esté transfiriendo un archivo grande sin imprimir nada.
    """
    stop = threading.Event()
    start = time.time()

    def _beat() -> None:
        while not stop.wait(heartbeat):
            mins = (time.time() - start) / 60.0
            logger.info("[heartbeat] %s: vivo, %.1f min transcurridos ...", label, mins)

    t = threading.Thread(target=_beat, daemon=True)
    t.start()
    try:
        proc = subprocess.run(cmd)
        return proc.returncode
    finally:
        stop.set()


# ---------------------------------------------------------------------------
# d) Sync S3
# ---------------------------------------------------------------------------

def sync_s3(aws: list[str], heartbeat: int = 30) -> None:
    """Sincroniza cada carpeta S3 a su destino (reanudable e idempotente)."""
    for name, (uri, dest_rel) in S3_FOLDERS.items():
        dest = DATA_ROOT / dest_rel
        dest.mkdir(parents=True, exist_ok=True)
        logger.info("─── S3 sync: %s → %s ───", uri, dest)
        rc = run_with_heartbeat(
            aws + ["s3", "sync", "--no-sign-request", uri, str(dest)],
            label=f"s3:{name}", heartbeat=heartbeat,
        )
        if rc != 0:
            logger.error("aws s3 sync de %s devolvió código %d (reanudable: re-corré).", name, rc)
        else:
            logger.info("✓ %s sincronizado.", name)
        report_download(dest, name)


# ---------------------------------------------------------------------------
# e) Kaggle
# ---------------------------------------------------------------------------

def _same_content(src: Path, dst: Path) -> bool:
    """True si dst ya tiene el mismo nº de archivos y tamaño total que src."""
    return dir_stats(src) == dir_stats(dst) and dir_stats(src)[0] > 0


def download_kaggle(heartbeat: int = 30) -> None:
    """Descarga (caché) cada dataset de Kaggle y lo COPIA al destino sin mover el caché."""
    import kagglehub

    for slug, dest_rel in KAGGLE_DATASETS.items():
        dest = DATA_ROOT / dest_rel
        dest.mkdir(parents=True, exist_ok=True)
        logger.info("─── Kaggle: %s ───", slug)
        cache = Path(kagglehub.dataset_download(slug))
        logger.info("Caché kagglehub: %s", cache)

        if _same_content(cache, dest):
            logger.info("Destino ya contiene una copia idéntica; se omite la copia.")
        else:
            logger.info("Copiando %s → %s ...", cache, dest)
            stop = threading.Event()
            start = time.time()

            def _beat() -> None:
                while not stop.wait(heartbeat):
                    logger.info("[heartbeat] copia %s: %.1f min ...", slug, (time.time()-start)/60)

            t = threading.Thread(target=_beat, daemon=True)
            t.start()
            try:
                shutil.copytree(cache, dest, dirs_exist_ok=True)
            finally:
                stop.set()
            logger.info("✓ Copiado (caché kagglehub intacto en %s).", cache)

        report_download(dest, slug)


# ---------------------------------------------------------------------------
# f) Reporte por descarga
# ---------------------------------------------------------------------------

def report_download(dest: Path, label: str) -> None:
    """Imprime ruta, nº de archivos, tamaño total y los primeros 5 archivos."""
    n, size = dir_stats(dest)
    logger.info("Reporte '%s': %s | %d archivos | %s", label, dest, n, _human(size))
    files = []
    for root, _dirs, fs in os.walk(dest):
        for f in sorted(fs):
            files.append(Path(root) / f)
        if len(files) >= 5:
            break
    for f in files[:5]:
        try:
            logger.info("    - %s  (%s)", f.relative_to(dest), _human(f.stat().st_size))
        except OSError:
            pass


# ---------------------------------------------------------------------------
# g) README de inventario
# ---------------------------------------------------------------------------

def _sample_csv_header(path: Path) -> str:
    """Devuelve las columnas del primer CSV hallado bajo `path` (o '—')."""
    for root, _dirs, files in os.walk(path):
        for f in sorted(files):
            if f.lower().endswith(".csv"):
                try:
                    with open(Path(root) / f, "r", encoding="utf-8", errors="replace") as fh:
                        header = next(csv.reader(fh))
                    return f"`{f}` → {', '.join(header)}"
                except Exception:  # noqa: BLE001
                    continue
    return "— (sin CSV de muestra todavía)"


def generate_readme() -> Path:
    """Escribe D:\\trading-data\\README.md con el inventario completo."""
    today = date.today().isoformat()
    lines = [
        "# Inventario de datasets externos — tesis trading",
        "",
        f"> Generado por `src/data/setup_external_data.py` el {today}. "
        "Estos datos viven fuera del repo (disco aparte).",
        "",
        f"Raíz: `{DATA_ROOT}`",
        "",
    ]

    inventory = [
        ("StockTwits NYU — symbol_sentiments", "s3://stocktwits-nyu (público)",
         Path("stocktwits_nyu") / "symbol_sentiments",
         "Sentimiento agregado por símbolo del dataset StockTwits de NYU."),
        ("StockTwits NYU — feature_wo_messages", "s3://stocktwits-nyu (público)",
         Path("stocktwits_nyu") / "feature_wo_messages",
         "Features derivadas sin el texto de los mensajes."),
        ("StockTwits NYU — messages", "s3://stocktwits-nyu (público)",
         Path("stocktwits_nyu") / "messages",
         "Mensajes crudos de StockTwits."),
        ("WSB Kaggle — kevinwang313", "kaggle:kevinwang313/wallstreetbets-dataset",
         Path("wsb_kaggle") / "kevinwang313_wallstreetbets",
         "Dataset de r/WallStreetBets (Kevin Wang)."),
        ("WSB Kaggle — unanimad", "kaggle:unanimad/reddit-rwallstreetbets",
         Path("wsb_kaggle") / "unanimad_reddit_rwallstreetbets",
         "Posts de r/WallStreetBets (unanimad)."),
        ("FNSPID", "manual (movido por el usuario)",
         Path("fnspid"),
         "Financial News and Stock Price Integration Dataset. Se mueve a mano."),
    ]

    for title, origin, rel, desc in inventory:
        dest = DATA_ROOT / rel
        n, size = dir_stats(dest)
        header = _sample_csv_header(dest)
        lines += [
            f"## {title}",
            f"- Origen: {origin}",
            f"- Ruta: `{dest}`",
            f"- Estado: {'descargado' if n > 0 else 'vacío (pendiente)'}",
            f"- Archivos: {n} | Tamaño: {_human(size)}",
            f"- Descripción: {desc}",
            f"- Columnas (muestra): {header}",
            "",
        ]

    free = free_space(DATA_ROOT)
    lines += ["---", f"Espacio libre en el disco de datos: **{_human(free)}**.", ""]

    out = DATA_ROOT / "README.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("README de inventario escrito en %s", out)
    return out


# ---------------------------------------------------------------------------
# Confirmación
# ---------------------------------------------------------------------------

def confirm(prompt: str, assume_yes: bool) -> bool:
    if assume_yes:
        logger.info("--yes activo: %s → sí (sin preguntar).", prompt)
        return True
    try:
        ans = input(f"{prompt} [y/n]: ").strip().lower()
    except EOFError:
        logger.warning("Sin entrada interactiva disponible; abortando. Usá --yes para forzar.")
        return False
    return ans in ("y", "yes", "s", "si", "sí")


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Setup de datasets externos en D:\\trading-data.")
    parser.add_argument("--yes", action="store_true", help="No preguntar; proceder a descargar.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo estructura + tamaños; no descarga nada.")
    parser.add_argument("--skip-aws-install", action="store_true",
                        help="No intentar instalar AWS CLI.")
    parser.add_argument("--only", choices=["structure", "s3", "kaggle", "readme"], default=None,
                        help="Ejecutar solo una etapa.")
    parser.add_argument("--heartbeat", type=int, default=30, help="Segundos entre latidos.")
    args = parser.parse_args()

    # a) estructura siempre
    ensure_structure()
    if args.only == "structure":
        return 0
    if args.only == "readme":
        generate_readme()
        return 0

    # b) espacio libre
    free_before = free_space(DATA_ROOT)
    logger.info("Espacio libre en %s: %s", DATA_ROOT.drive or DATA_ROOT, _human(free_before))

    aws = None
    s3_total = 0
    if args.only in (None, "s3"):
        aws = ensure_aws_cli(skip_install=args.skip_aws_install)
        if aws:
            s3_total = report_s3_sizes(aws)
            if s3_total > free_before:
                logger.warning(
                    "El total a descargar (%s) supera el espacio libre (%s).",
                    _human(s3_total), _human(free_before),
                )
        else:
            logger.warning("Sin AWS CLI no se puede medir ni sincronizar S3.")

    if args.dry_run:
        logger.info("--dry-run: plan mostrado, no se descarga. Espacio libre: %s",
                    _human(free_before))
        return 0

    # confirmación antes de descargar
    if not confirm("¿Proceder con la descarga de los datasets?", args.yes):
        logger.info("Cancelado por el usuario. No se descargó nada.")
        return 0

    # d) S3
    if args.only in (None, "s3"):
        if aws:
            sync_s3(aws, heartbeat=args.heartbeat)
        else:
            logger.error("No se puede sincronizar S3 sin AWS CLI (ver instrucciones arriba).")

    # e) Kaggle
    if args.only in (None, "kaggle"):
        try:
            download_kaggle(heartbeat=args.heartbeat)
        except Exception as exc:  # noqa: BLE001
            logger.error("Fallo en descarga Kaggle: %s", exc)

    # g) README + resumen final
    generate_readme()
    _final_summary(free_before)
    return 0


def _final_summary(free_before: int) -> None:
    logger.info("═══ RESUMEN FINAL ═══")
    print("\n  Dataset                              | archivos | tamaño")
    print("  " + "-" * 60)
    rows = [
        ("stocktwits_nyu/symbol_sentiments", Path("stocktwits_nyu") / "symbol_sentiments"),
        ("stocktwits_nyu/feature_wo_messages", Path("stocktwits_nyu") / "feature_wo_messages"),
        ("stocktwits_nyu/messages", Path("stocktwits_nyu") / "messages"),
        ("wsb_kaggle/kevinwang313", Path("wsb_kaggle") / "kevinwang313_wallstreetbets"),
        ("wsb_kaggle/unanimad", Path("wsb_kaggle") / "unanimad_reddit_rwallstreetbets"),
        ("fnspid (manual)", Path("fnspid")),
    ]
    for label, rel in rows:
        n, size = dir_stats(DATA_ROOT / rel)
        print(f"  {label:<36} | {n:>8} | {_human(size)}")
    free_after = free_space(DATA_ROOT)
    print("  " + "-" * 60)
    print(f"  Espacio libre: antes {_human(free_before)} → ahora {_human(free_after)} "
          f"(usado {_human(max(0, free_before - free_after))})\n")
    logger.info("Pasos manuales pendientes:")
    logger.info("  • Mover FNSPID a %s", DATA_ROOT / "fnspid")
    logger.info("  • Si AWS CLI se acababa de instalar y `aws` no respondía, abrí una "
                "terminal nueva y re-corré (reanudable).")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    sys.exit(main())
