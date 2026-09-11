"""
Sube los datos locales de la tesis a Google Drive.

Carpetas que sube:
  - D:/trading-data/  (o TESIS_DATA_ROOT)  → Drive:/tesis-trading/datos-externos/
  - data/raw/                               → Drive:/tesis-trading/data/raw/
  - data/processed/                         → Drive:/tesis-trading/data/processed/
  - models/                                 → Drive:/tesis-trading/models/

Uso:
    python -m src.data.drive_upload                         # sube todo (interactivo)
    python -m src.data.drive_upload --only raw              # solo data/raw/
    python -m src.data.drive_upload --only processed        # solo data/processed/
    python -m src.data.drive_upload --only models           # solo models/
    python -m src.data.drive_upload --only external         # solo D:/trading-data/
    python -m src.data.drive_upload --dry-run               # muestra plan sin subir
    python -m src.data.drive_upload --yes                   # sin confirmación

Autenticación (primera vez):
    1. Crear proyecto en https://console.cloud.google.com
    2. Habilitar Google Drive API
    3. Descargar credentials.json → colocar en la raíz del proyecto
    4. Al correr por primera vez, se abrirá el browser para autorizar
    5. Se guarda token.json (no commitear)

Variables de entorno opcionales:
    GOOGLE_CREDENTIALS_FILE=ruta/a/credentials.json   (default: credentials.json)
    GOOGLE_TOKEN_FILE=ruta/a/token.json               (default: token.json)
    TESIS_DATA_ROOT=D:/trading-data                   (default: D:/trading-data)
    DRIVE_FOLDER_NAME=tesis-trading                   (default: tesis-trading)
"""

from __future__ import annotations

import argparse
import logging
import mimetypes
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger("drive_upload")

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[3]
CREDENTIALS_FILE = Path(os.getenv("GOOGLE_CREDENTIALS_FILE", ROOT_DIR / "credentials.json"))
TOKEN_FILE = Path(os.getenv("GOOGLE_TOKEN_FILE", ROOT_DIR / "token.json"))
EXTERNAL_DATA_ROOT = Path(os.getenv("TESIS_DATA_ROOT", "D:/trading-data"))
DRIVE_FOLDER_NAME = os.getenv("DRIVE_FOLDER_NAME", "tesis-trading")

SCOPES = ["https://www.googleapis.com/auth/drive.file"]

# Extensiones de archivos grandes que siempre se suben (no se filtran)
INCLUDE_EXTENSIONS = {
    ".parquet", ".csv", ".json", ".txt", ".pkl", ".joblib",
    ".h5", ".hdf5", ".pt", ".pth", ".bin",
}

# Archivos/carpetas que se omiten
EXCLUDE_NAMES = {
    "__pycache__", ".git", ".venv", "venv", "node_modules",
    "*.pyc", "*.pyo", ".DS_Store", "Thumbs.db",
}

# Tamaño máximo de chunk para uploads reanudables (5 MB)
CHUNK_SIZE = 5 * 1024 * 1024


# ---------------------------------------------------------------------------
# Autenticación
# ---------------------------------------------------------------------------

def get_drive_service():
    """Obtiene el servicio autenticado de Google Drive."""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        logger.error(
            "Faltan dependencias. Instalá con:\n"
            "  pip install google-api-python-client google-auth-httplib2 "
            "google-auth-oauthlib"
        )
        sys.exit(1)

    creds = None

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Refrescando token de acceso...")
            creds.refresh(Request())
        else:
            if not CREDENTIALS_FILE.exists():
                logger.error(
                    "No se encontró '%s'.\n"
                    "  1. Ve a https://console.cloud.google.com\n"
                    "  2. Crea un proyecto → Habilita 'Google Drive API'\n"
                    "  3. Credenciales → 'OAuth 2.0 Client ID' (tipo Desktop)\n"
                    "  4. Descarga el JSON y guárdalo como: %s",
                    CREDENTIALS_FILE, CREDENTIALS_FILE,
                )
                sys.exit(1)
            logger.info("Abriendo navegador para autorizar acceso a Google Drive...")
            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_FILE), SCOPES
            )
            creds = flow.run_local_server(port=0)

        TOKEN_FILE.write_text(creds.to_json())
        logger.info("Token guardado en %s (no commitear este archivo)", TOKEN_FILE)

    return build("drive", "v3", credentials=creds)


# ---------------------------------------------------------------------------
# Operaciones sobre carpetas de Drive
# ---------------------------------------------------------------------------

def get_or_create_folder(service, name: str, parent_id: str | None = None) -> str:
    """Devuelve el ID de la carpeta en Drive, creándola si no existe."""
    query = f"name='{name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"

    results = service.files().list(q=query, fields="files(id, name)").execute()
    items = results.get("files", [])

    if items:
        folder_id = items[0]["id"]
        logger.debug("Carpeta existente: '%s' (id=%s)", name, folder_id)
        return folder_id

    # Crear carpeta nueva
    metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    if parent_id:
        metadata["parents"] = [parent_id]

    folder = service.files().create(body=metadata, fields="id").execute()
    folder_id = folder.get("id")
    logger.info("Carpeta creada en Drive: '%s' (id=%s)", name, folder_id)
    return folder_id


def file_exists_in_drive(service, name: str, parent_id: str) -> str | None:
    """Devuelve el ID del archivo en Drive si ya existe, None si no."""
    query = (
        f"name='{name}' and '{parent_id}' in parents "
        f"and trashed=false and mimeType!='application/vnd.google-apps.folder'"
    )
    results = service.files().list(q=query, fields="files(id, name, size, modifiedTime)").execute()
    items = results.get("files", [])
    return items[0]["id"] if items else None


# ---------------------------------------------------------------------------
# Upload de archivos
# ---------------------------------------------------------------------------

def upload_file(
    service,
    local_path: Path,
    parent_id: str,
    dry_run: bool = False,
) -> bool:
    """
    Sube un archivo a Drive con upload reanudable.
    Retorna True si se subió (o ya existía), False si falló.
    """
    from googleapiclient.http import MediaFileUpload

    file_size = local_path.stat().st_size
    size_str = _human(file_size)
    existing_id = file_exists_in_drive(service, local_path.name, parent_id)

    if existing_id:
        logger.info("  ⏭  Ya existe: %s (%s) — omitido", local_path.name, size_str)
        return True

    if dry_run:
        logger.info("  [dry-run] SUBIRÍA: %s (%s)", local_path.name, size_str)
        return True

    mime_type, _ = mimetypes.guess_type(str(local_path))
    mime_type = mime_type or "application/octet-stream"

    metadata = {"name": local_path.name, "parents": [parent_id]}
    media = MediaFileUpload(
        str(local_path),
        mimetype=mime_type,
        chunksize=CHUNK_SIZE,
        resumable=True,
    )

    logger.info("  ⬆  Subiendo: %s (%s)...", local_path.name, size_str)
    start = time.time()
    try:
        request = service.files().create(body=metadata, media_body=media, fields="id")
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                pct = int(status.progress() * 100)
                elapsed = time.time() - start
                logger.info("     %d%% — %.1fs", pct, elapsed)
        elapsed = time.time() - start
        logger.info("  ✓  Subido en %.1fs: %s", elapsed, local_path.name)
        return True
    except Exception as exc:
        logger.error("  ✗  Error subiendo %s: %s", local_path.name, exc)
        return False


# ---------------------------------------------------------------------------
# Upload recursivo de carpetas
# ---------------------------------------------------------------------------

def upload_directory(
    service,
    local_dir: Path,
    parent_drive_id: str,
    dry_run: bool = False,
    stats: dict | None = None,
) -> None:
    """Sube recursivamente el contenido de `local_dir` a Drive."""
    if stats is None:
        stats = {"uploaded": 0, "skipped": 0, "failed": 0, "bytes": 0}

    if not local_dir.exists():
        logger.warning("La carpeta no existe localmente: %s", local_dir)
        return

    for item in sorted(local_dir.iterdir()):
        # Omitir exclusiones
        if item.name in EXCLUDE_NAMES or item.name.startswith("."):
            continue

        if item.is_dir():
            sub_folder_id = get_or_create_folder(service, item.name, parent_drive_id)
            upload_directory(service, item, sub_folder_id, dry_run, stats)

        elif item.is_file():
            # Filtrar por extensión si no está en la lista permitida
            if item.suffix.lower() not in INCLUDE_EXTENSIONS and item.stat().st_size < 1024:
                logger.debug("  ⏭  Archivo pequeño ignorado: %s", item.name)
                stats["skipped"] += 1
                continue

            ok = upload_file(service, item, parent_drive_id, dry_run=dry_run)
            if ok:
                stats["uploaded"] += 1
                stats["bytes"] += item.stat().st_size
            else:
                stats["failed"] += 1


# ---------------------------------------------------------------------------
# Targets de upload
# ---------------------------------------------------------------------------

def _build_targets(args_only: str | None) -> list[tuple[Path, str]]:
    """
    Devuelve lista de (ruta_local, nombre_subcarpeta_en_drive).
    """
    all_targets = {
        "raw":      (ROOT_DIR / "data" / "raw",       "data/raw"),
        "processed":(ROOT_DIR / "data" / "processed", "data/processed"),
        "models":   (ROOT_DIR / "models",             "models"),
        "external": (EXTERNAL_DATA_ROOT,              "datos-externos"),
    }
    if args_only:
        if args_only not in all_targets:
            logger.error("Valor inválido para --only: '%s'. Opciones: %s",
                         args_only, list(all_targets.keys()))
            sys.exit(1)
        return [all_targets[args_only]]
    return list(all_targets.values())


def _resolve_nested_folder(service, root_id: str, path: str) -> str:
    """Crea/obtiene carpetas anidadas (ej. 'data/raw') bajo root_id."""
    current_id = root_id
    for part in path.split("/"):
        current_id = get_or_create_folder(service, part, current_id)
    return current_id


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _human(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def _count_files(path: Path) -> tuple[int, int]:
    """(nº archivos, bytes totales) en `path` de forma recursiva."""
    if not path.exists():
        return 0, 0
    count, total = 0, 0
    for f in path.rglob("*"):
        if f.is_file() and f.suffix.lower() in INCLUDE_EXTENSIONS:
            count += 1
            total += f.stat().st_size
    return count, total


def confirm(prompt: str, assume_yes: bool) -> bool:
    if assume_yes:
        return True
    try:
        ans = input(f"{prompt} [y/n]: ").strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes", "s", "si", "sí")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sube los datos de la tesis a Google Drive."
    )
    parser.add_argument(
        "--only",
        choices=["raw", "processed", "models", "external"],
        default=None,
        help="Subir solo una sección.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Muestra plan sin subir.")
    parser.add_argument("--yes", action="store_true", help="Sin confirmación interactiva.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    targets = _build_targets(args.only)

    # Mostrar resumen del plan
    print("\n📁 PLAN DE SUBIDA A GOOGLE DRIVE")
    print(f"   Carpeta raíz en Drive: '{DRIVE_FOLDER_NAME}'\n")
    total_files, total_bytes = 0, 0
    for local_path, drive_subpath in targets:
        n, b = _count_files(local_path)
        status = "✓ existe" if local_path.exists() else "⚠ no encontrada"
        print(f"   {status}  {local_path}")
        print(f"            → Drive: {DRIVE_FOLDER_NAME}/{drive_subpath}")
        print(f"            {n} archivos | {_human(b)}\n")
        total_files += n
        total_bytes += b

    print(f"   TOTAL: {total_files} archivos | {_human(total_bytes)}")
    if args.dry_run:
        print("\n   [dry-run] No se subirá nada.\n")
        return 0
    print()

    if not confirm("¿Proceder con la subida?", args.yes):
        print("Cancelado.")
        return 0

    # Autenticar y obtener servicio
    logger.info("Autenticando con Google Drive...")
    service = get_drive_service()

    # Carpeta raíz en Drive
    root_id = get_or_create_folder(service, DRIVE_FOLDER_NAME)
    logger.info("Carpeta raíz en Drive: '%s' (id=%s)", DRIVE_FOLDER_NAME, root_id)

    # Subir cada target
    global_stats = {"uploaded": 0, "skipped": 0, "failed": 0, "bytes": 0}
    for local_path, drive_subpath in targets:
        logger.info("─── Subiendo: %s → %s/%s ───", local_path, DRIVE_FOLDER_NAME, drive_subpath)
        folder_id = _resolve_nested_folder(service, root_id, drive_subpath)
        upload_directory(service, local_path, folder_id, dry_run=args.dry_run, stats=global_stats)

    # Resumen final
    print("\n═══ RESUMEN FINAL ═══")
    print(f"  Subidos:  {global_stats['uploaded']} archivos ({_human(global_stats['bytes'])})")
    print(f"  Fallidos: {global_stats['failed']}")
    print(f"  Saltados: {global_stats['skipped']}")
    if global_stats["failed"] > 0:
        print("\n  ⚠ Algunos archivos fallaron. Volvé a correr el script (es reanudable).")
    else:
        print(f"\n  ✓ Todo subido a Drive: '{DRIVE_FOLDER_NAME}/'")
    print()
    return 0 if global_stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
