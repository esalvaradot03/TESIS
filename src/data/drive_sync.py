"""
Descarga los datos de la tesis desde Google Drive al servidor/PC.

Uso (desde la raíz del proyecto con el venv activo):
    python -m src.data.drive_sync                    # descarga todo
    python -m src.data.drive_sync --only raw         # solo data/raw/
    python -m src.data.drive_sync --only processed   # solo data/processed/
    python -m src.data.drive_sync --only models      # solo models/
    python -m src.data.drive_sync --only external    # solo datos-externos/
    python -m src.data.drive_sync --dry-run          # muestra qué descargaría
    python -m src.data.drive_sync --yes              # sin confirmación

Variables de entorno:
    GOOGLE_CREDENTIALS_FILE  ruta al credentials.json  (default: credentials.json)
    GOOGLE_TOKEN_FILE        ruta al token.json         (default: token.json)
    TESIS_DATA_ROOT          destino datos externos     (default: D:/trading-data)
    DRIVE_FOLDER_NAME        nombre carpeta en Drive    (default: tesis-trading)
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time
from pathlib import Path

logger = logging.getLogger("drive_sync")

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[3]
CREDENTIALS_FILE = Path(os.getenv("GOOGLE_CREDENTIALS_FILE", ROOT_DIR / "credentials.json"))
TOKEN_FILE = Path(os.getenv("GOOGLE_TOKEN_FILE", ROOT_DIR / "token.json"))
EXTERNAL_DATA_ROOT = Path(os.getenv("TESIS_DATA_ROOT", "D:/trading-data"))
DRIVE_FOLDER_NAME = os.getenv("DRIVE_FOLDER_NAME", "tesis-trading")

SCOPES = ["https://www.googleapis.com/auth/drive.file"]

# Mapa: nombre_sección → (subcarpeta_en_drive, destino_local)
SYNC_TARGETS = {
    "raw":       ("data/raw",        ROOT_DIR / "data" / "raw"),
    "processed": ("data/processed",  ROOT_DIR / "data" / "processed"),
    "models":    ("models",          ROOT_DIR / "models"),
    "external":  ("datos-externos",  EXTERNAL_DATA_ROOT),
}


# ---------------------------------------------------------------------------
# Autenticación (reutilizada desde drive_upload)
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
                    "  Sigue la guía en docs/GUIA_GOOGLE_DRIVE.md para crear las credenciales.",
                    CREDENTIALS_FILE,
                )
                sys.exit(1)
            logger.info("Abriendo navegador para autorizar acceso a Google Drive...")
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_FILE.write_text(creds.to_json())
        logger.info("Token guardado en %s", TOKEN_FILE)

    return build("drive", "v3", credentials=creds)


# ---------------------------------------------------------------------------
# Navegación en Drive
# ---------------------------------------------------------------------------

def find_folder(service, name: str, parent_id: str | None = None) -> str | None:
    """Devuelve el ID de la carpeta si existe, None si no."""
    query = (
        f"name='{name}' and mimeType='application/vnd.google-apps.folder' "
        f"and trashed=false"
    )
    if parent_id:
        query += f" and '{parent_id}' in parents"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    items = results.get("files", [])
    return items[0]["id"] if items else None


def resolve_path(service, root_id: str, path: str) -> str | None:
    """Navega carpetas anidadas (ej. 'data/raw') y devuelve el ID final."""
    current_id = root_id
    for part in path.split("/"):
        current_id = find_folder(service, part, current_id)
        if current_id is None:
            return None
    return current_id


def list_folder_contents(service, folder_id: str) -> list[dict]:
    """Lista todos los archivos/carpetas dentro de un folder (paginando)."""
    items = []
    page_token = None
    while True:
        params = {
            "q": f"'{folder_id}' in parents and trashed=false",
            "fields": "nextPageToken, files(id, name, mimeType, size, modifiedTime)",
            "pageSize": 1000,
        }
        if page_token:
            params["pageToken"] = page_token
        response = service.files().list(**params).execute()
        items.extend(response.get("files", []))
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return items


# ---------------------------------------------------------------------------
# Descarga de archivos
# ---------------------------------------------------------------------------

def download_file(
    service,
    file_id: str,
    file_name: str,
    dest_path: Path,
    file_size: int,
    dry_run: bool = False,
) -> bool:
    """
    Descarga un archivo de Drive a `dest_path`.
    Si ya existe con el mismo tamaño, lo omite.
    Retorna True si se descargó (o ya existía), False si falló.
    """
    from googleapiclient.http import MediaIoBaseDownload

    size_str = _human(file_size)

    # Skip si ya existe con mismo tamaño
    if dest_path.exists() and dest_path.stat().st_size == file_size:
        logger.info("  ⏭  Ya existe: %s (%s)", file_name, size_str)
        return True

    if dry_run:
        logger.info("  [dry-run] DESCARGARÍA: %s (%s)", file_name, size_str)
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("  ⬇  Descargando: %s (%s)...", file_name, size_str)
    start = time.time()

    try:
        request = service.files().get_media(fileId=file_id)
        buf = io.FileIO(str(dest_path), mode="wb")
        downloader = MediaIoBaseDownload(buf, request, chunksize=5 * 1024 * 1024)

        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                pct = int(status.progress() * 100)
                elapsed = time.time() - start
                logger.info("     %d%% — %.1fs", pct, elapsed)

        buf.close()
        elapsed = time.time() - start
        logger.info("  ✓  Descargado en %.1fs: %s", elapsed, file_name)
        return True

    except Exception as exc:
        logger.error("  ✗  Error descargando %s: %s", file_name, exc)
        if dest_path.exists():
            dest_path.unlink()  # eliminar archivo incompleto
        return False


# ---------------------------------------------------------------------------
# Descarga recursiva
# ---------------------------------------------------------------------------

def sync_folder(
    service,
    drive_folder_id: str,
    local_dest: Path,
    dry_run: bool = False,
    stats: dict | None = None,
) -> None:
    """Descarga recursivamente el contenido de un folder de Drive a `local_dest`."""
    if stats is None:
        stats = {"downloaded": 0, "skipped": 0, "failed": 0, "bytes": 0}

    local_dest.mkdir(parents=True, exist_ok=True)
    items = list_folder_contents(service, drive_folder_id)

    for item in items:
        name = item["name"]
        mime = item.get("mimeType", "")
        item_id = item["id"]

        if mime == "application/vnd.google-apps.folder":
            # Subcarpeta: descender recursivamente
            sub_local = local_dest / name
            logger.debug("  📂 Entrando en subcarpeta: %s", name)
            sync_folder(service, item_id, sub_local, dry_run=dry_run, stats=stats)

        else:
            # Archivo
            file_size = int(item.get("size", 0))
            dest_path = local_dest / name
            ok = download_file(
                service, item_id, name, dest_path, file_size, dry_run=dry_run
            )
            if ok:
                if dest_path.exists() and dest_path.stat().st_size == file_size:
                    stats["skipped"] += 1  # ya existía
                else:
                    stats["downloaded"] += 1
                    stats["bytes"] += file_size
            else:
                stats["failed"] += 1


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _human(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


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
        description="Descarga datos de la tesis desde Google Drive."
    )
    parser.add_argument(
        "--only",
        choices=list(SYNC_TARGETS.keys()),
        default=None,
        help="Descargar solo una sección.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Muestra plan sin descargar.")
    parser.add_argument("--yes", action="store_true", help="Sin confirmación interactiva.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Seleccionar targets
    targets = (
        {args.only: SYNC_TARGETS[args.only]}
        if args.only
        else SYNC_TARGETS
    )

    # Mostrar plan
    print("\n📥 PLAN DE DESCARGA DESDE GOOGLE DRIVE")
    print(f"   Carpeta raíz en Drive: '{DRIVE_FOLDER_NAME}'\n")
    for key, (drive_sub, local_dest) in targets.items():
        print(f"   Drive: {DRIVE_FOLDER_NAME}/{drive_sub}")
        print(f"       → Local: {local_dest}\n")

    if args.dry_run:
        print("   [dry-run] No se descargará nada.\n")

    if not confirm("¿Proceder con la descarga?", args.yes):
        print("Cancelado.")
        return 0

    # Autenticar
    logger.info("Autenticando con Google Drive...")
    service = get_drive_service()

    # Buscar carpeta raíz en Drive
    root_id = find_folder(service, DRIVE_FOLDER_NAME)
    if not root_id:
        logger.error(
            "No se encontró la carpeta '%s' en Drive.\n"
            "  Primero ejecutá drive_upload.py desde tu PC local para subir los datos.",
            DRIVE_FOLDER_NAME,
        )
        return 1
    logger.info("Carpeta raíz encontrada: '%s' (id=%s)", DRIVE_FOLDER_NAME, root_id)

    # Descargar cada target
    global_stats = {"downloaded": 0, "skipped": 0, "failed": 0, "bytes": 0}

    for key, (drive_sub, local_dest) in targets.items():
        logger.info("─── Sincronizando: %s/%s → %s ───", DRIVE_FOLDER_NAME, drive_sub, local_dest)
        folder_id = resolve_path(service, root_id, drive_sub)
        if not folder_id:
            logger.warning("  ⚠ Carpeta '%s' no encontrada en Drive, saltando.", drive_sub)
            continue
        sync_folder(service, folder_id, local_dest, dry_run=args.dry_run, stats=global_stats)

    # Resumen
    print("\n═══ RESUMEN FINAL ═══")
    print(f"  Descargados: {global_stats['downloaded']} archivos ({_human(global_stats['bytes'])})")
    print(f"  Ya existían: {global_stats['skipped']}")
    print(f"  Fallidos:    {global_stats['failed']}")
    if global_stats["failed"] > 0:
        print("\n  ⚠ Algunos archivos fallaron. Volvé a correr el script (es reanudable).")
    else:
        print("\n  ✓ Sincronización completa.")
    print()
    return 0 if global_stats["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
