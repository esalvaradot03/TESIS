"""
Descarga y exploración del dataset histórico de StockTwits desde Kaggle.

Usa kagglehub para bajar `frankcaoyun/stocktwits-2020-2022-raw` y luego reporta
la estructura del dataset para poder mapear después el historical_loader:
carpetas, formato de archivos, columnas/dtypes de una muestra y conteo de
archivos por carpeta.

Credenciales: NO se incluyen en el código. kagglehub lee el token
automáticamente desde ~/.kaggle/access_token (o kaggle.json). Si falta, la
descarga lanzará un error de autenticación explícito.

Idempotencia: kagglehub cachea la descarga en ~/.cache/kagglehub. Si el dataset
ya está, dataset_download() devuelve la ruta cacheada sin volver a bajarlo.

Uso:
    python -m src.data.download_stocktwits_historical
"""

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Identificador del dataset en Kaggle (owner/slug)
DATASET_ID = "frankcaoyun/stocktwits-2020-2022-raw"

# Extensiones que consideramos "archivos de datos"
_DATA_EXTENSIONS = {".csv", ".json", ".jsonl", ".ndjson", ".txt", ".parquet"}

# Cuántos archivos listar por carpeta en el reporte
_FILES_PER_DIR = 5


# ---------------------------------------------------------------------------
# Descarga
# ---------------------------------------------------------------------------

def download_dataset(dataset_id: str = DATASET_ID) -> Path:
    """
    Descarga el dataset con kagglehub y devuelve la ruta local.

    kagglehub cachea: si ya está descargado, devuelve la ruta sin re-bajar.

    Args:
        dataset_id: Identificador 'owner/slug' del dataset en Kaggle.

    Returns:
        Ruta local a la raíz del dataset descargado.
    """
    import kagglehub  # import diferido: solo se necesita al descargar

    logger.info("Descargando (o resolviendo de caché) el dataset '%s'...", dataset_id)
    local_path = Path(kagglehub.dataset_download(dataset_id))
    logger.info("Dataset disponible en: %s", local_path)
    return local_path


# ---------------------------------------------------------------------------
# Exploración de la estructura
# ---------------------------------------------------------------------------

def _detect_format(files: list[str]) -> dict[str, int]:
    """Cuenta archivos por extensión (en minúsculas)."""
    counts: dict[str, int] = {}
    for name in files:
        ext = Path(name).suffix.lower() or "(sin extensión)"
        counts[ext] = counts.get(ext, 0) + 1
    return counts


def _find_sample_data_file(root: Path) -> Path | None:
    """Devuelve el primer archivo con extensión de datos hallado bajo root."""
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in sorted(filenames):
            if Path(name).suffix.lower() in _DATA_EXTENSIONS:
                return Path(dirpath) / name
    return None


def _read_sample(sample_path: Path, nrows: int = 3) -> pd.DataFrame | None:
    """
    Lee las primeras filas de un archivo de muestra según su extensión.

    Soporta csv, json/jsonl/ndjson, parquet y txt (intentado como csv).
    Devuelve None si no se pudo parsear como tabla.
    """
    ext = sample_path.suffix.lower()
    try:
        if ext == ".csv" or ext == ".txt":
            return pd.read_csv(sample_path, nrows=nrows)
        if ext == ".parquet":
            return pd.read_parquet(sample_path).head(nrows)
        if ext in (".json",):
            # Puede ser un JSON array o un JSON Lines: probar ambos
            try:
                return pd.read_json(sample_path, lines=False).head(nrows)
            except ValueError:
                return pd.read_json(sample_path, lines=True, nrows=nrows)
        if ext in (".jsonl", ".ndjson"):
            return pd.read_json(sample_path, lines=True, nrows=nrows)
    except Exception as exc:  # noqa: BLE001 — la exploración no debe abortar
        logger.warning("No se pudo leer la muestra %s: %s", sample_path, exc)
        return None
    return None


def explore_dataset(root: Path) -> None:
    """
    Recorre el dataset e imprime un reporte de su estructura.

    Reporta:
      - Cada carpeta con sus primeros _FILES_PER_DIR archivos.
      - Conteo de archivos por carpeta y por extensión (formato detectado).
      - Columnas, dtypes y primeras 3 filas de un archivo de muestra.

    Args:
        root: Ruta raíz del dataset descargado.
    """
    print("\n" + "=" * 78)
    print(f"ESTRUCTURA DEL DATASET: {root}")
    print("=" * 78)

    total_files = 0
    total_dirs = 0
    global_ext_counts: dict[str, int] = {}

    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        filenames = sorted(filenames)
        total_dirs += 1
        total_files += len(filenames)

        rel = os.path.relpath(dirpath, root)
        rel_display = "." if rel == "." else rel

        print(f"\n[DIR] {rel_display}/  ({len(filenames)} archivos, {len(dirnames)} subcarpetas)")

        ext_counts = _detect_format(filenames)
        for ext, n in ext_counts.items():
            global_ext_counts[ext] = global_ext_counts.get(ext, 0) + n
        if ext_counts:
            print(f"      Formatos: {ext_counts}")

        for name in filenames[:_FILES_PER_DIR]:
            full = Path(dirpath) / name
            try:
                size_kb = full.stat().st_size / 1024
            except OSError:
                size_kb = 0.0
            print(f"        - {name}  ({size_kb:,.1f} KB)")
        if len(filenames) > _FILES_PER_DIR:
            print(f"        ... y {len(filenames) - _FILES_PER_DIR} archivo(s) más")

    print("\n" + "-" * 78)
    print(f"TOTAL: {total_files} archivos en {total_dirs} carpeta(s).")
    print(f"Formato(s) detectado(s) global(es): {global_ext_counts}")
    print("-" * 78)

    # --- Muestra de un archivo de datos ---
    sample = _find_sample_data_file(root)
    if sample is None:
        print("\nNo se encontró ningún archivo de datos reconocible para muestrear.")
        return

    print(f"\nARCHIVO DE MUESTRA: {os.path.relpath(sample, root)}")
    print("-" * 78)
    df = _read_sample(sample, nrows=3)
    if df is None:
        print("No se pudo parsear el archivo de muestra como tabla.")
        return

    print(f"Columnas ({len(df.columns)}): {list(df.columns)}")
    print("\nDtypes:")
    print(df.dtypes.to_string())
    print("\nPrimeras 3 filas:")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.head(3).to_string(index=False))


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

def main() -> Path:
    """Descarga el dataset, imprime la ruta local y explora su estructura."""
    local_path = download_dataset()
    print(f"\nRuta local devuelta por kagglehub.dataset_download():\n  {local_path}")
    explore_dataset(local_path)
    return local_path


if __name__ == "__main__":
    import sys

    # La consola de Windows usa cp1252; forzar UTF-8 para evitar UnicodeEncodeError.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
