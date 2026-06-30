"""
Utilidades compartidas por los jobs del pipeline de larga duración.

Diseño orientado a ROBUSTEZ (correr 4 días sin supervisión):
  - Logging por job a D:\\trading-data\\logs\\<job>.log con timestamp por mensaje.
  - Checkpoints JSON atómicos (write-temp + rename) por unidad procesada
    (archivo o chunk); si el job crashea, al re-correr salta lo ya hecho.
  - Lectura tolerante de checkpoints corruptos (si no parsea, empieza de cero).
  - Helpers de Parquet, normalización de tickers y conjunto S&P 500.

NOTA de diseño: priorizamos resumibilidad por archivo/chunk (pandas en streaming)
sobre "todo con Dask", porque per-file checkpointing es lo más resistente a
crashes. Dask se usa para la CONSOLIDACIÓN out-of-core (groupby final), donde
aporta sin comprometer la recuperación ante fallos.
"""

import json
import logging
import os
import sys
import tempfile
from datetime import date
from pathlib import Path

from config.settings import EXTERNAL_DATA_ROOT

# --- Rutas ---
DATA_ROOT: Path = EXTERNAL_DATA_ROOT
MASTER_DIR: Path = DATA_ROOT / "master"
STAGING_DIR: Path = DATA_ROOT / "staging"
LOGS_DIR: Path = DATA_ROOT / "logs"
CKPT_DIR: Path = LOGS_DIR / "checkpoints"
PRICES_DIR: Path = MASTER_DIR / "prices"

# --- Rango objetivo ---
DATE_START: date = date(2008, 1, 1)
DATE_END: date = date(2024, 12, 31)


def ensure_dirs() -> None:
    for d in (MASTER_DIR, STAGING_DIR, LOGS_DIR, CKPT_DIR, PRICES_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(job_name: str) -> logging.Logger:
    """Logger que escribe a archivo (D:\\...\\logs\\<job>.log) y a consola, en UTF-8."""
    ensure_dirs()
    logger = logging.getLogger(f"longrun.{job_name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(LOGS_DIR / f"{job_name}.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# ---------------------------------------------------------------------------
# Checkpoint (JSON atómico)
# ---------------------------------------------------------------------------

class Checkpoint:
    """
    Registro persistente de unidades ya procesadas (archivos/chunks).

    Guarda un set de claves en un JSON. Escritura atómica (temp + replace) para
    no corromperse si hay corte de luz a mitad de escritura.
    """

    def __init__(self, job_name: str) -> None:
        ensure_dirs()
        self.path = CKPT_DIR / f"{job_name}.json"
        self._done: set[str] = set()
        self._meta: dict = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._done = set(data.get("done", []))
            self._meta = data.get("meta", {})
        except Exception:  # noqa: BLE001 — checkpoint corrupto: empezar de cero
            self._done = set()
            self._meta = {}

    def is_done(self, key: str) -> bool:
        return key in self._done

    def mark(self, key: str) -> None:
        self._done.add(key)
        self._flush()

    def set_meta(self, **kwargs) -> None:
        self._meta.update(kwargs)
        self._flush()

    @property
    def meta(self) -> dict:
        return dict(self._meta)

    def count(self) -> int:
        return len(self._done)

    def _flush(self) -> None:
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"done": sorted(self._done), "meta": self._meta}),
            encoding="utf-8",
        )
        os.replace(tmp, self.path)


# ---------------------------------------------------------------------------
# Tickers S&P 500
# ---------------------------------------------------------------------------

_SP500_CACHE: set[str] | None = None


def normalize_ticker(t: object) -> str:
    """Normaliza un ticker: mayúsculas, sin espacios, '.'→'-' (BRK.B→BRK-B)."""
    return str(t).strip().upper().replace(".", "-")


def sp500_set() -> set[str]:
    """
    Conjunto de tickers S&P 500 para filtrar (normalizado).

    LIMITACIÓN documentada: usa la lista ACTUAL del S&P 500 (sp500_loader, caché
    de Wikipedia). No incluye miembros históricos que salieron del índice entre
    2008-2024, por lo que esas menciones se descartan. Suficiente para acotar el
    volumen; ampliable con una lista histórica si se consigue.
    """
    global _SP500_CACHE
    if _SP500_CACHE is None:
        from src.universe.sp500_loader import get_tickers
        _SP500_CACHE = {normalize_ticker(t) for t in get_tickers()}
    return _SP500_CACHE


# ---------------------------------------------------------------------------
# Parquet seguro
# ---------------------------------------------------------------------------

def safe_to_parquet(df, path: Path, **kwargs) -> None:
    """Escribe Parquet de forma atómica (temp + replace) para no dejar archivos a medias."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False, **kwargs)
    os.replace(tmp, path)


def to_trading_date(series):
    """Convierte una columna a date (objeto date), tolerante a errores → NaT."""
    import pandas as pd
    return pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(None).dt.date
