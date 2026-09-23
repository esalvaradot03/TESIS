"""
Genera los lotes de etiquetado manual del corpus de prensa colombiana.

El muestreo, la partición y el reparto entre anotadores los hace
`research/corpus_partition.py`, que es la misma maquinaria que usa el corpus
del event study. Acá vive solo lo que es propio de este corpus:

  - la unidad de anotación es el par **(article_id, emisor)**;
  - las cuotas por emisor son **iguales**, no proporcionales: Ecopetrol tiene
    ~6x la cobertura del resto y un reparto proporcional daría un corpus de un
    solo emisor;
  - el filtro por `tipo_doc` es un flag, no un hardcode: qué entra al corpus
    (solo `noticia`, o también `regulatorio`) es una decisión abierta;
  - qué columnas ve el anotador, y el manifiesto.

Partición: L0 100 (compartido), doble 200 (compartido), test 300 (150/150),
train 400 (200/200). Los cuatro lotes son disjuntos y todos los pares de un
mismo `article_id` caen en el mismo lote, para que el mismo texto no quede a la
vez en train y en test.

Si el pool no alcanza para los 1000 pares, ABORTA: un corpus más chico generado
en silencio rompe el diseño pre-registrado. `--allow-partial` lo fuerza.

Uso:
    python -m research.colombia.build_corpus
    python -m research.colombia.build_corpus --tipos noticia regulatorio --unidad completo
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config.settings import DATA_DIR, SEED
from research.colombia.build_news_dataset import PARES_PATH
from research.colombia.emisores import ISSUERS
from research.corpus_partition import (CorpusSchema, annotator_seed,
                                       assign_to_annotators, batch_summary,
                                       partition, sample_stratified,
                                       scale_batches, sha256_file, sha256_pool)

logger = logging.getLogger(__name__)

# Esquema de este corpus para la maquinaria compartida: se anota un par
# (artículo, emisor); el artículo es lo que no puede partirse entre lotes; el
# emisor lleva las cuotas; el año estratifica.
SCHEMA = CorpusSchema(
    key=["article_id", "emisor"],
    group="article_id",
    klass="emisor",
    stratum="anio",
)

ANNOTATORS: list[str] = ["camilo", "esteban"]

TOTAL_PAIRS: int = 1000

# lote -> (tamaño, compartido entre ambos anotadores)
BATCHES: dict[str, tuple[int, bool]] = {
    "l0": (100, True),
    "doble": (200, True),
    "test": (300, False),
    "train": (400, False),
}

# Mínimo de pares por emisor dentro de cada lote de entrada (kappa).
BATCH_MIN_PER_ISSUER: dict[str, int] = {"l0": 10, "doble": 10}

OUTPUT_DIR: Path = DATA_DIR / "manual_labels_colombia"

DEFAULT_TYPES: list[str] = ["noticia"]

# Columnas del CSV que recibe el anotador; las 4 últimas van vacías.
OUTPUT_COLUMNS: list[str] = [
    "article_id", "emisor", "fecha_publicacion", "fuente", "titular", "cuerpo",
    "en_titular", "n_menciones", "label", "confianza", "base", "nota",
]

_BLANK: list[str] = ["label", "confianza", "base", "nota"]

# Columnas que definen el contenido del pool, para su hash de trazabilidad.
_HASH_COLUMNS: list[str] = ["article_id", "emisor", "fecha_publicacion", "titular"]


# ---------------------------------------------------------------------------
# Pool
# ---------------------------------------------------------------------------

def load_pool(
    path: Path = PARES_PATH,
    types: list[str] = DEFAULT_TYPES,
    unit: str = "titular",
) -> pd.DataFrame:
    """
    Carga los pares (article_id, emisor) elegibles para etiquetar.

    Args:
        path: Parquet de pares que escribe build_news_dataset.
        types: Valores de `tipo_doc` admitidos.
        unit: Unidad de anotación, `titular` o `completo`. Ambas columnas
            quedan en el CSV; esto solo decide cuál es obligatoria y no vacía.

    Returns:
        DataFrame con las columnas de salida más `anio`, sin fechas nulas.

    Raises:
        FileNotFoundError: si no existe el parquet de pares.
        ValueError: si `unit` no es válida.
    """
    if unit not in {"titular", "completo"}:
        raise ValueError(f"unidad debe ser 'titular' o 'completo', no {unit!r}")
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Corré primero: python -m research.colombia.build_news_dataset"
        )

    pairs = pd.read_parquet(path)
    before = len(pairs)
    pool = pairs[pairs["tipo_doc"].isin(types)].copy()
    pool = pool[pool["fecha_publicacion"].notna()]
    pool = pool[pool["titular"].str.strip() != ""]
    if unit == "completo":
        pool = pool[pool["cuerpo"].str.strip() != ""]

    pool["anio"] = pd.to_datetime(pool["fecha_publicacion"]).dt.year
    pool = pool.reset_index(drop=True)

    logger.info(
        "Pool: %d de %d pares (tipos=%s, unidad=%s) | emisores: %s | años: %s",
        len(pool), before, types, unit,
        pool["emisor"].value_counts().to_dict(),
        sorted(pool["anio"].unique().tolist()),
    )
    return pool


# ---------------------------------------------------------------------------
# Escritura de los lotes
# ---------------------------------------------------------------------------

def write_batches(
    assignments: dict[tuple[str, str], pd.DataFrame],
    output_dir: Path = OUTPUT_DIR,
    seed: int = SEED,
) -> dict[str, Path]:
    """
    Escribe un CSV por (lote, anotador) con las columnas de anotación vacías.

    Los lotes compartidos llevan los MISMOS pares en orden DISTINTO para cada
    anotador (semilla derivada por lote y anotador): reduce el sesgo de orden
    sin romper el merge del kappa, cuya llave es (article_id, emisor).

    Args:
        assignments: Salida de assign_to_annotators().
        output_dir: Directorio de salida.
        seed: Semilla base.

    Returns:
        Dict nombre_archivo → ruta escrita.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for (batch, annotator), frame in sorted(assignments.items()):
        shuffled = frame.sample(frac=1, random_state=annotator_seed(seed, batch, annotator))
        out = shuffled.reindex(columns=OUTPUT_COLUMNS)
        out[_BLANK] = ""

        path = output_dir / f"{batch}_{annotator}.csv"
        out.to_csv(path, index=False, encoding="utf-8")
        written[path.name] = path
        logger.info("Lote %s/%s: %d filas → %s", batch, annotator, len(out), path)
    return written


def write_manifest(
    pool: pd.DataFrame,
    sample: pd.DataFrame,
    assignments: dict[tuple[str, str], pd.DataFrame],
    written: dict[str, Path],
    definitions: dict[str, tuple[int, bool]] = BATCHES,
    types: list[str] = DEFAULT_TYPES,
    unit: str = "titular",
    seed: int = SEED,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """
    Escribe manifest.json con la trazabilidad del muestreo (pre-registro).

    Las secciones comunes a cualquier corpus (qué lote recibió cada anotador y
    con qué llaves) vienen de `batch_summary`; acá se agrega lo propio de este
    corpus: el filtro de tipo_doc, la unidad de anotación y el reparto por
    emisor y año.

    Args:
        pool: Pool del que se muestreó.
        sample: Pares muestreados.
        assignments: Reparto (lote, anotador) → filas.
        written: Archivos escritos, para hashear.
        definitions: Definición de lotes efectivamente usada.
        types: Filtro de tipo_doc aplicado.
        unit: Unidad de anotación elegida.
        seed: Semilla usada.
        output_dir: Directorio de salida.

    Returns:
        Ruta al manifest.json.
    """
    by_issuer_year: dict[str, dict[str, int]] = {}
    for (emisor, anio), n in sample.groupby(["emisor", "anio"]).size().items():
        by_issuer_year.setdefault(str(emisor), {})[str(anio)] = int(n)

    manifest = {
        "generado": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "unidad_de_muestreo": "(article_id, emisor)",
        "unidad_de_anotacion": unit,
        "tipos_doc_incluidos": types,
        "reparto_por_emisor": "cuotas iguales (los faltantes se reportan, no se rellenan)",
        "estratificacion": "por año dentro de cada emisor",
        "particion_agrupada_por_articulo": True,
        "total_pares": int(len(sample)),
        "pool_pares": int(len(pool)),
        "pool_sha256": sha256_pool(pool, _HASH_COLUMNS, SCHEMA.key),
        "pool_por_emisor": {
            str(e): int(n) for e, n in pool["emisor"].value_counts().sort_index().items()
        },
        **batch_summary(assignments, definitions, SCHEMA, ANNOTATORS),
        "por_emisor": {
            str(e): int(n) for e, n in sample["emisor"].value_counts().sort_index().items()
        },
        "por_emisor_y_anio": by_issuer_year,
        "archivos": {
            name: {
                "filas": int(len(pd.read_csv(path, dtype=str))),
                "sha256": sha256_file(path),
            }
            for name, path in sorted(written.items())
        },
    }

    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Manifiesto escrito en %s", path)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Genera los lotes de etiquetado del corpus de prensa colombiana."
    )
    parser.add_argument("--pares", type=Path, default=PARES_PATH, help="Parquet de pares.")
    parser.add_argument("--total", type=int, default=TOTAL_PAIRS, help="Pares a muestrear.")
    parser.add_argument("--tipos", nargs="+", default=DEFAULT_TYPES,
                        help="tipo_doc admitidos (default: noticia).")
    parser.add_argument("--unidad", choices=["titular", "completo"], default="titular",
                        help="Unidad de anotación; ambas columnas van igual en el CSV.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directorio de salida.")
    parser.add_argument("--seed", type=int, default=SEED, help="Semilla (default: 42).")
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Permite lotes proporcionalmente más chicos si el pool no alcanza "
             "(por defecto aborta, para no romper el diseño pre-registrado).",
    )
    args = parser.parse_args()

    pool = load_pool(args.pares, types=args.tipos, unit=args.unidad)
    if pool.empty:
        print(f"Pool vacío con tipos={args.tipos}. No se generó ningún lote.")
        sys.exit(1)

    sample = sample_stratified(pool, SCHEMA, args.total, ISSUERS, seed=args.seed)

    definitions = BATCHES
    if len(sample) < args.total:
        message = f"El pool solo permitió muestrear {len(sample)} de {args.total} pares pedidos."
        if not args.allow_partial:
            print(f"{message} Abortado para no romper el diseño pre-registrado. "
                  "Usá --allow-partial si querés lotes más chicos igual.")
            sys.exit(1)
        definitions = scale_batches(BATCHES, len(sample), args.total)
        logger.warning("%s Se generan lotes proporcionales: %s", message,
                       {k: v[0] for k, v in definitions.items()})

    batches_df = partition(sample, SCHEMA, definitions, BATCH_MIN_PER_ISSUER, ISSUERS)
    assignments = assign_to_annotators(batches_df, definitions, ANNOTATORS)
    written = write_batches(assignments, args.output_dir, seed=args.seed)
    manifest = write_manifest(
        pool, sample, assignments, written, definitions,
        types=args.tipos, unit=args.unidad, seed=args.seed, output_dir=args.output_dir,
    )

    print(f"\n{len(written)} archivos escritos en {args.output_dir}")
    for name in sorted(written):
        print(f"  {name}")
    print(f"Manifiesto: {manifest}")
