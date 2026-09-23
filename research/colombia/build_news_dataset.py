"""
Ingesta de la prensa colombiana: de exports de EMIS a pares (artículo, emisor).

Orquesta tres pasos, cada uno en su módulo:

    parse_emis.load_articles      exports .doc  → artículos únicos
    emisores.detect_issuers       artículo      → menciones de emisores
    reporte_cobertura             todo lo anterior → reporte_cobertura.md

Escribe:
    <EMIS_INTERIM_DIR>/articulos_colombia.parquet   (uno por artículo)
    <EMIS_INTERIM_DIR>/pares_colombia.parquet       (uno por par artículo-emisor)
    research/colombia/reporte_cobertura.md

Deliberadamente NO se implementa (decisiones abiertas; el reporte solo mide su
tamaño):
  - el mapeo de noticia a día de trading (fines de semana y festivos),
  - la deduplicación por hecho entre fuentes distintas,
  - el filtro de mención de paso vs. sujeto principal — pero `en_titular`,
    `n_menciones` y `posicion_primera_mencion` quedan guardados para poder
    decidir la regla después.

Uso:
    python -m research.colombia.build_news_dataset
    python -m research.colombia.build_news_dataset --raiz data/raw/emis
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config.settings import EMIS_INTERIM_DIR, EMIS_RAW_ROOT, ROOT_DIR, SEED
from research.colombia.emisores import ISSUERS, detect_issuers
from research.colombia.parse_emis import load_articles, verify
from research.colombia.reporte_cobertura import build_report

logger = logging.getLogger(__name__)

ARTICLES_FILE = "articulos_colombia.parquet"
PAIRS_FILE = "pares_colombia.parquet"

# Ruta por defecto del parquet de pares; build_corpus la usa como default de CLI.
PARES_PATH: Path = EMIS_INTERIM_DIR / PAIRS_FILE
REPORTE_PATH: Path = ROOT_DIR / "research" / "colombia" / "reporte_cobertura.md"

PAIR_COLUMNS: list[str] = [
    "article_id", "emisor", "emisores_buscados", "cruzada", "fecha_publicacion",
    "fuente", "tipo_doc", "titular", "cuerpo", "alias_matcheado", "nivel_alias",
    "es_descriptivo", "en_titular", "n_menciones_titular", "n_menciones_cuerpo",
    "n_menciones", "posicion_primera_mencion", "fragmento", "archivo",
]


# ---------------------------------------------------------------------------
# Detección de emisores y expansión a pares
# ---------------------------------------------------------------------------

def expand_pairs(articles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica la detección de emisores y expande a una fila por (artículo, emisor).

    Un artículo que nombra a Ecopetrol por dos alias distintos (p.ej.
    "Ecopetrol" y "Reficar") produce UN par, con el alias más mencionado como
    `alias_matcheado` y los conteos sumados; el detalle por alias se conserva
    para la auditoría.

    Cada fila se marca acá como `cruzada`: el artículo declaraba emisores de
    búsqueda y este no es ninguno de ellos. Se resuelve en este punto, donde el
    artículo y su procedencia están a la vista, para que la auditoría sea un
    filtro y no tenga que reconstruir la correspondencia. Un documento suelto no
    declara emisores, así que nunca es cruzado: no hay búsqueda que contradecir.

    Args:
        articles: Salida de load_articles().

    Returns:
        (pairs, detail): `pairs` tiene una fila por (artículo, emisor) con las
        columnas de PAIR_COLUMNS; `detail` una fila por (artículo, emisor,
        alias), que es lo que audita el reporte.
    """
    rows: list[dict] = []
    for row in articles.itertuples(index=False):
        searched = {e for e in str(row.emisores_busqueda).split("|") if e}
        for match in detect_issuers(row.titular, row.cuerpo):
            rows.append({
                **match.to_dict(),
                "article_id": row.article_id,
                "emisores_buscados": "|".join(sorted(searched)),
                "cruzada": bool(searched) and match.emisor not in searched,
                "fecha_publicacion": row.fecha_publicacion,
                "fuente": row.fuente,
                "tipo_doc": row.tipo_doc,
                "titular": row.titular,
                "cuerpo": row.cuerpo,
                "archivo": row.archivo,
            })

    if not rows:
        empty = pd.DataFrame(columns=PAIR_COLUMNS)
        return empty, empty

    detail = pd.DataFrame(rows)
    pairs = _collapse_by_issuer(detail)
    logger.info(
        "%d artículos → %d matches de alias → %d pares (artículo, emisor).",
        len(articles), len(detail), len(pairs),
    )
    return pairs, detail


def _first_position(series: pd.Series) -> int:
    """Índice de párrafo más temprano, ignorando los -1 (solo en titular)."""
    valid = series[series >= 0]
    return int(valid.min()) if not valid.empty else -1


def _collapse_by_issuer(detail: pd.DataFrame) -> pd.DataFrame:
    """
    Colapsa los matches de varios alias del mismo emisor en un solo par.

    Se queda con el alias de más menciones (desempate: el de menor nivel, o sea
    el nombre propio antes que el descriptivo) y suma los conteos.

    Args:
        detail: Una fila por (artículo, emisor, alias).

    Returns:
        Una fila por (artículo, emisor), con las columnas de PAIR_COLUMNS.
    """
    main = (
        detail.sort_values(
            ["article_id", "emisor", "n_menciones", "nivel_alias"],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(["article_id", "emisor"])
        .set_index(["article_id", "emisor"])
    )
    aggregated = detail.groupby(["article_id", "emisor"]).agg(
        n_menciones_titular=("n_menciones_titular", "sum"),
        n_menciones_cuerpo=("n_menciones_cuerpo", "sum"),
        n_menciones=("n_menciones", "sum"),
        en_titular=("en_titular", "any"),
        posicion_primera_mencion=("posicion_primera_mencion", _first_position),
    )
    pairs = main.drop(columns=aggregated.columns).join(aggregated).reset_index()
    return pairs[PAIR_COLUMNS]


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

def build(
    root: Path = EMIS_RAW_ROOT,
    interim_dir: Path = EMIS_INTERIM_DIR,
    reporte_path: Path = REPORTE_PATH,
    seed: int = SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Corre la ingesta completa y escribe los parquets y el reporte.

    Args:
        root: Raíz del repositorio de exports de EMIS.
        interim_dir: Destino de los parquets.
        reporte_path: Destino del reporte markdown.
        seed: Semilla para los ejemplos del reporte.

    Returns:
        (articles, pairs).
    """
    articles = load_articles(root, ISSUERS)
    summary = verify(articles)
    pairs, detail = expand_pairs(articles)

    interim_dir.mkdir(parents=True, exist_ok=True)
    articles.to_parquet(interim_dir / ARTICLES_FILE, index=False)
    pairs.to_parquet(interim_dir / PAIRS_FILE, index=False)
    logger.info("Parquets escritos en %s", interim_dir)

    reporte_path.parent.mkdir(parents=True, exist_ok=True)
    reporte_path.write_text(
        build_report(articles, pairs, detail, summary, seed=seed), encoding="utf-8"
    )
    logger.info("Reporte escrito en %s", reporte_path)
    return articles, pairs


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
        description="Ingesta de prensa colombiana (EMIS) + reporte de cobertura."
    )
    parser.add_argument("--raiz", type=Path, default=EMIS_RAW_ROOT,
                        help="Raíz del repositorio de exports.")
    parser.add_argument("--interim-dir", type=Path, default=EMIS_INTERIM_DIR,
                        help="Destino de los parquets.")
    parser.add_argument("--seed", type=int, default=SEED, help="Semilla (default: 42).")
    args = parser.parse_args()

    articles, pairs = build(args.raiz, args.interim_dir, seed=args.seed)
    print(f"\nArtículos: {len(articles)} | pares (artículo, emisor): {len(pairs)}")
    print(f"  {args.interim_dir / ARTICLES_FILE}")
    print(f"  {args.interim_dir / PAIRS_FILE}")
    print(f"  {REPORTE_PATH}")
