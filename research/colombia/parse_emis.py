"""
Parsea exports de EMIS (archivos con extensión .doc que en realidad son HTML)
a un DataFrame de artículos.

Cada export trae ~67 artículos con la misma estructura repetida:

    h1 > span                 -> titular
    span hermano              -> fecha en español ("31 de julio de 2022")
    a > p                     -> fuente
    div.html-reader-body      -> cuerpo, en <p>

El anclaje es `h1` + `div.html-reader-body`. Las clases `sc-*` y `css-*` son
hashes de styled-components y cambian entre exports: no se usan.

La lectura de archivos vive aislada en `read_exports()`, para poder mover la
fuente a Drive o a Railway sin tocar el resto del módulo.

Uso:
    python -m research.colombia.parse_emis data/raw/emis/ECOPETROL --emisor ECOPETROL
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterator

import pandas as pd
from bs4 import BeautifulSoup, Tag

logger = logging.getLogger(__name__)

EXTENSIONS: tuple[str, ...] = (".doc", ".html", ".htm")

COLUMNS: list[str] = [
    "article_id", "emisor_busqueda", "fecha_publicacion", "titular", "fuente",
    "paginas", "cuerpo", "n_chars_cuerpo", "tipo_doc", "archivo", "anio_carpeta",
]

MONTHS: dict[str, int] = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
}

_DATE_RE = re.compile(r"(\d{1,2})\s+de\s+([a-zñ]+)\s+de\s+(\d{4})")
_PAGE_RE = re.compile(r"p(?:a|á)g", re.IGNORECASE)

# Clasificación por fuente. Se CLASIFICA, no se filtra: en el archivo de
# muestra el 24% de los documentos no era periodismo, y qué entra al corpus es
# una decisión abierta que se toma con el reporte de cobertura a la vista.
SOURCE_TO_TYPE: dict[str, str] = {
    "colombia-licitaciones": "licitacion",
    "superfin - relevant facts": "regulatorio",
    "bvc - news": "regulatorio",
}
DEFAULT_TYPE: str = "noticia"
DOC_TYPES: tuple[str, ...] = ("noticia", "licitacion", "regulatorio")


# ---------------------------------------------------------------------------
# Utilidades de texto
# ---------------------------------------------------------------------------

def _strip_accents(text: str) -> str:
    """Minúsculas sin acentos, para comparar fechas y fuentes."""
    return "".join(
        c for c in unicodedata.normalize("NFD", text.lower())
        if not unicodedata.combining(c)
    )


def _node_text(node: Tag | None) -> str:
    """Texto de un nodo con los espacios colapsados."""
    if node is None:
        return ""
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()


def parse_date(text: str) -> date | None:
    """
    Convierte una fecha en español a `date`.

    Args:
        text: Cadena que contiene la fecha ("31 de julio de 2022").

    Returns:
        La fecha, o None si no matchea el patrón o el día es inválido.
    """
    m = _DATE_RE.search(_strip_accents(text))
    if not m:
        return None
    day, month_name, year = m.groups()
    month = MONTHS.get(month_name)
    if month is None:
        return None
    try:
        return date(int(year), month, int(day))
    except ValueError:
        return None


def classify_doc_type(source: str) -> str:
    """
    Clasifica un documento por su fuente en {noticia, licitacion, regulatorio}.

    Args:
        source: Nombre de la fuente tal como viene en el export.

    Returns:
        El tipo de documento; `noticia` para todo lo que no esté mapeado.
    """
    return SOURCE_TO_TYPE.get(_strip_accents(source).strip(), DEFAULT_TYPE)


def article_id(headline: str, published: date | None, source: str) -> str:
    """
    ID determinístico de un artículo, para deduplicar entre exports con rango
    de fechas solapado.

    Args:
        headline: Titular del artículo.
        published: Fecha de publicación.
        source: Fuente del artículo.

    Returns:
        Los primeros 16 hex de un SHA-256 sobre (headline, fecha, fuente)
        normalizados.
    """
    key = f"{_strip_accents(headline)}|{published}|{_strip_accents(source)}"
    key = re.sub(r"\s+", " ", key).strip()
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Lectura de archivos (única función que toca el sistema de archivos)
# ---------------------------------------------------------------------------

def read_exports(path: Path) -> Iterator[tuple[str, str]]:
    """
    Lee los exports de EMIS de un archivo o directorio, recursivamente.

    Único punto de contacto con el almacenamiento: para leer desde Drive,
    Railway o un bucket, se reemplaza esta función y nada más. El resto del
    pipeline solo ve pares (rel_path, html).

    La ruta se devuelve RELATIVA a `path` y con separadores "/", porque es la
    que lleva la organización del repositorio de exports
    (`{EMISOR}/{AÑO}/archivo.doc`) y el orquestador la usa para saber de qué
    emisor y de qué año viene cada archivo. Devolver solo el nombre perdería
    esa información.

    Args:
        path: Archivo de export, o directorio que los contiene (recursivo).

    Yields:
        Tuplas (ruta_relativa_posix, html).

    Raises:
        FileNotFoundError: si la ruta no existe o el directorio no tiene exports.
    """
    if not path.exists():
        raise FileNotFoundError(f"No existe la ruta {path}")

    if path.is_file():
        yield path.name, path.read_text(encoding="utf-8", errors="replace")
        return

    files = sorted(
        p for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(
            f"No hay exports ({', '.join(EXTENSIONS)}) en {path}"
        )

    for file_path in files:
        yield file_path.relative_to(path).as_posix(), file_path.read_text(
            encoding="utf-8", errors="replace"
        )


# ---------------------------------------------------------------------------
# Parseo de un export
# ---------------------------------------------------------------------------

def _body_text(node: Tag) -> str:
    """
    Texto del cuerpo conservando la separación de párrafos con "\\n".

    Los saltos importan: la posición del párrafo donde aparece la primera
    mención de un emisor es una de las señales del corpus (ver emisores.py).

    Args:
        node: El `div.html-reader-body` del artículo.

    Returns:
        Los párrafos unidos por "\\n"; si no hay `<p>`, el texto plano del div.
    """
    paragraphs = [p for p in (_node_text(p) for p in node.find_all("p")) if p]
    if not paragraphs:
        flat = _node_text(node)
        return flat
    return "\n".join(paragraphs)


def _meta_block(h1: Tag, body_node: Tag | None) -> tuple[str, str, str]:
    """
    Extrae fecha, fuente y páginas de la metadata entre el h1 y el cuerpo.

    Recorre el documento en orden desde el h1 y se detiene al llegar al cuerpo
    del artículo (o al h1 siguiente), sin depender de clases ni de la
    profundidad del anidamiento.

    Args:
        h1: El `h1` del titular.
        body_node: El `div.html-reader-body` del mismo artículo, o None.

    Returns:
        Tupla (fecha_texto, fuente, páginas); cadenas vacías si no aparecen.
    """
    date_text, source, pages = "", "", ""

    for node in h1.next_elements:
        if not isinstance(node, Tag):
            continue
        if node is body_node or node.name == "h1":
            break

        if node.name == "a" and not source:
            source = _node_text(node)
            continue
        if node.name not in {"span", "p", "div", "time"}:
            continue

        text = _node_text(node)
        if not text:
            continue
        if not date_text and _DATE_RE.search(_strip_accents(text)):
            # El span de fecha suele ser la hoja; se queda con el más corto que
            # contenga la fecha para no arrastrar el bloque entero de metadata.
            date_text = text
        if not pages and _PAGE_RE.search(text):
            pages = text

    return date_text, source, pages


def parse_export(
    html: str,
    issuer_searched: str,
    file_name: str,
    folder_year: int | None = None,
) -> list[dict]:
    """
    Extrae todos los artículos de un export de EMIS.

    Args:
        html: Contenido del export.
        issuer_searched: Emisor cuya carpeta contenía el export. Es el origen
            de la búsqueda, NO necesariamente el sujeto del artículo. Cadena
            vacía para los documentos sueltos, que no vienen de ninguna carpeta.
        file_name: Ruta relativa del archivo, para trazabilidad.
        folder_year: Año de la carpeta que contenía el export, si el
            repositorio está partido por año. Se guarda sin usarlo para
            corregir nada: sirve para detectar archivos mal archivados
            comparándolo contra el año de `fecha_publicacion`.

    Returns:
        Lista de dicts con las columnas de COLUMNS.
    """
    soup = BeautifulSoup(html, "html.parser")
    headlines = soup.find_all("h1")
    bodies = soup.select("div.html-reader-body")

    if len(headlines) != len(bodies):
        logger.warning(
            "%s: %d titulares y %d cuerpos (no coinciden).",
            file_name, len(headlines), len(bodies),
        )

    rows: list[dict] = []
    for i, h1 in enumerate(headlines):
        body_node = bodies[i] if i < len(bodies) else None
        date_text, source, pages = _meta_block(h1, body_node)

        headline = _node_text(h1)
        published = parse_date(date_text)
        body = _body_text(body_node) if body_node is not None else ""

        rows.append({
            "article_id": article_id(headline, published, source),
            "emisor_busqueda": issuer_searched,
            "fecha_publicacion": published,
            "titular": headline,
            "fuente": source,
            "paginas": pages,
            "cuerpo": body,
            "n_chars_cuerpo": len(body),
            "tipo_doc": classify_doc_type(source),
            "archivo": file_name,
            "anio_carpeta": folder_year,
        })
    return rows


# ---------------------------------------------------------------------------
# Lectura del repositorio completo de exports
# ---------------------------------------------------------------------------

def locate(rel_path: str, issuers: list[str]) -> tuple[str, int | None]:
    """
    Deduce el emisor y el año de la ruta de un export dentro del repositorio.

    Los tres layouts que conviven en el repositorio de EMIS:

        ECOPETROL/2022/export_03.doc   → ("ECOPETROL", 2022)
        ECOPETROL/export_03.doc        → ("ECOPETROL", None)
        suelto.doc                     → ("", None)

    Un documento suelto NO es un error: son los exports que llegan fuera de la
    búsqueda sistemática (una nota puntual, algo que mandó un asesor). Entran al
    corpus con `emisor_busqueda` vacío y el emisor lo decide la detección por
    alias, que es lo correcto: no hay búsqueda de origen que contradecir.

    Args:
        rel_path: Ruta del export relativa a la raíz, con "/".
        issuers: Códigos de emisor conocidos.

    Returns:
        Tupla (emisor_busqueda, anio_carpeta). Cadena vacía y None cuando la
        ruta no los declara.
    """
    folders = rel_path.split("/")[:-1]  # se descarta el nombre del archivo
    if not folders:
        return "", None

    issuer = folders[0].upper()
    if issuer not in issuers:
        logger.warning(
            "Carpeta '%s' no corresponde a ningún emisor conocido (%s); "
            "se trata como documento suelto.", folders[0], rel_path,
        )
        return "", None

    year = next((int(c) for c in folders[1:] if c.isdigit() and len(c) == 4), None)
    return issuer, year


def load_articles(root: Path, issuers: list[str]) -> pd.DataFrame:
    """
    Parsea todos los exports bajo `root` y deduplica por article_id.

    Recorre la raíz en una sola pasada y deduce emisor y año de la ruta de cada
    archivo (ver `locate`), así que soporta a la vez el repositorio partido
    por `{EMISOR}/{AÑO}/`, el partido solo por emisor y los documentos sueltos.

    Args:
        root: Raíz del repositorio de exports de EMIS.
        issuers: Códigos de emisor conocidos.

    Returns:
        DataFrame de artículos únicos, con la columna extra `emisores_busqueda`
        (los emisores en cuyas carpetas apareció el artículo, separados por "|";
        vacía si solo apareció suelto).

    Raises:
        FileNotFoundError: si `root` no existe o no tiene exports.
    """
    rows: list[dict] = []
    by_origin: dict[str, int] = {}
    for rel_path, html in read_exports(root):
        issuer, year = locate(rel_path, issuers)
        extracted = parse_export(html, issuer, rel_path, year)
        logger.info("%s: %d artículos (issuer=%s, año=%s)",
                    rel_path, len(extracted), issuer or "suelto", year)
        origin = issuer or "(sueltos)"
        by_origin[origin] = by_origin.get(origin, 0) + len(extracted)
        rows.extend(extracted)

    logger.info("Exports leídos por origen: %s", by_origin)
    if not rows:
        return pd.DataFrame(columns=[*COLUMNS, "emisores_busqueda"])

    raw_rows = pd.DataFrame(rows)[COLUMNS]

    # Un mismo artículo puede venir en exports de varios emisores (rango
    # solapado o mención cruzada) y además suelto: se conserva una sola copia y
    # se registran todos los orígenes de búsqueda. Los sueltos aportan cadena
    # vacía y se descartan del conjunto: no declaran emisor de origen.
    origins = (
        raw_rows.groupby("article_id")["emisor_busqueda"]
        .agg(lambda s: "|".join(sorted({e for e in s if e})))
        .rename("emisores_busqueda")
    )
    # Al deduplicar gana la copia que SÍ viene de una carpeta de emisor, para no
    # perder el año de carpeta por quedarse con la versión suelta.
    articles = (
        raw_rows.assign(_de_carpeta=raw_rows["emisor_busqueda"] != "")
        .sort_values("_de_carpeta", ascending=False, kind="stable")
        .drop_duplicates(subset="article_id")
        .drop(columns="_de_carpeta")
        .sort_index()
        .merge(origins, on="article_id", how="left")
        .reset_index(drop=True)
    )
    logger.info(
        "%d artículos crudos → %d únicos (%d duplicados entre exports).",
        len(raw_rows), len(articles), len(raw_rows) - len(articles),
    )
    return articles


# ---------------------------------------------------------------------------
# Verificación de integridad del parseo
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParseSummary:
    """
    Controles de integridad del parseo.

    Attributes:
        articles: Artículos únicos parseados.
        no_date: Artículos sin fecha de publicación.
        no_body: Artículos con cuerpo vacío.
        by_doc_type: Artículos por tipo de documento; suma `articles`.
    """

    articles: int
    no_date: int
    no_body: int
    by_doc_type: dict[str, int]


def verify(df: pd.DataFrame) -> ParseSummary:
    """
    Chequea los invariantes del parseo y loguea lo que falle.

    Args:
        df: Salida de load_articles().

    Returns:
        El ParseSummary.
    """
    no_date = df[df["fecha_publicacion"].isna()]
    no_body = df[df["n_chars_cuerpo"] == 0]

    for headline in no_date["titular"].head(20):
        logger.warning("Sin fecha: %s", headline[:120])
    for headline in no_body["titular"].head(20):
        logger.warning("Sin cuerpo: %s", headline[:120])

    by_type = {str(k): int(v) for k, v in df["tipo_doc"].value_counts().items()}
    if sum(by_type.values()) != len(df):
        logger.error("tipo_doc no suma el total (%d vs %d).", sum(by_type.values()), len(df))

    return ParseSummary(
        articles=int(len(df)),
        no_date=int(len(no_date)),
        no_body=int(len(no_body)),
        by_doc_type=by_type,
    )


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

    # Usa exactamente la misma función que el pipeline (load_articles), para
    # que inspeccionar una carpeta a mano no pueda dar un resultado distinto del
    # que produce la ingesta real.
    from research.colombia.emisores import ISSUERS

    parser = argparse.ArgumentParser(
        description="Parsea un repositorio de exports de EMIS a un DataFrame."
    )
    parser.add_argument("ruta", type=Path, help="Archivo o raíz del repositorio de exports.")
    parser.add_argument("--out", type=Path, default=None, help="Parquet de salida (opcional).")
    args = parser.parse_args()

    articles = load_articles(args.ruta, ISSUERS)
    summary = verify(articles)

    print(f"\nArtículos: {summary.articles}")
    print(f"Sin fecha: {summary.no_date} | sin cuerpo: {summary.no_body}")
    print(f"Por tipo_doc: {summary.by_doc_type}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        articles.to_parquet(args.out, index=False)
        print(f"Guardado en {args.out}")
