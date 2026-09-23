"""
Análisis y reporte del corpus de prensa colombiana.

Responde dos preguntas que se miran juntas pero son distintas:

  1. **Cobertura**: ¿alcanza la densidad de prensa para trabajar a frecuencia
     diaria, o hay que agregar a semanal? Se mide contra el calendario real de
     la BVC (~246 días/año), no contra la fracción 5/7, que sobreestima el
     denominador en ~7%.
  2. **Auditoría**: ¿el diccionario de alias y el repositorio de exports están
     bien calibrados? Un alias con falsos positivos contamina todo el corpus y
     después no se detecta.

Cada sección del reporte es una función `(análisis) -> list[str]`: el análisis
se calcula una vez, en un objeto tipado, y el renderizado no hace cuentas. Eso
mantiene separada la medición de su presentación, y permite consumir las
métricas desde otro lado (el dashboard) sin pasar por el markdown.

La ingesta que produce la entrada de este módulo vive en `build_news_dataset`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Iterable

import numpy as np
import pandas as pd

from config.settings import SEED
from research.colombia.calendario_bvc import is_trading_day, trading_days
from research.colombia.parse_emis import DOC_TYPES, ParseSummary

logger = logging.getLogger(__name__)

# Ejemplos de match por alias descriptivo que se listan para revisión manual.
N_DESCRIPTIVE_EXAMPLES: int = 20


# ---------------------------------------------------------------------------
# Cobertura temporal
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Coverage:
    """
    Cobertura temporal de un emisor.

    Attributes:
        issuer: Código del emisor.
        articles: Artículos únicos del emisor.
        by_doc_type: Artículos por tipo de documento.
        start: Primera fecha de publicación.
        end: Última fecha de publicación.
        by_year: Artículos por año.
        top_sources: Las 10 fuentes más frecuentes.
        n_trading_days: Días hábiles de la BVC en el rango.
        covered_days: Días hábiles con al menos un artículo.
        gap_median: Mediana del largo de los gaps, en días hábiles.
        gap_p90: Percentil 90 del largo de los gaps.
        gap_max: Hueco más largo.
        n_gaps: Cantidad de gaps.
        non_trading_articles: Artículos publicados en end de semana o festivo.
        duplication_days: Días con 3+ artículos de 3+ fuentes distintas.
    """

    issuer: str
    articles: int
    by_doc_type: dict[str, int]
    start: date
    end: date
    by_year: dict[int, int]
    top_sources: dict[str, int]
    n_trading_days: int
    covered_days: int
    gap_median: float
    gap_p90: float
    gap_max: int
    n_gaps: int
    non_trading_articles: int
    duplication_days: int

    @property
    def pct_coverage(self) -> float:
        """Fracción de días de trading con al menos una noticia."""
        return self.covered_days / self.n_trading_days if self.n_trading_days else float("nan")


def _gaps(with_news: set[date], business_days: list[date]) -> np.ndarray:
    """
    Largo de cada racha de días hábiles consecutivos sin noticia.

    Args:
        with_news: Días con al menos un artículo.
        business_days: Calendario de días de trading del rango.

    Returns:
        Array con el largo de cada hueco. Vacío si no hay gaps.
    """
    gaps: list[int] = []
    current = 0
    for day in business_days:
        if day in with_news:
            if current:
                gaps.append(current)
            current = 0
        else:
            current += 1
    if current:
        gaps.append(current)
    return np.array(gaps, dtype=int)


def coverage_by_issuer(issuer: str, issuer_pairs: pd.DataFrame) -> Coverage:
    """
    Calcula la cobertura temporal de un emisor.

    Args:
        issuer: Código del emisor.
        issuer_pairs: Pares de ese issuer, con fecha_publicacion no nula.

    Returns:
        La Coverage del emisor.
    """
    dates = pd.to_datetime(issuer_pairs["fecha_publicacion"]).dt.date
    start, end = dates.min(), dates.max()
    business_days = trading_days(start, end)
    with_news = set(dates)
    gaps = _gaps(with_news, business_days)

    by_day = issuer_pairs.assign(_dia=dates).groupby("_dia").agg(
        n_articulos=("article_id", "nunique"),
        n_fuentes=("fuente", "nunique"),
    )
    duplication = by_day[(by_day["n_articulos"] >= 3) & (by_day["n_fuentes"] >= 3)]

    return Coverage(
        issuer=issuer,
        articles=int(issuer_pairs["article_id"].nunique()),
        by_doc_type={t: int((issuer_pairs["tipo_doc"] == t).sum()) for t in DOC_TYPES},
        start=start,
        end=end,
        by_year={
            int(a): int(n) for a, n in
            pd.Series([d.year for d in dates]).value_counts().sort_index().items()
        },
        top_sources=issuer_pairs["fuente"].value_counts().head(10).to_dict(),
        n_trading_days=len(business_days),
        covered_days=sum(1 for d in business_days if d in with_news),
        gap_median=float(np.median(gaps)) if gaps.size else 0.0,
        gap_p90=float(np.percentile(gaps, 90)) if gaps.size else 0.0,
        gap_max=int(gaps.max()) if gaps.size else 0,
        n_gaps=int(gaps.size),
        non_trading_articles=sum(1 for d in dates if not is_trading_day(d)),
        duplication_days=int(len(duplication)),
    )


# ---------------------------------------------------------------------------
# Auditoría del repositorio de exports
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RepositoryAudit:
    """
    Cómo vinieron archivados los exports, independientemente de su contenido.

    Attributes:
        loose: Artículos que no vinieron de ninguna carpeta de issuer.
        misfiled: Artículos cuyo año de carpeta ≠ año de publicación.
        with_folder_year: Artículos que sí vinieron en una carpeta con año.
        by_folder: Conteo por (emisor_busqueda, anio_carpeta).
    """

    loose: pd.DataFrame
    misfiled: pd.DataFrame
    with_folder_year: int
    by_folder: pd.DataFrame


def repository_audit(articles: pd.DataFrame) -> RepositoryAudit:
    """
    Audita la organización del repositorio de exports.

    Dos cosas que solo se ven acá, y que en Railway importan más que en local
    porque allá los `.doc` los sube otra persona a Drive y nadie los mira:

      - **Documentos loose**: exports fuera de toda carpeta de issuer. Entran
        igual, pero sin issuer de búsqueda contra el cual contrastar la
        detección por alias.
      - **Archivos mal archivados**: el año de la carpeta no coincide con el de
        `fecha_publicacion`. No se corrige nada (la fecha parseada manda), pero
        un conteo alto significa que la partición por año no es confiable y no
        sirve para filtrar sin parsear.

    Args:
        articles: Salida de load_articles().

    Returns:
        El RepositoryAudit.
    """
    with_year = articles[
        articles["anio_carpeta"].notna() & articles["fecha_publicacion"].notna()
    ]
    misfiled = with_year[
        with_year["anio_carpeta"].astype(int)
        != pd.to_datetime(with_year["fecha_publicacion"]).dt.year
    ]
    return RepositoryAudit(
        loose=articles[articles["emisores_busqueda"] == ""],
        misfiled=misfiled,
        with_folder_year=int(len(with_year)),
        by_folder=(
            articles.groupby(["emisor_busqueda", "anio_carpeta"], dropna=False)
            .size().rename("articulos").reset_index()
            .sort_values(["emisor_busqueda", "anio_carpeta"])
        ),
    )


# ---------------------------------------------------------------------------
# Auditoría del diccionario de alias
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AliasAudit:
    """
    Qué matcheó realmente el diccionario de alias.

    Attributes:
        by_alias: Matches por (issuer, alias), ordenados.
        total_matches: Matches de alias en todo el corpus.
        descriptive: Matches que vinieron de un alias descriptivo.
        cross_mentions: Pares cuyo issuer no es ninguno de los buscados.
        multi_issuer: Artículos con 2+ emisores detectados.
        undetected: Artículos sin ningún issuer detectado.
        descriptive_examples: Muestra de matches descriptive para revisión.
    """

    by_alias: pd.DataFrame
    total_matches: int
    descriptive: pd.DataFrame
    cross_mentions: pd.DataFrame
    multi_issuer: pd.Series
    undetected: pd.DataFrame
    descriptive_examples: pd.DataFrame

    @property
    def pct_descriptive(self) -> float:
        """Porcentaje de matches que vinieron de un alias descriptivo."""
        return len(self.descriptive) / self.total_matches * 100 if self.total_matches else 0.0


def alias_audit(
    detail: pd.DataFrame,
    pairs: pd.DataFrame,
    articles: pd.DataFrame,
    seed: int = SEED,
) -> AliasAudit:
    """
    Audita el diccionario de alias contra lo que efectivamente matcheó.

    Args:
        detail: Una fila por (artículo, issuer, alias).
        pairs: Una fila por (artículo, issuer); trae la marca `cruzada`.
        articles: Todos los artículos parseados.
        seed: Semilla para los ejemplos aleatorios (SEED=42).

    Returns:
        El AliasAudit.
    """
    descriptive = detail[detail["es_descriptivo"]]
    per_article = pairs.groupby("article_id")["emisor"].nunique()

    return AliasAudit(
        by_alias=(
            detail.groupby(["emisor", "alias_matcheado", "nivel_alias", "es_descriptivo"])
            .size().rename("matches").reset_index()
            .sort_values("matches", ascending=False)
        ),
        total_matches=int(len(detail)),
        descriptive=descriptive,
        cross_mentions=pairs[pairs["cruzada"]],
        multi_issuer=per_article[per_article >= 2],
        undetected=articles[~articles["article_id"].isin(set(pairs["article_id"]))],
        descriptive_examples=descriptive.sample(
            n=min(N_DESCRIPTIVE_EXAMPLES, len(descriptive)), random_state=seed
        )[["emisor", "alias_matcheado", "fuente", "titular", "fragmento"]],
    )


# ---------------------------------------------------------------------------
# Renderizado: cada sección es (análisis) -> list[str], sin hacer cuentas
# ---------------------------------------------------------------------------

def _cell(value: object) -> str:
    """Celda markdown: escapa el pipe, aplana saltos y trunca."""
    return str(value).replace("|", "\\|").replace("\n", " ")[:120]


def _md_table(headers: list[str], rows: Iterable[Iterable[object]]) -> list[str]:
    """
    Tabla markdown a partir de encabezados y filas.

    Args:
        headers: Encabezados de columna.
        rows: Filas, cada una iterable de celdas.

    Returns:
        Líneas de markdown, con una línea en blanco al final.
    """
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(_cell(v) for v in row) + " |" for row in rows]
    lines.append("")
    return lines


def _table(df: pd.DataFrame, max_rows: int = 25) -> list[str]:
    """
    Renderiza un DataFrame como tabla markdown, truncando filas y celdas.

    Args:
        df: Datos a mostrar.
        max_rows: Filas a incluir antes de cortar.

    Returns:
        Líneas de markdown, con una línea en blanco al final.
    """
    if df.empty:
        return ["_(ninguno)_", ""]
    head = df.head(max_rows)
    lines = _md_table([str(c) for c in head.columns], head.itertuples(index=False))
    if len(df) > max_rows:
        lines.insert(-1, f"| _… {len(df) - max_rows} filas más_ |" + " |" * (len(head.columns) - 1))
    return lines


def _section_parsing(summary: ParseSummary, n_pares: int) -> list[str]:
    """Sección con los controles de integridad del parseo."""
    lines = [
        "## Parseo",
        "",
        *_md_table(["Control", "Valor"], [
            ("Artículos únicos", summary.articles),
            ("Sin fecha de publicación", summary.no_date),
            ("Sin cuerpo", summary.no_body),
            ("Pares (artículo, emisor)", n_pares),
        ]),
        "Desglose por `tipo_doc` (clasificado, **no filtrado** — qué entra al "
        "corpus es decisión pendiente):",
        "",
        *_md_table(["tipo_doc", "artículos", "%"], [
            (tipo, n, f"{n / summary.articles * 100 if summary.articles else 0:.1f}%")
            for tipo, n in summary.by_doc_type.items()
        ]),
    ]
    return lines


def _section_repository(repo: RepositoryAudit) -> list[str]:
    """Sección con la organización del repositorio de exports."""
    lines = [
        "## Organización del repositorio de exports",
        "",
        "Cómo vinieron archivados los `.doc`, independientemente de su contenido. "
        "Importa porque en Railway los sube otra persona a Drive y nadie los revisa "
        "a mano.",
        "",
        f"- Documentos sueltos (fuera de toda carpeta de emisor): "
        f"**{len(repo.loose)}** — entran al corpus, pero sin emisor de búsqueda "
        "contra el cual contrastar la detección por alias.",
        f"- Artículos en carpetas con año: **{repo.with_folder_year}**, de los cuales "
        f"**{len(repo.misfiled)}** tienen el año de carpeta distinto del año de "
        "publicación.",
        "",
        "Si los desalineados no son ~0, la partición por año del repositorio **no "
        "sirve para filtrar sin parsear**: manda siempre la fecha parseada del "
        "artículo, que es la que usa todo el pipeline.",
        "",
        "Artículos por carpeta (`emisor_busqueda` vacío = sueltos):",
        "",
    ]
    lines += _table(repo.by_folder, max_rows=60)
    if len(repo.misfiled):
        lines += ["Ejemplos de archivos mal archivados:", ""]
        lines += _table(
            repo.misfiled[["archivo", "anio_carpeta", "fecha_publicacion", "titular"]],
            max_rows=15,
        )
    return lines


def _section_coverage(coverages: list[Coverage]) -> list[str]:
    """Sección con la cobertura temporal de cada emisor."""
    lines = ["## Cobertura por emisor", ""]
    for c in coverages:
        lines += [
            f"### {c.issuer}",
            "",
            f"- Artículos: **{c.articles}** "
            f"({', '.join(f'{t}: {n}' for t, n in c.by_doc_type.items())})",
            f"- Rango: {c.start} → {c.end} ({c.n_trading_days} días de trading de la BVC)",
            f"- **Días de trading con al menos una noticia: {c.covered_days} "
            f"/ {c.n_trading_days} = {c.pct_coverage * 100:.1f}%**",
            f"- Huecos sin noticia (en días hábiles): mediana {c.gap_median:.0f}, "
            f"p90 {c.gap_p90:.0f}, máximo {c.gap_max} ({c.n_gaps} huecos)",
            f"- Artículos publicados en end de semana o festivo: {c.non_trading_articles} "
            "_(el mapeo a día de trading está pendiente)_",
            f"- Días con 3+ artículos de fuentes distintas: {c.duplication_days} "
            "_(magnitud de la duplicación por hecho; la deduplicación está pendiente)_",
            "",
            "Artículos por año: " + ", ".join(f"{a}: {n}" for a, n in c.by_year.items()),
            "",
            "Top 10 fuentes:",
            "",
            *_md_table(["fuente", "artículos"], c.top_sources.items()),
        ]
    return lines


def _section_alias(aud: AliasAudit, pairs: pd.DataFrame) -> list[str]:
    """Sección con la auditoría del diccionario de alias."""
    lines = [
        "## Auditoría de alias",
        "",
        "Un alias mal calibrado contamina todo el corpus y no se detecta después. "
        "Esta sección es tan importante como la cobertura.",
        "",
        f"- Matches totales: **{aud.total_matches}**",
        f"- De alias descriptive (sin nombre propio): **{len(aud.descriptive)}** "
        f"({aud.pct_descriptive:.1f}%)",
        f"- Artículos con 2+ emisores detectados: **{len(aud.multi_issuer)}**",
        f"- Artículos con emisor detectado ≠ emisor de la carpeta: **{len(aud.cross_mentions)}**",
        f"- Artículos SIN ningún emisor detectado pese a venir de una carpeta de "
        f"emisor: **{len(aud.undetected)}** _(fallo de cobertura del diccionario "
        "de alias)_",
        "",
        "### Matches por alias",
        "",
    ]
    lines += _table(aud.by_alias, max_rows=100)

    lines += ["### Menciones cruzadas (emisor detectado ≠ emisor de búsqueda)", ""]
    lines += _table(aud.cross_mentions[[
        "article_id", "emisor", "emisores_buscados", "alias_matcheado", "en_titular", "titular",
    ]])

    lines += ["### Artículos con 2+ emisores", ""]
    if aud.multi_issuer.empty:
        lines += ["_(ninguno)_", ""]
    else:
        lines += _table(
            pairs[pairs["article_id"].isin(set(aud.multi_issuer.index))]
            .groupby("article_id")
            .agg(emisores=("emisor", lambda s: ", ".join(sorted(s))),
                 titular=("titular", "first"))
            .reset_index()
        )

    lines += ["### Artículos sin ningún emisor detectado", ""]
    lines += _table(aud.undetected[["article_id", "emisores_busqueda", "fuente", "titular"]])

    lines += [
        f"### Ejemplos de match por alias descriptivo ({len(aud.descriptive_examples)})",
        "",
        "Para revisión manual: cada fila es un match que NO vino de un nombre propio. "
        "Si acá aparecen notas de contexto regional (PDVSA, Pemex), el alias hay que "
        "restringirlo o bajarlo de nivel.",
        "",
    ]
    lines += _table(aud.descriptive_examples, max_rows=N_DESCRIPTIVE_EXAMPLES)
    return lines


_SECTION_PENDING: list[str] = [
    "## Pendientes que este reporte solo mide",
    "",
    "1. **Mapeo de noticia a día de trading** — los artículos de end de semana y "
    "festivo están contados aparte por emisor, sin reasignar.",
    "2. **Deduplicación por hecho entre fuentes** — los días con 3+ artículos de "
    "fuentes distintas están contados, sin colapsar.",
    "3. **Mención de paso vs. sujeto principal** — `en_titular`, `n_menciones` y "
    "`posicion_primera_mencion` quedan guardados en `pares_colombia.parquet` para "
    "poder decidir la regla después.",
    "",
]


def build_report(
    articles: pd.DataFrame,
    pairs: pd.DataFrame,
    detail: pd.DataFrame,
    summary: ParseSummary,
    seed: int = SEED,
) -> str:
    """
    Calcula todo el análisis y arma el markdown del reporte.

    Args:
        articles: Artículos únicos.
        pairs: Pares (artículo, issuer).
        detail: Matches por alias.
        summary: Salida de verificar().
        seed: Semilla de los ejemplos.

    Returns:
        El reporte completo en markdown.
    """
    with_date = pairs[pairs["fecha_publicacion"].notna()]
    coverages = [
        coverage_by_issuer(issuer, with_date[with_date["emisor"] == issuer])
        for issuer in sorted(with_date["emisor"].unique())
    ]

    lines = [
        "# Reporte de cobertura — prensa colombiana (EMIS)",
        "",
        f"Generado con SEED={seed}. Ventana objetivo del pivote: 2020–2025.",
        "",
        *_section_parsing(summary, len(pairs)),
        *_section_repository(repository_audit(articles)),
        *_section_coverage(coverages),
        *_section_alias(alias_audit(detail, pairs, articles, seed=seed), pairs),
        *_SECTION_PENDING,
    ]
    return "\n".join(lines) + "\n"
