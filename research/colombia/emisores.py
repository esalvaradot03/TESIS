"""
Diccionario de emisores de la BVC y detección de menciones en texto de prensa.

Un emisor se detecta por sus alias: el nombre propio, las marcas y filiales que
la prensa usa como sinónimo del emisor, y (nivel 2) las descripciones sin
nombre propio ("la petrolera estatal"). Cada alias guarda su nivel y si es
descriptivo, porque los descriptivos son los de mayor riesgo de falso positivo
y el reporte de cobertura tiene que poder aislarlos.

La detección distingue titular de cuerpo y registra la posición (índice de
párrafo) de la primera mención en el cuerpo. Esa distinción es lo que después
permite separar "el artículo trata sobre este emisor" de "lo menciona de paso
en el párrafo 12", y NO se puede reconstruir una vez aplanado el texto.

Uso:
    from research.colombia.emisores import detect_issuers
    matches = detect_issuers(headline, body)
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field

# ---------------------------------------------------------------------------
# Diccionario de emisores
# ---------------------------------------------------------------------------

# nivel 1: nombre propio del emisor, sus marcas y filiales (alta precisión).
# nivel 2: descripciones sin nombre propio (alto recall, riesgo de falso
#          positivo: "la petrolera estatal" puede referirse a PDVSA o Pemex en
#          una nota de contexto regional). Se marcan `es_descriptivo=True`.
#
# Para agregar un nivel nuevo basta sumar la clave al dict de un emisor; el
# resto del módulo itera sobre los niveles presentes, no sobre una lista fija.
ALIASES_BY_LEVEL: dict[str, dict[int, list[str]]] = {
    "ECOPETROL": {
        1: [
            "Ecopetrol", "Grupo Ecopetrol", "Reficar", "Cenit", "Ocensa",
            "Hocol", "Esenttia", "Ecopetrol Permian",
        ],
        2: ["la petrolera estatal"],
    },
    "CIBEST": {
        1: [
            "Grupo Cibest", "Cibest", "Bancolombia", "Grupo Bancolombia",
            "Nequi", "Wompi", "Banistmo", "Bancoagrícola", "Renting Colombia",
            "Wenia",
        ],
        2: [],
    },
    "GEB": {
        1: [
            "GEB", "Grupo Energía Bogotá", "Grupo Energía de Bogotá", "EEB",
            "Empresa de Energía de Bogotá", "TGI", "Cálidda", "Calidda",
            "Contugas", "Electrodunas",
        ],
        2: [],
    },
    "GRUPOARGOS": {
        1: [
            "Grupo Argos", "Cementos Argos", "Celsia", "Odinsa", "Summa",
            "Situm", "Opain",
        ],
        2: [],
    },
    "PFDAVVND": {
        1: [
            "Davivienda", "Grupo Davivienda", "Banco Davivienda", "DaviPlata",
            "Corredores Davivienda", "Davivienda Corredores",
        ],
        2: [],
    },
}

ISSUERS: list[str] = list(ALIASES_BY_LEVEL)

# Los alias de nivel >= 2 son descripciones, no nombres propios.
_DESCRIPTIVE_LEVEL: int = 2


# ---------------------------------------------------------------------------
# Estructuras
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Alias:
    """Un alias de un emisor, con su nivel y su patrón compilado."""

    issuer: str
    text: str
    level: int
    descriptive: bool
    pattern: re.Pattern = field(compare=False, repr=False)

    @property
    def n_tokens(self) -> int:
        """Cantidad de tokens del alias; desempata cuál gana al solaparse."""
        return len(_normalize(self.text).split())


@dataclass
class IssuerMatch:
    """
    Una mención de un emisor en un artículo.

    Attributes:
        emisor: Código del emisor (clave de ALIASES_BY_LEVEL).
        alias_matcheado: Texto del alias tal como está en el diccionario.
        nivel_alias: Nivel del alias (1 = nombre propio, 2 = descriptivo).
        es_descriptivo: True si el alias no es un nombre propio.
        en_titular: True si el emisor aparece en el titular.
        n_menciones_titular: Menciones de este alias en el titular.
        n_menciones_cuerpo: Menciones de este alias en el cuerpo.
        posicion_primera_mencion: Índice (0-based) del párrafo del cuerpo con
            la primera mención, o -1 si el alias solo aparece en el titular.
        fragmento: Contexto textual de la primera mención (texto ORIGINAL, sin
            normalizar), para la auditoría manual de alias.
    """

    emisor: str
    alias_matcheado: str
    nivel_alias: int
    es_descriptivo: bool
    en_titular: bool
    n_menciones_titular: int
    n_menciones_cuerpo: int
    posicion_primera_mencion: int
    fragmento: str = ""

    @property
    def n_menciones(self) -> int:
        """Menciones totales (titular + cuerpo)."""
        return self.n_menciones_titular + self.n_menciones_cuerpo

    def to_dict(self) -> dict:
        """Versión serializable, con n_menciones materializado."""
        data = asdict(self)
        data["n_menciones"] = self.n_menciones
        return data


# ---------------------------------------------------------------------------
# Normalización y compilación de patrones
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """
    Pasa a minúsculas y quita acentos, para comparar sin depender de la
    ortografía de la fuente ("Cálidda" / "Calidda", "Bancoagrícola" en mayúsculas).

    El largo en caracteres se conserva (NFKD + descarte de combinantes no
    altera el conteo de caracteres base), así que los offsets del texto
    normalizado siguen siendo válidos sobre el texto original.

    Args:
        text: Texto crudo.

    Returns:
        Texto normalizado, mismo largo que el original.
    """
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


# Límites de palabra que no rompen con acentos ni con siglas pegadas a
# puntuación. `\b` de Python sobre text ya normalized alcanza, pero se
# explicita con lookarounds para que "GEB" no matchee dentro de "GEBCO" ni de
# "AGEB", y "TGI" no matchee dentro de "TGIF" ni de "STGI".
_LEFT = r"(?<![0-9a-z])"
_RIGHT = r"(?![0-9a-z])"


def _compile_alias(alias: str) -> re.Pattern:
    """
    Compila el patrón de un alias con límites de palabra obligatorios.

    Los espacios internos del alias aceptan cualquier espacio en blanco
    (la prensa parte nombres largos con saltos de línea).

    Args:
        alias: Texto del alias.

    Returns:
        Patrón compilado, para aplicar sobre texto ya normalizado.
    """
    parts = [re.escape(p) for p in _normalize(alias).split()]
    return re.compile(_LEFT + r"\s+".join(parts) + _RIGHT)


def build_aliases(
    aliases_by_level: dict[str, dict[int, list[str]]] = ALIASES_BY_LEVEL,
) -> list[Alias]:
    """
    Aplana el diccionario de emisores en una lista de Alias compilados.

    Los alias quedan ordenados de más largo a más corto (por tokens y luego por
    caracteres): al resolver solapamientos gana el más específico, así
    "Grupo Energía Bogotá" no se cuenta además como "GEB", ni "Grupo Ecopetrol"
    como "Ecopetrol".

    Args:
        aliases_by_level: Diccionario emisor → nivel → alias.

    Returns:
        Lista de Alias ordenada por especificidad descendente.
    """
    alias: list[Alias] = []
    for issuer, levels in aliases_by_level.items():
        for level, texts in levels.items():
            for text in texts:
                alias.append(
                    Alias(
                        issuer=issuer,
                        text=text,
                        level=level,
                        descriptive=level >= _DESCRIPTIVE_LEVEL,
                        pattern=_compile_alias(text),
                    )
                )
    # El texto del alias entra como desempate para que el orden sea estable:
    # dos alias que normalizan igual ("Cálidda" / "Calidda") compiten por el
    # mismo tramo y tiene que ganar siempre el mismo.
    alias.sort(key=lambda a: (a.n_tokens, len(a.text), a.text), reverse=True)
    return alias


_ALIASES: list[Alias] = build_aliases()


# ---------------------------------------------------------------------------
# Detección
# ---------------------------------------------------------------------------

def _find_non_overlapping(text: str, alias: list[Alias]) -> list[tuple[Alias, int, int]]:
    """
    Busca todos los alias en un texto, descartando los solapados.

    Recorre `alias` de más específico a menos (ya viene ordenado) y reserva los
    tramos ocupados: el alias largo gana y el corto contenido en él no vuelve a
    contarse.

    Args:
        text: Texto original (se normaliza acá dentro).
        alias: Alias compilados, ordenados por especificidad descendente.

    Returns:
        Lista de (alias, start, end) con los offsets sobre el texto ORIGINAL,
        ordenada por posición.
    """
    if not text:
        return []

    normalized = _normalize(text)
    occupied: list[tuple[int, int]] = []
    hits: list[tuple[Alias, int, int]] = []

    for item in alias:
        for m in item.pattern.finditer(normalized):
            start, end = m.span()
            if any(start < b and a < end for a, b in occupied):
                continue
            occupied.append((start, end))
            hits.append((item, start, end))

    hits.sort(key=lambda h: h[1])
    return hits


def _snippet(text: str, start: int, end: int, margin: int = 60) -> str:
    """
    Contexto original alrededor de un match, para la auditoría manual.

    Los offsets vienen del texto normalizado; NFD sin combinantes conserva el
    largo en el español, pero se acotan igual por si algún carácter exótico lo
    corriera.
    """
    start, end = min(start, len(text)), min(end, len(text))
    left = max(0, start - margin)
    right = min(len(text), end + margin)
    prefix = "…" if left > 0 else ""
    suffix = "…" if right < len(text) else ""
    return f"{prefix}{text[left:right].strip()}{suffix}"


def detect_issuers(
    headline: str,
    body: str,
    alias: list[Alias] = _ALIASES,
) -> list[IssuerMatch]:
    """
    Detecta los emisores mencionados en un artículo.

    Devuelve un IssuerMatch por (emisor, alias) efectivamente encontrado: si un
    artículo nombra a Ecopetrol y a Reficar, salen dos matches del mismo emisor
    y el consumidor decide si los colapsa. El conteo separado titular/cuerpo y
    la posición de la primera mención en el cuerpo se guardan acá porque no se
    pueden recuperar después.

    Args:
        headline: Titular del artículo.
        body: Cuerpo del artículo, con los párrafos separados por "\\n"
            (el formato que produce parse_emis).
        alias: Alias a usar (inyectable para tests).

    Returns:
        Lista de IssuerMatch ordenada por issuer y luego por alias.
    """
    paragraphs = [p for p in (body or "").split("\n")]

    headline_hits = _find_non_overlapping(headline or "", alias)
    body_hits: list[tuple[Alias, int, int, int]] = [
        (item, start, end, i)
        for i, paragraph in enumerate(paragraphs)
        for item, start, end in _find_non_overlapping(paragraph, alias)
    ]

    keys = {(h[0].issuer, h[0].text) for h in headline_hits}
    keys |= {(h[0].issuer, h[0].text) for h in body_hits}

    matches: list[IssuerMatch] = []
    for issuer, alias_text in sorted(keys):
        in_headline_hits = [h for h in headline_hits if h[0].issuer == issuer and h[0].text == alias_text]
        in_body_hits = [h for h in body_hits if h[0].issuer == issuer and h[0].text == alias_text]
        item = (in_headline_hits or in_body_hits)[0][0]

        if in_body_hits:
            _, start, end, idx = in_body_hits[0]
            position = idx
            snippet = _snippet(paragraphs[idx], start, end)
        else:
            _, start, end = in_headline_hits[0]
            position = -1
            snippet = _snippet(headline, start, end)

        matches.append(
            IssuerMatch(
                emisor=issuer,
                alias_matcheado=item.text,
                nivel_alias=item.level,
                es_descriptivo=item.descriptive,
                en_titular=bool(in_headline_hits),
                n_menciones_titular=len(in_headline_hits),
                n_menciones_cuerpo=len(in_body_hits),
                posicion_primera_mencion=position,
                fragmento=snippet,
            )
        )
    return matches
