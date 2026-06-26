"""
Preprocesador de texto para posts de r/WallStreetBets.

Pipeline de limpieza pensado específicamente para FinBERT (ProsusAI/finbert),
que es un modelo BERT uncased fine-tuned en texto financiero. Por eso:
  - Se conservan: números, negaciones, nombres de empresas/tickers, puntuación básica.
  - Se eliminan: URLs, emojis, markdown de Reddit, menciones de usuario, boilerplate
    de disclaimers ("not financial advice") y ruido estructural del foro.

El truncado a 512 tokens lo maneja el tokenizador de FinBERT; aquí no se trunca.
"""

import html
import logging
import re
from pathlib import Path

import pandas as pd

from config.settings import PROCESSED_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patrones compilados (una sola vez al importar el módulo)
# ---------------------------------------------------------------------------

# URLs (http/https/www y variantes sin protocolo)
_URL_RE = re.compile(
    r"https?://\S+"
    r"|www\.\S+"
    r"|i\.redd\.it/\S+"
    r"|v\.redd\.it/\S+",
    re.IGNORECASE,
)

# Menciones de usuario y subreddit de Reddit
_REDDIT_MENTION_RE = re.compile(r"/[ur]/\w+", re.IGNORECASE)

# Markdown de Reddit: negrita (**texto** o __texto__), cursiva, tachado, spoiler
_MARKDOWN_FORMAT_RE = re.compile(r"(\*{1,2}|_{1,2}|~~|>!|!<)")

# Citas de Reddit (líneas que empiezan con >)
_BLOCKQUOTE_RE = re.compile(r"^>+\s?", re.MULTILINE)

# Líneas horizontales de markdown
_HLINE_RE = re.compile(r"^[-*_]{3,}\s*$", re.MULTILINE)

# Prefijo $ de tickers: $TSLA → TSLA (el nombre del ticker se conserva)
_DOLLAR_TICKER_RE = re.compile(r"\$([A-Z]{1,5}(?:[.\-][A-Z]{1,2})?)\b")

# Emojis y símbolos Unicode fuera del rango de texto útil
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"   # emoticons clásicos
    "\U0001F300-\U0001F5FF"   # símbolos y pictogramas variados
    "\U0001F680-\U0001F6FF"   # transporte y mapas
    "\U0001F700-\U0001F77F"   # alquimia
    "\U0001F780-\U0001F7FF"   # figuras geométricas extendidas
    "\U0001F800-\U0001F8FF"   # flechas suplementarias
    "\U0001F900-\U0001F9FF"   # símbolos suplementarios
    "\U0001FA00-\U0001FA6F"   # piezas de ajedrez y dados
    "\U0001FA70-\U0001FAFF"   # símbolos adicionales
    "\U00002600-\U000026FF"   # símbolos misceláneos (☀, ☎…)
    "\U00002700-\U000027BF"   # dingbats (✂, ✈…)
    "\U0001F1E0-\U0001F1FF"   # banderas (pares de letras regionales)
    "\U0000FE00-\U0000FE0F"   # selectores de variación
    "\U000020D0-\U000020FF"   # combinando marcas diacríticas
    "\U0000200B-\U0000200F"   # espacios de ancho cero y similares
    "\U00002000-\U0000206F"   # puntuación general (espacios tipográficos)
    "\U00003000-\U00003004"   # puntuación CJK
    "]+",
    flags=re.UNICODE,
)

# Caracteres no ASCII que no son emojis (tras la limpieza de emojis quedan algunos
# caracteres latinos extendidos, letras con diacríticos, etc. que FinBERT no procesa bien)
_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]+")

# Caracteres especiales que no aportan información semántica para FinBERT.
# Se conservan: letras, dígitos, espacio, y .,-!?'%()
_NOISE_CHARS_RE = re.compile(r"[^a-z0-9 .,'!\?%\(\)\-]")

# Múltiples espacios / saltos de línea → un solo espacio
_WHITESPACE_RE = re.compile(r"\s+")

# ---------------------------------------------------------------------------
# Frases de boilerplate que no aportan señal de sentimiento financiero.
# Se eliminan DESPUÉS de pasar a minúsculas, antes del colapso de espacios.
# Son disclaimers y ruido estructural típico de WSB/Reddit.
# No se incluyen términos que SÍ tienen carga semántica ("bull", "bear", "loss", etc.)
# ---------------------------------------------------------------------------
_BOILERPLATE_PHRASES: list[str] = [
    "not financial advice",
    "this is not financial advice",
    "not a financial advisor",
    "not an investment advisor",
    "do your own research",
    "do your own due diligence",
    "dyor",
    "nfa",                       # "not financial advice" en acrónimo
    "this is not investment advice",
    "positions or ban",          # frase ritual de WSB
    "edit :",
    "edit:",
    "update :",
    "update:",
    "crossposted from",
    "cross posted from",
    "reposting because",
    "repost",
]

# Pre-compilar como regex para aplicar en un solo paso
_BOILERPLATE_RE = re.compile(
    "|".join(re.escape(p) for p in _BOILERPLATE_PHRASES),
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Funciones de limpieza individuales
# ---------------------------------------------------------------------------

def _decode_html(text: str) -> str:
    """Decodifica entidades HTML (&amp; → &, &lt; → <, &#39; → ', etc.)."""
    return html.unescape(text)


def _strip_reddit_structure(text: str) -> str:
    """Elimina elementos estructurales de Reddit: citas, líneas horizontales, menciones."""
    text = _BLOCKQUOTE_RE.sub(" ", text)
    text = _HLINE_RE.sub(" ", text)
    text = _REDDIT_MENTION_RE.sub(" ", text)
    return text


def _strip_urls(text: str) -> str:
    """Elimina URLs."""
    return _URL_RE.sub(" ", text)


def _strip_markdown(text: str) -> str:
    """Elimina marcadores de formato Markdown (***, __, ~~, >!)."""
    return _MARKDOWN_FORMAT_RE.sub(" ", text)


def _normalize_tickers(text: str) -> str:
    """Convierte $TSLA → TSLA para que FinBERT procese el nombre sin el símbolo $."""
    return _DOLLAR_TICKER_RE.sub(r"\1", text)


def _strip_emojis(text: str) -> str:
    """Elimina emojis y símbolos pictográficos Unicode."""
    return _EMOJI_RE.sub(" ", text)


def _strip_non_ascii(text: str) -> str:
    """Elimina caracteres no ASCII residuales (letras con diacríticos, etc.)."""
    return _NON_ASCII_RE.sub(" ", text)


def _lowercase(text: str) -> str:
    """Convierte a minúsculas (FinBERT usa el modelo uncased de BERT)."""
    return text.lower()


def _strip_boilerplate(text: str) -> str:
    """Elimina frases de boilerplate sin carga semántica."""
    return _BOILERPLATE_RE.sub(" ", text)


def _strip_noise_chars(text: str) -> str:
    """Elimina caracteres especiales conservando los que aportan contexto semántico."""
    return _NOISE_CHARS_RE.sub(" ", text)


def _collapse_whitespace(text: str) -> str:
    """Normaliza espacios múltiples y saltos de línea a un solo espacio."""
    return _WHITESPACE_RE.sub(" ", text).strip()


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

# Orden deliberado: primero quitar estructura, luego contenido, luego normalizar
_PIPELINE: list = [
    _decode_html,
    _strip_reddit_structure,
    _strip_urls,
    _strip_markdown,
    _normalize_tickers,
    _strip_emojis,
    _strip_non_ascii,
    _lowercase,
    _strip_boilerplate,
    _strip_noise_chars,
    _collapse_whitespace,
]


def clean_text(text: str) -> str:
    """
    Aplica el pipeline completo de limpieza a un string.

    Args:
        text: Texto crudo (título, cuerpo o ambos concatenados).

    Returns:
        Texto limpio listo para tokenizar con FinBERT. Puede ser cadena vacía
        si el texto original no contenía contenido semántico útil.
    """
    for step in _PIPELINE:
        text = step(text)
    return text


def build_input_text(title: str, body: str) -> str:
    """
    Combina título y cuerpo para construir el texto de entrada a FinBERT.

    El título va primero porque en WSB concentra la mayor densidad de señal
    (ticker, dirección, magnitud). El cuerpo añade contexto. Se separan con
    un punto para que el tokenizador de BERT respete el límite de oración.

    Args:
        title: Título del post.
        body: Cuerpo del post (puede ser cadena vacía).

    Returns:
        String combinado (puede tener solo el título si el cuerpo está vacío).
    """
    if body.strip():
        return f"{title.strip()}. {body.strip()}"
    return title.strip()


# ---------------------------------------------------------------------------
# Procesamiento batch sobre CSV
# ---------------------------------------------------------------------------

def preprocess_csv(
    input_path: Path,
    output_dir: Path = PROCESSED_DIR,
) -> Path:
    """
    Lee un CSV con tickers detectados, limpia el texto y guarda el resultado.

    Agrega la columna 'clean_text' con el texto listo para FinBERT.
    Registra estadísticas de posts vaciados por la limpieza.

    Args:
        input_path: Ruta al CSV de entrada (salida de ticker_detector.py).
                    Debe tener columnas 'title', 'body' y 'detected_tickers'.
        output_dir: Directorio de salida.

    Returns:
        Ruta al CSV generado (mismo stem + sufijo '_clean').
    """
    logger.info("Cargando CSV desde %s...", input_path)
    df = pd.read_csv(input_path, dtype=str).fillna("")

    required = {"title", "body", "detected_tickers"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"El CSV de entrada no tiene las columnas requeridas: {missing}. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    logger.info("Preprocesando %d posts...", len(df))

    df["clean_text"] = [
        clean_text(build_input_text(title, body))
        for title, body in zip(df["title"], df["body"])
    ]

    empty_count = (df["clean_text"].str.strip() == "").sum()
    if empty_count:
        logger.warning(
            "%d posts quedaron con texto vacío tras la limpieza y serán omitidos por FinBERT.",
            empty_count,
        )

    avg_len = df["clean_text"].str.len().mean()
    logger.info(
        "Longitud media del texto limpio: %.0f caracteres. "
        "Posts con texto: %d/%d.",
        avg_len,
        len(df) - empty_count,
        len(df),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_clean.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")
    logger.info("CSV preprocesado guardado en %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# CLI para pruebas rápidas
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if len(sys.argv) < 2:
        samples = [
            (
                "$TSLA 🚀🚀 TO THE MOON!! [DD]",
                "I've been holding **$TSLA** since $420. Check this out: "
                "https://i.redd.it/abc123. Not financial advice. Do your own research.",
            ),
            (
                "NVDA earnings tomorrow — YOLO or nah?",
                "IV is insane rn. OTM calls expiring Friday. This is not financial advice.\n\n"
                "> someone said buy\n\nEdit: added more context. /u/WallStreetBets_mod",
            ),
            (
                "Why I'm going long on AAPL and shorting GME",
                "The market is pricing in a 20% correction. ~~I was wrong before~~ "
                "but this time it's different. NFA. Positions: AAPL 150c, GME puts.",
            ),
            (
                "Daily Discussion Thread — May 2026",
                "",
            ),
        ]
        print(f"\n{'TEXTO LIMPIO':^70}\n{'─'*70}")
        for title, body in samples:
            raw = build_input_text(title, body)
            result = clean_text(raw)
            print(f"RAW  : {raw[:80]}")
            print(f"CLEAN: {result[:80]}")
            print()
    else:
        out = preprocess_csv(Path(sys.argv[1]))
        print(f"Preprocesado → {out}")
