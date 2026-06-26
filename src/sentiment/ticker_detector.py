"""
Detector de tickers del S&P 500 en posts de r/WallStreetBets.

ÁMBITO: este módulo se usa ÚNICAMENTE en el path histórico de Reddit/WSB, donde
el ticker no viene marcado y hay que inferirlo del texto libre. La fuente
principal del proyecto (StockTwits) NO lo usa: scraper_stocktwits.py extrae el
ticker directo del campo `symbols` de cada mensaje y emite ya la columna
`detected_tickers`, por lo que ese path va directo al preprocessor.

Estrategia de detección en dos capas:
  1. Notación $TICKER  — alta confianza, sin blocklist (el autor marcó explícitamente).
  2. Secuencias en MAYÚSCULAS con word boundaries — sujeta a blocklist para
     evitar falsos positivos con jerga de Reddit, verbos, preposiciones, etc.

Los tickers de una sola letra (A, C, F…) solo se detectan vía capa 1 ($A, $C…).
"""

import json
import logging
import re
from pathlib import Path

import pandas as pd

from config.settings import PROCESSED_DIR
from src.universe.sp500_loader import get_tickers

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Blocklist para detección bare (capa 2)
# Palabras en MAYÚSCULAS que aparecen frecuentemente en posts de WSB y que
# coinciden o podrían coincidir con tickers cortos del S&P 500.
# La capa $TICKER no usa esta lista.
# ---------------------------------------------------------------------------
_BARE_BLOCKLIST: frozenset[str] = frozenset({
    # Artículos, pronombres, preposiciones, conjunciones
    "A", "AN", "THE",
    "IN", "ON", "AT", "BY", "OF", "TO", "UP", "AS", "IF",
    "OR", "AND", "BUT", "FOR", "NOR", "SO", "YET",
    "I", "ME", "MY", "WE", "US", "HE", "IT", "ALL", "ANY",
    # Verbos y auxiliares
    "IS", "BE", "DO", "GO", "ARE", "WAS", "HAS", "HAD", "DID",
    "NOT", "NO", "YES", "OK",
    # Abreviaturas de Reddit / WSB / finanzas
    "DD",    # Due Diligence
    "OP",    # Original Poster
    "FD",    # Fool's Diligence (WSB slang)
    "IV",    # Implied Volatility
    "OTM",   # Out of the Money
    "ITM",   # In the Money
    "ATM",   # At the Money
    "DTE",   # Days to Expiration
    "ATH",   # All-Time High
    "CEO", "CFO", "CTO", "COO",
    "IPO", "ETF", "IMO", "TBH",
    "FOMO", "YOLO", "HODL", "TLDR",
    "WSB", "RH",              # WallStreetBets, Robinhood
    "SEC", "IRS", "FDA", "FED",
    "EPS", "PE", "GDP", "CPI", "PPI", "PCE",
    "YTD", "QTD", "MTD", "TTM",
    "DCA",   # Dollar Cost Average
    "PDT",   # Pattern Day Trader
    "AI", "ML", "API",
    "UTC", "EST", "PST", "CST",
    "AM", "PM", "AH",         # morning/afternoon/after-hours
    "LOL", "WTF", "EDIT", "UPDATE", "NEWS",
    "BTC", "ETH", "NFT",      # crypto, no son S&P 500 pero evita confusión
    "SPY", "QQQ", "DIA", "VIX",  # ETFs/índices comunes
    # Tickers cortos de alta ambigüedad semántica en texto de WSB
    "IT",    # "the IT sector" / "it is"   → ticker de Gartner
    "SO",    # "so bullish on…"            → ticker de Southern Co.
    "ARE",   # "are we there yet"          → ticker de Alexandria RE
    "NOW",   # "right now"                 → ticker de ServiceNow
    "KEY",   # "key support level"         → ticker de KeyCorp
    "ALL",   # "all in"                    → ticker de Allstate
    "RE",    # "RE: comment", "real estate" → ticker de Everest Re
    "PM",    # "PM me", "post meridiem"    → ticker de Philip Morris
    "IP",    # "IP address", "IP rights"   → ticker de International Paper
    "MO",    # "MO money", slang           → ticker de Altria
    "HI",    # saludo común               → ticker de Hawaiʻi Electric (salida del índice, pero precaución)
    "LO",    # "new lo(w)"                → posible ticker
    "CF",    # abreviatura de "cash flow"  → ticker de CF Industries
    "NI",    # "ni hao", expresión         → ticker de NiSource
    "WY",    # abreviatura de Wyoming      → ticker de Weyerhaeuser
    "MA",    # "ma position"              → ticker de Mastercard (MA es muy corto y ambiguo)
    "GS",    # pocas veces ambiguo, pero incluido por precaución si aparece como "GS..." en texto
    # Palabras que aparecen en CAPS en títulos sensacionalistas
    "BREAKING", "DAILY", "WEEKLY", "LIVE",
    "LONG", "SHORT", "CALL", "PUT",
    "BULL", "BEAR",
    "USD", "EUR", "GBP", "JPY",
})

# ---------------------------------------------------------------------------
# Patrones de expresión regular (compilados una sola vez al importar)
# ---------------------------------------------------------------------------

# Capa 1: $TICKER — acepta 1–5 letras, opcionalmente seguidas de punto/dash + 1-2 letras
# Ejemplos: $TSLA, $BRK.B, $BRK-B, $A
_DOLLAR_RE = re.compile(r"\$([A-Z]{1,5}(?:[.\-][A-Z]{1,2})?)\b")

# Capa 2: secuencias en MAYÚSCULAS de 2–5 letras con word boundaries
# Negative lookbehind para $ para no double-contar lo que ya captura _DOLLAR_RE
_BARE_RE = re.compile(r"(?<!\$)\b([A-Z]{2,5})\b")


class TickerDetector:
    """
    Detecta menciones de tickers del S&P 500 en texto libre.

    Carga la lista de tickers desde sp500_loader (con caché) y precompila
    los conjuntos de búsqueda en el constructor para que detect() sea rápido
    al procesar miles de posts.
    """

    def __init__(self, tickers: list[str] | None = None) -> None:
        """
        Inicializa el detector.

        Args:
            tickers: Lista de tickers del S&P 500. Si es None, se obtiene
                     automáticamente desde sp500_loader (usa caché de 7 días).
        """
        raw: list[str] = tickers if tickers is not None else get_tickers()

        self._sp500: frozenset[str] = frozenset(raw)

        # Mapa de variante-con-punto → ticker oficial con dash
        # (ej: "BRK.B" → "BRK-B") para normalizar lo que escriben los usuarios
        self._dot_to_dash: dict[str, str] = {
            t.replace("-", "."): t for t in raw if "-" in t
        }

        # Tickers de una sola letra: solo se detectan vía capa 1 ($A, $C…)
        self._single_letter: frozenset[str] = frozenset(t for t in raw if len(t) == 1)

        # Conjunto válido para la capa bare: excluye single-letter y blocklist
        self._bare_valid: frozenset[str] = (
            self._sp500 - self._single_letter - _BARE_BLOCKLIST
        )

        logger.debug(
            "TickerDetector inicializado: %d tickers totales, %d válidos para bare detection.",
            len(self._sp500),
            len(self._bare_valid),
        )

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def detect(self, text: str) -> list[str]:
        """
        Detecta tickers en un texto y devuelve una lista ordenada de tickers únicos.

        Args:
            text: Texto libre en el que buscar (puede ser título, cuerpo o ambos).

        Returns:
            Lista ordenada alfabéticamente de tickers detectados
            (e.g. ['AAPL', 'BRK-B', 'TSLA']).
        """
        found: set[str] = set()

        # Capa 1: notación $TICKER — alta confianza
        for m in _DOLLAR_RE.finditer(text):
            raw_token = m.group(1)
            # Normalizar BRK.B → BRK-B si aplica
            normalized = self._dot_to_dash.get(raw_token, raw_token)
            if normalized in self._sp500:
                found.add(normalized)

        # Capa 2: secuencias bare en MAYÚSCULAS
        for m in _BARE_RE.finditer(text):
            token = m.group(1)
            if token in self._bare_valid:
                found.add(token)

        return sorted(found)

    def detect_in_post(self, title: str, body: str) -> list[str]:
        """
        Detecta tickers en el título y cuerpo de un post (los combina internamente).

        Args:
            title: Título del post.
            body: Cuerpo del post (puede ser cadena vacía).

        Returns:
            Lista ordenada de tickers detectados.
        """
        return self.detect(f"{title} {body}")


# ---------------------------------------------------------------------------
# Función de procesamiento batch sobre CSV
# ---------------------------------------------------------------------------

def detect_tickers_in_csv(
    input_path: Path,
    output_dir: Path = PROCESSED_DIR,
    tickers: list[str] | None = None,
) -> Path:
    """
    Lee un CSV de posts crudos, detecta tickers en cada fila y guarda el resultado.

    Agrega la columna 'detected_tickers' con un JSON array de tickers
    (e.g. '["AAPL","TSLA"]'). Posts sin menciones quedan como '[]'.

    Args:
        input_path: Ruta al CSV de entrada (salida de scraper_reddit.py).
                    Debe tener columnas 'title' y 'body'.
        output_dir: Directorio donde se escribe el CSV de salida.
        tickers: Lista de tickers a usar; None carga desde sp500_loader.

    Returns:
        Ruta al CSV generado (mismo nombre que la entrada + sufijo '_tickers').
    """
    logger.info("Cargando posts desde %s...", input_path)
    df = pd.read_csv(input_path, dtype=str).fillna("")

    if "title" not in df.columns or "body" not in df.columns:
        raise ValueError(
            f"El CSV de entrada debe tener columnas 'title' y 'body'. "
            f"Columnas encontradas: {list(df.columns)}"
        )

    detector = TickerDetector(tickers)
    logger.info("Detectando tickers en %d posts...", len(df))

    df["detected_tickers"] = [
        json.dumps(detector.detect_in_post(title, body))
        for title, body in zip(df["title"], df["body"])
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_tickers.csv"
    df.to_csv(output_path, index=False, encoding="utf-8")

    posts_with_tickers = (df["detected_tickers"] != "[]").sum()
    logger.info(
        "Detección completada: %d/%d posts con al menos un ticker → %s",
        posts_with_tickers,
        len(df),
        output_path,
    )
    return output_path


# ---------------------------------------------------------------------------
# CLI mínimo para pruebas rápidas
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if len(sys.argv) < 2:
        # Demo interactivo sin archivo
        detector = TickerDetector()
        samples = [
            "$TSLA to the moon 🚀 also buying some $AAPL calls",
            "NVDA earnings tomorrow, loading up on GME too",
            "IT is what it is, SO tired of losing money ON options",
            "BRK.B looking strong, added to my position",
            "what do you think about A stock? it is really good",
            "YOLO all in on AMD and MSFT calls expiring Friday",
        ]
        print(f"{'Texto':<60} {'Tickers detectados'}")
        print("-" * 80)
        for sample in samples:
            detected = detector.detect(sample)
            print(f"{sample[:58]:<60} {detected}")
    else:
        output = detect_tickers_in_csv(Path(sys.argv[1]))
        print(f"Procesado → {output}")
