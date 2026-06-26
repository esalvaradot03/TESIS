"""
Módulo para obtener y cachear la lista de tickers del S&P 500 desde Wikipedia.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "sp500_tickers.csv"
_CACHE_TTL_DAYS = 7


def _fetch_from_wikipedia() -> list[str]:
    """Descarga la tabla de componentes del S&P 500 desde Wikipedia y devuelve los tickers."""
    logger.info("Descargando lista S&P 500 desde Wikipedia...")
    headers = {"User-Agent": "trading-sentiment-bot/1.0 (academic research)"}
    response = requests.get(_WIKIPEDIA_URL, headers=headers, timeout=15)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    table = soup.find("table", {"id": "constituents"})
    if table is None:
        raise ValueError("No se encontró la tabla 'constituents' en la página de Wikipedia.")

    rows = table.find("tbody").find_all("tr")
    tickers: list[str] = []
    for row in rows[1:]:  # omitir encabezado
        cols = row.find_all("td")
        if cols:
            # La primera columna contiene el ticker; algunos tienen puntos (e.g. BRK.B → BRK-B en Alpaca)
            raw = cols[0].get_text(strip=True)
            tickers.append(raw.replace(".", "-"))

    logger.info("Obtenidos %d tickers del S&P 500.", len(tickers))
    return tickers


def _cache_is_fresh() -> bool:
    """Devuelve True si el CSV de caché existe y tiene menos de _CACHE_TTL_DAYS días."""
    if not _CACHE_PATH.exists():
        return False
    mtime = datetime.fromtimestamp(_CACHE_PATH.stat().st_mtime)
    return datetime.now() - mtime < timedelta(days=_CACHE_TTL_DAYS)


def _load_from_cache() -> list[str]:
    """Lee los tickers desde el CSV de caché."""
    df = pd.read_csv(_CACHE_PATH)
    return df["ticker"].tolist()


def _save_to_cache(tickers: list[str]) -> None:
    """Guarda la lista de tickers en el CSV de caché."""
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": tickers}).to_csv(_CACHE_PATH, index=False)
    logger.info("Caché actualizado en %s", _CACHE_PATH)


def get_tickers(force_refresh: bool = False) -> list[str]:
    """
    Devuelve la lista de tickers del S&P 500.

    Usa el caché local (data/sp500_tickers.csv) si tiene menos de 7 días.
    Si el caché está desactualizado o no existe, descarga la lista desde Wikipedia
    y actualiza el caché.

    Args:
        force_refresh: Si True, ignora el caché y fuerza la descarga.

    Returns:
        Lista de strings con los símbolos bursátiles (e.g. ['AAPL', 'MSFT', ...]).
    """
    if not force_refresh and _cache_is_fresh():
        logger.info("Usando caché de tickers S&P 500 (%s).", _CACHE_PATH)
        return _load_from_cache()

    tickers = _fetch_from_wikipedia()
    _save_to_cache(tickers)
    return tickers


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = get_tickers()
    print(f"Total tickers: {len(result)}")
    print(result[:10])
