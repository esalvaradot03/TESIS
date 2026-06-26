"""
Descarga de precios históricos desde Alpaca y cálculo de indicadores técnicos.

Flujo:
  1. Descarga barras diarias para todos los tickers del S&P 500 en chunks
     de PRICE_CHUNK_SIZE símbolos por petición (bulk request).
  2. Añade un buffer de warm-up antes de la fecha de inicio para que los
     primeros valores de MACD/BB/RSI no sean NaN.
  3. Calcula RSI(14), MACD(12,26,9) y Bollinger Bands(20,2) con la librería ta
     usando operaciones vectorizadas de pandas por cada grupo de ticker.
  4. Filtra al rango de fechas pedido y devuelve un DataFrame plano.

Cachea los precios crudos en data/raw/prices_raw.parquet y los indicadores
en data/processed/indicators.parquet para evitar re-descargas innecesarias.

Columnas de salida:
  ticker, date, close, volume, rsi, macd, macd_signal, bb_upper, bb_lower, bb_middle
"""

import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from ta.momentum import RSIIndicator
from ta.trend import MACD as MACDIndicator
from ta.volatility import BollingerBands
from tqdm import tqdm

from config.settings import (
    ALPACA_API_KEY,
    ALPACA_API_SECRET,
    ALPACA_FEED,
    PRICE_CHUNK_SIZE,
    PROCESSED_DIR,
    RAW_DIR,
)
from src.universe.sp500_loader import get_tickers

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

_PRICE_CACHE = RAW_DIR / "prices_raw.parquet"
_INDICATORS_CACHE = PROCESSED_DIR / "indicators.parquet"

# Días calendario extra antes de la fecha de inicio para warm-up de indicadores.
# MACD necesita 26 barras (≈ 37 días calendario); usamos 60 para margen.
_WARMUP_CALENDAR_DAYS: int = 60

# Indicadores técnicos — parámetros canónicos
_RSI_WINDOW: int = 14
_MACD_FAST: int = 12
_MACD_SLOW: int = 26
_MACD_SIGNAL: int = 9
_BB_WINDOW: int = 20
_BB_STD: int = 2

# Pausa entre peticiones a Alpaca (segundos) para no superar el rate limit
_CHUNK_SLEEP: float = 0.35

# ---------------------------------------------------------------------------
# Cliente Alpaca
# ---------------------------------------------------------------------------


def _build_client() -> StockHistoricalDataClient:
    """Construye y devuelve el cliente de datos históricos de alpaca-py."""
    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        raise EnvironmentError(
            "ALPACA_API_KEY y ALPACA_API_SECRET deben estar definidos en .env"
        )
    return StockHistoricalDataClient(
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_API_SECRET,
    )


# ---------------------------------------------------------------------------
# Normalización de símbolos Wikipedia ↔ Alpaca
# ---------------------------------------------------------------------------

def _to_alpaca_symbol(ticker: str) -> str:
    """Convierte el formato Wikipedia ('BRK-B') al formato Alpaca ('BRK/B')."""
    return ticker.replace("-", "/")


def _from_alpaca_symbol(symbol: str) -> str:
    """Convierte el símbolo Alpaca ('BRK/B') al formato interno ('BRK-B')."""
    return symbol.replace("/", "-")


# ---------------------------------------------------------------------------
# Descarga de precios
# ---------------------------------------------------------------------------

def _download_chunk(
    client: StockHistoricalDataClient,
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """
    Descarga barras diarias para un chunk de tickers y devuelve un DataFrame plano.

    Si Alpaca no tiene datos para algún ticker (delisted, etc.) simplemente
    no aparece en el resultado; no se lanza excepción.

    Args:
        client: StockHistoricalDataClient autenticado (alpaca-py).
        tickers: Lista de tickers en formato interno ('BRK-B').
        start: Fecha de inicio ISO 'YYYY-MM-DD'.
        end: Fecha de fin ISO 'YYYY-MM-DD'.

    Returns:
        DataFrame con columnas: ticker, date, open, high, low, close, volume.
        Vacío si la petición no devuelve datos.
    """
    # El feed IEX gratuito no admite símbolos con "/" (BRK/B, BF/B…).
    # Un símbolo inválido tumba todo el chunk, así que los filtramos antes.
    alpaca_symbols = [_to_alpaca_symbol(t) for t in tickers]
    supported = [s for s in alpaca_symbols if "/" not in s]
    skipped = [s for s in alpaca_symbols if "/" in s]
    if skipped:
        logger.debug("Omitidos en feed IEX (slash symbols): %s", skipped)

    if not supported:
        return pd.DataFrame()

    # alpaca-py requiere datetime con timezone, no strings planas
    start_dt = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    end_dt = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)

    try:
        request = StockBarsRequest(
            symbol_or_symbols=supported,
            timeframe=TimeFrame.Day,
            start=start_dt,
            end=end_dt,
            adjustment="all",
            feed=ALPACA_FEED,
        )
        raw: pd.DataFrame = client.get_stock_bars(request).df
    except Exception as exc:
        logger.warning("Error descargando chunk %s…%s: %s", tickers[0], tickers[-1], exc)
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    # alpaca-py devuelve MultiIndex (symbol, timestamp)
    raw = raw.reset_index()
    raw.columns = raw.columns.str.lower()
    raw = raw.rename(columns={"symbol": "ticker", "timestamp": "date"})

    raw["ticker"] = raw["ticker"].map(_from_alpaca_symbol)
    raw["date"] = pd.to_datetime(raw["date"]).dt.date

    keep = ["ticker", "date", "open", "high", "low", "close", "volume"]
    available = [c for c in keep if c in raw.columns]
    return raw[available].copy()


def download_prices(
    tickers: list[str],
    start: date,
    end: date,
    chunk_size: int = PRICE_CHUNK_SIZE,
) -> pd.DataFrame:
    """
    Descarga barras diarias para todos los tickers en chunks bulk.

    Incluye automáticamente _WARMUP_CALENDAR_DAYS días antes de `start`
    para que el warm-up de los indicadores no genere NaN en el rango pedido.

    Args:
        tickers: Lista completa de tickers del S&P 500.
        start: Primer día del rango de interés.
        end: Último día del rango de interés.
        chunk_size: Número de símbolos por petición a Alpaca.

    Returns:
        DataFrame con columnas ticker, date, open, high, low, close, volume.
        Ordenado por (ticker, date).
    """
    fetch_start = (start - timedelta(days=_WARMUP_CALENDAR_DAYS)).isoformat()
    fetch_end = end.isoformat()

    logger.info(
        "Descargando precios para %d tickers desde %s hasta %s (incluye %d días warm-up).",
        len(tickers), fetch_start, fetch_end, _WARMUP_CALENDAR_DAYS,
    )

    client = _build_client()
    chunks = [tickers[i: i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    parts: list[pd.DataFrame] = []

    for chunk in tqdm(chunks, desc="Descargando precios", unit="chunk"):
        df_chunk = _download_chunk(client, chunk, fetch_start, fetch_end)
        if not df_chunk.empty:
            parts.append(df_chunk)
        time.sleep(_CHUNK_SLEEP)

    if not parts:
        raise RuntimeError("Alpaca no devolvió datos para ningún ticker.")

    prices = pd.concat(parts, ignore_index=True)
    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)

    n_tickers = prices["ticker"].nunique()
    n_rows = len(prices)
    logger.info(
        "Descarga completada: %d tickers, %d barras totales.", n_tickers, n_rows
    )
    return prices


# ---------------------------------------------------------------------------
# Cálculo de indicadores técnicos
# ---------------------------------------------------------------------------

def _compute_group_indicators(group: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula RSI, MACD y Bollinger Bands para el precio de cierre de un ticker.

    Requiere que `group` esté ordenado por fecha ascendente.

    Args:
        group: Sub-DataFrame de un único ticker con columna 'close'.

    Returns:
        El mismo DataFrame con las columnas de indicadores añadidas.
    """
    close = group["close"].astype(float)

    # RSI(14)
    group = group.copy()
    group["rsi"] = RSIIndicator(close=close, window=_RSI_WINDOW, fillna=False).rsi()

    # MACD(12, 26, 9)
    macd_obj = MACDIndicator(
        close=close,
        window_fast=_MACD_FAST,
        window_slow=_MACD_SLOW,
        window_sign=_MACD_SIGNAL,
        fillna=False,
    )
    group["macd"] = macd_obj.macd()
    group["macd_signal"] = macd_obj.macd_signal()

    # Bollinger Bands(20, 2)
    bb_obj = BollingerBands(
        close=close,
        window=_BB_WINDOW,
        window_dev=_BB_STD,
        fillna=False,
    )
    group["bb_upper"] = bb_obj.bollinger_hband()
    group["bb_middle"] = bb_obj.bollinger_mavg()
    group["bb_lower"] = bb_obj.bollinger_lband()

    return group


def compute_indicators(
    prices: pd.DataFrame,
    start: date | None = None,
) -> pd.DataFrame:
    """
    Aplica los indicadores técnicos a cada ticker del DataFrame de precios.

    Procesa cada ticker de forma independiente (groupby) para que los
    indicadores de un ticker no contaminen los de otro.

    Args:
        prices: DataFrame con columnas ticker, date, close (y opcionales open,
                high, low, volume). Debe estar ordenado por (ticker, date).
        start: Si se indica, filtra las filas anteriores a esta fecha del
               resultado final (elimina el período de warm-up).

    Returns:
        DataFrame con las columnas canónicas de salida, sin filas del warm-up.
    """
    logger.info("Calculando indicadores para %d tickers...", prices["ticker"].nunique())

    tickers_list = prices["ticker"].unique().tolist()
    parts: list[pd.DataFrame] = []

    for ticker in tqdm(tickers_list, desc="Indicadores", unit="ticker", leave=False):
        group = prices[prices["ticker"] == ticker].sort_values("date")
        if len(group) < _MACD_SLOW:
            logger.debug(
                "Ticker %s: solo %d barras disponibles (mínimo %d para MACD). Se omite.",
                ticker, len(group), _MACD_SLOW,
            )
            continue
        parts.append(_compute_group_indicators(group))

    if not parts:
        raise RuntimeError("No se pudieron calcular indicadores para ningún ticker.")

    result = pd.concat(parts, ignore_index=True)

    # Eliminar período de warm-up
    if start is not None:
        result = result[result["date"] >= start]

    # Descartar filas donde cualquier indicador sea NaN
    # (pueden quedar al inicio de series con pocos datos históricos)
    indicator_cols = ["rsi", "macd", "macd_signal", "bb_upper", "bb_middle", "bb_lower"]
    before = len(result)
    result = result.dropna(subset=indicator_cols)
    dropped = before - len(result)
    if dropped:
        logger.warning(
            "Se descartaron %d filas con NaN en indicadores (warm-up insuficiente).",
            dropped,
        )

    # Columnas canónicas de salida
    output_cols = [
        "ticker", "date", "close", "volume",
        "rsi", "macd", "macd_signal",
        "bb_upper", "bb_middle", "bb_lower",
    ]
    available = [c for c in output_cols if c in result.columns]
    result = result[available].reset_index(drop=True)

    logger.info(
        "Indicadores listos: %d tickers, %d filas (rango %s → %s).",
        result["ticker"].nunique(),
        len(result),
        result["date"].min(),
        result["date"].max(),
    )
    return result


# ---------------------------------------------------------------------------
# Caché de precios e indicadores
# ---------------------------------------------------------------------------

def _prices_cache_valid(path: Path, start: date, end: date, n_tickers: int) -> bool:
    """
    Devuelve True si el Parquet cacheado cubre el rango y el universo pedidos.

    Compara las fechas extremas y el número de tickers únicos del caché
    contra los parámetros de la llamada actual.
    """
    if not path.exists():
        return False
    try:
        cached = pd.read_parquet(path, columns=["ticker", "date"])
        cached_start = pd.to_datetime(cached["date"]).dt.date.min()
        cached_end = pd.to_datetime(cached["date"]).dt.date.max()
        cached_tickers = cached["ticker"].nunique()
        warmup_start = start - timedelta(days=_WARMUP_CALENDAR_DAYS)
        ok = (
            cached_start <= warmup_start
            and cached_end >= end
            and cached_tickers >= n_tickers * 0.95  # tolera un 5 % de tickers sin datos
        )
        if not ok:
            logger.info(
                "Caché de precios obsoleto (cubre %s→%s, %d tickers). Se re-descarga.",
                cached_start, cached_end, cached_tickers,
            )
        return ok
    except Exception as exc:
        logger.warning("No se pudo leer caché de precios: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Punto de entrada principal
# ---------------------------------------------------------------------------

def get_indicators(
    tickers: list[str] | None = None,
    start: date = date(2024, 1, 1),
    end: date = date(2024, 6, 30),
    use_price_cache: bool = True,
    use_indicator_cache: bool = False,
) -> pd.DataFrame:
    """
    Descarga precios y calcula indicadores técnicos para todos los tickers.

    Args:
        tickers: Lista de tickers. Si es None, se obtiene de sp500_loader.
        start: Primer día del rango de interés (warm-up se añade automáticamente).
        end: Último día del rango de interés.
        use_price_cache: Si True, reutiliza el Parquet de precios si ya cubre el rango.
        use_indicator_cache: Si True, devuelve el Parquet de indicadores si existe.

    Returns:
        DataFrame con columnas:
        ticker, date, close, volume, rsi, macd, macd_signal,
        bb_upper, bb_middle, bb_lower
    """
    if tickers is None:
        tickers = get_tickers()

    # --- Caché de indicadores ---
    if use_indicator_cache and _INDICATORS_CACHE.exists():
        logger.info("Cargando indicadores desde caché %s.", _INDICATORS_CACHE)
        return pd.read_parquet(_INDICATORS_CACHE)

    # --- Precios (con caché) ---
    if use_price_cache and _prices_cache_valid(_PRICE_CACHE, start, end, len(tickers)):
        logger.info("Cargando precios desde caché %s.", _PRICE_CACHE)
        prices = pd.read_parquet(_PRICE_CACHE)
        prices["date"] = pd.to_datetime(prices["date"]).dt.date
    else:
        prices = download_prices(tickers, start, end)
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(_PRICE_CACHE, index=False)
        logger.info("Precios cacheados en %s.", _PRICE_CACHE)

    # --- Indicadores ---
    indicators = compute_indicators(prices, start=start)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    indicators.to_parquet(_INDICATORS_CACHE, index=False)
    logger.info("Indicadores cacheados en %s.", _INDICATORS_CACHE)

    return indicators


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from datetime import date as Date

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    start_arg = Date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else Date(2024, 1, 1)
    end_arg = Date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else Date(2024, 6, 30)

    df = get_indicators(start=start_arg, end=end_arg)
    print(df.head(10).to_string(index=False))
    print(f"\nShape: {df.shape}  |  Tickers: {df['ticker'].nunique()}")
