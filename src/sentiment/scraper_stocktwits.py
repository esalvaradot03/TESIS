"""
Scraper de StockTwits — fuente principal de sentimiento del proyecto.

Usa el endpoint público read-only de StockTwits (no requiere API key):

  - Streams por símbolo:  https://api.stocktwits.com/api/2/streams/symbol/{SYMBOL}.json
  - Trending symbols:     https://api.stocktwits.com/api/2/trending/symbols.json

Cada mensaje de StockTwits trae el/los ticker(s) directamente en el campo
`symbols` (cashtags que el autor marcó), por lo que NO se usa ticker_detector:
el ticker se extrae del mensaje, no por regex sobre el texto.

Normaliza cada mensaje al MISMO esquema interno que produce scraper_reddit.py,
de modo que el resto del pipeline (preprocessor → finbert_scorer → feature_engine
→ xgboost_model) funcione sin cambios. Columnas de salida:

    post_id, timestamp, title, body, score, num_comments,
    detected_tickers, stocktwits_sentiment

Notas de mapeo respecto a Reddit:
  - `title`  ← cuerpo del mensaje de StockTwits (StockTwits no tiene título/cuerpo
               separados; el texto va en `title` para que build_input_text() del
               preprocessor lo use sin prefijos espurios).
  - `body`   ← "" (vacío).
  - `score`  ← likes del mensaje (engagement, análogo a los upvotes de Reddit).
  - `num_comments` ← 0 (el stream público no expone el conteo de respuestas).
  - `detected_tickers`   ← JSON array con los símbolos del campo `symbols`
                           (ya en el formato que esperan preprocessor/finbert_scorer,
                           lo que permite SALTAR ticker_detector).
  - `stocktwits_sentiment` ← label nativo entity.sentiment.basic: "Bullish",
                           "Bearish" o "None".

Rate limiting: la API pública tolera ~200 req/hora sin autenticar. Se limita a
~150 req/hora (sleep mínimo entre llamadas) y se respeta el header Retry-After
ante respuestas 429.

Uso desde línea de comandos:
    python -m src.sentiment.scraper_stocktwits           # modo demo (AAPL)
    python -m src.sentiment.scraper_stocktwits AAPL NVDA TSLA   # símbolos dados
    python -m src.sentiment.scraper_stocktwits --trending 15    # top trending
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from config.settings import (
    RAW_DIR,
    STOCKTWITS_MAX_MESSAGES_PER_SYMBOL,
    STOCKTWITS_MAX_REQUESTS_PER_HOUR,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
_BASE_URL = "https://api.stocktwits.com/api/2"
_STREAM_URL = _BASE_URL + "/streams/symbol/{symbol}.json"
_TRENDING_URL = _BASE_URL + "/trending/symbols.json"

# StockTwits devuelve hasta 30 mensajes por página
_PAGE_SIZE = 30

# Cabecera de navegador: el endpoint público filtra clientes sin User-Agent.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

# Reintentos ante errores de red/servidor
_MAX_RETRIES = 3
_RETRY_BACKOFF = 5.0  # segundos; se duplica en cada reintento

# Columnas de salida — esquema interno compatible con el resto del pipeline
_OUTPUT_COLUMNS = [
    "post_id",
    "timestamp",
    "title",
    "body",
    "score",
    "num_comments",
    "detected_tickers",
    "stocktwits_sentiment",
]


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class _RateLimiter:
    """
    Espaciador simple de peticiones para no superar un máximo por hora.

    Garantiza un intervalo mínimo entre llamadas consecutivas
    (3600 / max_per_hour segundos).
    """

    def __init__(self, max_per_hour: int) -> None:
        self._min_interval = 3600.0 / max(max_per_hour, 1)
        self._last_call: float | None = None

    def wait(self) -> None:
        """Bloquea hasta que sea seguro hacer la siguiente petición."""
        if self._last_call is not None:
            elapsed = time.monotonic() - self._last_call
            remaining = self._min_interval - elapsed
            if remaining > 0:
                logger.debug("Rate limiter: durmiendo %.1fs.", remaining)
                time.sleep(remaining)
        self._last_call = time.monotonic()


# ---------------------------------------------------------------------------
# Cliente HTTP con manejo de errores y 429
# ---------------------------------------------------------------------------

def _request_json(
    url: str,
    session: requests.Session,
    limiter: _RateLimiter,
    params: dict | None = None,
) -> dict:
    """
    Realiza una petición GET y devuelve el JSON decodificado.

    Respeta el rate limiter, maneja respuestas 429 (Too Many Requests) leyendo
    el header Retry-After y reintenta ante errores de red/servidor.

    Args:
        url: URL completa del endpoint.
        session: Sesión de requests reutilizable.
        limiter: Rate limiter compartido entre llamadas.
        params: Parámetros de query (e.g. {"max": <cursor>}).

    Returns:
        Diccionario JSON de la respuesta.

    Raises:
        requests.HTTPError: Si tras los reintentos no se obtuvo una respuesta válida.
    """
    backoff = _RETRY_BACKOFF
    last_exc: Exception | None = None

    for attempt in range(1, _MAX_RETRIES + 1):
        limiter.wait()
        try:
            resp = session.get(url, params=params, headers=_HEADERS, timeout=20)

            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", backoff))
                logger.warning(
                    "429 Too Many Requests en %s. Esperando %ds (intento %d/%d).",
                    url, retry_after, attempt, _MAX_RETRIES,
                )
                time.sleep(retry_after)
                backoff *= 2
                continue

            resp.raise_for_status()
            return resp.json()

        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            logger.warning(
                "Error de red en %s (intento %d/%d): %s. Reintentando en %.0fs.",
                url, attempt, _MAX_RETRIES, exc, backoff,
            )
            time.sleep(backoff)
            backoff *= 2
        except requests.HTTPError as exc:
            # Errores 4xx/5xx que no son 429: no tiene sentido reintentar 4xx
            last_exc = exc
            status = exc.response.status_code if exc.response is not None else None
            if status is not None and 400 <= status < 500 and status != 429:
                logger.error("Error HTTP %s en %s: no se reintenta.", status, url)
                raise
            logger.warning(
                "Error HTTP en %s (intento %d/%d): %s.",
                url, attempt, _MAX_RETRIES, exc,
            )
            time.sleep(backoff)
            backoff *= 2

    raise requests.HTTPError(
        f"Fallaron todos los reintentos para {url}: {last_exc}"
    )


# ---------------------------------------------------------------------------
# Normalización de mensajes
# ---------------------------------------------------------------------------

def _parse_created_at(value: str) -> str:
    """
    Convierte el `created_at` de StockTwits (ISO 8601, e.g. '2024-01-15T14:30:00Z')
    al formato interno ISO UTC usado por el resto del pipeline.
    """
    dt = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(dt):
        # Fallback: marca temporal actual si el campo viene corrupto
        dt = pd.Timestamp.now(tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _extract_symbols(message: dict) -> list[str]:
    """
    Extrae los tickers del campo `symbols` del mensaje (NO usa regex).

    Normaliza a mayúsculas y convierte el separador '.' al '-' que usa Alpaca
    (BRK.B → BRK-B). Los símbolos de cripto de StockTwits ('BTC.X') quedan como
    'BTC-X' y simplemente no harán match con el universo S&P 500 aguas abajo.
    """
    symbols = message.get("symbols") or []
    out: list[str] = []
    for sym in symbols:
        raw = (sym.get("symbol") or "").strip().upper()
        if not raw:
            continue
        out.append(raw.replace(".", "-"))
    # Únicos y ordenados para reproducibilidad
    return sorted(set(out))


def _extract_native_sentiment(message: dict) -> str:
    """
    Extrae el label nativo de sentimiento del mensaje.

    StockTwits expone entities.sentiment.basic ∈ {"Bullish", "Bearish"} o null.
    Devuelve "Bullish", "Bearish" o "None".
    """
    entities = message.get("entities") or {}
    sentiment = entities.get("sentiment")
    if isinstance(sentiment, dict):
        basic = sentiment.get("basic")
        if basic in ("Bullish", "Bearish"):
            return basic
    return "None"


def normalize_message(message: dict) -> dict:
    """
    Normaliza un mensaje crudo de StockTwits al esquema interno del proyecto.

    Args:
        message: Dict de un mensaje tal como lo devuelve la API
                 (clave 'messages' del stream).

    Returns:
        Dict con las columnas de _OUTPUT_COLUMNS.
    """
    likes = message.get("likes") or {}
    return {
        "post_id": str(message.get("id", "")),
        "timestamp": _parse_created_at(message.get("created_at", "")),
        "title": (message.get("body") or "").strip(),
        "body": "",
        "score": int(likes.get("total", 0) or 0),
        "num_comments": 0,
        "detected_tickers": json.dumps(_extract_symbols(message)),
        "stocktwits_sentiment": _extract_native_sentiment(message),
    }


def normalize_messages(messages: list[dict]) -> list[dict]:
    """Normaliza una lista de mensajes crudos descartando los sin ticker."""
    rows: list[dict] = []
    for msg in messages:
        row = normalize_message(msg)
        # Sin símbolos no hay ticker al que asignar el sentimiento → se descarta
        if row["detected_tickers"] != "[]":
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Trending symbols
# ---------------------------------------------------------------------------

def fetch_trending_symbols(
    limit: int = 30,
    session: requests.Session | None = None,
    limiter: _RateLimiter | None = None,
) -> list[str]:
    """
    Trae los símbolos en tendencia (trending) de StockTwits.

    Args:
        limit: Número máximo de símbolos a devolver.
        session: Sesión de requests reutilizable (se crea una si es None).
        limiter: Rate limiter compartido (se crea uno si es None).

    Returns:
        Lista de tickers activos (e.g. ['AAPL', 'NVDA', ...]), en mayúsculas
        y con '.' → '-'.
    """
    session = session or requests.Session()
    limiter = limiter or _RateLimiter(STOCKTWITS_MAX_REQUESTS_PER_HOUR)

    logger.info("Consultando trending symbols (limit=%d)...", limit)
    data = _request_json(_TRENDING_URL, session, limiter)

    symbols = data.get("symbols") or []
    tickers: list[str] = []
    for sym in symbols:
        raw = (sym.get("symbol") or "").strip().upper()
        if raw:
            tickers.append(raw.replace(".", "-"))

    tickers = tickers[:limit]
    logger.info("Trending: %d símbolos activos: %s", len(tickers), tickers)
    return tickers


# ---------------------------------------------------------------------------
# Stream por símbolo (con paginación por cursor `max`)
# ---------------------------------------------------------------------------

def fetch_symbol_messages(
    symbol: str,
    max_messages: int = STOCKTWITS_MAX_MESSAGES_PER_SYMBOL,
    session: requests.Session | None = None,
    limiter: _RateLimiter | None = None,
) -> list[dict]:
    """
    Trae mensajes recientes de un símbolo, paginando con el cursor `max`.

    StockTwits devuelve ~30 mensajes por página y un objeto `cursor` con el
    campo `max` (id del mensaje más antiguo de la página) y `more` (bool).
    Para pedir la página siguiente se pasa `?max=<cursor.max>`.

    Args:
        symbol: Ticker a consultar (e.g. 'AAPL').
        max_messages: Tope de mensajes a recolectar para el símbolo.
        session: Sesión de requests reutilizable.
        limiter: Rate limiter compartido.

    Returns:
        Lista de mensajes crudos (dicts de la API), sin normalizar.
    """
    session = session or requests.Session()
    limiter = limiter or _RateLimiter(STOCKTWITS_MAX_REQUESTS_PER_HOUR)

    url = _STREAM_URL.format(symbol=symbol.upper())
    collected: list[dict] = []
    max_cursor: int | None = None
    pages = 0
    max_pages = (max_messages + _PAGE_SIZE - 1) // _PAGE_SIZE

    while len(collected) < max_messages and pages < max_pages:
        params = {"max": max_cursor} if max_cursor is not None else None
        data = _request_json(url, session, limiter, params=params)

        messages = data.get("messages") or []
        if not messages:
            break
        collected.extend(messages)
        pages += 1

        cursor = data.get("cursor") or {}
        if not cursor.get("more"):
            break
        next_max = cursor.get("max")
        if next_max is None or next_max == max_cursor:
            break
        max_cursor = next_max

    logger.info(
        "Símbolo %s: %d mensajes recolectados en %d página(s).",
        symbol, len(collected), pages,
    )
    return collected[:max_messages]


# ---------------------------------------------------------------------------
# Orquestación: trending → mensajes → CSV normalizado
# ---------------------------------------------------------------------------

def _save(rows: list[dict], output_dir: Path) -> Path:
    """Guarda las filas normalizadas como CSV y devuelve la ruta creada."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"stocktwits_{timestamp_str}.csv"
    pd.DataFrame(rows, columns=_OUTPUT_COLUMNS).to_csv(
        path, index=False, encoding="utf-8"
    )
    logger.info("Guardadas %d filas en %s", len(rows), path)
    return path


def scrape(
    symbols: list[str] | None = None,
    trending_limit: int = 20,
    max_messages_per_symbol: int = STOCKTWITS_MAX_MESSAGES_PER_SYMBOL,
    output_dir: Path = RAW_DIR,
) -> Path:
    """
    Punto de entrada principal: recolecta mensajes de StockTwits y los persiste.

    Si `symbols` es None, primero consulta los trending symbols y opera sobre
    ellos. Luego trae los mensajes recientes de cada símbolo, los normaliza al
    esquema interno y los guarda en un CSV.

    Args:
        symbols: Lista de tickers a consultar. Si None, usa trending.
        trending_limit: Cuántos trending symbols usar cuando symbols es None.
        max_messages_per_symbol: Tope de mensajes por símbolo.
        output_dir: Directorio de salida.

    Returns:
        Ruta al CSV generado.
    """
    session = requests.Session()
    limiter = _RateLimiter(STOCKTWITS_MAX_REQUESTS_PER_HOUR)

    if symbols is None:
        symbols = fetch_trending_symbols(trending_limit, session, limiter)
        if not symbols:
            logger.warning("Trending vacío; no hay símbolos que consultar.")

    all_rows: list[dict] = []
    seen_ids: set[str] = set()

    for symbol in symbols:
        try:
            raw_messages = fetch_symbol_messages(
                symbol, max_messages_per_symbol, session, limiter
            )
        except requests.HTTPError as exc:
            logger.error("No se pudo consultar %s: %s. Se omite.", symbol, exc)
            continue

        for row in normalize_messages(raw_messages):
            # Deduplicar: un mensaje puede aparecer en el stream de varios símbolos
            if row["post_id"] not in seen_ids:
                seen_ids.add(row["post_id"])
                all_rows.append(row)

    logger.info(
        "Total recolectado: %d mensajes únicos sobre %d símbolos.",
        len(all_rows), len(symbols),
    )
    return _save(all_rows, output_dir)


# ---------------------------------------------------------------------------
# Mensajes de muestra para el modo demo offline
# ---------------------------------------------------------------------------
# Estructura idéntica a la que devuelve la API real, para poder demostrar la
# normalización aun cuando el endpoint no sea accesible (p.ej. tras Cloudflare).
_DEMO_MESSAGES: list[dict] = [
    {
        "id": 600000001,
        "body": "$AAPL breaking out above resistance, calls printing 🚀",
        "created_at": "2024-06-20T14:32:10Z",
        "symbols": [{"symbol": "AAPL", "title": "Apple Inc."}],
        "entities": {"sentiment": {"basic": "Bullish"}},
        "likes": {"total": 12},
    },
    {
        "id": 600000002,
        "body": "$AAPL looks overextended here, taking profits and shorting.",
        "created_at": "2024-06-20T15:01:44Z",
        "symbols": [{"symbol": "AAPL", "title": "Apple Inc."}],
        "entities": {"sentiment": {"basic": "Bearish"}},
        "likes": {"total": 3},
    },
    {
        "id": 600000003,
        "body": "$AAPL earnings next week, no position yet. What's everyone thinking?",
        "created_at": "2024-06-20T15:20:05Z",
        "symbols": [{"symbol": "AAPL", "title": "Apple Inc."}],
        "entities": {"sentiment": None},
        "likes": {"total": 1},
    },
    {
        "id": 600000004,
        "body": "Rotating from $AAPL into $NVDA, AI momentum is unstoppable.",
        "created_at": "2024-06-20T16:05:33Z",
        "symbols": [
            {"symbol": "AAPL", "title": "Apple Inc."},
            {"symbol": "NVDA", "title": "NVIDIA Corp."},
        ],
        "entities": {"sentiment": {"basic": "Bullish"}},
        "likes": {"total": 27},
    },
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_demo() -> None:
    """
    Modo demo: imprime las primeras filas normalizadas de AAPL.

    Intenta usar la API real; si no es accesible (red/Cloudflare) cae a un
    conjunto de mensajes de muestra embebidos para demostrar la normalización.
    """
    print("\nModo demo - primeras filas normalizadas de $AAPL\n" + "-" * 70)
    try:
        raw = fetch_symbol_messages("AAPL", max_messages=10)
        if not raw:
            raise RuntimeError("respuesta vacía")
        source = "API real"
    except Exception as exc:  # noqa: BLE001 — el demo nunca debe fallar
        logger.warning(
            "No se pudo acceder a la API de StockTwits (%s). "
            "Usando mensajes de muestra embebidos.", exc,
        )
        raw = _DEMO_MESSAGES
        source = "muestra embebida (offline)"

    rows = normalize_messages(raw)
    df = pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)

    print(f"Fuente: {source}\n")
    print(df.head(10).to_string(index=False))
    print(f"\nColumnas de salida: {_OUTPUT_COLUMNS}")


if __name__ == "__main__":
    import argparse
    import sys

    # La consola de Windows usa cp1252 por defecto; los mensajes pueden traer
    # emojis/Unicode. Forzar UTF-8 para evitar UnicodeEncodeError al imprimir.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(
        description="Scraper de StockTwits (endpoint público read-only)."
    )
    parser.add_argument(
        "symbols", nargs="*", default=None,
        help="Símbolos a consultar (e.g. AAPL NVDA). Sin argumentos: modo demo.",
    )
    parser.add_argument(
        "--trending", type=int, default=None, metavar="N",
        help="Usar los N trending symbols en vez de una lista fija.",
    )
    parser.add_argument(
        "--max-messages", type=int, default=STOCKTWITS_MAX_MESSAGES_PER_SYMBOL,
        help="Tope de mensajes por símbolo.",
    )

    args = parser.parse_args()

    if not args.symbols and args.trending is None:
        _run_demo()
    else:
        out = scrape(
            symbols=args.symbols if args.symbols else None,
            trending_limit=args.trending if args.trending is not None else 20,
            max_messages_per_symbol=args.max_messages,
        )
        result = pd.read_csv(out)
        print(f"\nScraping completado → {out}  ({len(result)} mensajes)")
        print(result.head(8).to_string(index=False))
