"""
Verifica si el mercado NYSE esta abierto usando Alpaca API.
Alpaca ya considera festivos automaticamente.

Exit code 0 = mercado abierto (el pipeline debe correr)
Exit code 1 = mercado cerrado (GitHub Actions cancela el pipeline)

Uso:
    python scripts/check_market_open.py
"""

import logging
import sys

from alpaca.trading.client import TradingClient

from config.settings import ALPACA_API_KEY, ALPACA_API_SECRET

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def check_market_open() -> bool:
    """Devuelve True si el mercado NYSE esta abierto ahora mismo."""
    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        logger.error("ALPACA_API_KEY / ALPACA_API_SECRET no configurados.")
        return False

    client = TradingClient(
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_API_SECRET,
        paper=True,
    )
    clock = client.get_clock()

    if clock.is_open:
        close_str = clock.next_close.strftime("%H:%M UTC") if clock.next_close else "?"
        logger.info("MERCADO ABIERTO - cierre a las %s", close_str)
    else:
        open_str = clock.next_open.strftime("%Y-%m-%d %H:%M UTC") if clock.next_open else "?"
        logger.info("MERCADO CERRADO - proxima apertura: %s", open_str)

    return bool(clock.is_open)


if __name__ == "__main__":
    is_open = check_market_open()
    print("OPEN" if is_open else "CLOSED")
    sys.exit(0 if is_open else 1)
