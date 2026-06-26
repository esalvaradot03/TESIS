"""
Configuración global del sistema: carga variables de entorno y define parámetros.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Rutas ---
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"

# --- Reproducibilidad ---
SEED: int = int(os.getenv("SEED", "42"))

# --- Reddit (legacy: la API fue negada en 2026; se mantiene solo para el
#     path histórico de WSB en historical_loader.py) ---
REDDIT_CLIENT_ID: str = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET: str = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT: str = os.getenv("REDDIT_USER_AGENT", "trading_sentiment_bot/1.0")

# --- StockTwits (fuente principal de sentimiento; endpoint público read-only) ---
# La API tolera ~200 req/h sin autenticar; se limita por debajo para no abusar.
STOCKTWITS_MAX_REQUESTS_PER_HOUR: int = int(
    os.getenv("STOCKTWITS_MAX_REQUESTS_PER_HOUR", "150")
)
STOCKTWITS_MAX_MESSAGES_PER_SYMBOL: int = int(
    os.getenv("STOCKTWITS_MAX_MESSAGES_PER_SYMBOL", "60")
)

# --- Alpaca ---
ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
ALPACA_API_SECRET: str = os.getenv("ALPACA_API_SECRET", "")
ALPACA_BASE_URL: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
# La URL de datos es independiente del entorno paper/live
ALPACA_DATA_URL: str = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
# "iex" = feed gratuito; "sip" = feed completo (plan de pago)
ALPACA_FEED: str = os.getenv("ALPACA_FEED", "iex")
# Tickers por petición bulk a la API de Alpaca
PRICE_CHUNK_SIZE: int = int(os.getenv("PRICE_CHUNK_SIZE", "100"))

# --- Filtro de universo ---
MIN_MENTIONS_PER_DAY: int = 5  # tickers con menos menciones se ignoran ese día

# --- Executor de paper trading ---
MAX_OPEN_POSITIONS: int = int(os.getenv("MAX_OPEN_POSITIONS", "20"))
# Mínimo en USD por orden (Alpaca rechaza órdenes notional < $1)
MIN_ORDER_NOTIONAL: float = float(os.getenv("MIN_ORDER_NOTIONAL", "1.0"))

# --- FinBERT ---
FINBERT_BATCH_SIZE: int = 32
FINBERT_MODEL_NAME: str = "ProsusAI/finbert"
# Longitud máxima de tokens. Los mensajes de StockTwits son cortos (≤280 chars
# ≈ 70 tokens), así que 128 captura el texto completo y acelera la inferencia en
# CPU varias veces frente a 512. Configurable por si se procesan textos largos.
FINBERT_MAX_LENGTH: int = int(os.getenv("FINBERT_MAX_LENGTH", "128"))
