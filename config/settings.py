"""
Configuración global del sistema: carga variables de entorno y define parámetros.
"""

import logging
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

# --- Datos externos (FUERA del repo, en disco aparte D:\trading-data) ---
# Pesan decenas/cientos de GB; los descarga src/data/setup_external_data.py.
# La raíz es configurable con TESIS_DATA_ROOT por si el disco cambia de letra.
EXTERNAL_DATA_ROOT: Path = Path(os.getenv("TESIS_DATA_ROOT", "D:/trading-data"))

# Sub-rutas derivadas (no se crean aquí; las crea el script de setup)
STOCKTWITS_NYU_SYMBOL_SENTIMENTS: Path = EXTERNAL_DATA_ROOT / "stocktwits_nyu" / "symbol_sentiments"
STOCKTWITS_NYU_FEATURES: Path = EXTERNAL_DATA_ROOT / "stocktwits_nyu" / "feature_wo_messages"
STOCKTWITS_NYU_MESSAGES: Path = EXTERNAL_DATA_ROOT / "stocktwits_nyu" / "messages"
WSB_KEVIN: Path = EXTERNAL_DATA_ROOT / "wsb_kaggle" / "kevinwang313_wallstreetbets"
WSB_UNANIMAD: Path = EXTERNAL_DATA_ROOT / "wsb_kaggle" / "unanimad_reddit_rwallstreetbets"
FNSPID: Path = EXTERNAL_DATA_ROOT / "fnspid"

# Verificación NO bloqueante: si las rutas externas no existen, solo advierte.
_EXTERNAL_PATHS = {
    "EXTERNAL_DATA_ROOT": EXTERNAL_DATA_ROOT,
    "STOCKTWITS_NYU_SYMBOL_SENTIMENTS": STOCKTWITS_NYU_SYMBOL_SENTIMENTS,
    "STOCKTWITS_NYU_FEATURES": STOCKTWITS_NYU_FEATURES,
    "STOCKTWITS_NYU_MESSAGES": STOCKTWITS_NYU_MESSAGES,
    "WSB_KEVIN": WSB_KEVIN,
    "WSB_UNANIMAD": WSB_UNANIMAD,
    "FNSPID": FNSPID,
}
_missing_external = [name for name, p in _EXTERNAL_PATHS.items() if not p.exists()]
if _missing_external:
    logging.getLogger(__name__).warning(
        "Datos externos no encontrados (%d/%d rutas). Corré "
        "`python -m src.data.setup_external_data`. Faltan: %s",
        len(_missing_external), len(_EXTERNAL_PATHS), ", ".join(_missing_external),
    )

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

# --- Linear probing sobre FinBERT (pivote agosto 2026: comparación de enfoques) ---
FINBERT_FINETUNED_MODEL_PATH: Path = MODELS_DIR / "finbert_finetuned"
FINETUNE_LR_HEAD: float = float(os.getenv("FINETUNE_LR_HEAD", "1e-4"))
FINETUNE_EPOCHS: int = int(os.getenv("FINETUNE_EPOCHS", "3"))
FINETUNE_BATCH_SIZE: int = int(os.getenv("FINETUNE_BATCH_SIZE", "16"))

# --- Labeling manual / active learning ---
LABELED_STORE_FILE: Path = DATA_DIR / "labeled" / "labeled_posts.csv"
UNCERTAINTY_SAMPLE_SIZE: int = int(os.getenv("UNCERTAINTY_SAMPLE_SIZE", "100"))
