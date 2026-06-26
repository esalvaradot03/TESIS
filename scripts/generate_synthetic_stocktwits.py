"""
Genera un CSV sintético de mensajes de StockTwits para probar el pipeline
sin depender del acceso en vivo a la API (que puede estar tras Cloudflare).

Construye mensajes CRUDOS con la misma estructura que devuelve la API real y
los pasa por scraper_stocktwits.normalize_messages(), de modo que se ejercita
exactamente la misma normalización que en producción. El resultado tiene el
esquema interno StockTwits:

  post_id, timestamp, title, body, score, num_comments,
  detected_tickers, stocktwits_sentiment

Para pasar el filtro MIN_MENTIONS_PER_DAY=5:
  - 2000 mensajes concentrados en 8 tickers populares del S&P 500
  - Distribuidos en 30 días dentro del rango de indicadores (2024-01-02 → 2024-06-28)
  → ~2000 / 30 / 8 ≈ 8 menciones por (ticker, día) ✓

Uso:
  cd c:\dev\Tesis
  python scripts/generate_synthetic_stocktwits.py
"""

import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sentiment.scraper_stocktwits import (  # noqa: E402
    _OUTPUT_COLUMNS,
    normalize_messages,
)

RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

# Tickers S&P 500 populares en StockTwits — todos en el caché de indicadores
FOCUS_TICKERS = ["AAPL", "NVDA", "TSLA", "AMD", "GME", "MSFT", "META", "AMZN"]

RANGE_START = datetime(2024, 1, 2, tzinfo=timezone.utc)
RANGE_END = datetime(2024, 6, 28, tzinfo=timezone.utc)
RANGE_DAYS = (RANGE_END - RANGE_START).days

_all_offsets = list(range(RANGE_DAYS + 1))
random.shuffle(_all_offsets)
ACTIVE_DAYS = [RANGE_START + timedelta(days=d) for d in sorted(_all_offsets[:30])]

# (plantilla de mensaje, label nativo de StockTwits)
BULLISH_TEMPLATES = [
    "${t} breaking out above resistance, calls printing",
    "Loading up on ${t}, this runs to new highs",
    "${t} earnings beat, bullish into next week",
    "${t} short squeeze setting up, longs in control",
    "Adding to ${t} every dip, easy hold",
    "${t} momentum is unstoppable here",
]

BEARISH_TEMPLATES = [
    "${t} overextended, taking profits and shorting",
    "${t} looks toppy, puts loaded",
    "${t} guidance weak, fading this pop",
    "Short ${t}, valuation makes no sense",
    "${t} breaking support, more downside coming",
    "${t} insiders selling, bearish",
]

NONE_TEMPLATES = [
    "${t} earnings tomorrow, no position yet",
    "Watching ${t} at this level, thoughts?",
    "${t} chart looks interesting, undecided",
    "Anyone following ${t} today?",
    "${t} consolidating, waiting for a signal",
]


def _make_raw_message(ticker: str, base_dt: datetime, msg_id: int) -> dict:
    """Construye un mensaje crudo con la estructura de la API de StockTwits."""
    kind = random.choices(["Bullish", "Bearish", "None"], weights=[0.45, 0.30, 0.25])[0]
    if kind == "Bullish":
        body = random.choice(BULLISH_TEMPLATES)
        sentiment = {"basic": "Bullish"}
    elif kind == "Bearish":
        body = random.choice(BEARISH_TEMPLATES)
        sentiment = {"basic": "Bearish"}
    else:
        body = random.choice(NONE_TEMPLATES)
        sentiment = None

    body = body.replace("${t}", f"${ticker}")
    dt = base_dt.replace(hour=random.randint(8, 22), minute=random.randint(0, 59))

    # Ocasionalmente un segundo cashtag, como ocurre en mensajes reales
    symbols = [{"symbol": ticker, "title": ticker}]
    if random.random() < 0.15:
        other = random.choice([t for t in FOCUS_TICKERS if t != ticker])
        symbols.append({"symbol": other, "title": other})

    return {
        "id": msg_id,
        "body": body,
        "created_at": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "symbols": symbols,
        "entities": {"sentiment": sentiment},
        "likes": {"total": random.randint(0, 80)},
    }


def generate(n: int = 2000) -> Path:
    raw_messages = []
    for i in range(n):
        ticker = random.choice(FOCUS_TICKERS)
        day = random.choice(ACTIVE_DAYS)
        raw_messages.append(_make_raw_message(ticker, day, msg_id=700_000_000 + i))

    rows = normalize_messages(raw_messages)
    df = pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)

    out = RAW_DIR / "stocktwits_synthetic.csv"
    df.to_csv(out, index=False, encoding="utf-8")

    print(f"Generados {len(df)} mensajes sintéticos de StockTwits -> {out}")
    print(f"Tickers usados: {FOCUS_TICKERS}")
    print(f"Días activos: {len(ACTIVE_DAYS)}")
    print(f"Menciones esperadas por (ticker, día): ~{n / len(ACTIVE_DAYS) / len(FOCUS_TICKERS):.1f}")
    dist = df["stocktwits_sentiment"].value_counts().to_dict()
    print(f"Distribución de sentimiento nativo: {dist}")
    return out


if __name__ == "__main__":
    generate(2000)
