"""
Genera un CSV sintético de posts de r/WallStreetBets para probar el pipeline
sin necesitar la API de Reddit.

Produce el mismo formato que scraper_reddit.py:
  post_id, timestamp, title, body, score, num_comments

Para pasar el filtro MIN_MENTIONS_PER_DAY=5 con datos sintéticos:
  - 2000 posts concentrados en 5 tickers del S&P 500 populares en WSB
  - Distribuidos en 30 días aleatorios del rango de indicadores (2024-01-02 → 2024-06-28)
  → ~2000 posts / 30 días / 5 tickers ≈ 13 menciones por (ticker, día) ✓

Uso:
  cd c:\dev\Tesis
  python scripts/generate_synthetic_posts.py
"""

import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)

# Tickers S&P 500 populares en WSB — todos en el caché de indicadores
FOCUS_TICKERS = ["AAPL", "NVDA", "TSLA", "AMD", "GME", "MSFT", "META", "AMZN"]

# 30 días de trading activos dentro del rango de indicadores
RANGE_START = datetime(2024, 1, 2, tzinfo=timezone.utc)
RANGE_END = datetime(2024, 6, 28, tzinfo=timezone.utc)
RANGE_DAYS = (RANGE_END - RANGE_START).days

# Seleccionar 30 días fijos (reproducibles con seed=42)
_all_offsets = list(range(RANGE_DAYS + 1))
random.shuffle(_all_offsets)
ACTIVE_DAYS = [RANGE_START + timedelta(days=d) for d in sorted(_all_offsets[:30])]

POSITIVE_TEMPLATES = [
    "{t} is going to the moon 🚀 loaded up on calls",
    "Just bought more ${t}. Not financial advice.",
    "{t} earnings beat expectations. Bullish AF.",
    "Why ${t} will hit new ATH by end of month — my DD",
    "{t} short squeeze incoming. Apes together strong.",
    "Bought the dip on {t}. This is the way.",
    "{t} breaking out of resistance. Technical analysis inside.",
    "My {t} calls are printing. 500% gains so far.",
    "Bull case for {t}: strong revenue growth, expanding margins.",
    "${t} is massively undervalued right now. Loading up.",
    "Adding to my {t} position every week. DCA gang.",
    "{t} to $500 EOY. Screenshot this.",
]

NEGATIVE_TEMPLATES = [
    "{t} is going to zero. Here is why.",
    "Lost everything on {t} puts. Market is rigged.",
    "{t} earnings miss. Dumping premarket.",
    "Short {t} — overvalued by any metric. My bear thesis.",
    "{t} insiders selling. Red flags everywhere.",
    "The {t} bubble is about to pop. Get out now.",
    "{t} guidance cut. This is not good.",
    "Why I lost $50k on {t} options in one day.",
    "{t} downgrade incoming. Bears win.",
    "Sold all my {t}. This company is cooked.",
]

NEUTRAL_TEMPLATES = [
    "{t} earnings tomorrow. What's everyone's position?",
    "Daily discussion: {t} — thoughts?",
    "Anyone else watching {t} today?",
    "{t} options flow analysis for this week.",
    "Confused about {t} valuation. Can someone explain?",
    "Is {t} a buy at current levels?",
    "What's the fair value for {t}?",
    "{t} weekly chart looks interesting.",
    "Rate my {t} position — am I cooked?",
    "{t} vs the market — who wins?",
]

BODIES = [
    "Not financial advice. Do your own research. I'm just an ape.",
    "I've done extensive DD on this. The fundamentals are strong and the technicals "
    "are aligning perfectly. RSI is oversold, MACD crossing bullish.",
    "Look at the chart. Clear cup and handle pattern forming. Target: +40%.",
    "Bought 100 contracts expiring Friday. Either I retire or I eat ramen.",
    "The institutional flow data shows massive call buying.",
    "Revenue up 23% YoY. EPS beat by $0.15. Margins expanding.",
    "Short interest at 25%. Any positive catalyst and this thing rockets.",
    "Management has been buying shares on the open market. Bullish.",
    "",
    "",
]


def _make_post(ticker: str, base_dt: datetime) -> dict:
    sentiment = random.choices(
        ["positive", "negative", "neutral"], weights=[0.45, 0.30, 0.25]
    )[0]

    if sentiment == "positive":
        title = random.choice(POSITIVE_TEMPLATES).format(t=ticker)
    elif sentiment == "negative":
        title = random.choice(NEGATIVE_TEMPLATES).format(t=ticker)
    else:
        title = random.choice(NEUTRAL_TEMPLATES).format(t=ticker)

    dt = base_dt.replace(hour=random.randint(8, 22), minute=random.randint(0, 59))

    return {
        "post_id": uuid.uuid4().hex[:7],
        "timestamp": dt.isoformat(),
        "title": title,
        "body": random.choice(BODIES),
        "score": random.randint(1, 15000),
        "num_comments": random.randint(0, 500),
    }


def generate(n: int = 2000) -> Path:
    rows = []
    for _ in range(n):
        ticker = random.choice(FOCUS_TICKERS)
        day = random.choice(ACTIVE_DAYS)
        rows.append(_make_post(ticker, day))

    df = pd.DataFrame(rows, columns=[
        "post_id", "timestamp", "title", "body", "score", "num_comments"
    ])

    out = RAW_DIR / "wsb_synthetic.csv"
    df.to_csv(out, index=False, encoding="utf-8")

    # Resumen para verificar que el filtro de menciones se cumplirá
    print(f"Generados {len(df)} posts sintéticos → {out}")
    print(f"Tickers usados: {FOCUS_TICKERS}")
    print(f"Días activos: {len(ACTIVE_DAYS)}")
    print(f"Menciones esperadas por (ticker, día): ~{n / len(ACTIVE_DAYS) / len(FOCUS_TICKERS):.1f}")
    return out


if __name__ == "__main__":
    generate(2000)
