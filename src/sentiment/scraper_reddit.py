"""
Scraper de r/WallStreetBets usando PRAW.

Recolecta posts de los feeds hot, new y rising sin filtrar por ticker.
El análisis de tickers se delega a ticker_detector.py.
Guarda resultados en data/raw/wsb_<YYYYMMDD_HHMMSS>.csv.
"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import praw
import prawcore

from config.settings import (
    RAW_DIR,
    REDDIT_CLIENT_ID,
    REDDIT_CLIENT_SECRET,
    REDDIT_USER_AGENT,
)

logger = logging.getLogger(__name__)

# Feeds que se recolectan en cada ejecución
_FEEDS: list[str] = ["hot", "new", "rising"]

# Máximo de posts por feed (Reddit permite hasta 1000 con paginación, pero
# en la práctica hot/rising raramente superan 300 resultados únicos)
_LIMIT_PER_FEED: int = 500

# Pausa entre llamadas a la API para respetar el rate limit de 60 req/min
_SLEEP_BETWEEN_FEEDS: float = 2.0

# Reintentos ante errores de red/servidor
_MAX_RETRIES: int = 3
_RETRY_BACKOFF: float = 5.0  # segundos; se duplica en cada reintento


def _build_reddit_client() -> praw.Reddit:
    """Instancia y devuelve un cliente PRAW en modo read-only."""
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET:
        raise EnvironmentError(
            "REDDIT_CLIENT_ID y REDDIT_CLIENT_SECRET deben estar definidos en .env"
        )
    return praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT,
        ratelimit_seconds=300,  # espera hasta 5 min si PRAW detecta rate limit
    )


def _fetch_feed(
    subreddit: praw.models.Subreddit,
    feed: str,
    limit: int,
) -> list[dict]:
    """
    Descarga hasta `limit` posts de un feed específico con reintentos.

    Args:
        subreddit: Objeto subreddit de PRAW.
        feed: Nombre del feed ('hot', 'new', 'rising').
        limit: Número máximo de posts a obtener.

    Returns:
        Lista de dicts con los campos del post.
    """
    feed_fn = getattr(subreddit, feed)
    records: list[dict] = []
    attempt = 0
    backoff = _RETRY_BACKOFF

    while attempt < _MAX_RETRIES:
        try:
            for submission in feed_fn(limit=limit):
                records.append(_submission_to_dict(submission))
            logger.info("Feed '%s': %d posts recolectados.", feed, len(records))
            return records
        except prawcore.exceptions.TooManyRequests as exc:
            wait = int(exc.response.headers.get("Retry-After", backoff))
            logger.warning("Rate limit alcanzado en feed '%s'. Esperando %ds.", feed, wait)
            time.sleep(wait)
        except prawcore.exceptions.ServerError as exc:
            logger.warning(
                "Error de servidor en feed '%s' (intento %d/%d): %s",
                feed, attempt + 1, _MAX_RETRIES, exc,
            )
            time.sleep(backoff)
            backoff *= 2
        except prawcore.exceptions.RequestException as exc:
            logger.warning(
                "Error de red en feed '%s' (intento %d/%d): %s",
                feed, attempt + 1, _MAX_RETRIES, exc,
            )
            time.sleep(backoff)
            backoff *= 2
        attempt += 1

    logger.error("Feed '%s': fallaron todos los reintentos. Se omite.", feed)
    return records


def _submission_to_dict(submission: praw.models.Submission) -> dict:
    """Extrae los campos relevantes de un submission de PRAW."""
    return {
        "post_id": submission.id,
        "timestamp": datetime.fromtimestamp(submission.created_utc, tz=timezone.utc).isoformat(),
        "title": submission.title,
        "body": submission.selftext if submission.selftext != "[removed]" else "",
        "score": submission.score,
        "num_comments": submission.num_comments,
    }


def _deduplicate(records: list[dict]) -> list[dict]:
    """Elimina posts repetidos (un mismo post puede aparecer en varios feeds)."""
    seen: set[str] = set()
    unique: list[dict] = []
    for rec in records:
        if rec["post_id"] not in seen:
            seen.add(rec["post_id"])
            unique.append(rec)
    return unique


def _save(records: list[dict], output_dir: Path) -> Path:
    """Guarda la lista de posts como CSV y devuelve la ruta del archivo creado."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"wsb_{timestamp_str}.csv"
    columns = ["post_id", "timestamp", "title", "body", "score", "num_comments"]
    pd.DataFrame(records, columns=columns).to_csv(path, index=False, encoding="utf-8")
    logger.info("Guardados %d posts en %s", len(records), path)
    return path


def scrape(
    limit_per_feed: int = _LIMIT_PER_FEED,
    output_dir: Path = RAW_DIR,
) -> Path:
    """
    Punto de entrada principal: recolecta posts de r/WallStreetBets y los persiste.

    Recorre los feeds hot, new y rising, deduplica por post_id y guarda
    el resultado en un CSV dentro de output_dir.

    Args:
        limit_per_feed: Máximo de posts a pedir por feed.
        output_dir: Directorio donde se escribe el CSV de salida.

    Returns:
        Ruta al CSV generado.
    """
    reddit = _build_reddit_client()
    wsb = reddit.subreddit("wallstreetbets")
    all_records: list[dict] = []

    for i, feed in enumerate(_FEEDS):
        logger.info("Recolectando feed '%s'...", feed)
        records = _fetch_feed(wsb, feed, limit_per_feed)
        all_records.extend(records)
        if i < len(_FEEDS) - 1:
            time.sleep(_SLEEP_BETWEEN_FEEDS)

    unique = _deduplicate(all_records)
    logger.info(
        "Total tras deduplicación: %d posts (%d duplicados eliminados).",
        len(unique),
        len(all_records) - len(unique),
    )
    return _save(unique, output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    output_path = scrape()
    print(f"Scraping completado → {output_path}")
