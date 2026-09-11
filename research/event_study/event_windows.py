"""
Descarga fechas de eventos catalizadores (earnings, upgrades/downgrades de
analistas) para los 6 tickers del event study, vía yfinance.

Este módulo solo descarga y persiste los eventos crudos en
data/events/eventos_{TICKER}.csv. La construcción de ventanas [-5,+5] días
de trading alrededor de cada evento y el cálculo de CAR (CAPM/SPY) se hacen
en car_analysis.py, una vez exista docs/pre_registro_event_study.md — por
protocolo de pre-registro del proyecto, ninguna hipótesis sobre CAR se corre
sin la dirección comprometida por escrito primero.
"""

import logging
from pathlib import Path

import pandas as pd

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)

# Los 6 tickers del event study (mismo universo que build_corpus.py).
TICKERS: list[str] = ["NCLH", "DIS", "CRWD", "TGT", "CMG", "DDOG"]

_EVENTS_DIR = DATA_DIR / "events"

_OUTPUT_COLUMNS: list[str] = ["fecha", "tipo_evento", "descripcion"]


# ---------------------------------------------------------------------------
# Earnings
# ---------------------------------------------------------------------------

def fetch_earnings_events(ticker: str) -> pd.DataFrame:
    """
    Descarga fechas de earnings vía yfinance.Ticker(ticker).earnings_dates.

    Args:
        ticker: Símbolo bursátil (e.g. "DIS").

    Returns:
        DataFrame con columnas _OUTPUT_COLUMNS, tipo_evento="earnings".
        Vacío (con warning) si yfinance no está instalado o la consulta falla
        — no lanza excepción, para que build_events() pueda seguir con la
        otra fuente aunque esta no esté disponible.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance no está instalado en este entorno. Sin earnings para %s.", ticker)
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    try:
        dates_df = yf.Ticker(ticker).earnings_dates
    except Exception as exc:  # noqa: BLE001 — yfinance cambia su API entre versiones
        logger.warning("earnings_dates falló para %s (%s).", ticker, exc)
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    if dates_df is None or dates_df.empty:
        logger.warning("Sin earnings_dates para %s.", ticker)
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    rows: list[dict] = []
    for idx, row in dates_df.iterrows():
        fecha = pd.Timestamp(idx).date().isoformat()
        eps_est = row.get("EPS Estimate")
        eps_rep = row.get("Reported EPS")
        surprise = row.get("Surprise(%)")
        descripcion = f"EPS estimado={eps_est}, reportado={eps_rep}, sorpresa%={surprise}"
        rows.append({"fecha": fecha, "tipo_evento": "earnings", "descripcion": descripcion})

    result = pd.DataFrame(rows, columns=_OUTPUT_COLUMNS).drop_duplicates(subset=["fecha", "tipo_evento"])
    logger.info("%s: %d fechas de earnings.", ticker, len(result))
    return result


# ---------------------------------------------------------------------------
# Upgrades / downgrades de analistas
# ---------------------------------------------------------------------------

def fetch_upgrade_downgrade_events(ticker: str) -> pd.DataFrame:
    """
    Descarga upgrades/downgrades de analistas vía
    yfinance.Ticker(ticker).upgrades_downgrades — fuente gratuita, ya es
    dependencia del proyecto (ver requirements.txt), sin necesidad de una
    API adicional de pago.

    Args:
        ticker: Símbolo bursátil.

    Returns:
        DataFrame con columnas _OUTPUT_COLUMNS, tipo_evento="upgrade_downgrade".
        Vacío (con warning) si yfinance no está instalado, el atributo no
        existe en la versión instalada, o la consulta falla.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance no está instalado en este entorno. Sin upgrades/downgrades para %s.", ticker)
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    try:
        ud_df = yf.Ticker(ticker).upgrades_downgrades
    except Exception as exc:  # noqa: BLE001 — yfinance cambia su API entre versiones
        logger.warning("upgrades_downgrades falló para %s (%s).", ticker, exc)
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    if ud_df is None or ud_df.empty:
        logger.warning("Sin upgrades/downgrades para %s.", ticker)
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    rows: list[dict] = []
    for idx, row in ud_df.iterrows():
        fecha = pd.Timestamp(idx).date().isoformat()
        firm = row.get("Firm", "")
        from_grade = row.get("FromGrade", "")
        to_grade = row.get("ToGrade", "")
        action = row.get("Action", "")
        descripcion = f"{firm}: {from_grade} → {to_grade} ({action})"
        rows.append({"fecha": fecha, "tipo_evento": "upgrade_downgrade", "descripcion": descripcion})

    result = pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)
    logger.info("%s: %d eventos de upgrade/downgrade.", ticker, len(result))
    return result


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

def build_events(ticker: str) -> pd.DataFrame:
    """
    Combina earnings + upgrades/downgrades de un ticker, ordenados por fecha.

    Args:
        ticker: Símbolo bursátil.

    Returns:
        DataFrame con columnas _OUTPUT_COLUMNS. Puede estar vacío si ambas
        fuentes fallaron o no devolvieron datos.
    """
    events = pd.concat(
        [fetch_earnings_events(ticker), fetch_upgrade_downgrade_events(ticker)],
        ignore_index=True,
    )
    if events.empty:
        return events
    return events.sort_values("fecha").reset_index(drop=True)


def save_events(ticker: str, events: pd.DataFrame, output_dir: Path = _EVENTS_DIR) -> Path:
    """
    Persiste los eventos de un ticker en data/events/eventos_{TICKER}.csv.

    Args:
        ticker: Símbolo bursátil.
        events: DataFrame de build_events().
        output_dir: Directorio de salida.

    Returns:
        Ruta al CSV escrito.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"eventos_{ticker}.csv"
    events.to_csv(path, index=False, encoding="utf-8")
    logger.info("%s: %d eventos guardados en %s.", ticker, len(events), path)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    target_tickers = sys.argv[1:] if len(sys.argv) > 1 else TICKERS

    for ticker in target_tickers:
        events = build_events(ticker)
        if events.empty:
            print(f"{ticker}: sin eventos (ver warnings arriba).")
            continue
        result_path = save_events(ticker, events)
        print(f"{ticker}: {len(events)} eventos → {result_path}")
