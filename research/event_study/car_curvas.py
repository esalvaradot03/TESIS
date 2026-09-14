"""
Curvas de CAR día a día alrededor del evento, para el selector de ventana del dashboard.

`car_analysis.py` calcula los CAR agregados de ventanas fijas. El dashboard
muestra la **trayectoria**: el CAR acumulado en cada día relativo, con ventanas
de ±5, ±10 y ±20 días hábiles. Esto solo usa precios, así que es barato y no
requiere volver a puntuar texto.

Mismo modelo de mercado que el estudio: CAPM contra SPY, estimación en
[-250, -30] días hábiles con mínimo 100 observaciones.

OJO con las ventanas largas: el alpha estimado se extrapola a todos los días
de la ventana, así que en ±20 días (41 sesiones) domina la acumulación. Con
CRWD, por ejemplo, un alpha de 0,22% diario arrastra −9% en 41 días, más que
el CAR observado. Por eso cada curva se reporta también **sin alpha**
(retorno ajustado solo por mercado), que es lo que conviene leer en ±20.
La ventana [0,+5] del estudio principal casi no sufre este efecto.

Salida: research/event_study/car_curvas.csv con una fila por
(ticker, ventana, dia_rel) y el CAR medio de los eventos de ese ticker.

Uso:
    python -m research.event_study.car_curvas
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from research.event_study.car_analysis import (
    ESTIMACION,
    MIN_DIAS_ESTIMACION,
    SEGMENTO,
    cargar_eventos,
    cargar_precios,
)

logger = logging.getLogger(__name__)

VENTANAS: tuple[int, ...] = (5, 10, 20)
_SALIDA = Path("research/event_study/car_curvas.csv")


def trayectoria(
    retornos: pd.DataFrame, ticker: str, fecha: pd.Timestamp, ventana: int,
) -> pd.DataFrame | None:
    """
    CAR acumulado en cada día relativo de la ventana, para un evento.

    Args:
        retornos: Retornos diarios (cargar_precios de car_analysis).
        ticker: Ticker del evento.
        fecha: Fecha del evento.
        ventana: Semiancho en días hábiles.

    Returns:
        DataFrame indexado por día relativo (−ventana..+ventana) con las
        columnas `car` (modelo de mercado completo) y `car_sin_alpha`
        (ajustado solo por mercado), o None si no hay datos suficientes para
        estimar el modelo.
    """
    if ticker not in retornos.columns:
        return None
    serie = retornos[[ticker, "SPY"]].dropna()
    posteriores = serie.index[serie.index >= fecha]
    if posteriores.empty:
        return None
    t0 = serie.index.get_loc(posteriores[0])

    ini_est, fin_est = t0 + ESTIMACION[0], t0 + ESTIMACION[1]
    if ini_est < 0 or fin_est <= ini_est:
        return None
    estimacion = serie.iloc[ini_est:fin_est]
    if len(estimacion) < MIN_DIAS_ESTIMACION:
        return None

    beta, alpha = np.polyfit(estimacion["SPY"].to_numpy(), estimacion[ticker].to_numpy(), 1)

    ini, fin = t0 - ventana, t0 + ventana + 1
    if ini < 0 or fin > len(serie):
        return None
    tramo = serie.iloc[ini:fin]
    anormal = tramo[ticker] - (alpha + beta * tramo["SPY"])
    sin_alpha = tramo[ticker] - beta * tramo["SPY"]
    return pd.DataFrame(
        {"car": anormal.cumsum().to_numpy(), "car_sin_alpha": sin_alpha.cumsum().to_numpy()},
        index=pd.Index(range(-ventana, ventana + 1), name="dia_rel"),
    )


def correr(salida: Path = _SALIDA, ventanas: tuple[int, ...] = VENTANAS) -> pd.DataFrame:
    """
    Calcula las curvas medias por ticker y ventana, y escribe el CSV.

    Args:
        salida: CSV de salida.
        ventanas: Semianchos a calcular.

    Returns:
        DataFrame con ticker, segmento, ventana, dia_rel, car_medio y n_eventos.
    """
    eventos = cargar_eventos()
    retornos = cargar_precios()

    filas: list[dict] = []
    for ventana in ventanas:
        curvas: dict[str, list[pd.Series]] = {}
        for _, evento in eventos.iterrows():
            curva = trayectoria(retornos, evento["ticker"], evento["fecha"], ventana)
            if curva is not None:
                curvas.setdefault(evento["ticker"], []).append(curva)

        for ticker, lista in curvas.items():
            media = sum(lista) / len(lista)
            for dia_rel, fila in media.iterrows():
                filas.append({
                    "ticker": ticker,
                    "segmento": SEGMENTO[ticker],
                    "ventana": ventana,
                    "dia_rel": int(dia_rel),
                    "car_medio": round(float(fila["car"]), 6),
                    "car_medio_sin_alpha": round(float(fila["car_sin_alpha"]), 6),
                    "n_eventos": len(lista),
                })
        logger.info("Ventana ±%d: %d tickers con curva.", ventana, len(curvas))

    tabla = pd.DataFrame(filas)
    salida.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(salida, index=False, encoding="utf-8")
    logger.info("Curvas escritas en %s (%d filas).", salida, len(tabla))
    return tabla


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    tabla = correr()
    print("\n=== CAR final por ticker y ventana (%) ===")
    final = tabla[tabla["dia_rel"] == tabla.groupby(["ticker", "ventana"])["dia_rel"].transform("max")]
    print((final.pivot(index="ticker", columns="ventana", values="car_medio") * 100).round(2).to_string())
    print("\nEventos por ticker:")
    print(tabla.groupby("ticker")["n_eventos"].max().to_string())
