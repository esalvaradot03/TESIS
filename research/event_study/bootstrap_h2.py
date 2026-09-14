"""
Contraste de H2 del pre-registro: ¿la correlación reactiva supera a la anticipatoria?

H2 dice que el sentimiento POSTERIOR al evento correlaciona con el CAR más que
el sentimiento PREVIO. Reportar las dos correlaciones por separado no lo prueba:
hay que contrastar la diferencia, y las dos comparten la misma variable de
retorno, así que no son independientes.

Método (sección 4.3 del pre-registro): bootstrap pareado sobre eventos. En cada
remuestreo se toman 55 eventos con reemplazo y se recalculan ρ_post y ρ_pre
sobre esa misma muestra, de modo que la dependencia entre ambas se preserva. El
p-valor unilateral es la fracción de remuestreos donde la diferencia no es
positiva.

Lee research/event_study/car_eventos.csv, que produce car_analysis.py, así que
no necesita volver a puntuar mensajes.

Uso:
    python -m research.event_study.bootstrap_h2
    python -m research.event_study.bootstrap_h2 --repeticiones 10000
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config.settings import SEED

logger = logging.getLogger(__name__)

ENFOQUES: tuple[str, ...] = ("finbert", "probe", "vader")
_EVENTOS = Path("research/event_study/car_eventos.csv")
_SALIDA = Path("research/event_study/bootstrap_h2.csv")
REPETICIONES = 10_000


def contrastar(
    eventos: pd.DataFrame, enfoque: str, repeticiones: int = REPETICIONES, seed: int = SEED,
) -> dict:
    """
    Bootstrap pareado de la diferencia ρ_post − ρ_pre para un enfoque.

    Args:
        eventos: Tabla por evento de car_analysis.py.
        enfoque: Nombre del enfoque (columnas sent_pre_<enfoque> y sent_post_<enfoque>).
        repeticiones: Remuestreos del bootstrap.
        seed: Semilla.

    Returns:
        Dict con las correlaciones observadas, la diferencia, su IC95 y el
        p-valor unilateral de H2 (post > pre).
    """
    columnas = [f"sent_pre_{enfoque}", f"sent_post_{enfoque}", "car_post"]
    datos = eventos[columnas].dropna()
    pre = datos[columnas[0]].to_numpy()
    post = datos[columnas[1]].to_numpy()
    car = datos["car_post"].to_numpy()

    rho_pre = float(spearmanr(pre, car).statistic)
    rho_post = float(spearmanr(post, car).statistic)
    observada = rho_post - rho_pre

    rng = np.random.default_rng(seed)
    n = len(datos)
    diferencias = np.empty(repeticiones)
    for i in range(repeticiones):
        idx = rng.integers(0, n, n)
        # Un remuestreo sin variación en alguna serie deja rho indefinido: se descarta.
        if len(np.unique(car[idx])) < 3 or len(np.unique(post[idx])) < 3 or len(np.unique(pre[idx])) < 3:
            diferencias[i] = np.nan
            continue
        diferencias[i] = (spearmanr(post[idx], car[idx]).statistic
                          - spearmanr(pre[idx], car[idx]).statistic)

    validas = diferencias[~np.isnan(diferencias)]
    p_unilateral = float((validas <= 0).mean())
    bajo, alto = np.percentile(validas, [2.5, 97.5])

    return {
        "enfoque": enfoque,
        "n_eventos": n,
        "rho_pre": round(rho_pre, 4),
        "rho_post": round(rho_post, 4),
        "diferencia": round(observada, 4),
        "ic95_bajo": round(float(bajo), 4),
        "ic95_alto": round(float(alto), 4),
        "p_unilateral": round(p_unilateral, 4),
        "remuestreos_validos": int(len(validas)),
        "h2_se_sostiene": bool(p_unilateral < 0.05),
    }


def correr(
    eventos_path: Path = _EVENTOS, repeticiones: int = REPETICIONES, salida: Path = _SALIDA,
) -> pd.DataFrame:
    """
    Corre el contraste para los tres enfoques y escribe el CSV.

    Args:
        eventos_path: CSV por evento de car_analysis.py.
        repeticiones: Remuestreos del bootstrap.
        salida: CSV de resultados.

    Returns:
        DataFrame con una fila por enfoque.

    Raises:
        FileNotFoundError: si no existe el CSV de eventos.
    """
    if not eventos_path.exists():
        raise FileNotFoundError(
            f"No existe {eventos_path}. Corré primero: python -m research.event_study.car_analysis"
        )
    eventos = pd.read_csv(eventos_path)
    logger.info("Eventos cargados: %d | bootstrap de %d remuestreos.", len(eventos), repeticiones)

    tabla = pd.DataFrame([contrastar(eventos, e, repeticiones) for e in ENFOQUES])
    salida.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(salida, index=False, encoding="utf-8")
    logger.info("Resultados escritos en %s", salida)
    return tabla


if __name__ == "__main__":
    import argparse
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Bootstrap de la diferencia rho_post - rho_pre (H2).")
    parser.add_argument("--repeticiones", type=int, default=REPETICIONES,
                        help="Remuestreos del bootstrap (default: 10000).")
    args = parser.parse_args()

    tabla = correr(repeticiones=args.repeticiones)
    print("\n=== H2: ¿rho_post > rho_pre? (bootstrap pareado) ===")
    print(tabla.to_string(index=False))
    print("\nCriterio del pre-registro: H2 se sostiene si p unilateral < 0.05.")
