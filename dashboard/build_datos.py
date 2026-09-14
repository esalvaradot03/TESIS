"""
Arma el paquete de datos REALES que consume el dashboard del event study.

El HTML original (`Dashboard-Sentimiento-Retorno.html`) generaba todo con un
generador pseudoaleatorio: precios, capitalizaciones, mensajes, eventos y
métricas eran sintéticos. Este script reemplaza esa capa por los resultados
del repositorio y escribe un único `dashboard/datos.json`.

Fuentes (todas reales, ninguna simulada):
  - research/event_study/sentimiento_diario.csv  sentimiento por (ticker, día)
  - research/event_study/car_curvas.csv          trayectorias de CAR por ventana
  - research/event_study/car_eventos.csv         los 55 eventos con su CAR
  - research/event_study/car_resultados.csv      correlaciones, direccional, placebos
  - research/event_study/bootstrap_h2.csv        contraste rho_post vs rho_pre
  - research/event_study/comparacion_enfoques.csv  accuracy y macro-F1
  - research/event_study/curva_aprendizaje.csv   ¿etiquetar más mejora?
  - research/event_study/kappa_doble/…           acuerdo entre anotadores
  - data/processed/precios_event_study.parquet   precios diarios (6 + SPY + ETF)

Límites que el paquete declara explícitamente, para que el HTML los muestre:
el texto termina el 2022-12-31, los earnings arrancan en 2020-10 (límite de
yfinance) y en ventanas largas el CAR con alpha está dominado por el arrastre
del modelo, por lo que se incluye también la versión ajustada solo por mercado.

Uso:
    python -m dashboard.build_datos
"""

import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_RESULTADOS = Path("research/event_study")
_PRECIOS = Path("data/processed/precios_event_study.parquet")
_SALIDA = Path("dashboard/datos.json")
_PLANTILLA = Path("dashboard/sr_terminal.template.html")
_HTML = Path("dashboard/sr_terminal.html")

TICKERS: list[dict] = [
    {"sym": "NCLH", "nombre": "Norwegian Cruise Line", "sector": "Consumo discrecional", "liquidez": "alta"},
    {"sym": "DIS", "nombre": "The Walt Disney Co.", "sector": "Comunicación", "liquidez": "alta"},
    {"sym": "CRWD", "nombre": "CrowdStrike", "sector": "Tecnología", "liquidez": "media"},
    {"sym": "TGT", "nombre": "Target Corp.", "sector": "Consumo discrecional", "liquidez": "media"},
    {"sym": "CMG", "nombre": "Chipotle", "sector": "Consumo discrecional", "liquidez": "baja"},
    {"sym": "DDOG", "nombre": "Datadog", "sector": "Tecnología", "liquidez": "baja"},
]
REFERENCIAS: list[dict] = [
    {"sym": "SPY", "nombre": "S&P 500"},
    {"sym": "VGK", "nombre": "Europa desarrollada"},
    {"sym": "EWJ", "nombre": "Japón"},
    {"sym": "AAXJ", "nombre": "Asia ex-Japón"},
]
ENFOQUES: tuple[str, ...] = ("finbert", "probe", "vader")
UMBRAL_SENTIMIENTO = 0.02   # por debajo de esto no se declara dirección
UMBRAL_RETORNO = 0.0005


def _limpiar(valor):
    """
    Reemplaza NaN e infinitos por None, recursivamente.

    Es imprescindible: `json.dumps` los escribe como `NaN`/`Infinity`, que son
    válidos para Python pero **no** para JSON. El navegador aborta el
    `JSON.parse` en el primero que encuentra y el tablero queda en blanco.
    Las tablas de resultados vienen llenas de huecos (las filas de correlación
    no tienen las columnas de placebo y viceversa), así que esto no es un caso
    raro sino el habitual.

    Args:
        valor: Estructura anidada de dicts, listas y escalares.

    Returns:
        La misma estructura con los no-finitos convertidos a None.
    """
    if isinstance(valor, float):
        return valor if math.isfinite(valor) else None
    if isinstance(valor, dict):
        return {k: _limpiar(v) for k, v in valor.items()}
    if isinstance(valor, list):
        return [_limpiar(v) for v in valor]
    return valor


def _leer(nombre: str) -> pd.DataFrame:
    """Lee un CSV de resultados; devuelve DataFrame vacío si aún no existe."""
    path = _RESULTADOS / f"{nombre}.csv"
    if not path.exists():
        logger.warning("Falta %s; el paquete se arma sin esa sección.", path)
        return pd.DataFrame()
    return pd.read_csv(path)


def retornos_diarios() -> pd.DataFrame:
    """Retornos simples diarios de los 6 tickers, SPY y los ETF regionales."""
    precios = pd.read_parquet(_PRECIOS).sort_index()
    return precios.pct_change().dropna(how="all")


def bloque_calendario(retornos: pd.DataFrame) -> dict:
    """
    Une sentimiento diario con el retorno real del mismo día.

    Para cada (ticker, día) marca si el signo del sentimiento coincide con el
    del retorno. Los días sin mensajes suficientes o sin sesión quedan como
    `sin_datos`, nunca se inventan.

    Returns:
        Dict ticker → lista de días con sentimiento por enfoque, retorno y estado.
    """
    diario = _leer("sentimiento_diario")
    if diario.empty:
        return {}

    salida: dict[str, list[dict]] = {}
    for ticker, grupo in diario.groupby("ticker"):
        if ticker not in retornos.columns:
            continue
        serie = retornos[ticker]
        filas = []
        for _, fila in grupo.iterrows():
            fecha = pd.Timestamp(fila["dia"])
            ret = float(serie.get(fecha)) if fecha in serie.index and pd.notna(serie.get(fecha)) else None
            dia = {
                "d": fila["dia"],
                "n": int(fila["n_mensajes"]),
                "ret": round(ret, 6) if ret is not None else None,
            }
            for enfoque in ENFOQUES:
                valor = float(fila[enfoque])
                dia[enfoque] = round(valor, 4)
                if ret is None or abs(valor) < UMBRAL_SENTIMIENTO or abs(ret) < UMBRAL_RETORNO:
                    estado = "sin_datos"
                else:
                    estado = "coincide" if (valor > 0) == (ret > 0) else "diverge"
                dia[f"estado_{enfoque}"] = estado
            filas.append(dia)
        salida[ticker] = sorted(filas, key=lambda x: x["d"])
    logger.info("Calendario: %d tickers, %d días en total.",
                len(salida), sum(len(v) for v in salida.values()))
    return salida


def bloque_rendimiento(retornos: pd.DataFrame, desde: str = "2020-01-02") -> dict:
    """Series base 100 de los 6 tickers y las referencias, desde `desde`."""
    tramo = retornos[retornos.index >= desde]
    series: dict[str, list[float]] = {}
    for sym in [t["sym"] for t in TICKERS] + [r["sym"] for r in REFERENCIAS]:
        if sym not in tramo.columns:
            continue
        acumulado = (1 + tramo[sym].fillna(0)).cumprod() * 100
        series[sym] = [round(float(v), 2) for v in acumulado]
    return {"fechas": [d.strftime("%Y-%m-%d") for d in tramo.index], "series": series}


def bloque_car() -> dict:
    """Curvas de CAR por ventana y la tabla de eventos con su sentimiento."""
    curvas, eventos = _leer("car_curvas"), _leer("car_eventos")
    salida: dict = {"curvas": [], "eventos": []}
    if not curvas.empty:
        salida["curvas"] = curvas.to_dict("records")
    if not eventos.empty:
        columnas = ["ticker", "t0", "tipo_evento", "descripcion", "segmento",
                    "car_pre", "car_post", "car_m5_p5", "n_estimacion"]
        columnas += [f"sent_{v}_{e}" for v in ("pre", "post") for e in ENFOQUES]
        salida["eventos"] = eventos[[c for c in columnas if c in eventos.columns]].round(6).to_dict("records")
    return salida


def bloque_modelos() -> dict:
    """Métricas de comparación, acuerdo entre anotadores y curva de aprendizaje."""
    kappa = {}
    reporte = _RESULTADOS / "kappa_doble" / "kappa_camilo_vs_esteban.md"
    if reporte.exists():
        texto = reporte.read_text(encoding="utf-8")
        fila = re.search(r"\*\*Primario\*\*.*?\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)%", texto)
        if fila:
            kappa = {"n": int(fila.group(1)), "kappa": float(fila.group(2)),
                     "acuerdo_exacto": float(fila.group(3)) / 100}
    return {
        "comparacion": _leer("comparacion_enfoques").to_dict("records"),
        "correlaciones": _leer("car_resultados").to_dict("records"),
        "h2": _leer("bootstrap_h2").to_dict("records"),
        "curva_aprendizaje": _leer("curva_aprendizaje").to_dict("records"),
        "kappa_doble": kappa,
    }


def construir(salida: Path = _SALIDA) -> dict:
    """Arma el paquete completo y lo escribe como JSON."""
    retornos = retornos_diarios()
    calendario = bloque_calendario(retornos)

    paquete = {
        "meta": {
            "generado": datetime.now(timezone.utc).isoformat(),
            "fuente": "resultados reales del repositorio; ninguna serie es sintética",
            "rango_texto": "2020-01-01 a 2022-12-31",
            "rango_eventos": "2020-10-21 a 2022-11-29",
            "n_eventos": 55,
            "advertencias": [
                "El corpus de texto termina el 2022-12-31: no hay datos de 2023 en adelante.",
                "Los earnings arrancan en 2020-10 por el límite de yfinance.earnings_dates.",
                "En ventanas largas el CAR con alpha está dominado por el arrastre del modelo; "
                "se incluye la versión ajustada solo por mercado (car_medio_sin_alpha).",
                "El sentimiento diario usa un tope de 20 mensajes por (ticker, día).",
            ],
        },
        "tickers": TICKERS,
        "referencias": REFERENCIAS,
        "calendario": calendario,
        "rendimiento": bloque_rendimiento(retornos),
        "car": bloque_car(),
        "modelos": bloque_modelos(),
    }

    paquete = _limpiar(paquete)
    salida.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False hace que falle acá si algo se escapó, en vez de escribir
    # un JSON que el navegador rechaza en silencio.
    salida.write_text(
        json.dumps(paquete, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    logger.info("Paquete escrito en %s (%.1f KB).", salida, salida.stat().st_size / 1024)
    return paquete


def escribir_html(paquete: dict, plantilla: Path = _PLANTILLA, salida: Path = _HTML) -> Path:
    """
    Inyecta el paquete en la plantilla y escribe el HTML final.

    Los datos van **incrustados** y no en un fetch: un HTML abierto con doble
    clic corre sobre file:// y ahí el navegador bloquea la petición al JSON.
    Incrustar deja el archivo autocontenido, que es lo que se quiere para
    compartirlo o versionarlo.

    Args:
        paquete: Salida de construir().
        plantilla: HTML con el marcador /*__DATOS__*/.
        salida: HTML final.

    Returns:
        Ruta del HTML escrito.

    Raises:
        FileNotFoundError: si no existe la plantilla.
        ValueError: si la plantilla no trae el marcador.
    """
    if not plantilla.exists():
        raise FileNotFoundError(f"No existe la plantilla {plantilla}")
    html = plantilla.read_text(encoding="utf-8")
    if "/*__DATOS__*/" not in html:
        raise ValueError(f"La plantilla {plantilla} no tiene el marcador /*__DATOS__*/")

    # Escapar '<' evita que un texto con '</script>' corte el bloque de datos.
    datos = json.dumps(
        _limpiar(paquete), ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).replace("<", "\\u003c")
    salida.write_text(html.replace("/*__DATOS__*/", datos), encoding="utf-8")
    logger.info("HTML escrito en %s (%.1f KB).", salida, salida.stat().st_size / 1024)
    return salida


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    paquete = construir()
    html = escribir_html(paquete)
    print(f"\nHTML autocontenido: {html}")
    print("\n=== CONTENIDO DEL PAQUETE ===")
    print(f"calendario: {len(paquete['calendario'])} tickers, "
          f"{sum(len(v) for v in paquete['calendario'].values())} días")
    print(f"rendimiento: {len(paquete['rendimiento']['series'])} series, "
          f"{len(paquete['rendimiento']['fechas'])} fechas")
    print(f"car: {len(paquete['car']['curvas'])} puntos de curva, {len(paquete['car']['eventos'])} eventos")
    print(f"modelos: comparacion={len(paquete['modelos']['comparacion'])}, "
          f"h2={len(paquete['modelos']['h2'])}, kappa={paquete['modelos']['kappa_doble']}")
