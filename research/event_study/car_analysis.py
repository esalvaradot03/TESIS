"""
Estudio de eventos: retorno anormal acumulado (CAR) contra sentimiento.

Implementa la sección 4 de docs/pre_registro_event_study.md para los 6 tickers,
con los tres enfoques de sentimiento (FinBERT off-the-shelf, linear probing y
VADER) y los dos controles placebo obligatorios.

Pregunta operativa: ¿el sentimiento **antes** del evento anticipa el signo del
CAR posterior, o sólo el sentimiento **posterior** correlaciona con él? El marco
previo del proyecto (fases 1-6) dice que el sentimiento es coincidente y
reactivo, no anticipatorio; acá se contrasta a nivel de evento.

DESVIACIONES del pre-registro, por límites de las fuentes (declaradas acá y en
la salida del módulo):

  1. **Precios: yfinance, no Alpaca IEX.** El feed IEX gratuito arranca en
     ~2020-07-27, insuficiente para la ventana de estimación [-250, -30] de los
     eventos de 2020-2021.
  2. **Earnings desde 2020-10, no desde 2019.** `yfinance.earnings_dates` sólo
     devuelve ~25 fechas por ticker y la más antigua es de octubre de 2020. En
     la ventana con texto (hasta 2022-12-31) quedan 54 earnings, no los ~96 que
     suponía el pre-registro. Los upgrades/downgrades sí cubren 2019-2022 y se
     pueden sumar como conjunto secundario con --incluir-upgrades.

Uso:
    python -m research.event_study.car_analysis
    python -m research.event_study.car_analysis --max-por-dia 40 --incluir-upgrades
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest, spearmanr

from config.settings import DATA_DIR, SEED
from research.event_study.linear_probe import (
    HPARAMS_PREREGISTRADOS,
    cargar_split,
)
from src.sentiment.finbert_finetune import FinBERTHeadTrainer, _LABEL_TO_IDX
from src.sentiment.finbert_scorer import FinBERTScorer
from src.sentiment.lexicon_scorer import LexiconScorer

logger = logging.getLogger(__name__)

TICKERS: list[str] = ["NCLH", "DIS", "CRWD", "TGT", "CMG", "DDOG"]

# Segmentos de liquidez (decididos en el pivote, ver AGENTS.md).
SEGMENTO: dict[str, str] = {
    "NCLH": "alta", "DIS": "alta",
    "CRWD": "media", "TGT": "media",
    "CMG": "baja", "DDOG": "baja",
}

# Evento idiosincrático incluido explícitamente por el pre-registro.
EVENTOS_MANUALES: list[dict] = [
    {"ticker": "DIS", "fecha": "2022-11-20", "tipo_evento": "iger", "descripcion": "Regreso de Bob Iger"},
]

RANGO_TEXTO = ("2019-01-01", "2022-12-31")

_EVENTS_DIR = DATA_DIR / "events"
_PRECIOS = DATA_DIR / "processed" / "precios_event_study.parquet"
_POOL = DATA_DIR / "processed" / "nyu_pool_v2_CMG_CRWD_DDOG_DIS_NCLH_TGT.parquet"
_SALIDA_EVENTOS = Path("research/event_study/car_eventos.csv")
_SALIDA_RESUMEN = Path("research/event_study/car_resultados.csv")

# Ventanas en días hábiles relativos al día del evento (t=0).
VENTANA_EVENTO = (-5, 5)
VENTANA_PRE = (-5, -1)
VENTANA_POST = (0, 5)
ESTIMACION = (-250, -30)
MIN_DIAS_ESTIMACION = 100
MIN_MENSAJES_VENTANA = 3

ENFOQUES = ("finbert", "probe", "vader")


# ---------------------------------------------------------------------------
# Eventos y precios
# ---------------------------------------------------------------------------

def cargar_eventos(incluir_upgrades: bool = False, events_dir: Path = _EVENTS_DIR) -> pd.DataFrame:
    """
    Carga las fechas de evento dentro del rango con cobertura de texto.

    Args:
        incluir_upgrades: Suma los upgrades/downgrades como conjunto secundario.
        events_dir: Directorio con los CSV de event_windows.py.

    Returns:
        DataFrame con ticker, fecha (Timestamp), tipo_evento y descripcion.
    """
    frames = []
    for path in sorted(events_dir.glob("eventos_*.csv")):
        df = pd.read_csv(path, dtype=str)
        df["ticker"] = path.stem.replace("eventos_", "")
        frames.append(df)
    eventos = pd.concat(frames, ignore_index=True)
    eventos["fecha"] = pd.to_datetime(eventos["fecha"], utc=True, errors="coerce").dt.tz_localize(None).dt.normalize()

    tipos = ["earnings"] + (["upgrade_downgrade"] if incluir_upgrades else [])
    eventos = eventos[eventos["tipo_evento"].isin(tipos)]
    eventos = pd.concat([eventos, pd.DataFrame(EVENTOS_MANUALES).assign(
        fecha=lambda d: pd.to_datetime(d["fecha"]))], ignore_index=True)

    eventos = eventos[eventos["fecha"].between(*RANGO_TEXTO)]
    eventos = eventos[eventos["ticker"].isin(TICKERS)]
    eventos = eventos.drop_duplicates(["ticker", "fecha", "tipo_evento"]).reset_index(drop=True)
    logger.info("Eventos en rango: %d | %s", len(eventos), eventos["tipo_evento"].value_counts().to_dict())
    return eventos


def cargar_precios(path: Path = _PRECIOS) -> pd.DataFrame:
    """
    Carga precios ajustados diarios y devuelve retornos simples.

    Args:
        path: Parquet con una columna por ticker más SPY.

    Returns:
        DataFrame de retornos indexado por fecha.
    """
    precios = pd.read_parquet(path).sort_index()
    return precios.pct_change()


def car_de_evento(
    retornos: pd.DataFrame, ticker: str, fecha: pd.Timestamp,
) -> dict | None:
    """
    Calcula los CAR de un evento con el modelo de mercado (CAPM vs SPY).

    Estima alpha y beta por MCO sobre [-250, -30] días hábiles y acumula el
    residuo en las ventanas de evento.

    Args:
        retornos: Retornos diarios (salida de cargar_precios).
        ticker: Ticker del evento.
        fecha: Fecha del evento.

    Returns:
        Dict con los CAR y metadatos, o None si no hay datos suficientes.
    """
    if ticker not in retornos.columns:
        return None
    serie = retornos[[ticker, "SPY"]].dropna()
    calendario = serie.index
    posteriores = calendario[calendario >= fecha]
    if posteriores.empty:
        return None
    t0 = calendario.get_loc(posteriores[0])

    ini_est, fin_est = t0 + ESTIMACION[0], t0 + ESTIMACION[1]
    if ini_est < 0 or fin_est <= ini_est:
        return None
    estimacion = serie.iloc[ini_est:fin_est]
    if len(estimacion) < MIN_DIAS_ESTIMACION:
        return None

    beta, alpha = np.polyfit(estimacion["SPY"].to_numpy(), estimacion[ticker].to_numpy(), 1)

    ini_ev, fin_ev = t0 + VENTANA_EVENTO[0], t0 + VENTANA_EVENTO[1] + 1
    if ini_ev < 0 or fin_ev > len(serie):
        return None
    ventana = serie.iloc[ini_ev:fin_ev].copy()
    ventana["ar"] = ventana[ticker] - (alpha + beta * ventana["SPY"])
    ventana["rel"] = range(VENTANA_EVENTO[0], VENTANA_EVENTO[1] + 1)

    def car(desde: int, hasta: int) -> float:
        tramo = ventana[(ventana["rel"] >= desde) & (ventana["rel"] <= hasta)]
        return float(tramo["ar"].sum())

    return {
        "t0": serie.index[t0],
        "alpha": float(alpha),
        "beta": float(beta),
        "n_estimacion": len(estimacion),
        "car_m5_p5": car(-5, 5),
        "car_m1_p1": car(-1, 1),
        "car_post": car(*VENTANA_POST),
        "car_pre": car(*VENTANA_PRE),
        "dias_pre": [d.date().isoformat() for d in ventana.index[ventana["rel"] < 0]],
        "dias_post": [d.date().isoformat() for d in ventana.index[ventana["rel"] >= 0]],
    }


# ---------------------------------------------------------------------------
# Sentimiento por ventana
# ---------------------------------------------------------------------------

def mensajes_de_ventanas(
    dias_por_ticker: dict[str, set[str]], max_por_dia: int, seed: int = SEED, pool_path: Path = _POOL,
) -> pd.DataFrame:
    """
    Extrae del pool NYU los mensajes que caen en los días de ventana pedidos.

    Args:
        dias_por_ticker: ticker → conjunto de días (ISO) a recolectar.
        max_por_dia: Tope de mensajes por (ticker, día); acota el costo de FinBERT.
        seed: Semilla del muestreo cuando hay más mensajes que el tope.
        pool_path: Caché del pool NYU.

    Returns:
        DataFrame con ticker, dia, message_id y texto.
    """
    pool = pd.read_parquet(pool_path, columns=["message_id", "created_at", "symbol_list", "message_body"])
    pool["dia"] = pool["created_at"].str[:10]

    filas = []
    for mid, dia, symbols_json, cuerpo in zip(
        pool["message_id"], pool["dia"], pool["symbol_list"], pool["message_body"]
    ):
        for ticker in json.loads(symbols_json):
            if ticker in dias_por_ticker and dia in dias_por_ticker[ticker]:
                filas.append({"ticker": ticker, "dia": dia, "message_id": mid, "texto": cuerpo})
    df = pd.DataFrame(filas)
    logger.info("Mensajes en ventanas: %d", len(df))

    if max_por_dia > 0 and not df.empty:
        df = (df.groupby(["ticker", "dia"], group_keys=False)
                .apply(lambda g: g.sample(min(len(g), max_por_dia), random_state=seed)))
        logger.info("Tras el tope de %d por (ticker, día): %d mensajes.", max_por_dia, len(df))
    return df.reset_index(drop=True)


def _probs_probe(textos: list[str], trainer: FinBERTHeadTrainer) -> np.ndarray:
    """Probabilidades [positive, negative, neutral] de la cabeza entrenada."""
    embeddings = trainer._embed_all(textos)
    trainer.head.eval()
    with torch.no_grad():
        return torch.softmax(trainer.head(embeddings), dim=-1).cpu().numpy()


def puntuar(textos: list[str], trainer: FinBERTHeadTrainer) -> pd.DataFrame:
    """
    Puntúa los textos con los tres enfoques y devuelve net_sentiment por enfoque.

    net_sentiment = prob_positive − prob_negative, en [-1, 1], comparable entre
    los tres (ver docstring de lexicon_scorer sobre por qué no se compara
    sentiment_score).

    Args:
        textos: Textos crudos de los mensajes.
        trainer: Cabeza entrenada para el enfoque `probe`.

    Returns:
        DataFrame con una columna por enfoque.
    """
    finbert = FinBERTScorer().score_texts(textos)
    vader = LexiconScorer().score_texts(textos)
    idx_pos, idx_neg = _LABEL_TO_IDX["positive"], _LABEL_TO_IDX["negative"]
    probs = _probs_probe(textos, trainer)

    return pd.DataFrame({
        "finbert": [r["prob_positive"] - r["prob_negative"] for r in finbert],
        "vader": [r["prob_positive"] - r["prob_negative"] for r in vader],
        "probe": probs[:, idx_pos] - probs[:, idx_neg],
    })


# ---------------------------------------------------------------------------
# Contrastes
# ---------------------------------------------------------------------------

def correlaciones(eventos: pd.DataFrame) -> pd.DataFrame:
    """
    Spearman entre sentimiento (pre y post) y CAR post, global y por segmento.

    Args:
        eventos: Tabla por evento con columnas sent_pre_*, sent_post_* y car_post.

    Returns:
        DataFrame con rho, p-valor y n por (enfoque, ventana, segmento).
    """
    filas = []
    grupos = [("todos", eventos)] + [(s, eventos[eventos["segmento"] == s]) for s in ("alta", "media", "baja")]
    for enfoque in ENFOQUES:
        for ventana in ("pre", "post"):
            columna = f"sent_{ventana}_{enfoque}"
            for nombre, grupo in grupos:
                datos = grupo[[columna, "car_post"]].dropna()
                if len(datos) < 5:
                    filas.append({"enfoque": enfoque, "ventana": ventana, "segmento": nombre,
                                  "n": len(datos), "rho": np.nan, "p": np.nan})
                    continue
                rho, p = spearmanr(datos[columna], datos["car_post"])
                filas.append({"enfoque": enfoque, "ventana": ventana, "segmento": nombre,
                              "n": len(datos), "rho": round(float(rho), 4), "p": round(float(p), 4)})
    return pd.DataFrame(filas)


def direccional(eventos: pd.DataFrame) -> pd.DataFrame:
    """
    Precisión direccional: ¿el signo del sentimiento pre-evento acierta el del CAR post?

    Args:
        eventos: Tabla por evento.

    Returns:
        DataFrame con aciertos, n, precisión e IC95 binomial por enfoque.
    """
    filas = []
    for enfoque in ENFOQUES:
        datos = eventos[[f"sent_pre_{enfoque}", "car_post"]].dropna()
        datos = datos[datos[f"sent_pre_{enfoque}"] != 0]
        if datos.empty:
            continue
        aciertos = int((np.sign(datos[f"sent_pre_{enfoque}"]) == np.sign(datos["car_post"])).sum())
        prueba = binomtest(aciertos, len(datos), 0.5)
        ic = prueba.proportion_ci(confidence_level=0.95)
        filas.append({"enfoque": enfoque, "n": len(datos), "aciertos": aciertos,
                      "precision": round(aciertos / len(datos), 4),
                      "ic95_bajo": round(float(ic.low), 4), "ic95_alto": round(float(ic.high), 4),
                      "p": round(float(prueba.pvalue), 4)})
    return pd.DataFrame(filas)


def placebos(
    eventos: pd.DataFrame, retornos: pd.DataFrame, repeticiones: int = 1000, seed: int = SEED,
) -> pd.DataFrame:
    """
    Los dos controles placebo de la regla 2 del pre-registro.

    1. **Permutación del sentimiento** entre eventos, manteniendo los CAR.
    2. **Fechas de evento falsas**: se recalcula el CAR en una fecha falsa del
       mismo ticker y año (a 30-180 días hábiles de la real) y se empareja con
       el sentimiento real.

    Args:
        eventos: Tabla por evento.
        retornos: Retornos diarios, para recalcular CAR en fechas falsas.
        repeticiones: Repeticiones de cada placebo.
        seed: Semilla.

    Returns:
        DataFrame con el |rho| observado y el percentil que ocupa en cada placebo.
    """
    rng = np.random.default_rng(seed)
    filas = []

    for enfoque in ENFOQUES:
        datos = eventos[[f"sent_post_{enfoque}", "car_post", "ticker", "t0"]].dropna()
        if len(datos) < 5:
            continue
        observado = abs(spearmanr(datos[f"sent_post_{enfoque}"], datos["car_post"]).statistic)

        nulos_perm = []
        valores = datos[f"sent_post_{enfoque}"].to_numpy()
        for _ in range(repeticiones):
            nulos_perm.append(abs(spearmanr(rng.permutation(valores), datos["car_post"]).statistic))

        nulos_fecha = []
        for _ in range(repeticiones):
            cars_falsos = []
            for ticker, t0 in zip(datos["ticker"], pd.to_datetime(datos["t0"])):
                desfase = int(rng.choice([-1, 1]) * rng.integers(30, 180))
                falso = car_de_evento(retornos, ticker, t0 + pd.Timedelta(days=desfase))
                cars_falsos.append(falso["car_post"] if falso else np.nan)
            serie = pd.Series(cars_falsos, index=datos.index)
            validos = serie.notna()
            if validos.sum() >= 5:
                nulos_fecha.append(abs(spearmanr(valores[validos.to_numpy()], serie[validos]).statistic))

        filas.append({
            "enfoque": enfoque,
            "rho_abs_observado": round(float(observado), 4),
            "percentil_en_placebo_permutacion": round(float((np.array(nulos_perm) < observado).mean() * 100), 1),
            "percentil_en_placebo_fechas": round(float((np.array(nulos_fecha) < observado).mean() * 100), 1)
            if nulos_fecha else np.nan,
            "supera_ambos_p95": bool(
                (np.array(nulos_perm) < observado).mean() > 0.95
                and nulos_fecha and (np.array(nulos_fecha) < observado).mean() > 0.95
            ),
        })
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------

def correr(max_por_dia: int = 40, incluir_upgrades: bool = False, repeticiones: int = 1000) -> pd.DataFrame:
    """
    Corre el estudio completo y escribe los CSV de salida.

    Args:
        max_por_dia: Tope de mensajes por (ticker, día).
        incluir_upgrades: Suma upgrades/downgrades al conjunto de eventos.
        repeticiones: Repeticiones de cada placebo.

    Returns:
        Tabla por evento.
    """
    eventos = cargar_eventos(incluir_upgrades)
    retornos = cargar_precios()

    registros = []
    for _, fila in eventos.iterrows():
        car = car_de_evento(retornos, fila["ticker"], fila["fecha"])
        if car is None:
            continue
        registros.append({**fila.to_dict(), **car, "segmento": SEGMENTO[fila["ticker"]]})
    tabla = pd.DataFrame(registros)
    logger.info("Eventos con CAR calculable: %d de %d", len(tabla), len(eventos))

    dias_por_ticker: dict[str, set[str]] = {}
    for _, fila in tabla.iterrows():
        dias_por_ticker.setdefault(fila["ticker"], set()).update(fila["dias_pre"] + fila["dias_post"])

    mensajes = mensajes_de_ventanas(dias_por_ticker, max_por_dia)

    logger.info("Entrenando la cabeza lineal sobre el corpus etiquetado...")
    train = cargar_split("train")
    trainer = FinBERTHeadTrainer(hparams=HPARAMS_PREREGISTRADOS)
    trainer.train(train["clean_text"].tolist(), train["clase"].tolist())

    logger.info("Puntuando %d mensajes con los 3 enfoques...", len(mensajes))
    scores = puntuar(mensajes["texto"].fillna("").astype(str).tolist(), trainer)
    mensajes = pd.concat([mensajes.reset_index(drop=True), scores], axis=1)
    promedios = mensajes.groupby(["ticker", "dia"])[list(ENFOQUES)].mean()
    conteos = mensajes.groupby(["ticker", "dia"]).size()

    for enfoque in ENFOQUES:
        for ventana, columna_dias in (("pre", "dias_pre"), ("post", "dias_post")):
            valores = []
            for _, fila in tabla.iterrows():
                llaves = [(fila["ticker"], d) for d in fila[columna_dias]]
                presentes = [k for k in llaves if k in promedios.index]
                n = sum(int(conteos[k]) for k in presentes)
                valores.append(float(promedios.loc[presentes, enfoque].mean())
                               if presentes and n >= MIN_MENSAJES_VENTANA else np.nan)
            tabla[f"sent_{ventana}_{enfoque}"] = valores

    _SALIDA_EVENTOS.parent.mkdir(parents=True, exist_ok=True)
    tabla.drop(columns=["dias_pre", "dias_post"]).to_csv(_SALIDA_EVENTOS, index=False, encoding="utf-8")

    corr = correlaciones(tabla)
    dire = direccional(tabla)
    plac = placebos(tabla, retornos, repeticiones)

    print("\n=== CORRELACIÓN DE SPEARMAN: sentimiento vs CAR [0,+5] ===")
    print(corr.pivot(index=["enfoque", "ventana"], columns="segmento", values="rho").to_string())
    print("\n(p-valores, segmento 'todos')")
    print(corr[corr.segmento.eq("todos")][["enfoque", "ventana", "n", "rho", "p"]].to_string(index=False))
    print("\n=== PRECISIÓN DIRECCIONAL (signo del sentimiento pre-evento vs signo del CAR post) ===")
    print(dire.to_string(index=False))
    print("\n=== PLACEBOS (rho observado vs distribuciones nulas) ===")
    print(plac.to_string(index=False))

    resumen = pd.concat([
        corr.assign(analisis="correlacion"),
        dire.assign(analisis="direccional"),
        plac.assign(analisis="placebo"),
    ], ignore_index=True)
    resumen.to_csv(_SALIDA_RESUMEN, index=False, encoding="utf-8")
    print(f"\nGuardado: {_SALIDA_EVENTOS} y {_SALIDA_RESUMEN}")
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

    parser = argparse.ArgumentParser(description="Estudio de eventos: CAR vs sentimiento.")
    parser.add_argument("--max-por-dia", type=int, default=40, help="Tope de mensajes por (ticker, día).")
    parser.add_argument("--incluir-upgrades", action="store_true", help="Suma upgrades/downgrades.")
    parser.add_argument("--repeticiones", type=int, default=1000, help="Repeticiones de cada placebo.")
    args = parser.parse_args()

    print("DESVIACIONES del pre-registro (ver docstring): precios de yfinance en vez de Alpaca IEX; "
          "earnings desde 2020-10 y no desde 2019, por el límite de yfinance.earnings_dates.\n")
    correr(max_por_dia=args.max_por_dia, incluir_upgrades=args.incluir_upgrades, repeticiones=args.repeticiones)
