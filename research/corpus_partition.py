"""
Maquinaria de muestreo y partición de corpus de etiquetado manual.

La comparten los dos corpus de la tesis, que difieren solo en cómo se llaman
sus columnas y en qué texto ve el anotador:

    event study (StockTwits)   unidad (post_id, target_ticker), estrato `year`
    pivote colombiano (EMIS)   unidad (article_id, emisor),     estrato `anio`

Todo lo demás es el mismo diseño y tiene que seguir siéndolo: cuotas por clase,
estratificación por año, partición agrupada para que un mismo texto no quede a
la vez en train y en test, lotes compartidos para el kappa, y semillas
derivadas por (lote, anotador). Duplicar esto en cada corpus significaba que un
arreglo en la partición se aplicara en un solo lado.

El esquema de columnas entra por `CorpusSchema`; ninguna función de este módulo
conoce un nombre de columna concreto.

Este módulo NO escribe archivos ni sabe qué columnas ve el anotador: eso es
decisión de cada corpus, que arma su propio CSV y su propio manifiesto con
`sha256_file` y `sha256_pool`.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CorpusSchema:
    """
    Nombres de columna del corpus sobre el que se muestrea.

    Attributes:
        key: Columnas que identifican una unidad de anotación, p.ej.
            ["article_id", "emisor"].
        group: Columna cuyos valores NO pueden partirse entre lotes, porque
            comparten el mismo texto (el artículo o el post). Todas las
            unidades con el mismo valor caen en el mismo lote.
        klass: Columna sobre la que se reparten las cuotas (el emisor o el
            ticker).
        stratum: Columna de estratificación dentro de cada clase (el año).
    """

    key: list[str]
    group: str
    klass: str
    stratum: str


# ---------------------------------------------------------------------------
# Reparto de cuotas
# ---------------------------------------------------------------------------

def largest_remainder(weights: dict, total: int, caps: dict) -> dict:
    """
    Reparte `total` unidades proporcionalmente a `weights`, respetando `caps`.

    Método del resto mayor: la parte entera primero, y el sobrante por parte
    fraccionaria descendente, saltando las claves que ya llegaron a su tope.

    Args:
        weights: Peso relativo de cada clave (típicamente su disponibilidad).
        total: Unidades a repartir.
        caps: Tope por clave.

    Returns:
        Dict clave → unidades asignadas. La suma es min(total, sum(caps)).
    """
    total_weight = sum(weights.values())
    if total <= 0 or total_weight <= 0:
        return {k: 0 for k in weights}

    exact = {k: total * w / total_weight for k, w in weights.items()}
    allocated = {k: min(int(v), caps[k]) for k, v in exact.items()}

    order = sorted(exact, key=lambda k: exact[k] - int(exact[k]), reverse=True)
    while sum(allocated.values()) < total:
        progressed = False
        for key in order:
            if sum(allocated.values()) >= total:
                break
            if allocated[key] < caps[key]:
                allocated[key] += 1
                progressed = True
        if not progressed:  # todas las claves llegaron a su tope
            break
    return allocated


def allocate_by_cell(
    pool: pd.DataFrame,
    schema: CorpusSchema,
    total: int,
    classes: list[str],
    equal_quotas: bool = True,
    min_per_class: int = 0,
) -> dict[tuple[str, int], int]:
    """
    Decide cuántas unidades tomar de cada celda (clase, estrato).

    Con `equal_quotas`, cada clase recibe total / n_clases: la pregunta de
    investigación se responde por clase, así que hace falta precisión
    comparable en todas, y muestrear en proporción importaría al corpus el
    sesgo de cobertura de la fuente. Una clase sin material para su cuota se
    REPORTA y el faltante NO se rellena con otras, porque rellenar
    reintroduciría exactamente ese sesgo.

    Sin `equal_quotas`, se garantiza `min_per_class` a cada clase y el resto
    se reparte proporcional a la disponibilidad.

    Args:
        pool: Pool de unidades disponibles.
        schema: Nombres de columna del corpus.
        total: Unidades a muestrear en total.
        classes: Universo de clases (emisores o tickers).
        equal_quotas: Cuotas iguales por clase en vez de proporcionales.
        min_per_class: Mínimo garantizado por clase; solo aplica si las cuotas
            NO son iguales.

    Returns:
        Dict (clase, estrato) → cantidad a muestrear.

    Raises:
        ValueError: si el mínimo por clase supera el total pedido.
    """
    available = pool.groupby([schema.klass, schema.stratum]).size().to_dict()
    caps = {c: sum(n for (cl, _), n in available.items() if cl == c) for c in classes}
    caps = {c: n for c, n in caps.items() if n > 0}

    missing = [c for c in classes if c not in caps]
    if missing:
        logger.warning("Clases sin ninguna unidad en el pool: %s", missing)

    if equal_quotas:
        quota = largest_remainder({c: 1 for c in classes}, total, {c: total for c in classes})
        per_class = {c: min(quota[c], caps.get(c, 0)) for c in classes}
        short = {c: quota[c] - n for c, n in per_class.items() if n < quota[c]}
        if short:
            logger.warning(
                "Clases sin material para su cuota (faltante, no se rellena): %s", short
            )
    else:
        base = {c: min(min_per_class, cap) for c, cap in caps.items()}
        remaining = total - sum(base.values())
        if remaining < 0:
            raise ValueError(
                f"El mínimo por clase ({min_per_class} × {len(base)} clases) supera "
                f"el total pedido ({total})."
            )
        residual = {c: caps[c] - base[c] for c in base}
        extra = largest_remainder(residual, remaining, residual)
        per_class = {c: base[c] + extra.get(c, 0) for c in base}

    cells: dict[tuple[str, int], int] = {}
    for klass, n_class in per_class.items():
        stratum_caps = {e: n for (cl, e), n in available.items() if cl == klass}
        per_stratum = largest_remainder(stratum_caps, n_class, stratum_caps)
        cells.update({
            (klass, stratum): n for stratum, n in per_stratum.items() if n > 0
        })
    return cells


def sample_stratified(
    pool: pd.DataFrame,
    schema: CorpusSchema,
    total: int,
    classes: list[str],
    seed: int,
    equal_quotas: bool = True,
    min_per_class: int = 0,
) -> pd.DataFrame:
    """
    Muestrea `total` unidades estratificadas por (clase, estrato).

    Args:
        pool: Pool de unidades disponibles.
        schema: Nombres de columna del corpus.
        total: Unidades a muestrear.
        classes: Universo de clases.
        seed: Semilla (SEED=42 del proyecto).
        equal_quotas: Ver allocate_by_cell.
        min_per_class: Ver allocate_by_cell.

    Returns:
        DataFrame barajado con las unidades seleccionadas, mismas columnas que
        el pool. Puede tener menos de `total` filas si el pool no alcanza; el
        llamador decide si eso aborta.
    """
    if pool.empty:
        return pool.copy()

    cells = allocate_by_cell(pool, schema, total, classes, equal_quotas, min_per_class)
    rng = np.random.default_rng(seed)

    picked: list[int] = []
    for (klass, stratum), n in sorted(cells.items()):
        candidates = pool.index[
            (pool[schema.klass] == klass) & (pool[schema.stratum] == stratum)
        ].to_numpy()
        picked.extend(rng.choice(candidates, size=n, replace=False).tolist())

    sample = pool.loc[picked].sample(frac=1, random_state=seed).reset_index(drop=True)
    logger.info(
        "Muestreo: %d unidades de %d pedidas (%d celdas %s×%s).",
        len(sample), total, len(cells), schema.klass, schema.stratum,
    )
    return sample


# ---------------------------------------------------------------------------
# Partición en lotes
# ---------------------------------------------------------------------------

def partition(
    sample: pd.DataFrame,
    schema: CorpusSchema,
    batches: dict[str, tuple[int, bool]],
    minimums: dict[str, int],
    classes: list[str],
) -> dict[str, pd.DataFrame]:
    """
    Corta la muestra en lotes disjuntos, en el orden en que vienen definidos.

    Todas las unidades que comparten `schema.group` van al mismo lote: si dos
    unidades salen del mismo texto, separarlas filtraría información de train a
    test. Para cada lote se cubre primero su mínimo por clase y después se
    completa siguiendo el orden aleatorio de la muestra; un grupo que no entra
    pasa al lote siguiente, así los tamaños quedan exactos.

    Args:
        sample: Salida de sample_stratified() (ya barajada).
        schema: Nombres de columna del corpus.
        batches: Definición lote → (tamaño, compartido).
        minimums: Mínimo de unidades por clase dentro de cada lote. Los lotes
            que no aparecen no tienen mínimo.
        classes: Clases sobre las que aplica el mínimo.

    Returns:
        Dict lote → DataFrame, con los lotes en el orden de `batches`.

    Raises:
        ValueError: si la muestra no alcanza, si no se cubre algún mínimo, o si
            un lote no se puede completar sin partir un grupo.
    """
    needed = sum(size for size, _ in batches.values())
    if len(sample) < needed:
        raise ValueError(
            f"La muestra tiene {len(sample)} unidades pero los lotes requieren {needed}."
        )

    positions = sample.groupby(schema.group).indices
    pending = [positions[g] for g in pd.unique(sample[schema.group])]
    class_of = sample[schema.klass].to_numpy()

    result: dict[str, pd.DataFrame] = {}
    for name, (size, _) in batches.items():
        taken = [False] * len(pending)
        count = 0

        minimum = minimums.get(name, 0)
        for klass in (classes if minimum else []):
            have = sum(
                int((class_of[g] == klass).sum())
                for g, t in zip(pending, taken) if t
            )
            for i, group in enumerate(pending):
                if have >= minimum:
                    break
                if taken[i] or count + len(group) > size:
                    continue
                if not (class_of[group] == klass).any():
                    continue
                taken[i] = True
                count += len(group)
                have += int((class_of[group] == klass).sum())
            if have < minimum:
                raise ValueError(
                    f"Lote '{name}': solo {have} unidades de {klass} (mínimo {minimum})."
                )

        for i, group in enumerate(pending):
            if count >= size:
                break
            if not taken[i] and count + len(group) <= size:
                taken[i] = True
                count += len(group)
        if count < size:
            raise ValueError(
                f"No se pudo completar el lote '{name}' ({count}/{size}) sin "
                f"partir un {schema.group}."
            )

        chosen = [g for g, t in zip(pending, taken) if t]
        result[name] = sample.iloc[np.concatenate(chosen)].reset_index(drop=True)
        pending = [g for g, t in zip(pending, taken) if not t]
    return result


def scale_batches(
    batches: dict[str, tuple[int, bool]], obtained: int, requested: int
) -> dict[str, tuple[int, bool]]:
    """
    Reduce proporcionalmente los tamaños de lote cuando el pool no alcanzó.

    Solo se usa detrás de un `--allow-partial` explícito: generar un corpus más
    chico en silencio rompe el diseño pre-registrado.

    Args:
        batches: Definición original lote → (tamaño, compartido).
        obtained: Unidades que se pudieron muestrear.
        requested: Unidades pedidas.

    Returns:
        Definición escalada, con al menos 1 unidad por lote.
    """
    scale = obtained / requested if requested else 0
    return {n: (max(1, int(size * scale)), shared) for n, (size, shared) in batches.items()}


# ---------------------------------------------------------------------------
# Reparto entre anotadores
# ---------------------------------------------------------------------------

def assign_to_annotators(
    batches_df: dict[str, pd.DataFrame],
    definitions: dict[str, tuple[int, bool]],
    annotators: list[str],
) -> dict[tuple[str, str], pd.DataFrame]:
    """
    Reparte cada lote entre los anotadores.

    Los lotes compartidos van completos a ambos — son la base del kappa, así que
    tienen que llevar exactamente las mismas llaves. Los demás se parten en
    bloques disjuntos del mismo tamaño.

    Args:
        batches_df: Salida de partition().
        definitions: Definición lote → (tamaño, compartido).
        annotators: Nombres de los anotadores.

    Returns:
        Dict (lote, anotador) → DataFrame.
    """
    result: dict[tuple[str, str], pd.DataFrame] = {}
    for name, frame in batches_df.items():
        if definitions[name][1]:
            for annotator in annotators:
                result[(name, annotator)] = frame.copy()
            continue
        chunk = len(frame) // len(annotators)
        for i, annotator in enumerate(annotators):
            result[(name, annotator)] = (
                frame.iloc[i * chunk:(i + 1) * chunk].reset_index(drop=True)
            )
    return result


def annotator_seed(seed: int, batch: str, annotator: str) -> int:
    """
    Semilla determinista y distinta por (lote, anotador), derivada de `seed`.

    Sirve para barajar los lotes compartidos en orden distinto para cada
    anotador: mismos pares, distinto orden, lo que reduce el sesgo de orden sin
    romper el merge del kappa.

    Args:
        seed: Semilla base del proyecto.
        batch: Nombre del lote.
        annotator: Nombre del anotador.

    Returns:
        Entero de 32 bits.
    """
    digest = hashlib.sha256(f"{seed}|{batch}|{annotator}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


# ---------------------------------------------------------------------------
# Trazabilidad
# ---------------------------------------------------------------------------

def sha256_file(path) -> str:
    """
    SHA-256 del contenido de un archivo.

    Args:
        path: Ruta al archivo.

    Returns:
        Digest hexadecimal.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_pool(pool: pd.DataFrame, columns: list[str], key: list[str]) -> str:
    """
    SHA-256 del pool, independiente del orden de las filas.

    Args:
        pool: Pool del que se muestreó.
        columns: Columnas que entran al hash (las que definen el contenido).
        key: Columnas por las que ordenar antes de hashear.

    Returns:
        Digest hexadecimal.
    """
    canon = pool[columns].sort_values(key)
    return hashlib.sha256(canon.to_csv(index=False).encode("utf-8")).hexdigest()


def batch_summary(
    assignments: dict[tuple[str, str], pd.DataFrame],
    definitions: dict[str, tuple[int, bool]],
    schema: CorpusSchema,
    annotators: list[str],
) -> dict:
    """
    Arma las secciones del manifiesto que son comunes a cualquier corpus.

    Las CLAVES del dict devuelto van en español porque son parte del manifiesto
    (un artefacto que se lee a mano, no identificadores de código).

    Args:
        assignments: Salida de assign_to_annotators().
        definitions: Definición lote → (tamaño, compartido).
        schema: Nombres de columna del corpus.
        annotators: Nombres de los anotadores.

    Returns:
        Dict con `lotes` (tamaño, si es compartido y cuántas unidades recibió
        cada anotador), `por_lote_y_clase` e `ids_por_lote`, que es lo que
        permite reconstruir exactamente qué se le dio a quién.
    """
    ids_por_lote: dict[str, dict[str, list[list[str]]]] = {}
    por_lote_y_clase: dict[str, dict[str, int]] = {}

    for batch in definitions:
        frames = {a: assignments[(batch, a)] for a in annotators if (batch, a) in assignments}
        ids_por_lote[batch] = {
            a: sorted([str(v) for v in row] for row in f[schema.key].itertuples(index=False))
            for a, f in frames.items()
        }
        union = pd.concat(frames.values()).drop_duplicates(schema.key)
        por_lote_y_clase[batch] = {
            str(c): int(n) for c, n in union[schema.klass].value_counts().sort_index().items()
        }

    return {
        "lotes": {
            batch: {
                "unidades": int(size),
                "compartido": shared,
                "por_anotador": {
                    a: int(len(assignments[(batch, a)]))
                    for a in annotators if (batch, a) in assignments
                },
            }
            for batch, (size, shared) in definitions.items()
        },
        "por_lote_y_clase": por_lote_y_clase,
        "ids_por_lote": ids_por_lote,
    }
