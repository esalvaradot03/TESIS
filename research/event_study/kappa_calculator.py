"""
Acuerdo entre anotadores sobre los lotes del corpus del event study.

Toma los dos CSV de un mismo lote (uno por anotador), los une por la llave
(post_id, target_ticker) y reporta:

  - **Kappa primario**: Cohen's kappa sobre 3 clases (bullish/bearish/
    neutral), EXCLUYENDO los pares que algún anotador marcó `unusable`.
    Un desacuerdo sobre si un post es etiquetable o no es una pregunta
    distinta de un desacuerdo de polaridad; mezclarlas infla el acuerdo.
  - **Kappa secundario**: las mismas 4 clases incluyendo `unusable`,
    reportado aparte para no perder esa información.
  - Matriz de confusión entre anotadores (4 clases).
  - Tasa de `unusable` por anotador.
  - Desacuerdos ordenados por confianza combinada descendente: donde ambos
    estaban seguros y aun así difieren es donde el codebook falla.
  - Control de integridad: pares presentes en un archivo y no en el otro,
    y pares todavía sin etiquetar.

El kappa primario se reporta sobre el lote `doble` (aleatorio). Los lotes
de uncertainty sampling son más difíciles por construcción y no sirven como
kappa de referencia.

Uso:
    python -m research.event_study.kappa_calculator <csv_a> <csv_b> <dir_salida>
"""

import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

logger = logging.getLogger(__name__)

_KEYS: list[str] = ["post_id", "target_ticker"]

_LABELS_MAIN: list[str] = ["bullish", "bearish", "neutral"]
_UNUSABLE: str = "unusable"
_LABELS_ALL: list[str] = _LABELS_MAIN + [_UNUSABLE]


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------

def annotator_name(path: Path) -> str:
    """
    Deriva el nombre del anotador del nombre de archivo (`doble_camilo.csv`
    → `camilo`). Si no hay separador, usa el stem completo.
    """
    stem = path.stem
    return stem.rsplit("_", 1)[-1] if "_" in stem else stem


def load_annotations(path: Path) -> pd.DataFrame:
    """
    Carga un CSV de etiquetado de un anotador.

    Args:
        path: Ruta al CSV (schema de build_corpus.write_batches).

    Returns:
        DataFrame con dtype=str; los vacíos quedan como "".

    Raises:
        ValueError: si faltan post_id, target_ticker o label.
    """
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = {"post_id", "target_ticker", "label"} - set(df.columns)
    if missing:
        raise ValueError(
            f"Columnas faltantes en {path}: {missing}. Columnas encontradas: {list(df.columns)}"
        )
    df["label"] = df["label"].str.strip().str.lower()
    return df


def merge_annotations(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    name_a: str,
    name_b: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Une los dos CSV por (post_id, target_ticker) y mide la integridad del cruce.

    Args:
        df_a: Anotaciones del primer anotador.
        df_b: Anotaciones del segundo.
        name_a: Nombre del primer anotador (sufijo de columnas).
        name_b: Nombre del segundo.

    Returns:
        (merged, integridad) donde merged solo tiene los pares presentes en
        AMBOS archivos y con label puesta por ambos, e integridad reporta
        pares faltantes de cada lado y pares sin etiquetar.
    """
    merged = pd.merge(
        df_a, df_b, on=_KEYS, how="outer", suffixes=(f"_{name_a}", f"_{name_b}"),
        indicator=True,
    )

    label_a, label_b = f"label_{name_a}", f"label_{name_b}"
    solo_a = int((merged["_merge"] == "left_only").sum())
    solo_b = int((merged["_merge"] == "right_only").sum())

    common = merged[merged["_merge"] == "both"].drop(columns="_merge")
    sin_etiquetar_a = int((common[label_a] == "").sum())
    sin_etiquetar_b = int((common[label_b] == "").sum())

    scored = common[(common[label_a] != "") & (common[label_b] != "")].copy()

    integridad = {
        "pares_totales": int(len(merged)),
        "pares_en_ambos": int(len(common)),
        f"solo_en_{name_a}": solo_a,
        f"solo_en_{name_b}": solo_b,
        f"sin_etiquetar_{name_a}": sin_etiquetar_a,
        f"sin_etiquetar_{name_b}": sin_etiquetar_b,
        "pares_evaluables": int(len(scored)),
    }

    if solo_a or solo_b:
        logger.warning(
            "Integridad: %d pares solo en %s y %d solo en %s (deberían ser 0 en un lote compartido).",
            solo_a, name_a, solo_b, name_b,
        )
    return scored, integridad


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def kappa_scores(scored: pd.DataFrame, name_a: str, name_b: str) -> dict:
    """
    Calcula el kappa primario (3 clases, sin `unusable`) y el secundario
    (4 clases, con `unusable`).

    Args:
        scored: Pares etiquetados por ambos (salida de merge_annotations).
        name_a: Nombre del primer anotador.
        name_b: Nombre del segundo.

    Returns:
        Dict con n_primario, kappa_primario, acuerdo_primario, n_secundario,
        kappa_secundario, acuerdo_secundario. Los kappa son NaN si no hay
        pares suficientes.
    """
    label_a, label_b = f"label_{name_a}", f"label_{name_b}"

    principal = scored[
        ~scored[label_a].eq(_UNUSABLE) & ~scored[label_b].eq(_UNUSABLE)
    ]

    def _kappa(frame: pd.DataFrame, labels: list[str]) -> tuple[float, float]:
        if frame.empty:
            return float("nan"), float("nan")
        kappa = float(cohen_kappa_score(frame[label_a], frame[label_b], labels=labels))
        acuerdo = float((frame[label_a] == frame[label_b]).mean())
        return kappa, acuerdo

    kappa_main, acuerdo_main = _kappa(principal, _LABELS_MAIN)
    kappa_all, acuerdo_all = _kappa(scored, _LABELS_ALL)

    logger.info(
        "Kappa primario (3 clases, n=%d): %.3f | secundario (4 clases, n=%d): %.3f",
        len(principal), kappa_main, len(scored), kappa_all,
    )
    return {
        "n_primario": int(len(principal)),
        "kappa_primario": kappa_main,
        "acuerdo_primario": acuerdo_main,
        "n_secundario": int(len(scored)),
        "kappa_secundario": kappa_all,
        "acuerdo_secundario": acuerdo_all,
    }


def confusion(scored: pd.DataFrame, name_a: str, name_b: str) -> pd.DataFrame:
    """
    Matriz de confusión entre anotadores sobre las 4 clases.

    Args:
        scored: Pares etiquetados por ambos.
        name_a: Nombre del primer anotador (filas).
        name_b: Nombre del segundo (columnas).

    Returns:
        DataFrame 4×4 indexado por las etiquetas de `name_a`.
    """
    matrix = confusion_matrix(
        scored[f"label_{name_a}"], scored[f"label_{name_b}"], labels=_LABELS_ALL
    )
    return pd.DataFrame(matrix, index=_LABELS_ALL, columns=_LABELS_ALL)


def unusable_rates(scored: pd.DataFrame, name_a: str, name_b: str) -> dict:
    """
    Fracción de pares marcados `unusable` por cada anotador.

    Args:
        scored: Pares etiquetados por ambos.
        name_a: Nombre del primer anotador.
        name_b: Nombre del segundo.

    Returns:
        Dict anotador → tasa en [0, 1].
    """
    if scored.empty:
        return {name_a: float("nan"), name_b: float("nan")}
    return {
        name_a: float(scored[f"label_{name_a}"].eq(_UNUSABLE).mean()),
        name_b: float(scored[f"label_{name_b}"].eq(_UNUSABLE).mean()),
    }


def disagreements(scored: pd.DataFrame, name_a: str, name_b: str) -> pd.DataFrame:
    """
    Pares donde las etiquetas difieren, ordenados por confianza combinada.

    Los desacuerdos donde ambos anotadores marcaron confianza alta son los
    que revelan ambigüedad real del codebook, no descuido, así que van
    primero para priorizar la adjudicación.

    Args:
        scored: Pares etiquetados por ambos.
        name_a: Nombre del primer anotador.
        name_b: Nombre del segundo.

    Returns:
        DataFrame con la llave, ambas etiquetas, ambas confianzas, su suma,
        el texto y ambas notas. Vacío si no hay desacuerdos.
    """
    label_a, label_b = f"label_{name_a}", f"label_{name_b}"
    diff = scored[scored[label_a] != scored[label_b]].copy()
    if diff.empty:
        logger.info("Sin desacuerdos entre %s y %s.", name_a, name_b)
        return diff

    conf_a, conf_b = f"confianza_{name_a}", f"confianza_{name_b}"
    for column in (conf_a, conf_b):
        if column not in diff.columns:
            diff[column] = ""
    diff["confianza_total"] = (
        pd.to_numeric(diff[conf_a], errors="coerce").fillna(0)
        + pd.to_numeric(diff[conf_b], errors="coerce").fillna(0)
    )

    text_column = next(
        (c for c in (f"clean_text_{name_a}", "clean_text", f"clean_text_{name_b}") if c in diff.columns),
        None,
    )
    columns = [*_KEYS, label_a, label_b, conf_a, conf_b, "confianza_total"]
    if text_column:
        columns.append(text_column)
    for note in (f"nota_{name_a}", f"nota_{name_b}"):
        if note in diff.columns:
            columns.append(note)

    diff = diff.sort_values("confianza_total", ascending=False)
    logger.info("%d desacuerdos de %d pares evaluables.", len(diff), len(scored))
    return diff[columns].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Reporte
# ---------------------------------------------------------------------------

def write_report(
    output_dir: Path,
    name_a: str,
    name_b: str,
    integridad: dict,
    kappas: dict,
    matriz: pd.DataFrame,
    tasas: dict,
    diff: pd.DataFrame,
) -> Path:
    """
    Escribe el reporte markdown del acuerdo entre anotadores.

    Args:
        output_dir: Directorio de salida.
        name_a: Nombre del primer anotador.
        name_b: Nombre del segundo.
        integridad: Control de integridad del cruce.
        kappas: Salida de kappa_scores().
        matriz: Salida de confusion().
        tasas: Salida de unusable_rates().
        diff: Salida de disagreements().

    Returns:
        Ruta al .md escrito.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"kappa_{name_a}_vs_{name_b}.md"

    lines = [
        f"# Acuerdo entre anotadores: {name_a} vs {name_b}",
        "",
        "## Kappa",
        "",
        "| Métrica | Clases | n | Kappa | Acuerdo exacto |",
        "|---|---|---|---|---|",
        f"| **Primario** (sin `unusable`) | bullish/bearish/neutral | {kappas['n_primario']} | "
        f"{kappas['kappa_primario']:.3f} | {kappas['acuerdo_primario'] * 100:.1f}% |",
        f"| Secundario (con `unusable`) | 4 clases | {kappas['n_secundario']} | "
        f"{kappas['kappa_secundario']:.3f} | {kappas['acuerdo_secundario'] * 100:.1f}% |",
        "",
        "## Tasa de `unusable`",
        "",
        "| Anotador | Tasa |",
        "|---|---|",
    ]
    lines += [f"| {who} | {rate * 100:.1f}% |" for who, rate in tasas.items()]

    lines += [
        "",
        "## Matriz de confusión",
        "",
        f"Filas = `{name_a}`, columnas = `{name_b}`.",
        "",
        "| | " + " | ".join(matriz.columns) + " |",
        "|---" * (len(matriz.columns) + 1) + "|",
    ]
    lines += [
        f"| **{idx}** | " + " | ".join(str(v) for v in row) + " |"
        for idx, row in matriz.iterrows()
    ]

    lines += [
        "",
        "## Integridad del cruce",
        "",
        "| Control | Valor |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in integridad.items()]

    lines += [
        "",
        "## Desacuerdos",
        "",
        f"{len(diff)} desacuerdos sobre {kappas['n_secundario']} pares evaluables. "
        "Ordenados por confianza combinada descendente — los primeros son los que "
        "más probablemente señalan un problema del codebook.",
        "",
    ]
    if not diff.empty:
        head = diff.head(15)
        lines += [
            "| " + " | ".join(head.columns) + " |",
            "|---" * len(head.columns) + "|",
        ]
        lines += [
            "| " + " | ".join(str(v).replace("|", "\\|")[:80] for v in row) + " |"
            for row in head.itertuples(index=False)
        ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Reporte escrito en %s", path)
    return path


def write_disagreements(output_dir: Path, name_a: str, name_b: str, diff: pd.DataFrame) -> Path:
    """
    Escribe el CSV completo de desacuerdos para adjudicación.

    Args:
        output_dir: Directorio de salida.
        name_a: Nombre del primer anotador.
        name_b: Nombre del segundo.
        diff: Salida de disagreements().

    Returns:
        Ruta al CSV escrito.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"desacuerdos_{name_a}_vs_{name_b}.csv"
    diff.to_csv(path, index=False, encoding="utf-8")
    logger.info("Desacuerdos escritos en %s (%d filas).", path, len(diff))
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

    if len(sys.argv) < 4:
        print(
            "Uso: python -m research.event_study.kappa_calculator <csv_a> <csv_b> <dir_salida>\n"
            "Ejemplo: python -m research.event_study.kappa_calculator "
            "data/manual_labels/doble_camilo.csv data/manual_labels/doble_esteban.csv "
            "research/event_study/"
        )
        sys.exit(1)

    path_a, path_b, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    who_a, who_b = annotator_name(path_a), annotator_name(path_b)
    if who_a == who_b:
        who_a, who_b = f"{who_a}_a", f"{who_b}_b"

    scored, integridad = merge_annotations(
        load_annotations(path_a), load_annotations(path_b), who_a, who_b
    )
    if scored.empty:
        print(
            f"No hay pares etiquetados por ambos anotadores.\nIntegridad: {integridad}\n"
            "¿Los CSV todavía están sin llenar?"
        )
        sys.exit(1)

    kappas = kappa_scores(scored, who_a, who_b)
    matriz = confusion(scored, who_a, who_b)
    tasas = unusable_rates(scored, who_a, who_b)
    diff = disagreements(scored, who_a, who_b)

    report_path = write_report(out_dir, who_a, who_b, integridad, kappas, matriz, tasas, diff)
    diff_path = write_disagreements(out_dir, who_a, who_b, diff)

    print(f"\nKappa primario (3 clases): {kappas['kappa_primario']:.3f}  (n={kappas['n_primario']})")
    print(f"Kappa secundario (4 clases): {kappas['kappa_secundario']:.3f}  (n={kappas['n_secundario']})")
    print(f"Desacuerdos: {len(diff)}")
    print(f"Reporte  → {report_path}")
    print(f"CSV      → {diff_path}")
