"""
Acuerdo entre anotadores sobre los lotes del corpus del event study.

Todo el cálculo vive en `research/inter_annotator.py`, que comparten los dos
corpus de la tesis. Acá solo se declara qué distingue a este: la llave que une
los dos CSV, (post_id, target_ticker), y la columna que trae el texto que vio el anotador.

Uso:
    python -m research.event_study.kappa_calculator <csv_a> <csv_b> <dir_salida>
"""

import sys

from research.inter_annotator import main

# Llave de una unidad de anotación en este corpus.
KEYS: list[str] = ["post_id", "target_ticker"]

# Columna de texto, en orden de preferencia.
# El corpus del event study trae el texto en `clean_text`.
TEXT_COLUMNS: tuple[str, ...] = ("clean_text",)

USAGE = (
    "Uso: python -m research.event_study.kappa_calculator <csv_a> <csv_b> <dir_salida>\n"
    "Ejemplo: python -m research.event_study.kappa_calculator "
    "data/manual_labels/doble_camilo.csv data/manual_labels/doble_esteban.csv research/event_study/"
)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], KEYS, TEXT_COLUMNS, USAGE))
