"""
Acuerdo entre anotadores sobre los lotes del corpus de prensa colombiana.

Todo el cálculo vive en `research/inter_annotator.py`, que comparten los dos
corpus de la tesis. Acá solo se declara qué distingue a este: la llave que une
los dos CSV, (article_id, emisor), y la columna que trae el texto que vio el anotador.

Uso:
    python -m research.colombia.kappa_calculator <csv_a> <csv_b> <dir_salida>
"""

import sys

from research.inter_annotator import main

# Llave de una unidad de anotación en este corpus.
KEYS: list[str] = ["article_id", "emisor"]

# Columna de texto, en orden de preferencia.
# El corpus colombiano trae el texto en `titular`; `clean_text` queda
# por compatibilidad con los CSV del event study.
TEXT_COLUMNS: tuple[str, ...] = ("titular", "clean_text")

USAGE = (
    "Uso: python -m research.colombia.kappa_calculator <csv_a> <csv_b> <dir_salida>\n"
    "Ejemplo: python -m research.colombia.kappa_calculator "
    "data/manual_labels_colombia/doble_camilo.csv data/manual_labels_colombia/doble_esteban.csv research/colombia/"
)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], KEYS, TEXT_COLUMNS, USAGE))
