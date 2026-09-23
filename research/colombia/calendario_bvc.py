"""
Calendario de días hábiles de la Bolsa de Valores de Colombia.

La cobertura de prensa se mide contra los días en que efectivamente hay
mercado (~246 al año), no contra la fracción 5/7 del calendario: Colombia tiene
18 festivos oficiales, la mayoría trasladados al lunes siguiente por la Ley 51
de 1983 ("Ley Emiliani"), y usar 5/7 sobreestima el denominador en ~7%.

La BVC no opera sábados, domingos ni festivos nacionales. No se modelan los
cierres extraordinarios (subastas suspendidas, jornadas especiales): son
puntuales y no mueven el porcentaje de cobertura.

Uso:
    from research.colombia.calendario_bvc import trading_days
    habiles = trading_days(date(2020, 1, 1), date(2025, 12, 31))
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

# Festivos de fecha fija (mes, día).
_FIXED: tuple[tuple[int, int], ...] = (
    (1, 1),    # Año Nuevo
    (5, 1),    # Día del Trabajo
    (7, 20),   # Independencia
    (8, 7),    # Batalla de Boyacá
    (12, 8),   # Inmaculada Concepción
    (12, 25),  # Navidad
)

# Festivos fijos que SÍ se trasladan al lunes siguiente (Ley 51 de 1983).
_FIXED_MOVABLE: tuple[tuple[int, int], ...] = (
    (1, 6),    # Reyes Magos
    (3, 19),   # San José
    (6, 29),   # San Pedro y San Pablo
    (8, 15),   # Asunción
    (10, 12),  # Día de la Raza
    (11, 1),   # Todos los Santos
    (11, 11),  # Independencia de Cartagena
)

# Festivos móviles relativos al Domingo de Pascua: (días de offset, si se traslada).
_EASTER_RELATIVE: tuple[tuple[int, bool], ...] = (
    (-3, False),  # Jueves Santo
    (-2, False),  # Viernes Santo
    (43, True),   # Ascensión
    (64, True),   # Corpus Christi
    (71, True),   # Sagrado Corazón
)


def _easter(year: int) -> date:
    """
    Domingo de Pascua por el algoritmo de Meeus/Jones/Butcher (gregoriano).

    Args:
        year: Año.

    Returns:
        Fecha del Domingo de Pascua.
    """
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    lun = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * lun) // 451
    month, day = divmod(h + lun - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _to_monday(day: date) -> date:
    """Traslada una fecha al lunes siguiente (o la deja si ya es lunes)."""
    return day + timedelta(days=(7 - day.weekday()) % 7)


@lru_cache(maxsize=None)
def holidays(year: int) -> frozenset[date]:
    """
    Festivos nacionales de Colombia de un año, con la Ley Emiliani aplicada.

    Args:
        year: Año a calcular.

    Returns:
        Conjunto de fechas festivas (18 por año).
    """
    days: set[date] = {date(year, month, day) for month, day in _FIXED}
    days |= {_to_monday(date(year, month, day)) for month, day in _FIXED_MOVABLE}

    easter = _easter(year)
    for offset, moves in _EASTER_RELATIVE:
        day = easter + timedelta(days=offset)
        days.add(_to_monday(day) if moves else day)
    return frozenset(days)


def is_trading_day(day: date) -> bool:
    """
    True si la BVC opera ese día (lunes a viernes y no festivo nacional).

    Args:
        day: Fecha a evaluar.

    Returns:
        True si es día de trading.
    """
    return day.weekday() < 5 and day not in holidays(day.year)


def trading_days(start: date, end: date) -> list[date]:
    """
    Días de trading de la BVC en un rango cerrado [start, end].

    Args:
        start: Primer día del rango.
        end: Último día del rango (incluido).

    Returns:
        Lista ordenada de días hábiles; vacía si end < start.
    """
    days: list[date] = []
    day = start
    while day <= end:
        if is_trading_day(day):
            days.append(day)
        day += timedelta(days=1)
    return days


if __name__ == "__main__":
    for year in range(2020, 2026):
        print(f"{year}: {len(holidays(year))} festivos, "
              f"{len(trading_days(date(year, 1, 1), date(year, 12, 31)))} días de trading")
