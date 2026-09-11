# Fase 6 — cuantificación del join de labels intradía (30 min)

Join `feature_wo_messages` (timestamp) × `symbol_sentiments` (label) por `message_id`, TSLA/AMD, 2020-08→2022-12, horario de mercado ET.

**Regla fijada ANTES de ver el resultado:** mediana de mensajes etiquetados por ventana >=10 en AMBOS → NET etiquetado es feature principal; <10 en alguno → ese activo usa volumen+aceleración como principales y NET como secundaria.

| activo | ventanas | msgs | etiquetados | frac label (mediana / p10) | etiquetados/ventana (mediana / p10) | decisión |
|--------|----------|------|-------------|----------------------------|-------------------------------------|----------|
| TSLA | 8,189 | 1,423,335 | 813,478 | 0.569 / 0.467 | **76.0** / 36.0 | NET_principal |
| AMD | 8,180 | 385,079 | 196,503 | 0.5 / 0.318 | **18.0** / 6.0 | NET_principal |

**Resultado de la regla:** el NET etiquetado ES la feature principal en ambos activos (ver columna decisión). Esta elección queda fijada para el pre-registro de H6.
