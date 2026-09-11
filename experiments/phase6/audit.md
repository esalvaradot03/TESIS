# Fase 6 — Tarea 1: auditoría de viabilidad de datos intradía

## Resumen ejecutivo

- **StockTwits** tiene timestamps intradía reales (en `feature_wo_messages`, `created_at` ISO-8601 UTC con `Z`; convertidos a ET America/New_York con DST). `symbol_sentiments` es fecha-solo, NO sirve para intradía.
- **OJO — las etiquetas Bull/Bear NO están en el archivo intradía.** `feature_wo_messages` trae timestamp pero su columna `sentiment` viene vacía (0/3.8M etiquetadas). El label nativo vive en `symbol_sentiments` (fecha-solo), keyado por `message_id`. Para el sentimiento neto intradía hay que **unir por `message_id`** ambos archivos. El conteo por ventana de abajo es de **volumen de mensajes** (todos), no del subconjunto etiquetado; la densidad del NET etiquetado se cuantifica en Tarea 2 tras el join (los volúmenes de `symbol_sentiments` para TSLA/AMD son comparables al volumen intradía, así que se espera cobertura de label alta).
- **Noticias FNSPID: fecha-solo** (99.8% de los artículos de TSLA/AMD sellados a `00:00`) → **quedan FUERA de la Fase 6**.
- **Precios Alpaca IEX (gratuito): solo desde 2020-07-27.** Sin barras intradía 2016–jul2020. Cobertura 2020H2–2022 ≈ 99.8% de las 13 ventanas de 30min/día. **Este es el cuello de botella: el target intradía solo existe ~ago2020–dic2022.**

## Distribución de mensajes StockTwits por ventana (horario de mercado ET)

`mediana/p90` sobre ventanas con ≥1 mensaje; `%<5` sobre TODO el universo de ventanas de mercado (incluye vacías); `cobertura` = ventanas con actividad / universo. Universo `barras` = ventanas con barra Alpaca (exacto, 2020H2+); `aprox` = díashábiles×N (2016-2019, sin barras).

| ticker | año | src | msgs (vol) | 30m cob% | 30m mediana | 30m p90 | 30m %<5 | 60m mediana | 60m p90 | 60m %<5 |
|--------|-----|-----|-----------|----------|-------------|---------|---------|-------------|---------|---------|
| TSLA | 2016 | sin precio | 70,335 | — | 14 | 43 | — | 27 | 80 | — |
| TSLA | 2017 | sin precio | 155,593 | — | 35 | 88 | — | 67 | 164 | — |
| TSLA | 2018 | sin precio | 226,311 | — | 47 | 135 | — | 88 | 257 | — |
| TSLA | 2019 | sin precio | 214,815 | — | 45 | 129 | — | 84 | 240 | — |
| TSLA | 2020 | barras | 312,696 | 100 | 138 | 374 | 0 | 253 | 715 | 0 |
| TSLA | 2021 | barras | 563,225 | 100 | 134 | 300 | 0 | 250 | 563 | 0.1 |
| TSLA | 2022 | barras | 546,936 | 100 | 131 | 275 | 0 | 250 | 529 | 0 |
| AMD | 2016 | sin precio | 72,925 | — | 10 | 59 | — | 18 | 106 | — |
| AMD | 2017 | sin precio | 357,100 | — | 77 | 223 | — | 145 | 412 | — |
| AMD | 2018 | sin precio | 297,708 | — | 63 | 174 | — | 120 | 321 | — |
| AMD | 2019 | sin precio | 184,099 | — | 41 | 106 | — | 76 | 198 | — |
| AMD | 2020 | barras | 62,255 | 100 | 38 | 82 | 0 | 70 | 151 | 0 |
| AMD | 2021 | barras | 171,537 | 99.9 | 38 | 94 | 0.1 | 72 | 172 | 0.1 |
| AMD | 2022 | barras | 154,891 | 100 | 38 | 85 | 0.1 | 70 | 161 | 0 |

## Recomendación

**Zona horaria:** timestamps StockTwits en UTC (`...Z`) → convertidos a ET (America/New_York, con DST). Verificado contra barras Alpaca (ambos alinean en 09:30 ET apertura).

**Activos viables:** TSLA y AMD, **ambos con densidad de sobra**. En el período con precios (2020H2-2022) la cobertura de ventanas de 30 min es ~100% y el %<5 msgs ≈0: prácticamente ninguna ventana de mercado queda por debajo de 5 mensajes. Medianas: TSLA ~130 msgs/ventana-30min, AMD ~38; p90 TSLA ~300, AMD ~90.

**Rango de años (cuello de botella):** limitado por precios a **2020-07-27 → 2022-12-30** (IEX gratuito no da intradía antes). StockTwits cubre 2016-2022, pero sin barras no hay target intradía pre-ago2020. Split viable: **train 2020-08 → 2021-12 (~17 meses), test 2022 (12 meses)**.

**Ventana 30 vs 60 min:** **30 min es viable para ambos** (densidad no es problema). 60 min duplica la densidad y da más resolución estadística por ventana, pero **la mitad de observaciones** (menos poder en el test de ~1 año). Recomendación: **30 min como principal** (más observaciones para el ya corto test 2022), con 60 min como chequeo de robustez.

**Aviso metodológico para Tarea 2:** el NET de sentimiento intradía exige unir `feature_wo_messages` (timestamp) con `symbol_sentiments` (label Bull/Bear) por `message_id` — el archivo intradía no trae el label. Hay que cuantificar la fracción etiquetada por ventana antes de pre-registrar (afecta la densidad efectiva del NET).

**Nota:** extender a 2016-2019 exigiría un feed intradía pago (SIP/Polygon); con datos gratuitos, la Fase 6 es un estudio de ~2.4 años (2020H2-2022).
