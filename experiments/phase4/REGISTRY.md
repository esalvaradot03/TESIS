# Fase 4 — PRE-REGISTRO (escrito ANTES de correr cualquier modelo)

> Documento vinculante. Toda decisión de diseño, criterio de éxito y manejo de
> datos queda fijada acá antes de ver un solo resultado. No se ajusta nada
> post-hoc. Una sola corrida por experimento. Lo que dé negativo se reporta
> negativo.

## Motivación del pivote
Las Fases 1-3 dieron negativo sobre el universo amplio del S&P 500 (clasificación
y asset-pricing). Fase 4 concentra el estudio en **5 activos fijos** de alta
actividad social y de noticias, y por primera vez usa el **sentimiento de noticias
FNSPID**. Se comparan **dos fuentes de sentimiento** con **modelos independientes**.

## Activos (seleccionados data-driven en Tarea 1, ver asset_selection.md)
TSLA, AMD, DIS, BA, GILD. Criterio de selección: alta actividad simultánea en
StockTwits y en noticias FNSPID, cobertura de precio 2015-2022 completa, y
diversidad sectorial (consumo/EV, semiconductores, media, industriales, salud).

## Dos modelos por activo (10 experimentos)
- **Modelo A — StockTwits:** sentimiento nativo Bullish/Bearish del dataset
  StockTwits NYU (`symbol_sentiments`, columna `sentiment` ∈ {−1,0,+1}).
- **Modelo B — Noticias FNSPID:** sentimiento FinBERT (`ProsusAI/finbert`) aplicado
  al **título** de cada artículo de `nasdaq_exteral_data.csv` para el activo.

Cada modelo usa **solo** las features de su fuente. **Sin features técnicas** (el
objetivo es aislar el efecto del sentimiento). 5 activos × 2 fuentes = **10 modelos
+ 10 placebos**.

Nota de diseño (declarada a priori): las dos fuentes usan su señal de sentimiento
más fiable y point-in-time —etiqueta nativa de la multitud en StockTwits, FinBERT
sobre titulares en noticias—. Es una **comparación de fuentes**, no de motores de
NLP; la asimetría metodológica es intencional y se reporta como tal.

## Precios y target
- Precios: **yfinance** (cierre ajustado, `auto_adjust=True`), 2015-01-01 … 2022-12-31.
  Se usa yfinance —no el `full_history` local de FNSPID— para evitar el artefacto
  de truncamiento 2020 documentado en Tarea 1.
- Calendario: días de trading definidos por el índice de precios de yfinance del activo.
- **Target binario:** `y_T = 1` si `close_{T+1} > close_T`, si no `0`. Empates
  (retorno exactamente 0, rarísimo) → `0`.

## Alineación temporal (sin leakage)
- Las features del día de trading `T` agregan todos los mensajes/artículos con
  **fecha de calendario T**. Items en días no-trading (finde/feriado) se asignan al
  **siguiente día de trading** (convención del proyecto).
- Las features de `T` predicen el retorno de `close_T → close_{T+1}`. Ninguna feature
  usa información posterior a T.

## Features por fuente (idénticas en estructura para A y B; 5 por fuente)
Sea `net_T` el sentimiento neto del día T de esa fuente:
- StockTwits: `net_T = (bull − bear) / (bull + bear)` con
  `bull=#(sentiment>0)`, `bear=#(sentiment<0)` ese día; ∈ [−1, +1].
- Noticias: `net_T = media_T(prob_pos − prob_neg)` de FinBERT sobre los títulos del
  día; ∈ [−1, +1].

Features (prefijo `st_` para Modelo A, `nw_` para Modelo B):
1. `net`     — sentimiento neto del día (0 si no hay actividad ese día).
2. `net_3d`  — media móvil de `net` de los últimos 3 días de trading (min 1 obs).
3. `vol`     — `log(1 + nº de mensajes/artículos del día)`.
4. `mom`     — momentum: `net_T − media(net_{T-1..T-3})` (sobre la serie rellenada).
5. `has`     — binaria: 1 si hubo ≥1 mensaje/artículo ese día, 0 si no.

## Manejo de días SIN actividad (decisión FIJADA antes de ver resultados)
**Requisito explícito del pre-registro.**

- **Días sin noticias (frecuente en FNSPID):** el sentimiento se pone en
  **neutro/cero** (`nw_net = 0`, y `nw_net_3d`/`nw_mom` se computan sobre la serie
  rellenada con 0), `nw_vol = 0`, y la binaria **`nw_has = 0`**.
  Justificación: (a) 0 es el punto neutro de la escala de `net`, así que no inyecta
  señal direccional falsa; (b) la binaria `has` permite a XGBoost distinguir
  "sin cobertura" de "cubierto-pero-neutro", que son estados genuinamente
  distintos; (c) es el tratamiento estándar de features de eventos dispersos y
  **conserva todos los días de trading** (no condiciona la muestra a la llegada de
  noticias, lo que sesgaría la comparación contra el Modelo A).
  Alternativas consideradas y **rechazadas** a priori:
    - *Forward-fill* del último sentimiento → rechazada: filtra sentimiento rancio
      hacia adelante y difumina la señal diaria que queremos aislar.
    - *Descartar días sin noticias* → rechazada: reduce drásticamente la muestra del
      Modelo B y condiciona la muestra a que haya cobertura, sesgando la
      comparación A vs B.
- **Días sin mensajes de StockTwits (raro en estos 5 activos de alta actividad):**
  **misma política** — `st_net = 0`, `st_vol = 0`, `st_has = 0`, `st_net_3d`/`st_mom`
  sobre la serie rellenada. La feature `st_has` queda igualmente definida aunque en
  la práctica sea casi siempre 1.

## Split temporal (estricto, sin shuffle)
- **Train:** días de trading `T` en **[2015-01-01, 2020-12-31]**.
- **Test:**  días de trading `T` en **[2021-01-01, 2022-12-31]**.
- Se **excluye del train el último día de trading de 2020**, porque su etiqueta
  proviene del cierre de 2021 (evita que la etiqueta cruce la frontera train/test).
- Las fechas exactas de corte (primer/último día de trading de cada tramo por activo)
  se documentan en el reporte al construir el dataset.
- Validación interna para early stopping: **último 15% del train por fecha**
  (temporal, nunca aleatorio). El test NO se toca en entrenamiento.

## Modelo (hiperparámetros FIJOS, sin tuning)
XGBoost clasificación binaria: `n_estimators=400` (early stopping paciencia 40),
`max_depth=3`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`,
`min_child_weight=5`, `eval_metric=logloss`, `random_state=42`. Idénticos para los
10 modelos y sus placebos.

## Control placebo (idéntico a fases anteriores; obligatorio)
Para cada (activo, fuente): pipeline idéntico pero con las **mismas features
permutadas temporalmente** — se baraja el orden de las filas del bloque de features
respecto a las fechas/targets, **por separado dentro del train y dentro del test**
(semilla 42). Esto preserva el balance de clases y la distribución marginal de cada
feature, pero **destruye la alineación temporal real** con los retornos. Se entrena
el placebo sobre el train permutado y se evalúa sobre el test permutado.
**Ningún resultado cuenta sin su placebo.**

## Hipótesis direccional por experimento
Para cada activo `a` ∈ {TSLA, AMD, DIS, BA, GILD} y fuente `s` ∈ {StockTwits, Noticias}:

> **H_{a,s}:** el sentimiento de la fuente `s` predice la dirección del retorno del
> día siguiente de `a`. Signo comprometido a priori: mayor sentimiento *bullish*
> (net más alto) → mayor probabilidad de día siguiente al alza (confirmación, no
> contrarian).

Prior honesto dado el resultado de Fases 1-3: se **espera negativo** en la mayoría o
en todos. El criterio no cambia por esa expectativa.

## Criterio de éxito (pre-registrado, por experimento)
Un experimento (activo, fuente) es **ÉXITO** si y solo si:
1. **AUC en test > 0.55**, y
2. **AUC en test > AUC del placebo** (misma partición, features permutadas).

Se reporta además, como contexto, si el placebo por sí solo alcanza 0.55. Si el AUC
real no supera 0.55, o el placebo lo iguala/supera → **NEGATIVO**. Sin excepciones,
sin reajuste de umbral.

## Entregables (reporte)
Por cada uno de los 10 modelos: AUC test, accuracy, precisión por clase (sube/baja),
AUC placebo, veredicto ÉXITO/NEGATIVO, y top de importancia de features. Tabla
resumen de los 10 modelos + 10 placebos. Comparación agregada A (StockTwits) vs
B (Noticias).
