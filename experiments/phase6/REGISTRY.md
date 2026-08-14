# FASE 6 — PRE-REGISTRO: Sentimiento intradía (StockTwits, ventanas de 30 min)

> Documento vinculante, escrito ANTES de correr el experimento predictivo. Mismo
> protocolo que Fases 3-5: criterio fijo, placebo obligatorio, una corrida, sin ajuste
> post-hoc. Aislado en `src/phase6/` y `experiments/phase6/`; no toca nada existente.

## Motivación
5.1 mostró correlación contemporánea diaria fuerte (Spearman hasta ρ=0.51) sin poder
predictivo diario (Fases 1-4). Hipótesis: la dinámica ocurre DENTRO del día y la
agregación diaria la destruye. Fase 6 baja la frecuencia a 30 min.

## Limitaciones de datos (conocidas y ACEPTADAS)
- **Rango 2020-08-01 → 2022-12-30, forzado por Alpaca IEX** (feed gratuito, sin barras
  intradía antes de 2020-07-27). No se paga feed; NO se extiende a 2016-2019.
- **Régimen train ≠ test:** el train (2020-08 → 2021-12) es mayormente **alcista**; el
  test (2022) es un **régimen bajista** distinto. Limitación reconocida: un negativo puede
  deberse en parte al cambio de régimen, y un positivo sería más creíble por atravesarlo.
  Se mitiga con el walk-forward (abajo).
- **Noticias FNSPID quedan fuera** (timestamps fecha-solo, verificado en Tarea 1).
- **Activos:** TSLA y AMD (densidad intradía sobrada; ver audit.md).

## Sentimiento: decisión del join de labels (fijada por regla ANTES de ver el resultado)
El label Bull/Bear no está en el archivo intradía; se une por `message_id` con
`symbol_sentiments`. Mediana de mensajes ETIQUETADOS por ventana de 30 min:
**TSLA = 76, AMD = 18** (ambos ≥10; frac etiquetada mediana 0.57 / 0.50, ver
label_join.md). Por la regla pre-fijada (≥10 en ambos), **el NET etiquetado es la feature
de sentimiento principal en ambos activos** (`net = (bull − bear)/(bull + bear)` sobre los
mensajes etiquetados de la ventana). Volumen y aceleración entran como actividad complementaria.

## Ventanas, retornos y emparejamiento (sin leakage)
- Barras Alpaca de 30 min; la barra que empieza en `t` cubre `[t, t+30)`.
  `ret(t) = close(t)/open(t) − 1`; `range(t) = (high(t) − low(t))/open(t)`.
- **Objetivo:** signo del `ret` de la ventana `[t, t+30]` (`y = 1` si `ret > 0`, si no 0).
- **Sentimiento predictor:** agregados de la ventana **anterior** `[t−30, t]` (disponibles
  al inicio de `[t, t+30]`).
- **Regla intradía (FIJADA):** el par predictor→objetivo `[t−30,t]→[t,t+30]` es
  **estrictamente del mismo día**; se **excluye la primera ventana objetivo de cada día**
  (su ventana de sentimiento caería en la sesión previa / overnight). El par nunca cruza
  el cierre ni el overnight.
- **Baseline autorregresivo:** `ret` y `range` de las **1-5 ventanas previas**. Para
  ventanas objetivo temprano en el día, los lags 2-5 pueden alcanzar la sesión anterior;
  se **retienen y se marcan** con la feature `n_gap_lags` (nº de lags que caen en un día
  previo). El lag1 (sentimiento y AR) es siempre del mismo día por la exclusión anterior.

## Features
- **Baseline (autorregresivo), 11:** `ret_lag1..5`, `range_lag1..5`, `n_gap_lags`.
- **Sentimiento/actividad (se añaden en base+sent), 5:** `net_lag1` (NET de `[t−30,t]`,
  principal), `net_accel` (`net_lag1 − net_lag2`), `vol_lag1` (`log1p` nº mensajes),
  `vol_accel` (`vol_lag1 − vol_lag2`), `lab_lag1` (`log1p` nº etiquetados).

## Modelos (por activo × tamaño de ventana)
- **BASELINE:** solo las 11 autorregresivas.
- **BASELINE+SENT:** + las 5 de sentimiento/actividad.
- **Placebo:** base+sent con el **bloque de sentimiento permutado temporalmente**
  (barajado por filas dentro de train y dentro de test, semilla 42), autorregresivas
  intactas. XGBoost con los hiperparámetros/semilla de Fases 4-5
  (`max_depth=3`, `lr=0.05`, `n_estimators=400`/es=40, `subsample=colsample=0.8`,
  `min_child_weight=5`, `seed=42`), validación interna temporal (último 15% del train).

## H6 (hipótesis)
> Las features de sentimiento/actividad de StockTwits de `[t−30, t]` predicen el signo del
> retorno de `[t, t+30]`, por encima del baseline autorregresivo y del placebo permutado.

## Criterio de éxito (pre-registrado, fijo)
Por (activo, ventana): **ÉXITO ⇔ AUC(base+sent) − AUC(base) > 0.02 en test
Y AUC(base+sent) > AUC(placebo).** Sin ajustes posteriores.

## Split y robustez
- **Principal:** train 2020-08-01 → 2021-12-31; test 2022-01-01 → 2022-12-30.
- **Robustez walk-forward:** re-entrenar cada 3 meses y probar el trimestre siguiente
  (ventana expansiva); reportar ΔAUC por trimestre.
- **Robustez secundaria:** repetir todo con ventanas de **60 min** (mismo esquema).

## Diagnóstico contemporáneo (incluido, como en 5.1)
Correlación de Spearman del NET de `[t−30, t]` con:
- `ret` de `[t−30, t]`  → **coincidente**;
- `ret` de `[t−60, t−30]` → **reactivo** (¿el sentimiento sigue al precio con 30 min de rezago?);
- `ret` de `[t, t+30]`   → **anticipatorio** (lo que el modelo intentaría explotar).
Mapea la estructura temporal a 30 min aunque el modelo predictivo dé negativo.
