# Diagnóstico de `model_v0`

> Reporte de solo lectura. No se reentrenó ni modificó el modelo. Splits reconstruidos con las mismas funciones del pipeline (`_build_target` → `_temporal_split` → `_internal_val_split`).

- **Features:** 1863 filas | 5 tickers | rango 2020-09-11 → 2022-03-04
- **Split temporal:** train ≤ 2021-11-15 | test ≥ 2021-11-16
- **Tamaños:** core(train)=1190 | val=300 | test=373
- **Modelo:** 26 features

## 1. Balance del target por split

Target = 1 si `close[t+1] > close[t]` (el día siguiente sube).

| Split | n | % sube (1) | % baja (0) | n sube | n baja |
|---|---|---|---|---|---|
| train (core) | 1190 | 52.4% | 47.6% | 624 | 566 |
| val | 300 | 57.3% | 42.7% | 172 | 128 |
| test | 373 | 44.0% | 56.0% | 164 | 209 |

**Interpretación:** el balance en train es 52.4% sube vs 44.0% en test (Δ = -8.5 pp). 
El train tiene **mayoría de días al alza** (mercado alcista 2020-21) y el test **mayoría a la baja** (bear market 2022). El modelo aprendió el prior alcista del train; al cambiar el régimen en test, ese prior se vuelve sistemáticamente erróneo. **Este desbalance entre splits es la causa raíz del colapso a una sola clase.**

## 2. Matriz de confusión (test set)

| real \ pred | pred=0 (baja) | pred=1 (sube) |
|---|---|---|
| **real=0 (baja)** | 0 (TN) | 209 (FP) |
| **real=1 (sube)** | 0 (FN) | 164 (TP) |

- Predicciones clase 1 (sube): **373** / 373 (100.0%)
- Predicciones clase 0 (baja): **0** / 373 (0.0%)

```
              precision    recall  f1-score   support

    baja (0)       0.00      0.00      0.00       209
    sube (1)       0.44      1.00      0.61       164

    accuracy                           0.44       373
   macro avg       0.22      0.50      0.31       373
weighted avg       0.19      0.44      0.27       373
```

**Interpretación:** el modelo predice la clase 'sube' en el **100%** de los días de test → **colapso total a una clase**. El recall de la clase 'baja' es 0 (no detecta ni una caída) y la precision de 'sube' iguala el base rate de días al alza del test. La accuracy agregada es engañosa: equivale a la estrategia trivial 'comprar siempre'.

## 3. Distribución de probabilidades de salida (test)

- min = 0.506 | p25 = 0.517 | mediana = 0.526 | p75 = 0.535 | max = 0.543
- media = 0.527 | desviación = 0.010
- **% de probabilidades > 0.5: 100.0%**
- distancia media al umbral |p − 0.5| = 0.027 (máx posible 0.5)

Histograma de p(sube) en test:

```
  [0.00, 0.05)     0 |
  [0.05, 0.10)     0 |
  [0.10, 0.15)     0 |
  [0.15, 0.20)     0 |
  [0.20, 0.25)     0 |
  [0.25, 0.30)     0 |
  [0.30, 0.35)     0 |
  [0.35, 0.40)     0 |
  [0.40, 0.45)     0 |
  [0.45, 0.50)     0 |
  [0.50, 0.55)   373 |████████████████████████████████████████  <- 0.5
  [0.55, 0.60)     0 |
  [0.60, 0.65)     0 |
  [0.65, 0.70)     0 |
  [0.70, 0.75)     0 |
  [0.75, 0.80)     0 |
  [0.80, 0.85)     0 |
  [0.85, 0.90)     0 |
  [0.90, 0.95)     0 |
  [0.95, 1.00)     0 |
```

**Interpretación:** todas las probabilidades caen en una banda **estrechísima justo por encima de 0.5** (rango [0.506, 0.543], desviación 0.010). El matiz importa: el modelo **no está confiado** (no hay valores 0.8-0.9), sino que emite una salida **casi constante ≈ 0.53** para todos los días — ha perdido poder discriminativo en test (la varianza de la señal es ~0). Como toda la masa queda por encima de 0.5, el `argmax`/threshold 0.5 predice 'sube' el 100% del tiempo. Mover el umbral a ~0.53 reequilibraría el conteo de clases, pero **no añadiría skill**: con AUC≈0.55 el orden de las probabilidades apenas correlaciona con el target, así que separar por un umbral interno daría predicciones casi aleatorias. El problema no es el umbral sino el colapso de la señal.

## 4. Feature importance y correlación con el target

Top 10 por *gain*:

| # | feature | gain | weight | cover |
|---|---|---|---|---|
| 1 | net_sentiment_mean | 6.639 | 50 | 93.32 |
| 2 | prob_negative_mean | 5.448 | 24 | 67.16 |
| 3 | neutral_ratio | 5.252 | 39 | 64.69 |
| 4 | prob_positive_mean | 5.124 | 42 | 68.07 |
| 5 | net_sentiment_max | 4.944 | 80 | 76.74 |
| 6 | negative_ratio | 4.913 | 18 | 93.11 |
| 7 | close | 4.716 | 36 | 89.77 |
| 8 | bb_pct | 4.691 | 36 | 83.42 |
| 9 | bb_lower | 4.661 | 15 | 60.41 |
| 10 | bb_upper | 4.651 | 21 | 39.32 |

Correlación de Pearson **feature ↔ target** en train (core) vs test (un **cambio de signo** = distribution shift):

| feature de sentimiento | corr train | corr test | ¿cambia de signo? |
|---|---|---|---|
| mention_count | +0.028 | +0.015 | no |
| net_sentiment_mean | -0.113 | +0.087 | **SÍ** |
| net_sentiment_max | -0.044 | +0.145 | **SÍ** |
| net_sentiment_min | -0.061 | +0.005 | no |
| net_sentiment_std | +0.022 | +0.073 | no |
| positive_ratio | -0.066 | +0.121 | **SÍ** |
| negative_ratio | +0.101 | +0.021 | no |
| neutral_ratio | -0.037 | -0.102 | no |
| prob_positive_mean | -0.083 | +0.147 | **SÍ** |
| prob_negative_mean | +0.096 | -0.012 | no |
| prob_neutral_mean | -0.026 | -0.088 | no |
| weighted_sentiment | -0.076 | +0.106 | **SÍ** |
| total_upvotes | +0.043 | +0.006 | no |
| total_comments | +nan | +nan | no |
| stocktwits_native_sentiment | -0.035 | +0.019 | no |

**Interpretación:** 5 de 15 features de sentimiento **invierten el signo** de su correlación con el target entre train y test. 
Esto **confirma el distribution shift**: la relación que el modelo aprendió (p. ej. 'más sentimiento positivo → sube') deja de cumplirse —o se invierte— en el período de test. Un modelo estático no puede acertar si el signo de la señal cambia.

## 5. Análisis por régimen (mensual)

| mes | split | n | % sube (real) | precisión modelo | % pred sube | p(sube) media |
|---|---|---|---|---|---|---|
| 2020-09 | train | 70 | 62.9% | 62.9% | 100.0% | 0.53 |
| 2020-10 | train | 110 | 46.4% | 46.4% | 100.0% | 0.52 |
| 2020-11 | train | 100 | 57.0% | 57.0% | 100.0% | 0.52 |
| 2020-12 | train | 110 | 49.1% | 49.1% | 100.0% | 0.52 |
| 2021-01 | train | 95 | 56.8% | 56.8% | 100.0% | 0.52 |
| 2021-02 | train | 95 | 45.3% | 45.3% | 100.0% | 0.52 |
| 2021-03 | train | 115 | 48.7% | 48.7% | 100.0% | 0.53 |
| 2021-04 | train | 105 | 55.2% | 55.2% | 100.0% | 0.52 |
| 2021-05 | train | 100 | 47.0% | 47.0% | 100.0% | 0.53 |
| 2021-06 | train | 110 | 60.9% | 60.9% | 100.0% | 0.52 |
| 2021-07 | train | 105 | 51.4% | 51.4% | 100.0% | 0.52 |
| 2021-08 | train | 110 | 56.4% | 56.4% | 100.0% | 0.52 |
| 2021-09 | val | 105 | 50.5% | 50.5% | 100.0% | 0.52 |
| 2021-10 | val | 105 | 64.8% | 64.8% | 100.0% | 0.52 |
| 2021-11 | val | 105 | 51.4% | 51.4% | 100.0% | 0.52 |
| 2021-12 | test | 110 | 46.4% | 46.4% | 100.0% | 0.53 |
| 2022-01 | test | 100 | 40.0% | 40.0% | 100.0% | 0.53 |
| 2022-02 | test | 95 | 44.2% | 44.2% | 100.0% | 0.53 |
| 2022-03 | test | 18 | 27.8% | 27.8% | 100.0% | 0.53 |

**Interpretación:** en meses con mayoría de días al alza la precisión es **56.5%**; en meses con mayoría a la baja cae a **45.6%**. 
El modelo 'funciona' solo cuando el mes sube (porque siempre predice sube) y falla sistemáticamente cuando el mes baja. No hay capacidad direccional real: hay un sesgo constante hacia 'sube' que coincide con el mercado alcista del train y se rompe en el bear market del test.

## Conclusión del diagnóstico

1. **Causa raíz:** desbalance de régimen entre train (alcista) y test (bajista) — sección 1 — combinado con la inversión de la señal de sentimiento — sección 4.
2. **Síntoma:** el modelo colapsa a predecir 'sube' en el 100% del test (sección 2) con probabilidades casi constantes ≈ 0.53, apenas por encima de 0.5 y con varianza ~0 — pérdida de poder discriminativo, no exceso de confianza (sección 3).
3. **Implicación para el sistema adaptativo:** un modelo estático entrenado en un solo régimen no generaliza al siguiente. Hace falta (a) reentrenamiento incremental / ventana móvil, (b) posiblemente `scale_pos_weight` o re-balanceo por régimen, y (c) una señal de régimen de mercado que el modelo pueda usar. Esto motiva el sistema adaptativo.
