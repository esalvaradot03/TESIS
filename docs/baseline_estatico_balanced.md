# Baseline estático mejorado (balanceado) + walk-forward

> NO se modifica el sistema adaptativo. Objetivo: una comparación justa estático-vs-adaptativo más adelante. El modelo balanceado se guarda en `models/model_v0_balanced` (mismo split y mismas features que `model_v0`).

## 1. `model_v0_balanced` — `scale_pos_weight` ajustado al ratio del train

- Train (core): 624 sube / 566 baja → **scale_pos_weight = neg/pos = 0.9071**.
- Sin cambiar features (26) ni splits. Todo lo demás idéntico a `model_v0`.

### Comparación en el MISMO test set

| | model_v0 (original) | model_v0_balanced |
|---|---|---|
| Predicciones 'sube' | 100.0% | 59.8% |
| Accuracy | 0.4397 | 0.5469 |
| AUC-ROC | 0.5541 | 0.5570 |
| Rango p(sube) | [0.506, 0.543] | [0.482, 0.520] |

Matriz de confusión — `model_v0` (original):

| real \ pred | pred=0 (baja) | pred=1 (sube) |
|---|---|---|
| **real=0** | 0 (TN) | 209 (FP) |
| **real=1** | 0 (FN) | 164 (TP) |

Matriz de confusión — `model_v0_balanced`:

| real \ pred | pred=0 (baja) | pred=1 (sube) |
|---|---|---|
| **real=0** | 95 (TN) | 114 (FP) |
| **real=1** | 55 (FN) | 109 (TP) |

```
              precision    recall  f1-score   support

    baja (0)       0.63      0.45      0.53       209
    sube (1)       0.49      0.66      0.56       164

    accuracy                           0.55       373
   macro avg       0.56      0.56      0.55       373
weighted avg       0.57      0.55      0.54       373
```

**¿El balanceo arregla el colapso?** **SÍ**: las predicciones 'sube' pasaron de 100.0% a 59.8%; la matriz de confusión deja de ser degenerada (ahora hay TN y FN).

**¿Cambia el AUC?** Prácticamente no: 0.5541 → 0.5570 (Δ = +0.0029). El balanceo **re-centra el umbral** (las probabilidades ahora cruzan 0.5) pero **no añade poder discriminativo**: el AUC sigue cerca de 0.55 (apenas mejor que el azar 0.50). La 'mejora' de accuracy es por dejar de apostar todo a una clase en un test bajista, no por skill real.

## 2. Walk-forward del modelo estático (baseline justo)

Una sola entrenada al inicio (con `scale_pos_weight` del train inicial), predicción **día a día** sobre todo el horizonte forward, **sin reentrenar**.

- Origen de entrenamiento: hasta **2021-06-08** (train inicial 930 filas; core 740, val 190).
- Horizonte forward: **2021-06-09 → 2022-03-04** (933 predicciones (ticker, día)).
- scale_pos_weight = 0.9023 | best_iteration = 0

### Métricas agregadas (todo el horizonte forward)

- Accuracy: **0.4673** | AUC-ROC: **0.4722** | predicciones 'sube': 34.6% | base rate 'sube': 51.2%

| real \ pred | pred=0 (baja) | pred=1 (sube) |
|---|---|---|
| **real=0** | 284 (TN) | 171 (FP) |
| **real=1** | 326 (FN) | 152 (TP) |

### Métricas por mes

| mes | n | % sube (real) | accuracy | AUC | % pred sube | p(sube) media |
|---|---|---|---|---|---|---|
| 2021-06 | 80 | 61.3% | 0.625 | 0.699 | 41.2% | 0.501 |
| 2021-07 | 105 | 51.4% | 0.524 | 0.570 | 30.5% | 0.498 |
| 2021-08 | 110 | 56.4% | 0.364 | 0.376 | 32.7% | 0.498 |
| 2021-09 | 105 | 50.5% | 0.410 | 0.454 | 31.4% | 0.498 |
| 2021-10 | 105 | 64.8% | 0.390 | 0.402 | 38.1% | 0.500 |
| 2021-11 | 105 | 51.4% | 0.457 | 0.479 | 33.3% | 0.498 |
| 2021-12 | 110 | 46.4% | 0.455 | 0.387 | 28.2% | 0.498 |
| 2022-01 | 100 | 40.0% | 0.510 | 0.480 | 43.0% | 0.500 |
| 2022-02 | 95 | 44.2% | 0.516 | 0.511 | 31.6% | 0.500 |
| 2022-03 | 18 | 27.8% | 0.500 | 0.477 | 55.6% | 0.503 |

**¿La señal por mes sigue plana o aparece skill?** AUC mensual medio = **0.483** (mediana 0.478); solo **2/10** meses superan AUC 0.55. 
La señal **sigue esencialmente plana**: el balanceo evita el colapso degenerado pero no recupera capacidad direccional. El modelo estático no generaliza a través del cambio de régimen — justo lo que motivará al sistema adaptativo.

## Conclusión

1. **El balanceo arregla el colapso** (sube%: 100.0% → 59.8%) pero **no añade skill** (AUC 0.554 → 0.557).
2. El **walk-forward estático** confirma que la capacidad direccional es ~aleatoria a lo largo del horizonte (AUC agregado 0.472, mensual medio 0.483).
3. Este es el **baseline justo** (mismo armazón walk-forward) contra el que se medirá el sistema adaptativo: si el adaptativo no supera claramente estas cifras, no aporta.
