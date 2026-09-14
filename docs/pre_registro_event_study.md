
---

## 1. Qué ya se corrió antes de este documento (y por lo tanto es exploratorio)

Honestidad metodológica primero: los siguientes análisis **ya se ejecutaron y
sus resultados ya se vieron**, el 2026-09-13, antes de existir este
pre-registro. Por la regla 1 del protocolo quedan marcados como
**exploratorios** y no pueden presentarse como confirmatorios, ni usarse para
fijar los criterios de la sección 4 de forma que los favorezca.

| Análisis | Resultado observado | Dónde |
|---|---|---|
| Kappa de `l0` | 0.628 primario (n=77), 75.3% acuerdo exacto | `research/event_study/kappa_l0/` |
| Kappa de `doble` | 0.788 primario (n=155), 86.5% acuerdo exacto | `research/event_study/kappa_doble/` |
| Comparación de 3 enfoques (train/test de Esteban) | probe 0.531 > FinBERT 0.370 > VADER 0.357 > placebo 0.329 (macro-F1) | `comparacion_enfoques_exploratoria.csv` |
| Comparación con ambos anotadores | probe (ambos) 0.474–0.505; transferencia cruzada por debajo del placebo | `comparacion_enfoques_v2.csv` |
| Diagnóstico de hiperparámetros | Los defaults del repo (3 épocas, lr 1e-4) dejan la cabeza en azar | `linear_probe_diagnostico.csv` |

**Consecuencia:** la comparación de enfoques se reporta como análisis
exploratorio. La corrida confirmatoria es la de la sección 4.3, sobre el corpus
corregido (sección 3) y con los hiperparámetros fijados acá.

---

## 2. Pregunta e hipótesis

**Pregunta:** ¿cómo se comparan tres enfoques de análisis de sentimiento en su
capacidad de capturar señal en ventanas alrededor de eventos catalizadores,
para 6 tickers del S&P 500 segmentados por liquidez?

Enfoques: (1) FinBERT off-the-shelf, (2) FinBERT con linear probing sobre el
corpus etiquetado, (3) lexicón VADER.

**Marco previo (establecido, no se re-litiga):** las fases 1–6 concluyeron que
el sentimiento retail de StockTwits **no predice** retornos (AUC ~0.50) y que
es un indicador coincidente y reactivo. Este estudio es de **medición**, no de
predicción.

### Hipótesis

- **H1 (dirección).** El sentimiento pre-evento, medido con cualquiera de los
  tres enfoques, **no** predice el signo del CAR post-evento. Predicción
  direccional esperada ≈ 50%, consistente con el marco previo.
- **H2 (reacción).** El sentimiento **post**-evento sí correlaciona con el CAR
  contemporáneo, y esa correlación es mayor que la anticipatoria. Es la
  hipótesis coincidente/reactiva, replicada a nivel de evento.
- **H3 (liquidez).** La magnitud de la correlación reactiva de H2 difiere entre
  segmentos de liquidez: alta (NCLH, DIS), media (CRWD, TGT), baja (CMG, DDOG).
  Sin dirección pre-especificada; es exploratoria dentro del confirmatorio.
- **H4 (enfoques).** El linear probing captura la señal reactiva de H2 mejor
  que FinBERT off-the-shelf, y este mejor que VADER, medido como |ρ| entre el
  score de sentimiento y el CAR contemporáneo.

---

## 3. Datos, y una decisión pendiente

**Corpus de texto:** bucket NYU (`symbol_sentiments` ⋈ `messages`), 2010-06 a
2022-12-31. Pool de 1.130.992 pares (post, ticker): 420.043 etiquetados por el
autor y 710.949 sin etiqueta, con fecha interpolada desde `message_id`
(validación: 99.997% en el día correcto sobre holdout del 20%, error máximo 1
día).

**Corpus etiquetado a mano:** 1000 pares, cuotas iguales por ticker, lotes
`l0`/`doble`/`test`/`train`, dos anotadores.


**Eventos (fijados antes de mirar resultados).** Los datos de texto terminan el
2022-12-31, así que los catalizadores de 2024–25 quedan fuera del estudio:

1. **Earnings 2019–2022** de los 6 tickers, vía `event_windows.py` (yfinance).
   Es el conjunto principal: fecha exacta, recurrente y comparable entre
   tickers.
2. **Regreso de Bob Iger a Disney** (2022-11-20), evento idiosincrático.
3. **Earnings de TGT de noviembre 2022** (reacción de −25%), ya incluido en (1)
   y además analizado como caso.

Quedan **excluidos** por estar fuera del rango de datos: apagón de CRWD
(2024-07), split de CMG (2024-06), proxy battle de Peltz (2023–24), entrada de
DDOG al S&P 500 (2025-07).

---

## 4. Análisis, especificados antes de correrlos

### 4.1 Retorno anormal acumulado (CAR)

- Modelo de mercado **CAPM con SPY** como proxy de mercado.
- Ventana de estimación: **[-250, -30]** días hábiles respecto al evento.
- Ventana de evento: **[-5, +5]** días hábiles. Se reportan además [-1, +1] y
  [0, +5].
- Precios: Alpaca, feed IEX. Eventos sin al menos 100 días de estimación
  disponibles se descartan y se reporta cuántos.

### 4.2 Sentimiento por ventana

- Score diario por (ticker, día) para cada uno de los tres enfoques, sobre los
  mensajes del pool NYU de ese ticker y día.
- `net_sentiment = prob_positive − prob_negative`.
- **Pre-evento:** media de [-5, -1]. **Post-evento:** media de [0, +5].
- Días sin mensajes: se excluyen del promedio, no se imputan como 0. Se reporta
  la cobertura (proporción de días con al menos un mensaje) por ticker.

### 4.3 Contrastes y criterios de éxito

Fijados **antes** de ver los resultados. Si no se cumplen, se reportan igual y
se busca interpretación; no se cambia el criterio.

| Hipótesis | Contraste | Criterio |
|---|---|---|
| H1 | Precisión direccional del signo del CAR [0,+5] a partir del sentimiento pre-evento | Se declara "no predice" si el IC del 95% incluye 50% |
| H2 | ρ de Spearman entre sentimiento post-evento y CAR [0,+5], contra ρ pre-evento vs CAR | H2 se sostiene si ρ_post > ρ_pre con p < 0.05 (test de diferencia por bootstrap, 10.000 remuestreos) |
| H3 | ρ_post por segmento de liquidez | Se reporta con IC por bootstrap; sin umbral, es descriptivo |
| H4 | \|ρ_post\| por enfoque | Orden esperado: probe > FinBERT > VADER. Se reporta la diferencia con IC por bootstrap |

**Hiperparámetros del linear probing, congelados acá** (los defaults del repo
dejan la cabeza en azar a este tamaño de corpus):

```
epochs = 200
lr_head = 1e-3
batch_size = 16
val_split = 0.15
max_length = 128
backbone = ProsusAI/finbert, congelado
seed = 42
```

### 4.4 Controles placebo (regla 2, obligatorios)

Todo resultado numérico de 4.3 se acompaña de:

1. **Permutación del sentimiento:** se barajan los scores entre eventos,
   manteniendo las fechas. 1.000 repeticiones.
2. **Shuffling de fechas de evento:** se reasignan fechas falsas dentro del
   mismo ticker y año. 1.000 repeticiones.

Un resultado solo se considera real si supera el percentil 95 de ambas
distribuciones placebo.

---

## 5. Lo que NO se va a hacer

- No se reintenta el enfoque predictivo diario/intraday de las fases 1–6.
- No se ajustan hiperparámetros mirando el test.
- No se agregan ni se quitan tickers.
- No se extiende el rango de eventos más allá de 2022-12-31 mientras la fuente
  de texto termine ahí.

---

## 6. Registro de cambios

| Fecha | Cambio | Motivo |
|---|---|---|
| 2026-09-13 | Versión inicial (borrador) | Pre-registro del estudio de eventos, posterior a la comparación exploratoria de enfoques (sección 1) |
