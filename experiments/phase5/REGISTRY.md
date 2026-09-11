# Fase 5 — PRE-REGISTRO (escrito ANTES de correr cada experimento)

> Documento vinculante. Mismo protocolo que Fases 3-4: hipótesis y criterio fijados
> antes de ver resultados, control placebo obligatorio, una sola corrida, sin ajuste
> post-hoc. Aislado en `src/phase5/` y `experiments/phase5/`; no toca nada existente.
> Reutiliza el dataset de Fase 4 (5 activos: TSLA, AMD, DIS, BA, GILD; mismas features
> de sentimiento `st_*` / `nw_*`; mismos cortes train 2015-2020 / test 2021-2022).

---

## EXPERIMENTO 5.1 — Diagnóstico de correlación contemporánea (mismo día)

**Pregunta:** ¿el sentimiento agregado diario se mueve *junto* con el retorno del
MISMO día? Es diagnóstico de contenido informativo, **no** predicción.

**H5.1:** el sentimiento diario (StockTwits y noticias, por separado) correlaciona
**positivamente** con el retorno del mismo día del activo.

### Variables
- Sentimiento diario = feature `net` de cada fuente (`st_net`, `nw_net`), tal como se
  construyó en Fase 4 (StockTwits: (bull−bear)/(bull+bear); Noticias: media diaria de
  `prob_pos − prob_neg` de FinBERT sobre titulares).
- Retorno mismo día: `ret_T = close_T / close_{T-1} − 1` (cierre ajustado yfinance).
- Retorno de ayer: `ret_{T-1}` (para el chequeo de reactividad, ver abajo).

### Submuestra (fijada a priori)
Las correlaciones se computan **solo sobre días con actividad real de esa fuente**
(`*_has == 1`), para que el valor de sentimiento sea una medición y no la imputación
neutra de días sin actividad (ver política de Fase 4). Se reporta N por celda.
Para StockTwits esto es ~100% de los días; para noticias filtra los días sin cobertura.

### Métricas (por activo × fuente)
- **Spearman** ρ y p, y **Pearson** r y p, en: período completo, train, y test por separado.
- **Reactividad al precio:** Spearman de `net_T` contra `ret_{T-1}` (período completo).
  Si es alta y positiva, el sentimiento es *reactivo* al movimiento previo del precio.
- **Placebo:** test de permutación — se baraja `net` contra `ret` (200 permutaciones,
  semilla 42) y se computa la fracción con |ρ| ≥ |ρ real| → p empírico. Placebo ≈ 0.

### Criterio de éxito (pre-registrado)
Un par (activo, fuente) **CONFIRMA H5.1** si y solo si:
1. Spearman **p < 0.01** en el período completo, y
2. **ρ > 0** (signo predicho), y
3. **signo estable**: ρ_train y ρ_test tienen el mismo signo (positivo) y ambos con p<0.01, y
4. **supera al placebo**: p empírico de permutación < 0.01.

Si falla cualquiera → no confirma. Nota: una correlación contemporánea positiva NO
implica poder predictivo (Fase 4 ya fue negativa en predicción); 5.1 sólo mide si el
sentimiento *contiene/refleja* el movimiento del día. Una correlación fuerte con
`ret_{T-1}` (reactividad) matiza aún más la interpretación.

---

## EXPERIMENTO 5.2 — ¿Sentimiento → actividad (volumen/volatilidad) en vez de dirección?

**Pregunta:** ¿el sentimiento predice ACTIVIDAD de mañana (volumen anormal, rango
anormal) por encima de un baseline autorregresivo? Volumen y volatilidad están
altamente autocorrelacionados, así que el sentimiento debe agregar valor SOBRE su
propio pasado, no sobre azar.

**H5.2a:** el volumen de mensajes y la intensidad del sentimiento de hoy predicen
**volumen anormal alto** de mañana.
**H5.2b:** ídem para el **rango anormal alto** de mañana.

### Datos
Mismos 5 activos (TSLA, AMD, DIS, BA, GILD), mismos cortes de Fase 4 (train
2015-2020 / test 2021-2022, frontera 2020-12-31 excluida). OHLCV completo de
**yfinance** (`auto_adjust=True`; el dataset de Fase 4 solo tenía `close`, así que se
re-descarga volumen/high/low). Features de sentimiento tomadas del dataset de Fase 4
(las 10: `st_*` + `nw_*`), usadas **juntas** (5.1 ya estableció la jerarquía de fuentes).

### Definiciones (todo point-in-time, sin leakage)
- Volumen relativo: `relvol_t = volume_t / SMA20(volume)_t` (media móvil trailing 20d).
- Rango: `range_t = (high_t − low_t) / close_t`.
- **Target (a) volumen anormal:** `y_t = 1` si `relvol_{t+1} > umbral_vol`, si no 0.
- **Target (b) rango anormal:** `y_t = 1` si `range_{t+1} > umbral_range`, si no 0.
- Umbrales = **mediana calculada solo en train** (del valor continuo `relvol_{t+1}` /
  `range_{t+1}` sobre filas de train), aplicada a train y test (sin leakage del test).

### Features
- **Autorregresivas (baseline), 10:** `relvol` y `range` rezagados 1-5 días
  (`relvol_l1..l5`, `range_l1..l5`; `l1` = valor de hoy `t`, `l5` = `t−4`, todos
  conocidos al cierre de `t`).
- **Sentimiento, 10:** las features `st_*` y `nw_*` del día `t` (Fase 4).

### Dos modelos por (activo × target) + placebo
- **(i) BASELINE:** solo las 10 autorregresivas.
- **(ii) BASELINE+SENT:** las 10 autorregresivas + las 10 de sentimiento.
- **Placebo:** BASELINE+SENT con las **features de sentimiento permutadas
  temporalmente** (barajadas por filas dentro de train y dentro de test, semilla 42),
  manteniendo **intactas** las autorregresivas. Aísla el aporte real del sentimiento.

XGBoost con los mismos hiperparámetros y semilla que Fase 4 (`max_depth=3`,
`lr=0.05`, `n_estimators=400`/es=40, `subsample=colsample=0.8`, `min_child_weight=5`,
`seed=42`), validación interna temporal (último 15% del train) para early stopping.

### Criterio de éxito (pre-registrado)
Un (activo, target) es **ÉXITO** si y solo si:
1. **AUC(baseline+sent) − AUC(baseline) > 0.02** en test, **y**
2. **AUC(baseline+sent) > AUC(placebo)** (mismo test, sentimiento permutado).

El aporte se mide como mejora incremental SOBRE el baseline autorregresivo, no sobre
azar. Sin ajuste post-hoc, una sola corrida.

### Entregables
Por (activo × target): AUC baseline, AUC baseline+sent, ΔAUC, AUC placebo, veredicto,
y top features (gain) del modelo completo. Tabla de los 10 settings (5 activos × 2
targets) con sus 3 modelos.
