# Sanity check con target multi-día (1 / 3 / 5)

> Solo diagnóstico, sin adaptativo. `target[T]=1 ⇔ close[T+H]>close[T]`. Se pierden las últimas H filas por ticker (shift), de ahí que n baje un poco con H. Modelos con `scale_pos_weight` balanceado para que no colapsen.

- Features técnicas (11): ['close', 'volume', 'rsi', 'macd', 'macd_signal', 'macd_hist', 'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_pct']
- Features híbridas (26): técnicas + 15 de sentimiento

## Horizonte H = 1 día(s)

- Filas: **1863** | split: train≤2021-11-15, test≥2021-11-16

**1. Balance del target (% sube):**

| split | n | % sube | % baja |
|---|---|---|---|
| train (core) | 1190 | 52.4% | 47.6% |
| val | 300 | 57.3% | 42.7% |
| test | 373 | 44.0% | 56.0% |

**2. Modelo solo-técnico — AUC test: `0.5445`** (pred sube 53.1%)

**3. Modelo híbrido — AUC test: `0.5570`** (pred sube 59.8%)

**4. Top 5 Spearman feature[T] vs retorno[T+H]:**

| # | feature | ρ | ρ² | p | tipo |
|---|---|---|---|---|---|
| 1 | prob_negative_mean | +0.0554 | 0.3% | 0.017 | sentimiento |
| 2 | negative_ratio | +0.0554 | 0.3% | 0.017 | sentimiento |
| 3 | net_sentiment_mean | -0.0540 | 0.3% | 0.020 | sentimiento |
| 4 | bb_middle | -0.0486 | 0.2% | 0.036 | técnica |
| 5 | close | -0.0475 | 0.2% | 0.041 | técnica |

_Máx |ρ| = 0.0554 (ρ² ≈ 0.3%)._

## Horizonte H = 3 día(s)

- Filas: **1853** | split: train≤2021-11-11, test≥2021-11-12

**1. Balance del target (% sube):**

| split | n | % sube | % baja |
|---|---|---|---|
| train (core) | 1180 | 53.0% | 47.0% |
| val | 300 | 58.3% | 41.7% |
| test | 373 | 42.6% | 57.4% |

**2. Modelo solo-técnico — AUC test: `0.5003`** (pred sube 48.8%)

**3. Modelo híbrido — AUC test: `0.5550`** (pred sube 56.6%)

**4. Top 5 Spearman feature[T] vs retorno[T+H]:**

| # | feature | ρ | ρ² | p | tipo |
|---|---|---|---|---|---|
| 1 | bb_middle | -0.0823 | 0.7% | 0.000 | técnica |
| 2 | close | -0.0796 | 0.6% | 0.001 | técnica |
| 3 | bb_lower | -0.0789 | 0.6% | 0.001 | técnica |
| 4 | bb_upper | -0.0763 | 0.6% | 0.001 | técnica |
| 5 | net_sentiment_mean | -0.0665 | 0.4% | 0.004 | sentimiento |

_Máx |ρ| = 0.0823 (ρ² ≈ 0.7%)._

## Horizonte H = 5 día(s)

- Filas: **1843** | split: train≤2021-11-10, test≥2021-11-11

**1. Balance del target (% sube):**

| split | n | % sube | % baja |
|---|---|---|---|
| train (core) | 1180 | 53.6% | 46.4% |
| val | 295 | 61.0% | 39.0% |
| test | 368 | 40.8% | 59.2% |

**2. Modelo solo-técnico — AUC test: `0.5615`** (pred sube 51.1%)

**3. Modelo híbrido — AUC test: `0.5315`** (pred sube 48.6%)

**4. Top 5 Spearman feature[T] vs retorno[T+H]:**

| # | feature | ρ | ρ² | p | tipo |
|---|---|---|---|---|---|
| 1 | bb_middle | -0.1019 | 1.0% | 0.000 | técnica |
| 2 | close | -0.1007 | 1.0% | 0.000 | técnica |
| 3 | bb_lower | -0.0971 | 0.9% | 0.000 | técnica |
| 4 | bb_upper | -0.0962 | 0.9% | 0.000 | técnica |
| 5 | prob_negative_mean | +0.0723 | 0.5% | 0.002 | sentimiento |

_Máx |ρ| = 0.1019 (ρ² ≈ 1.0%)._

## Comparativa entre horizontes

| H | n | % sube test | AUC técnico | AUC híbrido | máx |ρ| | máx ρ² |
|---|---|---|---|---|---|---|
| 1 | 1863 | 44.0% | 0.5445 | 0.5570 | 0.0554 | 0.3% |
| 3 | 1853 | 42.6% | 0.5003 | 0.5550 | 0.0823 | 0.7% |
| 5 | 1843 | 40.8% | 0.5615 | 0.5315 | 0.1019 | 1.0% |

## Veredicto

Desglose de la correlación máxima por TIPO de feature en cada horizonte:

| H | máx |ρ| total | feature top | ¿nivel de precio? | máx |ρ| SENTIMIENTO |
|---|---|---|---|---|---|
| 1 | 0.0554 | prob_negative_mean | no | 0.0554 (ρ²≈0.3%) |
| 3 | 0.0823 | bb_middle | **SÍ** | 0.0665 (ρ²≈0.4%) |
| 5 | 0.1019 | bb_middle | **SÍ** | 0.0723 (ρ²≈0.5%) |

**Lectura crítica:**
- La correlación que **crece** con el horizonte (0.055 → 0.082 → 0.102) está dominada por features de **nivel de precio** (`close`, `bb_middle/upper/lower`, que son casi la misma variable), todas con ρ **negativo**. Eso **no es señal predictiva generalizable**: es el artefacto de que un precio absoluto alto en el boom de 2021 precede a caídas en el crash de 2022 — un **proxy de régimen no estacionario**. Un modelo que se apoya en el nivel de precio memoriza ese ciclo concreto y no transferirá a otro período.
- Las features de **sentimiento se mantienen planas** en todos los horizontes (máx |ρ| ≲ 0.07, ρ² ≲ 0.5%). Alargar el horizonte **no** despierta la señal de sentimiento.
- Los **AUC son inconsistentes** (H=3 técnico 0.50 vs híbrido 0.56; H=5 técnico 0.56 vs híbrido 0.53): si hubiera señal robusta, añadir sentimiento no debería empeorar el AUC. Esa volatilidad entre 0.50 y 0.56 es **ruido de un único test set**, no skill estable.

**Conclusión:** alargar el horizonte a 3/5 días **no destapa señal de sentimiento real**. La única correlación que sube es un **artefacto de nivel de precio / régimen** que no generaliza. Antes del adaptativo, las palancas correctas son: (1) **target de exceso sobre el mercado/sector** (resta el beta común y el efecto nivel-de-precio que ensucia todo), (2) **quitar features de nivel absoluto** (usar solo derivadas estacionarias: rsi, macd_hist, bb_pct, bb_width) para no memorizar el ciclo, y (3) **ampliar el universo de tickers**. El adaptativo no arregla ausencia de señal.
