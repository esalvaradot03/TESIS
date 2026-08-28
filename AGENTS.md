# Proyecto: Sistema de Trading Algorítmico con Análisis de Sentimiento

## Equipo
Tesis de grado de Ingeniería de Sistemas (Universidad de los Andes, Colombia)
de Esteban Alvarado y Camilo Linares. Dos asesores, uno por programa:
- **Sergio Cabrales** — asesor de Ingeniería Industrial.
- **Camilo Escobar** — asesor de Ingeniería de Sistemas (dueño del pendiente
  de dashboard interactivo, ver [Pivote actual](#pivote-actual-agosto-2026)).

## Descripción
Sistema de trading algorítmico que combina análisis de sentimiento de **StockTwits** (mensajes con cashtags y label nativo Bullish/Bearish) usando FinBERT con indicadores técnicos (RSI, MACD, Bollinger Bands) en un modelo XGBoost para generar señales de compra/venta. Opera en paper trading via Alpaca API.

## Universo de acciones
Todas las acciones del S&P 500 (~503 tickers). Se obtiene la lista desde Wikipedia y se cachea en `data/sp500_tickers.csv` (TTL 7 días). El sistema filtra dinámicamente: solo opera sobre acciones con mínimo 5 menciones en redes sociales en las últimas 24h (`MIN_MENTIONS_PER_DAY=5` en settings).

## Estado de implementación — COMPLETO Y VERIFICADO

Todos los módulos corren exitosamente de punta a punta (verificado con datos en
vivo de StockTwits y con datos sintéticos).

### Módulos implementados

| Módulo | Archivo | Estado |
|--------|---------|--------|
| Cargador S&P 500 | `src/universe/sp500_loader.py` | ✓ Corriendo |
| **Scraper StockTwits** | `src/sentiment/scraper_stocktwits.py` | ✓ Corriendo (fuente principal, endpoint público) |
| Scraper Reddit | `src/sentiment/scraper_reddit.py` | ⚠ Legacy (API negada en 2026) |
| Cargador histórico | `src/sentiment/historical_loader.py` | ✓ Corriendo (detecta StockTwits y Reddit) |
| Detector de tickers | `src/sentiment/ticker_detector.py` | ✓ Solo path histórico WSB (StockTwits no lo usa) |
| Preprocesador | `src/sentiment/preprocessor.py` | ✓ Corriendo |
| Scorer FinBERT | `src/sentiment/finbert_scorer.py` | ✓ Corriendo |
| Indicadores técnicos | `src/trading/indicators.py` | ✓ Corriendo (498 tickers, 61563 barras) |
| Feature engine | `src/trading/feature_engine.py` | ✓ Corriendo |
| Modelo XGBoost | `src/trading/xgboost_model.py` | ✓ Corriendo |
| Métricas | `src/evaluation/metrics.py` | ✓ Implementado |
| Backtester | `src/evaluation/backtester.py` | ✓ Implementado |
| Executor Alpaca | `src/trading/alpaca_executor.py` | ✓ Implementado |

## Fuente de datos de sentimiento
**StockTwits** — fuente principal. Cambio respecto al diseño original (Reddit):
la solicitud de API de Reddit fue **negada en 2026**, por lo que se migró a
StockTwits, que expone un **endpoint público read-only sin API key**:

  - Streams por símbolo: `https://api.stocktwits.com/api/2/streams/symbol/{SYMBOL}.json`
  - Trending symbols:    `https://api.stocktwits.com/api/2/trending/symbols.json`

Ventajas frente a Reddit/WSB para esta tesis:
  - El ticker viene **explícito** en el campo `symbols` de cada mensaje (cashtags),
    así que NO hace falta inferirlo por regex → `ticker_detector` queda solo para
    el path histórico de WSB.
  - Cada mensaje trae un **label nativo de sentimiento** (Bullish/Bearish/None) que
    se usa como feature adicional (`stocktwits_native_sentiment`) junto a FinBERT.
  - Sin costo ni aprobación: el endpoint público es de solo lectura.

Twitter/X sigue descartado por costo prohibitivo de la API ($200/mes mínimo).
La literatura de NLP financiero (Bollen et al. 2011; Oliveira et al. 2017) respalda
los foros de inversión retail como predictores directos para acciones individuales;
StockTwits es además específico de mercados y mejor estructurado que Reddit.

### Rate limiting StockTwits
El endpoint público tolera ~200 req/hora sin autenticar. El scraper se limita a
`STOCKTWITS_MAX_REQUESTS_PER_HOUR=150` (sleep mínimo entre llamadas) y respeta el
header `Retry-After` ante respuestas 429. Requiere un User-Agent de navegador
(de lo contrario Cloudflare responde 403). Paginación por el cursor `max`.

## Stack técnico
- Python 3.11 (NO 3.12+, NO 3.14 — incompatible con numpy/torch de estas versiones)
- FinBERT: `ProsusAI/finbert` (modelo público, ~438MB, se descarga automáticamente)
- XGBoost 2.0.3
- requests 2.32.3 (StockTwits, endpoint público read-only — sin API key)
- praw 7.7.1 (Reddit API — legacy, solo para datasets históricos de WSB)
- **alpaca-py 0.43.4** (NO alpaca-trade-api — deprecado)
- pandas 2.2.2, numpy 1.26.4, ta 0.11.0, scikit-learn 1.4.2
- pyarrow 16.1.0 (necesario para `.to_parquet()`)
- scipy 1.13.1 (pruebas de hipótesis)

## Arquitectura
Dos agentes modulares:
1. **SENTIMENT_QUANT_AGENT** — Recolecta mensajes de StockTwits (endpoint público), toma el ticker directo del campo `symbols`, preprocesa texto, aplica FinBERT y captura el label nativo Bullish/Bearish; genera score de sentimiento por ticker/día.
2. **TRADING_ENGINE_AGENT** — Combina scores de sentimiento + indicadores técnicos en XGBoost, emite señales, envía órdenes a Alpaca paper trading.

## Estructura del proyecto
```
trading-sentiment/
├── CLAUDE.md
├── requirements.txt
├── .env.example
├── .env                          # NO commitear — tiene keys reales
├── config/
│   └── settings.py               # API keys, parámetros globales
├── data/
│   ├── raw/
│   │   ├── prices_raw.parquet        # Precios cacheados de Alpaca (498 tickers)
│   │   ├── stocktwits_synthetic.csv  # Mensajes sintéticos StockTwits para testing
│   │   ├── stocktwits_<ts>.csv       # Salida del scraper en vivo
│   │   └── wsb_synthetic.csv         # Posts sintéticos Reddit (legacy)
│   ├── processed/
│   │   ├── indicators.parquet    # RSI, MACD, BB para 498 tickers
│   │   ├── sentiment_scores.csv  # Scores FinBERT por (post, ticker)
│   │   ├── features.parquet      # Features combinadas listas para XGBoost
│   │   └── strategy_returns.parquet  # Retornos del backtest
│   └── sp500_tickers.csv         # Lista cacheada de tickers
├── models/
│   ├── xgboost_model.json        # Modelo entrenado
│   ├── feature_names.json        # Lista de features en orden de entrenamiento
│   ├── split_info.json           # Fechas del split temporal
│   └── feature_importance.csv    # Importancia de features (gain, weight, cover)
├── scripts/
│   ├── generate_synthetic_stocktwits.py  # Genera mensajes StockTwits de test
│   └── generate_synthetic_posts.py       # Genera posts Reddit de test (legacy)
├── src/
│   ├── universe/
│   │   └── sp500_loader.py
│   ├── data/
│   │   └── download_stocktwits_historical.py  # Descarga dataset Kaggle (kagglehub)
│   ├── sentiment/
│   │   ├── scraper_stocktwits.py   # Fuente principal (en vivo)
│   │   ├── scraper_reddit.py       # Legacy
│   │   ├── historical_loader.py    # StockTwits/Reddit + dataset Kaggle (directorio)
│   │   ├── ticker_detector.py      # Solo path histórico WSB
│   │   ├── preprocessor.py
│   │   └── finbert_scorer.py
│   ├── trading/
│   │   ├── indicators.py
│   │   ├── feature_engine.py
│   │   ├── xgboost_model.py
│   │   └── alpaca_executor.py
│   └── evaluation/
│       ├── metrics.py
│       └── backtester.py
└── tests/
```

## Runbook completo (desde cero, con datos reales — StockTwits)

Nota: el path StockTwits **NO usa ticker_detector** (el ticker viene en el mensaje);
se va directo del scraper al preprocessor.

```powershell
# Activar entorno (siempre primero)
.\.venv\Scripts\Activate.ps1

# 1. Universo S&P 500
python -m src.universe.sp500_loader

# 2a. Scraping StockTwits en vivo (endpoint público, sin API key)
python -m src.sentiment.scraper_stocktwits --trending 20      # trending → mensajes
#   o por símbolos fijos:  python -m src.sentiment.scraper_stocktwits AAPL NVDA TSLA
# Output: data/raw/stocktwits_YYYYMMDD_HHMMSS.csv

# 2b. Alternativamente: cargar dataset histórico (StockTwits o Kaggle/Pushshift)
python -m src.sentiment.historical_loader ruta/al/dataset.csv --start 2024-01-01 --end 2024-06-30
# Detecta el formato por columnas. Output: data/raw/stocktwits_historical.csv (o wsb_historical.csv)

# 3. Preprocesamiento para FinBERT (StockTwits ya trae detected_tickers → sin ticker_detector)
python -m src.sentiment.preprocessor data/raw/stocktwits_YYYYMMDD_HHMMSS.csv
# Output: data/processed/stocktwits_..._clean.csv

# 4. Scoring con FinBERT (acumula en sentiment_scores.csv con caché por post_id;
#    conserva la columna nativa stocktwits_sentiment)
python -m src.sentiment.finbert_scorer data/processed/stocktwits_..._clean.csv

# 5. Precios e indicadores técnicos (fechas configurables)
python -m src.trading.indicators 2024-01-01 2024-06-30
# Output: data/raw/prices_raw.parquet + data/processed/indicators.parquet

# 6. Combinar features (añade stocktwits_native_sentiment cuando la fuente es StockTwits)
python -m src.trading.feature_engine
# Output: data/processed/features.parquet

# 7. Entrenar modelo
python -m src.trading.xgboost_model
# Output: models/xgboost_model.json + feature_importance.csv

# 8. Backtest comparativo (3 estrategias + p-value)
python -m src.evaluation.backtester
```

### Modelo base real `model_v0` (dataset histórico Kaggle StockTwits)

Dataset: `frankcaoyun/stocktwits-2020-2022-raw` (362 CSVs, ~6 GB, 5 carpetas por
ticker: AAPL, AMZN, FB, NVDA, TSLA). FB es correcto para la época (pre-META).

```powershell
# 1. Descargar el dataset (kagglehub; token en ~/.kaggle/access_token)
python -m src.data.download_stocktwits_historical
#    Devuelve la ruta de caché ~/.cache/kagglehub/.../versions/1

# 2. Normalizar el directorio completo → esquema interno (parseo ast.literal_eval,
#    cap por (ticker,día) para acotar FinBERT). Pasar la carpeta, no un CSV:
python -m src.sentiment.historical_loader "<ruta_kaggle>/StockTwits_2020_2022_Raw" `
    --start 2020-01-01 --end 2022-03-31 --max-per-ticker-day 40
#    Output: data/raw/stocktwits_historical.csv + stats (incl. % label nativo no-nulo)

# 3. Indicadores SOLO para los 5 tickers en el rango del dataset
python -m src.trading.indicators 2020-01-01 2022-03-31    # tras fijar tickers en el código/llamada
#    OJO: el feed IEX gratuito de Alpaca solo da historia desde ~2020-07-27,
#    así que el join efectivo arranca ~2020-09 (warm-up de indicadores incluido).

# 4. preprocessor → finbert_scorer → feature_engine
python -m src.sentiment.preprocessor data/raw/stocktwits_historical.csv
python -m src.sentiment.finbert_scorer data/processed/stocktwits_historical_clean.csv
python -m src.trading.feature_engine

# 5. Entrenar y guardar el modelo base estático en models/model_v0
python -c "from pathlib import Path; from src.trading.xgboost_model import train_and_evaluate; train_and_evaluate(model_dir=Path('models/model_v0'))"

# 6. Backtest comparativo contra model_v0
python -m src.evaluation.backtester data/processed/features.parquet models/model_v0
```

Notas del dataset Kaggle:
- Columnas `symbols` y `entities` son **literales de Python** (comillas simples), se
  parsean con `ast.literal_eval` (nunca `json.loads`/`eval`). El ticker sale del
  campo `symbol` de cada elemento de `symbols`, NO del nombre de carpeta (que es
  inconsistente, ej. `AMZN2019-2022`).
- Label nativo en `entities['sentiment']['basic']`; **~52% de los 4.17M mensajes en
  rango traen label no-nulo** (no "casi siempre None").
- El cap por (ticker, día) acota el costo de FinBERT manteniendo >> 5 menciones/día.

### Path histórico de Reddit/WSB (legacy)
Solo para datasets históricos de Kaggle/Pushshift. Este path SÍ requiere
`ticker_detector` (el ticker no viene marcado):

```powershell
python -m src.sentiment.historical_loader dataset_wsb.csv --start 2024-01-01 --end 2024-06-30
python -m src.sentiment.ticker_detector data/raw/wsb_historical.csv
python -m src.sentiment.preprocessor data/processed/wsb_historical_tickers.csv
python -m src.sentiment.finbert_scorer data/processed/wsb_historical_tickers_clean.csv
# ... resto igual (indicators → feature_engine → xgboost_model → backtester)
```

## Runbook con datos sintéticos (sin acceso a la API)

```powershell
python scripts/generate_synthetic_stocktwits.py     # 2000 mensajes StockTwits en 2024
python -m src.sentiment.preprocessor data/raw/stocktwits_synthetic.csv
python -m src.sentiment.finbert_scorer data/processed/stocktwits_synthetic_clean.csv
python -m src.trading.indicators 2024-01-01 2024-06-30
python -m src.trading.feature_engine
python -m src.trading.xgboost_model
python -m src.evaluation.backtester
```

> Si la consola de Windows lanza `UnicodeEncodeError` al imprimir (cp1252),
> exportar `PYTHONIOENCODING=utf-8` antes de correr los módulos.

## Convenciones de código
- Docstrings en español
- Type hints en todas las funciones
- Logging con módulo `logging` (no print)
- Variables de entorno para API keys (.env, nunca hardcoded)
- Semillas aleatorias fijas (SEED=42)
- Procesamiento en batches para volumen de 500+ tickers
- Todos los comandos se corren desde `c:\dev\Tesis` con el venv activo

## Consideraciones de escala (S&P 500 completo)
- **Scraping:** Combinar trending symbols + streams por símbolo de StockTwits. El ticker viene en el mensaje (campo `symbols`), no se infiere por regex. Rate limit ~150 req/h con paginación por cursor `max`
- **FinBERT:** Batch inference de 32 textos. Cachea por `post_id` en `sentiment_scores.csv`
- **Indicadores técnicos:** Batch vectorizado con pandas/ta. Descarga en chunks de 100 tickers
- **Precios:** Alpaca bulk requests. Feed IEX (gratuito). BRK/B y BF/B **no soportados en IEX** — se omiten automáticamente
- **Filtro de actividad:** Solo generar señales para tickers con >= 5 menciones/día

## Decisiones de diseño importantes

### Sentimiento
- `net_sentiment = prob_positive - prob_negative` → señal continua [-1, +1] (no usar `sentiment_score` que es siempre positivo)
- Mensajes de fines de semana se asignan al siguiente día de trading
- Ponderación por engagement (likes de StockTwits / upvotes de Reddit): `weighted_sentiment`
- **Label nativo de StockTwits**: `stocktwits_native_sentiment` = promedio diario por ticker de Bullish=+1 / Bearish=−1 / None=0. Es una feature **opcional**: solo existe cuando la fuente es StockTwits; con datos de Reddit la columna no aparece y se omite en todo el pipeline (la selección de features filtra por columnas presentes)

### Esquema interno y flujo de columnas
- El scraper de StockTwits normaliza al MISMO esquema que producía Reddit, más dos columnas: `detected_tickers` (JSON array tomado del campo `symbols`) y `stocktwits_sentiment` (Bullish/Bearish/None). Mapeos: `title` ← cuerpo del mensaje, `body` ← "", `score` ← likes, `num_comments` ← 0
- Al traer ya `detected_tickers`, el path StockTwits salta `ticker_detector` y va directo al preprocessor
- `finbert_scorer` arrastra `stocktwits_sentiment` hasta `sentiment_scores.csv` (columna opcional en `_OUTPUT_COLUMNS`), y `feature_engine` la convierte en `stocktwits_native_sentiment`

### Modelo
- Split temporal estricto: 80% train / 20% test, sin shuffle, por fechas únicas
- Validación interna (20% del train) solo para early stopping — el test nunca toca el modelo
- Target: `1` si `close[T+1] > close[T]`, calculado por ticker independientemente

### Alpaca API
- Usar `alpaca-py` (no `alpaca-trade-api` que está deprecado)
- `TradingClient(paper=True)` para órdenes
- `StockHistoricalDataClient` para datos históricos
- Símbolos con "/" (BRK/B) no soportados en feed IEX → filtrar antes del request
- `ALPACA_BASE_URL` debe ser `https://paper-api.alpaca.markets` (sin `/v2` al final)

### Parquet
- Requiere `pyarrow` instalado (`pip install pyarrow`)
- Todos los DataFrames intermedios grandes se guardan en Parquet (no CSV)

## Datos
- Período histórico: enero–junio 2024 (precios reales de Alpaca, 498 tickers, 61563 barras)
- Evaluación paper trading: agosto–noviembre 2026

## Métricas objetivo
- Precisión direccional > 55%
- Sharpe Ratio > 1.0
- Max Drawdown < 15%
- p < 0.05 vs modelo solo con indicadores técnicos (t-test pareado + Wilcoxon)

## Variables de entorno (.env)
```
# StockTwits — fuente principal. Endpoint público read-only: NO requiere API key.
# Solo parámetros opcionales de rate limiting:
STOCKTWITS_MAX_REQUESTS_PER_HOUR=150
STOCKTWITS_MAX_MESSAGES_PER_SYMBOL=60

# Reddit — legacy (API negada en 2026); solo para datasets históricos de WSB:
REDDIT_CLIENT_ID=...
REDDIT_CLIENT_SECRET=...
REDDIT_USER_AGENT=trading_sentiment_bot/1.0
ALPACA_API_KEY=...
ALPACA_API_SECRET=...
ALPACA_BASE_URL=https://paper-api.alpaca.markets    # sin /v2
ALPACA_DATA_URL=https://data.alpaca.markets
ALPACA_FEED=iex
PRICE_CHUNK_SIZE=100
MAX_OPEN_POSITIONS=20
MIN_ORDER_NOTIONAL=1.0
HF_TOKEN=...    # para HuggingFace (opcional, FinBERT es público)
SEED=42
```
## Pivote actual (agosto 2026)

### Resultado principal ya establecido (no re-litigar)
Fases 1–6 del proyecto (E1–E8/F1–F4 sobre dataset_v1 [475 tickers S&P 500,
231.305 filas, 2010–2022] y dataset_v2 [survivors-only small/mid-cap]; Fase 3
con hipótesis contrarian Baker-Wurgler + Fama-French 3 factores sobre panel
mensual; Fases 4–6 con 5 tickers fijos TSLA/AMD/DIS/BA/GILD a frecuencia
diaria e intraday 30/60min) concluyeron consistentemente que **el sentimiento
retail de StockTwits no predice retornos** — AUC ~0.50 en todas las
configuraciones (el placebo con sentimiento permutado tuvo el AUC más alto
en E8), y correlación anticipatoria ~0.02 (p>0.15) mientras que la
correlación contemporánea/reactiva sí es fuerte (ρ hasta 0.51 para TSLA).
Interpretación: el sentimiento es un indicador **coincidente y reactivo**,
no anticipatorio — consistente con eficiencia de mercado a frecuencias
diarias e intraday.

**Este es el hallazgo principal de la tesis, no un fracaso a corregir.**
Trabajo nuevo no debe intentar "mejorar el AUC" ni resucitar el enfoque
predictivo diario/intraday.

### La nueva pregunta (medición, no predicción)
¿Cómo se comparan tres enfoques de análisis de sentimiento en su capacidad
de capturar señal en ventanas alrededor de eventos catalizadores, para 6
tickers del S&P 500 segmentados por liquidez?
1. FinBERT off-the-shelf (baseline existente, `src/sentiment/finbert_scorer.py`)
2. FinBERT con linear probing: backbone congelado + cabeza clasificadora
   entrenada sobre ~2.500 pares (post, target_ticker) etiquetados a mano
3. Baseline neutro con lexicón (VADER)

Entregables: (1) el corpus etiquetado como recurso reusable, (2) la
comparación metodológica de los 3 enfoques, (3) un estudio de eventos con
retorno anormal acumulado (CAR, CAPM/SPY) en ventanas [-5,+5] días, (4) un
dashboard interactivo (pendiente de Sistemas, ver abajo).

### Los 6 tickers (decididos, no cambiar)
- **Alta liquidez** (ADV >15M acciones/día): **NCLH** (Norwegian Cruise
  Line — ADV ~21M, beta 2.1, earnings + anuncios de flota + movimientos de
  deuda), **DIS** (Walt Disney — ADV ~15M, cobertura NYU 2012–2022, saga
  Iger salida/regreso 2022, proxy battle Peltz 2024, earnings con
  reacciones de ±12%)
- **Liquidez media** (ADV 3–10M/día): **CRWD** (CrowdStrike — ADV ~4.5M,
  el apagón global del 19-jul-2024 es el catalizador más limpio del
  corpus: fecha exacta, –38% peak-to-trough, recuperación total),
  **TGT** (Target — ADV ~5-8M, el ticker más frecuente en los run logs del
  live system, reacciones de –25% en earnings nov-2022)
- **Baja liquidez** (ADV <3M/día, relativo al S&P 500): **CMG** (Chipotle
  — ADV ~1.5M, cobertura NYU, split 50:1 jun-2024 como catalizador de
  sentimiento puro), **DDOG** (Datadog — ADV ~6M pero seguimiento
  institucional bajo, sentimiento FinBERT más alto de todo el corpus vivo
  +0.304, entrada al S&P 500 jul-2025 +13.6% en un día)

Solo **DIS y CMG** tienen cobertura histórica de texto confirmada (bucket
`stocktwits_nyu/messages/`). Para NCLH/CRWD/TGT/DDOG la única fuente en vivo
es el stream reciente de StockTwits (no hay búsqueda histórica) — gap
metodológico real para el muestreo LR estratificado por año en esos 4
tickers, a resolver con Sergio.

### Protocolo de pre-registro (no negociable)
1. Toda hipótesis nueva se compromete por escrito en
   `docs/pre_registro_event_study.md` **antes** de correr el análisis que la prueba.
2. Todo análisis con resultados numéricos lleva un control placebo idéntico
   (permutación del sentimiento, o shuffling de fechas de evento).
3. Criterios de éxito fijados antes de ver resultados — si no se cumplen,
   se reportan igual y se busca interpretación, no se cambia el criterio.
4. Semilla aleatoria fija SEED=42 en todo experimento.

### Estado de la implementación del pivote
```
research/event_study/
├── build_corpus.py           ✓ pool NYU + StockTwits en vivo, sample_random(),
│                                escribe lotes de etiquetado (l0, lr, lu1-4)
├── kappa_calculator.py       ✓ Cohen's kappa entre dos CSVs de etiquetado
├── event_windows.py          ✓ earnings + upgrades/downgrades vía yfinance
├── linear_probe.py           ⏳ pendiente — OJO: build_corpus.py etiqueta por
│                                (post_id, target_ticker); es un schema distinto
│                                al de src/labeling/labeled_store.py y
│                                src/sentiment/finbert_finetune.py (por post_id
│                                solo) — hay que decidir si se reconcilian o se
│                                mantienen como dos corpus separados
├── evaluate_approaches.py    ⏳ pendiente
├── car_analysis.py           ⏳ pendiente (bloqueado por el pre-registro, regla 1)
└── results_event_study.md    ⏳ pendiente

data/manual_labels/           ⏳ vacío — falta correr build_corpus.py con acceso
                                  real a D:\trading-data\ y/o StockTwits en vivo
docs/codebook_etiquetado.md   ⏳ referenciado en conversaciones previas pero no
                                  existe en el repo — falta escribirlo/commitearlo
docs/pre_registro_event_study.md  ⏳ pendiente

# Ya construido (subsistema anterior de comparación de sentimiento, por post_id):
src/sentiment/finbert_finetune.py           ✓ linear probing sobre embeddings [CLS]
src/sentiment/finbert_finetuned_scorer.py   ✓ scoring con la cabeza entrenada
src/sentiment/lexicon_scorer.py             ✓ baseline VADER
src/labeling/ (labeled_store, uncertainty_sampling, inter_rater)  ✓
src/evaluation/sentiment_comparison.py      ✓ accuracy/macro-F1/matrices de confusión
```

### Pendiente — Ingeniería de Sistemas: dashboard interactivo
Responsabilidad de Camilo Escobar. Extiende `dashboard/app.py` /
`dashboard/data_loader.py` (Streamlit) ya existentes. Alcance a diseñar:
- Comparación visual del portafolio de 6 tickers vs. el S&P 500 y ETFs de
  otras regiones (a definir el set exacto — e.g. EFA/EEM/VGK).
- Posts y noticias relevantes por ticker (o ticker relacionado), navegables
  en el tiempo — no solo el score agregado.
- Calendario de eventos con los aciertos/desaciertos del modelo: si el
  sentimiento pre-evento (de cada uno de los 3 enfoques) predijo la
  dirección del CAR post-evento observado.
- Word cloud de las palabras más usadas en los posts, por ticker y/o
  ventana de evento.
- (lista abierta — completar con Camilo Escobar)