# Datos externos de la tesis

Los datasets externos pesan decenas/cientos de GB y **viven fuera del repo**, en
un disco aparte: `D:\trading-data\`. El repo solo contiene el código que los
descarga (`src/data/setup_external_data.py`) y las rutas (`config/settings.py`).

## Estructura de `D:\trading-data\`

```
D:\trading-data\
├── README.md                                  # inventario autogenerado
├── stocktwits_nyu\                            # bucket público s3://stocktwits-nyu
│   ├── symbol_sentiments\                     #   sentimiento agregado por símbolo
│   ├── feature_wo_messages\                   #   features sin el texto
│   └── messages\                              #   mensajes crudos
├── wsb_kaggle\
│   ├── kevinwang313_wallstreetbets\           # kaggle: kevinwang313/wallstreetbets-dataset
│   └── unanimad_reddit_rwallstreetbets\       # kaggle: unanimad/reddit-rwallstreetbets
└── fnspid\                                     # FNSPID — se mueve A MANO (ver abajo)
```

La raíz es configurable con la variable de entorno `TESIS_DATA_ROOT` (por si el
disco cambia de letra). Por defecto `D:/trading-data`.

## Cómo correr el script de setup

Desde `c:\dev\Tesis` con el venv activo:

```powershell
# Plan: crea estructura, instala AWS CLI si falta, mide tamaños del bucket S3.
python -m src.data.setup_external_data --dry-run

# Descarga real (pregunta y/n con los tamaños a la vista):
python -m src.data.setup_external_data

# Sin preguntar (para desatendido):
python -m src.data.setup_external_data --yes

# Solo una etapa:
python -m src.data.setup_external_data --only s3
python -m src.data.setup_external_data --only kaggle
python -m src.data.setup_external_data --only readme
```

Características:
- **Idempotente / reanudable:** `aws s3 sync` solo baja lo que falta; si se corta,
  se re-corre y continúa. Kaggle usa el caché de kagglehub y solo copia si el
  destino no es idéntico.
- **AWS CLI:** se instala solo (MSI silencioso) si no está. En Windows el PATH se
  refresca únicamente en terminales nuevas; el script busca `aws.exe` también en
  `C:\Program Files\Amazon\AWSCLIV2\` para evitar tener que reiniciar la terminal.
- **Heartbeat:** durante descargas largas imprime un latido cada N segundos.
- **Confirmación:** muestra espacio libre y total a descargar, y pide `y/n` antes
  de bajar nada (salvo `--yes`).

## Cómo lee el resto del pipeline estas rutas

`config/settings.py` expone las rutas (no las crea):

```python
from config.settings import (
    EXTERNAL_DATA_ROOT,
    STOCKTWITS_NYU_SYMBOL_SENTIMENTS,
    STOCKTWITS_NYU_FEATURES,
    STOCKTWITS_NYU_MESSAGES,
    WSB_KEVIN,
    WSB_UNANIMAD,
    FNSPID,
)
```

Al importar `settings`, si alguna ruta externa no existe se emite un **warning**
(no un error) recordando correr el script de setup. La integración de estos
datasets en el pipeline (`historical_loader`, etc.) se hará en un paso aparte;
por ahora solo quedan las rutas listas.

## FNSPID se mueve a mano

FNSPID (Financial News and Stock Price Integration Dataset) ya está descargado
manualmente. El script **solo crea la carpeta vacía** `D:\trading-data\fnspid\`;
el usuario mueve ahí sus archivos. El script nunca toca su contenido.

## Datasets descartados (y por qué)

- **Twitter/X API:** costo prohibitivo (≥ $200/mes el plan básico). Descartado
  desde el diseño original.
- **Reddit API en vivo (PRAW):** la solicitud de API fue negada en 2026. Se
  conserva solo el path histórico (datasets de WSB en Kaggle), no acceso en vivo.
- Por eso la fuente principal en vivo es **StockTwits** (endpoint público), y los
  datasets externos de WSB/StockTwits-NYU/FNSPID se usan solo para análisis e
  investigación histórica, no para trading en tiempo real.

## Garantía de que nada entra al repo

`.gitignore` tiene reglas **defensivas**: ignora `trading-data/`, `**/trading-data/`,
`data/raw/external/`, `data/external/`, `**/external_data/`, y CSV/TSV pesados,
por si alguien crea un junction (`mklink`) o copia datos dentro del árbol del repo.
