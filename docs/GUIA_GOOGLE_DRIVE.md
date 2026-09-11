# Guía: Subir datos a Google Drive y conectarlos al proyecto

Esta guía explica cómo mover los datasets locales de tu PC a **Google Drive** y
luego descargarlos en un **servidor en la nube** para correr el pipeline de ML.

> **Idea general:** tu PC sube los datos a Drive una vez → el servidor los baja
> con un solo comando cada vez que lo necesites. Drive actúa de puente y de backup.

---

## 📋 Índice

1. [Arquitectura de la solución](#1-arquitectura)
2. [Requisitos previos](#2-requisitos-previos)
3. [Crear credenciales de Google Drive](#3-crear-credenciales)
4. [Instalar dependencias](#4-instalar-dependencias)
5. [Subir datos desde tu PC](#5-subir-datos-desde-tu-pc)
6. [Descargar datos en el servidor](#6-descargar-datos-en-el-servidor)
7. [Conectar los datos al pipeline](#7-conectar-los-datos)
8. [Preguntas frecuentes](#8-faq)

---

## 1. Arquitectura

```
┌──────────────┐      drive_upload.py      ┌─────────────────┐
│  Tu PC local │ ────────────────────────► │  Google Drive   │
│  (Windows)   │                           │  tesis-trading/ │
└──────────────┘                           └────────┬────────┘
                                                     │
                                        drive_sync.py│
                                                     ▼
                                          ┌─────────────────┐
                                          │  Servidor nube  │
                                          │  (Railway/      │
                                          │   Oracle/etc.)  │
                                          └─────────────────┘
```

**Estructura que se crea en Drive:**

```
tesis-trading/
├── data/
│   ├── raw/              # prices_raw.parquet, CSVs del scraper, sintéticos
│   └── processed/        # indicators.parquet, features.parquet, sentiment_scores.csv
├── models/               # xgboost_model.json, model_v0, feature_importance.csv
└── datos-externos/       # el contenido de D:/trading-data (StockTwits NYU, WSB, FNSPID)
```

---

## 2. Requisitos previos

- Una cuenta de Google (Gmail).
- **Espacio en Drive:** la cuenta gratuita da **15 GB**. Si tus datasets externos
  superan eso (el dataset StockTwits NYU + FNSPID pueden pesar decenas de GB),
  tienes dos opciones:
  - **Google One 100 GB → ~$2/mes** (lo más barato y suficiente).
  - Subir **solo** `data/processed/` y `models/` a Drive (los datos ya procesados
    pesan poco), y descargar los datasets crudos directamente en el servidor con
    `setup_external_data.py` / `download_stocktwits_historical.py`.

> 💡 **Recomendación para tesis:** sube a Drive solo lo que cuesta regenerar
> (features procesadas + modelos entrenados). Los datasets crudos de Kaggle/S3
> se bajan directo en el servidor porque son públicos.

---

## 3. Crear credenciales

Necesitas un archivo `credentials.json` (OAuth). Solo se hace **una vez**:

1. Entra a [Google Cloud Console](https://console.cloud.google.com).
2. Arriba, crea un **proyecto nuevo** (ej. `tesis-trading`).
3. En el buscador, escribe **"Google Drive API"** → **Habilitar**.
4. Menú lateral → **APIs y servicios → Pantalla de consentimiento OAuth**:
   - Tipo de usuario: **Externo** → Crear.
   - Nombre de la app: `tesis-trading`, tu email, guardar.
   - En **Usuarios de prueba**, agrega tu propio correo de Gmail.
5. Menú lateral → **Credenciales → Crear credenciales → ID de cliente de OAuth**:
   - Tipo de aplicación: **Aplicación de escritorio**.
   - Nombre: `tesis-desktop` → Crear.
6. Descarga el JSON (botón ⬇) y **renómbralo a `credentials.json`**.
7. Colócalo en la **raíz del proyecto** (junto a `requirements.txt`).

> ⚠️ **`credentials.json` y `token.json` ya están en `.gitignore`.** Nunca los subas
> al repositorio.

---

## 4. Instalar dependencias

Desde la raíz del proyecto, con tu entorno virtual activo:

```bash
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

O simplemente reinstala todo el `requirements.txt` (ya incluye estas librerías):

```bash
pip install -r requirements.txt
```

---

## 5. Subir datos desde tu PC

Desde la raíz del proyecto (con `credentials.json` en su sitio):

```bash
# 1. Ver primero qué se subiría (sin subir nada)
python -m src.data.drive_upload --dry-run

# 2. Subir todo
python -m src.data.drive_upload

# O subir solo una parte:
python -m src.data.drive_upload --only processed   # solo data/processed/
python -m src.data.drive_upload --only models       # solo models/
python -m src.data.drive_upload --only raw          # solo data/raw/
python -m src.data.drive_upload --only external     # solo D:/trading-data/
```

**La primera vez** se abrirá el navegador para que autorices el acceso. Se genera
un `token.json` que evita repetir el login en las siguientes ejecuciones.

Características del script:
- **Reanudable:** si se corta, vuelve a correrlo — omite lo ya subido.
- **Sin duplicados:** detecta archivos que ya existen en Drive por nombre.
- **Chunks de 5 MB:** soporta archivos grandes (.parquet de varios GB).

---

## 6. Descargar datos en el servidor

En tu servidor en la nube (después de clonar el repo e instalar dependencias):

```bash
# Copia tu credentials.json y token.json al servidor primero, luego:

# Ver plan
python -m src.data.drive_sync --dry-run

# Descargar todo
python -m src.data.drive_sync

# O solo lo necesario para entrenar/evaluar
python -m src.data.drive_sync --only processed
python -m src.data.drive_sync --only models
```

> 💡 **Copiar las credenciales al servidor:** transfiere `credentials.json` y
> `token.json` de forma segura, por ejemplo con `scp`:
> ```bash
> scp credentials.json token.json usuario@IP_SERVIDOR:/ruta/al/proyecto/
> ```

---

## 7. Conectar los datos

El proyecto ya está preparado para encontrar los datos automáticamente:

- **`data/raw/` y `data/processed/`**: se descargan directo en el árbol del repo,
  donde `config/settings.py` ya los busca (`RAW_DIR`, `PROCESSED_DIR`).
- **`models/`**: igual, se descargan donde el pipeline los espera (`MODELS_DIR`).
- **Datos externos** (`datos-externos/`): configura la variable de entorno
  `TESIS_DATA_ROOT` apuntando a donde los descargaste. En un servidor Linux:

  ```bash
  export TESIS_DATA_ROOT=/home/usuario/TESIS/trading-data
  ```

  Si no la defines, `settings.py` usa `./trading-data` (relativo al repo) como
  fallback portable en Linux, y `D:/trading-data` si existe en Windows.

Verifica que todo quedó conectado:

```bash
python -c "from config import settings; print('Datos externos:', settings.EXTERNAL_DATA_ROOT); print('Existe:', settings.EXTERNAL_DATA_ROOT.exists())"
```

Luego corre el pipeline normalmente (ver `CLAUDE.md`), por ejemplo:

```bash
python -m src.trading.feature_engine
python -m src.trading.xgboost_model
python -m src.evaluation.backtester
```

---

## 8. FAQ

**¿Cuánto cuesta?**
Google Drive gratis da 15 GB. Si necesitas más, Google One 100 GB cuesta ~$2/mes.
Comparado con almacenamiento en AWS S3 o similar, Drive es de lo más económico
para una tesis.

**¿Los datasets crudos (Kaggle/S3) también van por Drive?**
No hace falta: son públicos. En el servidor, bájalos directo con
`python -m src.data.download_stocktwits_historical` o
`python -m src.data.setup_external_data`. Usa Drive solo para lo que tú generaste
(features procesadas, modelos entrenados) o lo que no quieras volver a descargar.

**¿Puedo automatizar la sincronización?**
Sí. En el servidor, agrega `python -m src.data.drive_sync --only processed --yes`
a un cron job, o llámalo al inicio de tus scripts de pipeline.

**El navegador no abre en el servidor (headless).**
Autoriza primero en tu PC (genera `token.json`) y copia ese `token.json` al
servidor. El servidor reutiliza el token sin necesitar navegador.

**¿Es seguro?**
El scope usado es `drive.file`: la app solo ve y gestiona los archivos que ella
misma sube, no todo tu Drive. `credentials.json` y `token.json` están en
`.gitignore` y nunca deben subirse al repo.

---

## Resumen de comandos

| Acción | Comando |
|---|---|
| Ver qué se subiría | `python -m src.data.drive_upload --dry-run` |
| Subir todo desde PC | `python -m src.data.drive_upload` |
| Subir solo procesados | `python -m src.data.drive_upload --only processed` |
| Descargar todo en servidor | `python -m src.data.drive_sync` |
| Descargar solo modelos | `python -m src.data.drive_sync --only models` |
| Verificar conexión | `python -c "from config import settings; print(settings.EXTERNAL_DATA_ROOT.exists())"` |
