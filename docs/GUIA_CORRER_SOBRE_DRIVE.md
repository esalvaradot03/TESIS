# Guía: Correr scripts "sobre los datos de Drive" (VS Code y Railway)

## El concepto clave

Google Drive **no es un disco** que se pueda montar y leer directamente. Ni VS Code
ni Railway pueden ejecutar pandas/FinBERT/XGBoost "sobre Drive". Lo que sí se hace
es un **ciclo de 3 pasos**:

```
  1. BAJAR de Drive   →   2. CORRER tu script   →   3. SUBIR resultados a Drive
    (drive_sync)            (sobre datos locales)      (drive_upload)
```

El script `src/data/run_on_drive.py` encadena esos 3 pasos en **un solo comando**,
y funciona igual en tu PC (VS Code) o en Railway.

---

## Requisito único: credenciales de Drive

Necesitas `credentials.json` + `token.json` en la raíz del proyecto (ver
`docs/GUIA_GOOGLE_DRIVE.md`). El `token.json` debe tener scope de **escritura** si
vas a subir resultados (`--push`). Genera el token una vez en tu PC autorizando en
el navegador; para Railway, pásalo por variables de entorno.

---

## Uso desde VS Code (local)

Abre la terminal integrada de VS Code (`Ctrl+ñ` o `View → Terminal`), activa el
venv, y corre un solo comando según lo que quieras:

```bash
# Entrenar el modelo: baja features → entrena → sube el modelo
python -m src.data.run_on_drive \
    --pull processed --run "python -m src.trading.xgboost_model" --push models

# Recalcular features: baja lo necesario → corre feature_engine → sube processed
python -m src.data.run_on_drive \
    --pull processed models --run "python -m src.trading.feature_engine" --push processed

# Backtest: baja todo → corre el backtester → sube resultados
python -m src.data.run_on_drive \
    --pull processed models --run "python -m src.evaluation.backtester" --push processed

# Solo bajar datos (sin correr nada)
python -m src.data.run_on_drive --pull processed models

# Ver el plan sin ejecutar
python -m src.data.run_on_drive --pull processed --run "python -m src.trading.xgboost_model" --push models --dry-run
```

### Atajo: tareas de VS Code (opcional)

Puedes crear `.vscode/tasks.json` para tener botones/atajos en VS Code:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Drive: entrenar modelo",
      "type": "shell",
      "command": "python -m src.data.run_on_drive --pull processed --run \"python -m src.trading.xgboost_model\" --push models",
      "problemMatcher": []
    },
    {
      "label": "Drive: bajar datos",
      "type": "shell",
      "command": "python -m src.data.run_on_drive --pull processed models",
      "problemMatcher": []
    }
  ]
}
```

Luego `Ctrl+Shift+P → Tasks: Run Task → Drive: entrenar modelo`.

---

## Uso desde Railway

En Railway hay dos formas, según lo que quieras:

### A) Ejecución puntual/manual — `railway run`
Con el [CLI de Railway](https://docs.railway.com/develop/cli) instalado, desde tu
PC ejecutas un comando **en el entorno de Railway** (usa sus variables de entorno):

```bash
railway run python -m src.data.run_on_drive \
    --pull processed --run "python -m src.trading.xgboost_model" --push models
```

### B) Ejecución programada — Cron Job de Railway
Para que corra solo (ej. reentrenar cada noche, o el pipeline intradía en horario
de mercado), Railway ofrece **Cron Jobs**:

1. En tu servicio → **Settings → Cron Schedule** → define el cron (ej. `0 22 * * 1-5`
   = 22:00 UTC de lunes a viernes).
2. El **Start Command** de ese servicio (o uno separado) sería, por ejemplo:
   ```bash
   python -m src.data.run_on_drive --pull processed --run "python scripts/run_intraday_pipeline.py" --push processed
   ```
3. Añade las variables `GOOGLE_CREDENTIALS_JSON` y `GOOGLE_TOKEN_JSON` (el arranque
   las recrea como archivos, ver `railway_start.sh`).

> ⚠️ El scope del token para `--push` debe ser de escritura
> (`https://www.googleapis.com/auth/drive`). El de solo lectura sirve solo para bajar.

---

## Flags de `run_on_drive.py`

| Flag | Qué hace |
|---|---|
| `--pull SECCIONES` | Baja de Drive antes de correr (`raw processed models external`) |
| `--run "COMANDO"` | Comando shell a ejecutar (entre comillas) |
| `--push SECCIONES` | Sube a Drive después de correr |
| `--dry-run` | Muestra el plan sin ejecutar |
| `--skip-pull-errors` | Continúa aunque falle una descarga |

**Comportamiento seguro:** si el comando de `--run` falla, **NO se sube nada** a
Drive (evita subir resultados corruptos).

---

## Preguntas frecuentes

**¿Puedo evitar bajar los datos cada vez?**
Sí. `drive_sync` **omite los archivos que ya existen** localmente con el mismo
tamaño. La primera vez baja todo; las siguientes solo lo que cambió.

**¿VS Code y Railway comparten los mismos datos?**
Comparten el mismo **Drive** como fuente de verdad. Cada entorno baja su copia
local. Si entrenas en VS Code y haces `--push models`, Railway obtiene ese modelo
nuevo la próxima vez que haga `--pull models`.

**¿Y los 80 GB de datos externos?**
Evita bajarlos a Railway (no caben en el plan Hobby de 5 GB). Procesa los datos
externos en tu PC (donde ya los tienes), sube solo `processed`/`models`, y deja que
Railway trabaje con esos artefactos livianos.
