# Guía: Desplegar TESIS en Railway + Google Drive (datos ~80 GB)

Esta guía es específica para tu caso: **stack Drive + Railway**, con **~80 GB de
datos** que subes **completos** a Google Drive (tienes espacio de sobra).

> 🎯 **Plan:** subes los 80 GB tal cual a Drive → Railway los baja para correr el
> pipeline y la app. Sin filtrado previo.

---

## 🗺️ Flujo completo

```
┌───────────────────┐  drive_upload.py  ┌────────────────────┐
│  TU PC             │ ────────────────► │  Google Drive      │
│  80 GB de datos    │                   │  tesis-trading/    │
│  StockTwits/WSB/   │                   │  (espacio de sobra)│
│  FNSPID + procesad.│                   └─────────┬──────────┘
└───────────────────┘                             │ drive_sync.py
                                                   ▼
                                        ┌────────────────────┐
                                        │  Railway           │
                                        │  pipeline + app +  │
                                        │  dashboard         │
                                        └────────────────────┘
```

---

## ⚠️ Lo único a tener en cuenta con 80 GB

Subir los 80 GB a Drive no es problema (tienes espacio). El punto a resolver es el
**disco de Railway**:

| Plan Railway | Límite de volumen | Implicación para 80 GB |
|---|---|---|
| **Hobby** ($5/mes) | **5 GB** | ❌ No caben los 80 GB en el volumen |
| **Pro** ($20/mes) | grande, a **~$0.15/GB/mes** | ✅ Caben; 80 GB ≈ **+$12/mes** de disco |

Fuentes: [Railway pricing](https://railway.com/pricing) (volumen ~$0.15/GB/mes; el
límite de 5 GB del plan Hobby está documentado por terceros como
[Sealos](https://sealos.io/pricing/)). *Contenido reformulado por cumplimiento de
licencias.*

**Dos formas de manejarlo:**

- **Opción A — Railway solo corre la app en vivo (recomendada, ~$5/mes).**
  Los 80 GB crudos viven en Drive como backup/almacén. En Railway sincronizas
  **solo `processed/` + `models/`** (lo que la app en vivo y el dashboard
  necesitan), que cabe en los 5 GB del plan Hobby. El procesamiento pesado de los
  80 GB lo haces en tu PC (o una VM temporal) y subes el resultado.

- **Opción B — Railway procesa los 80 GB completos (Pro, ~$32/mes).**
  Subes a Railway Pro, montas un volumen ≥80 GB y bajas todo con `drive_sync.py`.
  Solo tiene sentido si quieres correr el pipeline completo *dentro* de Railway.

> 💡 Para una tesis, la **Opción A** es lo normal: Drive guarda los 80 GB, Railway
> solo ejecuta la app ligera en vivo. Ahorras ~$27/mes.

---

## 🛠️ Paso a paso

### Fase 1 — Subir a Google Drive (desde tu PC)

Configura primero las credenciales de Drive — ver
[`GUIA_GOOGLE_DRIVE.md`](GUIA_GOOGLE_DRIVE.md) sección 3.

```bash
# Ver el plan de subida (sin subir)
python -m src.data.drive_upload --dry-run

# Subir TODO: datos externos (80 GB) + procesados + modelos
python -m src.data.drive_upload

# O por partes:
python -m src.data.drive_upload --only external     # los 80 GB de D:/trading-data
python -m src.data.drive_upload --only processed     # data/processed/
python -m src.data.drive_upload --only models        # models/
```

La subida de 80 GB tardará según tu conexión, pero el script es **reanudable**: si
se corta, vuelve a correrlo y omite lo ya subido.

### Fase 2 — Desplegar en Railway

1. **Crea cuenta** en [railway.com](https://railway.com) (login con GitHub).

2. **New Project → Deploy from GitHub repo** → selecciona `esalvaradot03/TESIS`.

3. **Variables de entorno** (pestaña *Variables*):
   ```
   TESIS_DATA_ROOT=/app/trading-data
   DRIVE_FOLDER_NAME=tesis-trading
   ALPACA_API_KEY=<tu_key>
   ALPACA_API_SECRET=<tu_secret>
   ALPACA_BASE_URL=https://paper-api.alpaca.markets
   ALPACA_DATA_URL=https://data.alpaca.markets
   ALPACA_FEED=iex
   SEED=42
   PYTHONIOENCODING=utf-8
   # Credenciales de Drive como texto (evita commitearlas):
   GOOGLE_CREDENTIALS_JSON=<contenido de credentials.json en una línea>
   GOOGLE_TOKEN_JSON=<contenido de token.json en una línea>
   ```

4. **Volumen** (pestaña *Volumes*): monta un volumen en `/app/trading-data`.
   - **Opción A (Hobby):** volumen de ~5 GB; sincroniza solo `processed/` + `models/`.
   - **Opción B (Pro):** volumen ≥80 GB para bajar todo.

5. **Deploy.** Railway detecta Python vía `requirements.txt`. El arranque
   (`railway_start.sh`, ver `GUIA_GOOGLE_DRIVE.md`) recrea credenciales, sincroniza
   desde Drive y lanza la app.

### Fase 3 — Sincronizar en Railway

Según la opción elegida, el arranque corre:

```bash
# Opción A — solo lo que la app necesita (cabe en Hobby)
python -m src.data.drive_sync --only processed --yes
python -m src.data.drive_sync --only models --yes

# Opción B — todo, incluidos los 80 GB (requiere Pro + volumen grande)
python -m src.data.drive_sync --yes
```

---

## 💰 Costos estimados

| Escenario | Componentes | Total/mes |
|---|---|---|
| **A — app en vivo** | Drive ($0) + Railway Hobby ($5) | **~$5** |
| **B — pipeline completo en Railway** | Drive ($0) + Railway Pro ($20) + volumen 80 GB (~$12) | **~$32** |

---

## ⚠️ Notas de operación en Railway

- **Headless (sin navegador):** autoriza Drive primero en tu PC para generar
  `token.json` y pásalo a Railway vía la variable `GOOGLE_TOKEN_JSON`. El servidor
  reutiliza ese token sin abrir navegador.
- **FinBERT en CPU:** Railway no ofrece GPU. Corre el scoring de sentimiento por
  lotes en horarios acotados, no en bucle infinito, para no disparar el consumo.
- **Paper trading:** un proceso liviano durante horas de mercado cabe holgado en
  el crédito de Hobby.

---

## Resumen ejecutivo

1. **Sube los 80 GB completos a Drive** → `python -m src.data.drive_upload`.
2. **Railway** baja de Drive con `drive_sync.py` y corre la app.
3. **Recomendado (~$5/mes):** Drive guarda los 80 GB; Railway Hobby sincroniza solo
   `processed/` + `models/` y ejecuta la app en vivo.
4. **Si necesitas el pipeline completo en Railway (~$32/mes):** plan Pro + volumen ≥80 GB.
