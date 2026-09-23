#!/usr/bin/env bash
#
# Script de arranque para Railway.
#
# El dashboard corre con lo que ya está versionado en el repo:
#   - models/            (model_v0, model_v0_balanced)  → feature importance
#   - data/live/reports/ (reportes EOD)                 → curva de equity
# Los datos en vivo de Alpaca se leen por API con las claves de entorno.
#
# La sincronización con Google Drive es OPCIONAL y está desactivada por defecto.
# Actívala poniendo la variable DRIVE_SYNC_TARGETS (ej. "processed models" o "all")
# y aportando GOOGLE_CREDENTIALS_JSON + GOOGLE_TOKEN_JSON.
#
# Variables de entorno relevantes (Railway → Variables):
#   ALPACA_API_KEY / ALPACA_API_SECRET   claves de paper trading (para datos en vivo)
#   DRIVE_SYNC_TARGETS                   opcional; si se define, baja datos de Drive
#   GOOGLE_CREDENTIALS_JSON / _TOKEN_JSON  opcional; credenciales OAuth de Drive
#   TESIS_DATA_ROOT                      opcional; destino de datos externos
#   EMIS_RAW_ROOT / EMIS_INTERIM_DIR     opcional; exports de EMIS y sus parquets
#   EMIS_INGEST                          auto | force | off  (default: auto)
#   PORT                                 lo asigna Railway automáticamente
#
set -euo pipefail

echo "──────────────────────────────────────────────"
echo " Railway start — TESIS trading dashboard"
echo "──────────────────────────────────────────────"

# 1. Sincronización OPCIONAL con Google Drive -------------------------------
# Solo se ejecuta si defines DRIVE_SYNC_TARGETS. Por defecto NO se toca Drive:
# el dashboard funciona con models/ y data/live/ que ya vienen en el repo.
if [[ -n "${DRIVE_SYNC_TARGETS:-}" ]]; then
  echo "→ DRIVE_SYNC_TARGETS='${DRIVE_SYNC_TARGETS}' — sincronizando desde Drive"
  [[ -n "${GOOGLE_CREDENTIALS_JSON:-}" ]] && printf '%s' "$GOOGLE_CREDENTIALS_JSON" > credentials.json
  [[ -n "${GOOGLE_TOKEN_JSON:-}" ]] && printf '%s' "$GOOGLE_TOKEN_JSON" > token.json

  if [[ -f "token.json" ]]; then
    if [[ "$DRIVE_SYNC_TARGETS" == "all" ]]; then
      python -m src.data.drive_sync --yes || echo "⚠ drive_sync falló (continuo igual)"
    else
      for t in $DRIVE_SYNC_TARGETS; do
        python -m src.data.drive_sync --only "$t" --yes || echo "⚠ drive_sync --only $t falló (continuo)"
      done
    fi
  else
    echo "⚠ DRIVE_SYNC_TARGETS definido pero falta token.json — se omite la sincronización."
  fi
else
  echo "→ Sin DRIVE_SYNC_TARGETS: el dashboard usa models/ y data/live/ del repo."
fi

# 2. Ingesta OPCIONAL de los exports de EMIS (prensa colombiana) ------------
# Parsear miles de .doc en cada arranque es caro y siempre da lo mismo: el
# resultado se cachea en los parquets y solo se recalcula si no existen o si
# se fuerza con EMIS_INGEST=force.
#
#   EMIS_RAW_ROOT      raíz de los .doc; apúntala al VOLUMEN (no al checkout,
#                      que es efímero) — ej. /app/trading-data/emis
#   EMIS_INTERIM_DIR   destino de los parquets; también en el volumen
#   EMIS_INGEST        "auto" (default: solo si faltan los parquets) | "force" | "off"
#
# Los defaults de las dos rutas los resuelve config/settings.py, que es el dueño
# de las rutas del proyecto; acá solo se replican para poder chequear la
# existencia de los archivos desde bash sin arrancar Python.
EMIS_INGEST="${EMIS_INGEST:-auto}"
EMIS_RAW_ROOT="${EMIS_RAW_ROOT:-data/raw/emis}"
EMIS_INTERIM_DIR="${EMIS_INTERIM_DIR:-data/interim}"
export EMIS_RAW_ROOT EMIS_INTERIM_DIR

if [[ "$EMIS_INGEST" != "off" ]]; then
  if [[ ! -d "$EMIS_RAW_ROOT" ]]; then
    echo "→ EMIS: no existe '$EMIS_RAW_ROOT' — se omite la ingesta."
    echo "   (¿falta 'emis' en DRIVE_SYNC_TARGETS, o el volumen sin montar?)"
  elif [[ -f "$EMIS_INTERIM_DIR/pares_colombia.parquet" && "$EMIS_INGEST" != "force" ]]; then
    echo "→ EMIS: parquets ya presentes en '$EMIS_INTERIM_DIR' — se reutilizan."
    echo "   (EMIS_INGEST=force para reprocesar tras subir exports nuevos)"
  else
    echo "→ EMIS: parseando exports de '$EMIS_RAW_ROOT'…"
    python -m research.colombia.build_news_dataset \
      || echo "⚠ la ingesta de EMIS falló (el dashboard arranca igual)"
  fi
fi

# 3. Arrancar el dashboard --------------------------------------------------
echo "→ Iniciando Streamlit en el puerto ${PORT:-8501}"
exec streamlit run dashboard/app.py \
  --server.port "${PORT:-8501}" \
  --server.address 0.0.0.0 \
  --server.headless true
