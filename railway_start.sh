#!/usr/bin/env bash
#
# Script de arranque para Railway.
#
# Hace, en orden:
#   1. Recrea credentials.json / token.json desde variables de entorno (si existen),
#      para que drive_sync.py pueda autenticarse sin commitear secretos.
#   2. Sincroniza los datos necesarios desde Google Drive al volumen de Railway.
#      Por defecto baja solo lo que la app en vivo necesita (processed + models),
#      que cabe en el volumen de 5 GB del plan Hobby.
#   3. Arranca el dashboard de Streamlit en el puerto que asigna Railway ($PORT).
#
# Variables de entorno relevantes (configúralas en Railway → Variables):
#   GOOGLE_CREDENTIALS_JSON   contenido de credentials.json (opcional)
#   GOOGLE_TOKEN_JSON         contenido de token.json (opcional)
#   DRIVE_SYNC_TARGETS        qué bajar de Drive; default: "processed models"
#                             (usa "all" para bajar todo, incl. datos externos)
#   TESIS_DATA_ROOT           destino de datos externos; default: /app/trading-data
#   PORT                      lo asigna Railway automáticamente
#
set -euo pipefail

echo "──────────────────────────────────────────────"
echo " Railway start — TESIS trading dashboard"
echo "──────────────────────────────────────────────"

# 1. Recrear credenciales de Google Drive desde variables de entorno --------
if [[ -n "${GOOGLE_CREDENTIALS_JSON:-}" ]]; then
  echo "→ Recreando credentials.json desde GOOGLE_CREDENTIALS_JSON"
  printf '%s' "$GOOGLE_CREDENTIALS_JSON" > credentials.json
fi
if [[ -n "${GOOGLE_TOKEN_JSON:-}" ]]; then
  echo "→ Recreando token.json desde GOOGLE_TOKEN_JSON"
  printf '%s' "$GOOGLE_TOKEN_JSON" > token.json
fi

# 2. Sincronizar datos desde Google Drive -----------------------------------
# Solo intenta sincronizar si hay token (si subiste a Drive manualmente y no
# configuraste OAuth, se salta este paso sin fallar).
TARGETS="${DRIVE_SYNC_TARGETS:-processed models}"
if [[ -f "token.json" ]]; then
  if [[ "$TARGETS" == "all" ]]; then
    echo "→ Sincronizando TODO desde Google Drive"
    python -m src.data.drive_sync --yes || echo "⚠ drive_sync falló (continuo igual)"
  else
    for t in $TARGETS; do
      echo "→ Sincronizando '$t' desde Google Drive"
      python -m src.data.drive_sync --only "$t" --yes || echo "⚠ drive_sync --only $t falló (continuo)"
    done
  fi
else
  echo "→ Sin token.json: se omite la sincronización con Drive."
  echo "  (Si subiste los datos por otro medio al volumen, ignora este aviso.)"
fi

# 3. Arrancar el dashboard --------------------------------------------------
echo "→ Iniciando Streamlit en el puerto ${PORT:-8501}"
exec streamlit run dashboard/app.py \
  --server.port "${PORT:-8501}" \
  --server.address 0.0.0.0 \
  --server.headless true
