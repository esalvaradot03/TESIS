@echo off
setlocal EnableExtensions
REM ============================================================
REM TesisLiveRunner — live_runner en LIVE (paper, ordenes a Alpaca)
REM Lo dispara la tarea programada "TesisLiveRunner" cada 30 min en
REM horario de mercado. live_runner se auto-gatea con el reloj de Alpaca
REM (sin --force respeta horario; sin --dry-run envia ordenes paper).
REM ============================================================

cd /d C:\dev\Tesis

REM --- Impedir sleep mientras opera (AC). Persistente: revertir con
REM     powercfg /change standby-timeout-ac 30  (o el valor que prefieras).
powercfg /change standby-timeout-ac 0 >nul 2>&1

REM --- La tarea corre como SYSTEM, cuyo perfil NO tiene el cache de FinBERT.
REM     Apuntar HF_HOME al cache del usuario (ya descargado) evita re-bajar
REM     ~438MB en cada corrida y que el limite de 10 min mate la tarea.
set "HF_HOME=C:\Users\ASUS\.cache\huggingface"
set PYTHONIOENCODING=utf-8

REM --- Log por corrida con timestamp (opcion a) ---
set "LOGDIR=C:\dev\Tesis\logs\live"
if not exist "%LOGDIR%" mkdir "%LOGDIR%"
for /f "usebackq delims=" %%i in (`powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"`) do set "TS=%%i"
set "LOG=%LOGDIR%\run_%TS%.log"

echo ============================================================>> "%LOG%"
echo [%TS%] INICIO live_runner (model_v0, max-symbols 15)>> "%LOG%"
"C:\dev\Tesis\.venv\Scripts\python.exe" -m src.trading.live_runner --model-dir models/model_v0 --max-symbols 15 >> "%LOG%" 2>&1
set "EXITCODE=%ERRORLEVEL%"
echo [%TS%] FIN exit code=%EXITCODE%>> "%LOG%"

REM --- Notificar si Python crasheo (exit != 0). Las corridas con mercado
REM     cerrado o sin datos terminan en 0 (no notifican). ---
if not "%EXITCODE%"=="0" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "C:\dev\Tesis\scripts\notify_failure.ps1" -ExitCode %EXITCODE% -LogPath "%LOG%" >nul 2>&1
)

REM --- Propagar el exit code de Python a Task Scheduler (no fallar silencioso) ---
exit /b %EXITCODE%
