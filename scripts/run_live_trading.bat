@echo off
REM Live trading launcher - corre desde Task Scheduler
REM Activa el venv y ejecuta live_runner. Logs en D:\trading-data\logs\

cd /d C:\dev\Tesis
call .venv\Scripts\activate.bat

set LOGDIR=D:\trading-data\logs
if not exist "%LOGDIR%" mkdir "%LOGDIR%"

set LOGFILE=%LOGDIR%\live_runner_%DATE:~-4%%DATE:~3,2%%DATE:~0,2%.log
echo ===== %DATE% %TIME% ===== >> "%LOGFILE%"

python -m src.trading.live_runner >> "%LOGFILE%" 2>&1
echo Exit code: %ERRORLEVEL% >> "%LOGFILE%"
