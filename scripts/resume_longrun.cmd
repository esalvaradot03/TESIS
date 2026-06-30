@echo off
REM Auto-resume del pipeline de larga duración (idempotente: lock + DONE marker).
REM Lo invoca la tarea programada "TesisLongrunResume" al iniciar sesión y cada 2h.
REM Si ya hay una instancia corriendo (lock) o ya completó (DONE), hace no-op.
cd /d C:\dev\Tesis
set PYTHONIOENCODING=utf-8
"C:\dev\Tesis\.venv\Scripts\python.exe" -m src.data.longrun.run_longrun >> "D:\trading-data\logs\scheduled.log" 2>&1
