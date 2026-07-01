# Estado de TesisLiveRunner: ultima corrida, exit codes, run logs de hoy,
# estado de la tarea y ultima orden (senales) del ultimo run_*.parquet.
$ErrorActionPreference = 'SilentlyContinue'
$repo    = 'C:\dev\Tesis'
$logdir  = Join-Path $repo 'logs\live'
$livedir = Join-Path $repo 'data\live'
$py      = Join-Path $repo '.venv\Scripts\python.exe'

Write-Host "==================== TesisLiveRunner - estado ====================" -ForegroundColor Cyan

# 1) Ultima corrida + exit codes de las ultimas 5
$logs = Get-ChildItem $logdir -Filter 'run_*.log' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
if ($logs) {
    Write-Host "`nUltimo log: $($logs[0].Name)   ($($logs[0].LastWriteTime))"
    Write-Host "`nExit code de las ultimas 5 corridas:"
    foreach ($l in ($logs | Select-Object -First 5)) {
        # Lee el exit code real: acepta "FIN exit code=N", "Exit code: N", etc.
        $m = Select-String -Path $l.FullName -Pattern '(?i)exit code[=:\s]+([0-9]+)' | Select-Object -Last 1
        if ($m) {
            $code = $m.Matches[0].Groups[1].Value
            # Solo FAIL si el exit code esta presente y != 0
            $mark = if ($code -eq '0') { 'OK        ' } else { 'FAIL      ' }
            Write-Host ("  [{0}] {1}  exit={2}" -f $mark, $l.Name, $code)
        } else {
            # Sin linea de exit code => INCOMPLETO (no es un FAIL): kill por el
            # timeout de 10 min, corrida aun en curso, o wrapper sin esa linea.
            Write-Host ("  [INCOMPLETO] {0}  (sin 'exit code' - kill por timeout de 10min o corrida en curso)" -f $l.Name)
        }
    }
} else {
    Write-Host "`n(sin logs todavia en $logdir)"
}

# 2) run_*.parquet generados HOY
$today = (Get-Date).ToString('yyyyMMdd')
$todayRuns = Get-ChildItem $livedir -Filter "run_$today*.parquet" -ErrorAction SilentlyContinue
Write-Host "`nrun_*.parquet generados HOY ($today): $(@($todayRuns).Count)"

# 3) Estado de la tarea
$task = Get-ScheduledTask -TaskName 'TesisLiveRunner' -ErrorAction SilentlyContinue
if ($task) {
    $info = Get-ScheduledTaskInfo -TaskName 'TesisLiveRunner'
    Write-Host "`nTarea TesisLiveRunner:"
    Write-Host "  State          : $($task.State)"
    Write-Host "  Proxima corrida: $($info.NextRunTime)"
    Write-Host "  Ultima corrida : $($info.LastRunTime)  (resultado=$($info.LastTaskResult))"
} else {
    Write-Host "`nTarea TesisLiveRunner: NO registrada (corre el schtasks de registro)."
}

# 4) Ultima orden / senales (ultimo run_*.parquet)
$lastP = Get-ChildItem $livedir -Filter 'run_*.parquet' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($lastP) {
    Write-Host "`nUltimo run log: $($lastP.Name)  ($($lastP.LastWriteTime))"
    Write-Host "Senales (signal=1 => orden de compra enviada en modo LIVE):"
    $env:LASTP = $lastP.FullName
    # PS 5.1 rompe el quoting al pasar -c a python.exe (se come las comillas).
    # Solucion robusta: escribir el snippet a un .py temporal y ejecutarlo.
    $tmp = Join-Path $env:TEMP ("checklive_" + $PID + ".py")
@'
import os, pandas as pd
d = pd.read_parquet(os.environ["LASTP"])
c = [x for x in ["ticker", "date", "signal", "confidence", "close"] if x in d.columns]
print(d[c].to_string(index=False) if len(d) else "  (run sin senales)")
'@ | Set-Content -Path $tmp -Encoding UTF8
    & $py $tmp
    Remove-Item $tmp -ErrorAction SilentlyContinue
} else {
    Write-Host "`n(sin run_*.parquet todavia)"
}

# 5) Ultimos fallos (feed persistente FAILURES.log)
$failLog = Join-Path $logdir 'FAILURES.log'
Write-Host "`n--- ULTIMOS FALLOS (FAILURES.log) ---" -ForegroundColor Yellow
if (Test-Path $failLog) {
    $last = Get-Content $failLog -Tail 5
    if ($last) { $last | ForEach-Object { Write-Host "  $_" } }
    else { Write-Host "  (FAILURES.log vacio)" }
} else {
    Write-Host "  (sin fallos registrados todavia)"
}

Write-Host "`n=================================================================" -ForegroundColor Cyan
