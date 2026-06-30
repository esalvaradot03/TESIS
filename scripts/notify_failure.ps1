param(
    [int]$ExitCode,
    [string]$LogPath
)
# Notifica cuando una corrida de TesisLiveRunner falla (exit != 0).
#
# Windows Home + Session 0: msg.exe NO llega desde SYSTEM a la sesion
# interactiva, asi que el feed CONFIABLE es un log persistente que se revisa a
# mano. Orden:
#   1) SIEMPRE: append a logs\live\FAILURES.log (no rota, no se borra).
#   2) Discord webhook si DISCORD_WEBHOOK_URL esta en .env (no se agrega aca).
#   3) Toast/balloon best-effort solo si NO corre como SYSTEM (prueba interactiva).

$repo    = 'C:\dev\Tesis'
$failLog = Join-Path $repo 'logs\live\FAILURES.log'
$logName = if ($LogPath) { Split-Path $LogPath -Leaf } else { 'desconocido' }
$ts      = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
$line    = "[$ts] exit=$ExitCode log=$logName"

# --- 1) Feed persistente de fallos (SIEMPRE) ---
try {
    $dir = Split-Path $failLog -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    Add-Content -Path $failLog -Value $line -Encoding UTF8
} catch {}

# --- 2) Discord webhook (solo si la variable esta en .env) ---
try {
    $envFile = Join-Path $repo '.env'
    if (Test-Path $envFile) {
        $raw = Get-Content $envFile | Where-Object { $_ -match '^\s*DISCORD_WEBHOOK_URL\s*=' } | Select-Object -First 1
        if ($raw) {
            $url = ($raw -replace '^\s*DISCORD_WEBHOOK_URL\s*=\s*', '').Trim().Trim('"').Trim("'")
            if ($url) {
                $body = @{ content = "TesisLiveRunner fallo: exit=$ExitCode, log=$logName" } | ConvertTo-Json -Compress
                Invoke-RestMethod -Uri $url -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 15 | Out-Null
            }
        }
    }
} catch {}

# --- 3) Notificacion interactiva best-effort (no aplica a SYSTEM/Home Session 0) ---
$whoami = ([Security.Principal.WindowsIdentity]::GetCurrent()).Name
if ($whoami -ne 'NT AUTHORITY\SYSTEM') {
    $msg = "TesisLiveRunner FALLO: exit code $ExitCode. Ver $LogPath"
    try {
        if (Get-Module -ListAvailable -Name BurntToast) {
            Import-Module BurntToast -ErrorAction Stop
            New-BurntToastNotification -Text "TesisLiveRunner falló", $msg
        } else {
            Add-Type -AssemblyName System.Windows.Forms
            $n = New-Object System.Windows.Forms.NotifyIcon
            $n.Icon = [System.Drawing.SystemIcons]::Error
            $n.Visible = $true
            $n.ShowBalloonTip(10000, "TesisLiveRunner falló", $msg, [System.Windows.Forms.ToolTipIcon]::Error)
            Start-Sleep -Seconds 6
            $n.Dispose()
        }
    } catch {}
}
