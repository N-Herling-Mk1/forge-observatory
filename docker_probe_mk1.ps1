# =====================================================================
# docker_probe_mk1.ps1
# Why: `docker compose` errors with "open //./pipe/dockerDesktopLinuxEngine:
# The system cannot find the file specified" = the Docker Desktop ENGINE
# is not running (client ok, backend absent).
#
# This probe: [1] client  [2] engine pipe  [3] start Docker Desktop if
# needed and WAIT on the pipe  [4] verify engine  [5] check the
# forge-observatory:cpu image  [6] verdict (+ WSL dump on failure).
#
#   .\docker_probe_mk1.ps1
# =====================================================================
$ErrorActionPreference = "Continue"
$PIPE = "\\.\pipe\dockerDesktopLinuxEngine"
$IMG  = "forge-observatory:cpu"

function Say([string]$m, [string]$c = "Cyan") { Write-Host $m -ForegroundColor $c }

Say "=== docker_probe_mk1 ===" "Yellow"

# ---- [1/6] client ----------------------------------------------------
Say "[1/6] docker client"
$cli = Get-Command docker -ErrorAction SilentlyContinue
if (-not $cli) { Say "     FAIL - docker CLI not on PATH. Install Docker Desktop." "Red"; exit 1 }
Say ("     ok  " + $cli.Source) "Green"

# ---- [2/6] engine pipe ------------------------------------------------
Say "[2/6] engine pipe ($PIPE)"
$pipeUp = Test-Path $PIPE
Say ($(if ($pipeUp) { "     ok  pipe present" } else { "     absent - engine not running" })) ($(if ($pipeUp) { "Green" } else { "DarkYellow" }))

# ---- [3/6] start Docker Desktop if needed ------------------------------
if (-not $pipeUp) {
    Say "[3/6] Docker Desktop process"
    $proc = Get-Process "Docker Desktop" -ErrorAction SilentlyContinue
    if ($proc) {
        Say "     running (still booting or stuck) - waiting on engine..." "DarkYellow"
    } else {
        $exe = @(
            (Join-Path $Env:ProgramFiles "Docker\Docker\Docker Desktop.exe"),
            (Join-Path $Env:LOCALAPPDATA "Programs\Docker\Docker\Docker Desktop.exe")
        ) | Where-Object { Test-Path $_ } | Select-Object -First 1
        if (-not $exe) {
            Say "     FAIL - Docker Desktop exe not found at standard paths." "Red"
            Say "     Open Docker Desktop from the Start menu manually, then re-run this probe." "Red"
            exit 1
        }
        Say ("     launching: " + $exe)
        Start-Process $exe | Out-Null
    }
    # wait up to 150 s for the engine pipe
    $limit = 150; $t = 0
    while (-not (Test-Path $PIPE) -and $t -lt $limit) {
        Write-Progress -Activity "docker_probe" -Status ("waiting for engine pipe... " + $t + "s / " + $limit + "s") -PercentComplete ([int]($t/$limit*100))
        Start-Sleep -Seconds 3; $t += 3
    }
    Write-Progress -Activity "docker_probe" -Completed
    $pipeUp = Test-Path $PIPE
    Say ($(if ($pipeUp) { "     ok  engine pipe up after ~" + $t + "s" } else { "     FAIL - engine never came up (" + $limit + "s)" })) ($(if ($pipeUp) { "Green" } else { "Red" }))
} else {
    Say "[3/6] Docker Desktop process - skip (engine already up)"
}

# ---- [4/6] engine answers ----------------------------------------------
Say "[4/6] docker version (server)"
$srvOk = $false
if ($pipeUp) {
    $v = (& docker version 2>&1) | Out-String
    $srvOk = ($v -match "Server:")
    if ($srvOk) {
        $line = ($v -split "`r?`n" | Where-Object { $_ -match "^\s*Version:" } | Select-Object -First 2) -join " | "
        Say ("     ok  " + $line.Trim()) "Green"
    } else { Say "     FAIL - pipe present but engine not answering" "Red" }
} else { Say "     skip - no pipe" "DarkYellow" }

# ---- [5/6] image present -------------------------------------------------
Say ("[5/6] image " + $IMG)
$imgOk = $false
if ($srvOk) {
    & docker image inspect $IMG *> $null
    $imgOk = ($LASTEXITCODE -eq 0)
    if ($imgOk) { Say "     ok  image exists - earlier build survived" "Green" }
    else { Say "     absent - earlier 'build' ran with the engine down; rebuild needed" "DarkYellow" }
} else { Say "     skip" "DarkYellow" }

# ---- [6/6] verdict --------------------------------------------------------
Say "[6/6] verdict" "Yellow"
if ($srvOk -and $imgOk) {
    Say "     ENGINE UP + IMAGE PRESENT. Run:" "Green"
    Say "       docker compose up -d genre" "Green"
    Say "       docker compose logs -f genre" "Green"
    exit 0
}
if ($srvOk -and -not $imgOk) {
    Say "     ENGINE UP, IMAGE MISSING. Run:" "Green"
    Say "       docker compose build genre" "Green"
    Say "       docker compose up -d genre ; docker compose logs -f genre" "Green"
    exit 0
}
# engine never answered -> dump WSL state for the next round
Say "     ENGINE DOWN. WSL state below - paste this whole output back." "Red"
Say "----- wsl --status -----"
& wsl --status 2>&1 | Out-String | Write-Host
Say "----- wsl -l -v --------"
& wsl -l -v 2>&1 | Out-String | Write-Host
Say "----- com.docker.service (may legitimately not exist) -----"
Get-Service com.docker.service -ErrorAction SilentlyContinue | Format-Table -AutoSize | Out-String | Write-Host
Say "Common fixes: approve the WSL2-update prompt in the Docker Desktop window;" "DarkYellow"
Say "or run 'wsl --update' in an admin PowerShell, then relaunch Docker Desktop." "DarkYellow"
exit 1
