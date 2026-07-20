# =====================================================================
# publish_mk2.ps1  -  the "door 2" button.
# Dev loop stays:  python apps\genre\app\server.py   (local, auto-port)
# When happy:      .\publish_mk2.ps1                 (this file)
#
# What it does, in order:
#   [1] Docker engine up (launches Docker Desktop + waits if needed)
#   [2] docker compose build genre        (bakes current code + artifacts)
#   [3] docker compose up -d genre        (container on 127.0.0.1:5601)
#   [4] smoke the container health endpoint
#   [5] tunnel: if .env carries TUNNEL_TOKEN -> up cloudflared + smoke the
#       public subdomain; if not -> prints the ONE-TIME setup and continues
#   [6] git add/commit/push               (updates the public Pages floor)
#
# Switches:  -NoPush   build+run+tunnel but skip git
#            -NoBuild  skip the image rebuild (rare)
# =====================================================================
param(
    [switch]$NoPush,
    [switch]$NoBuild
)
$ErrorActionPreference = "Continue"
$PIPE   = "\\.\pipe\dockerDesktopLinuxEngine"
$LOCAL  = "http://127.0.0.1:5601/api/health"
$PUBLIC = "https://genre.forge-observatory.com/api/health"

function Say([string]$m, [string]$c = "Cyan") { Write-Host $m -ForegroundColor $c }
function Probe([string]$url, [int]$tries, [int]$gap, [string]$label) {
    for ($i = 1; $i -le $tries; $i++) {
        Write-Progress -Activity "publish" -Status ("waiting on " + $label + " (" + $i + "/" + $tries + ")") -PercentComplete ([int]($i/$tries*100))
        try {
            $r = Invoke-WebRequest -UseBasicParsing -TimeoutSec 4 $url
            if ($r.StatusCode -eq 200) { Write-Progress -Activity "publish" -Completed; return $true }
        } catch { Start-Sleep -Seconds $gap }
    }
    Write-Progress -Activity "publish" -Completed
    return $false
}

Say "=== publish_mk2 ===" "Yellow"

# ---- [1/6] engine ------------------------------------------------------
Say "[1/6] docker engine"
if (-not (Test-Path $PIPE)) {
    $exe = @(
        (Join-Path $Env:ProgramFiles "Docker\Docker\Docker Desktop.exe"),
        (Join-Path $Env:LOCALAPPDATA "Programs\Docker\Docker\Docker Desktop.exe")
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not (Get-Process "Docker Desktop" -ErrorAction SilentlyContinue) -and $exe) {
        Say "     launching Docker Desktop..."
        Start-Process $exe | Out-Null
    }
    $t = 0
    while (-not (Test-Path $PIPE) -and $t -lt 150) {
        Write-Progress -Activity "publish" -Status ("engine starting... " + $t + "s") -PercentComplete ([int]($t/150*100))
        Start-Sleep 3; $t += 3
    }
    Write-Progress -Activity "publish" -Completed
}
if (-not (Test-Path $PIPE)) { Say "     FAIL - engine never came up. Run docker_probe_mk1.ps1 for the WSL dump." "Red"; exit 1 }
Say "     ok  engine up" "Green"

# ---- [2/6] build --------------------------------------------------------
if ($NoBuild) { Say "[2/6] build - skipped (-NoBuild)" "DarkYellow" }
else {
    Say "[2/6] docker compose build genre"
    & docker compose build genre
    if ($LASTEXITCODE -ne 0) { Say "     FAIL - build error above" "Red"; exit 1 }
    Say "     ok  image rebuilt" "Green"
}

# ---- [3/6] container ------------------------------------------------------
Say "[3/6] docker compose up -d genre"
& docker compose up -d genre
if ($LASTEXITCODE -ne 0) { Say "     FAIL" "Red"; exit 1 }
Say "     ok" "Green"

# ---- [4/6] local smoke -----------------------------------------------------
Say "[4/6] container smoke ($LOCAL)"
if (Probe $LOCAL 10 2 "container") { Say "     ok  container answering on 127.0.0.1:5601" "Green" }
else { Say "     FAIL - container not answering; docker compose logs genre" "Red"; exit 1 }

# ---- [5/6] tunnel ------------------------------------------------------------
Say "[5/6] tunnel"
$envFile = ".env"
$hasToken = (Test-Path $envFile) -and ((Get-Content $envFile -Raw) -match "TUNNEL_TOKEN=\S")
if ($hasToken) {
    & docker compose up -d --no-deps cloudflared
    if ($LASTEXITCODE -ne 0) { Say "     FAIL starting cloudflared" "Red"; exit 1 }
    Say "     cloudflared up - probing the public subdomain..."
    if (Probe $PUBLIC 15 4 "public subdomain") { Say "     ok  PUBLIC LIVE: https://genre.forge-observatory.com" "Green" }
    else { Say "     public not answering yet - check Zero Trust hostname genre -> genre:5000; docker compose logs cloudflared" "Red" }
} else {
    Say "     no TUNNEL_TOKEN in .env - PUBLIC SKIPPED. One-time setup:" "DarkYellow"
    Say "       1. one.dash.cloudflare.com -> Zero Trust -> Networks -> Tunnels -> Create (Cloudflared), name: forge" "DarkYellow"
    Say "       2. copy the token -> repo root .env:   TUNNEL_TOKEN=<paste>" "DarkYellow"
    Say "       3. tunnel Public Hostname: genre.forge-observatory.com -> HTTP -> genre:5000" "DarkYellow"
    Say "       4. re-run .\publish_mk2.ps1  (fully hands-off from then on)" "DarkYellow"
}

# ---- [6/6] push --------------------------------------------------------------
if ($NoPush) { Say "[6/6] git - skipped (-NoPush)" "DarkYellow" }
else {
    Say "[6/6] git push (updates the public Pages floor)"
    & git add -A
    $stamp = Get-Date -Format "yyyy-MM-dd HH:mm"
    & git commit -m ("publish " + $stamp) 2>&1 | Out-String | Write-Host
    & git push
    if ($LASTEXITCODE -ne 0) { Say "     push FAILED (see above)" "Red" } else { Say "     ok  pushed" "Green" }
}

Say ""
Say "PUBLISH COMPLETE." "Yellow"
Say ("  local container : http://127.0.0.1:5601")
Say ("  public          : " + $(if ($hasToken) { "https://genre.forge-observatory.com  +  https://forge-observatory.com" } else { "skipped (no tunnel token yet)" }))
Say "  dev loop untouched: python apps\genre\app\server.py"
exit 0
