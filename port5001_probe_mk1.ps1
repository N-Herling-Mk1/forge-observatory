# =====================================================================
# port5001_probe_mk1.ps1 - who owns 5001, and how to take it back.
# Three suspects: (a) a process listening, (b) a Windows/WSL "excluded
# port range" reservation (nothing listens, bind still fails),
# (c) docker's own half-dead container binding.
#
#   .\port5001_probe_mk1.ps1          # diagnose only
#   .\port5001_probe_mk1.ps1 -Kill    # also stop a found listener
# =====================================================================
param([switch]$Kill, [int]$Port = 5001)
$ErrorActionPreference = "Continue"
function Say([string]$m, [string]$c="Cyan"){ Write-Host $m -ForegroundColor $c }

Say ("=== port" + $Port + "_probe ===") "Yellow"

# ---- [1/4] listeners / connections on the port -------------------------
Say "[1/4] TCP state on port $Port"
$conns = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
if ($conns) {
    $conns | ForEach-Object {
        $proc = Get-Process -Id $_.OwningProcess -ErrorAction SilentlyContinue
        $name = if ($proc) { $proc.ProcessName } else { "?" }
        Say ("     " + $_.State + "  PID " + $_.OwningProcess + "  (" + $name + ")") "DarkYellow"
        if ($Kill -and $_.State -eq "Listen") {
            Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
            Say ("     KILLED PID " + $_.OwningProcess) "Red"
        }
    }
} else { Say "     no TCP entries - nothing is listening on $Port" "Green" }

# ---- [2/4] Windows excluded port ranges --------------------------------
Say "[2/4] excluded port ranges (Hyper-V/WSL reservations)"
$excl = netsh interface ipv4 show excludedportrange protocol=tcp 2>&1 | Out-String
$hit = $false
foreach ($line in ($excl -split "`r?`n")) {
    if ($line -match "^\s*(\d+)\s+(\d+)") {
        $lo=[int]$Matches[1]; $hi=[int]$Matches[2]
        if ($Port -ge $lo -and $Port -le $hi) { $hit=$true; Say ("     RESERVED: $Port falls in " + $lo + "-" + $hi) "Red" }
    }
}
if (-not $hit) { Say "     $Port is not inside any reserved range" "Green" }

# ---- [3/4] docker's view -------------------------------------------------
Say "[3/4] docker containers touching $Port"
& docker ps -a --format "{{.Names}}  {{.Status}}  {{.Ports}}" 2>&1 |
    ForEach-Object { if ($_ -match $Port -or $_ -match "genre") { Say ("     " + $_) "DarkYellow" } }

# ---- [4/4] verdict --------------------------------------------------------
Say "[4/4] verdict" "Yellow"
if ($conns -and -not $Kill) {
    Say "     A process holds $Port. Re-run with -Kill, or close that app, then:" "Green"
    Say "       .\publish_mk1.ps1 -NoBuild -NoPush" "Green"
} elseif ($conns -and $Kill) {
    Say "     Listener killed. Re-run:  .\publish_mk1.ps1 -NoBuild -NoPush" "Green"
} elseif ($hit) {
    Say "     Nothing listens - Windows RESERVED the range. Two ways out:" "DarkYellow"
    Say "       (a) admin PowerShell:  net stop winnat ; net start winnat" "DarkYellow"
    Say "           (clears dynamic reservations; run publish again after)" "DarkYellow"
    Say "       (b) tell me and I remap the container's host port in compose +" "DarkYellow"
    Say "           publish script - the tunnel does not care about the host port." "DarkYellow"
} else {
    Say "     No listener, no reservation - stale docker state. Try:" "DarkYellow"
    Say "       docker compose down ; .\publish_mk1.ps1 -NoBuild -NoPush" "DarkYellow"
}
exit 0
