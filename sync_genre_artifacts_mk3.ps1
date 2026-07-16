# =====================================================================
# sync_genre_artifacts_mk3.ps1   (mk2 + raw-wav exclusion + clean dry-run)
# Copies the genre runtime artifacts the deploy repo does NOT carry in
# git (models discipline / derived data) from the experiments checkout:
#
#   data\  (jsons + before/after, EXCLUDING data\raw)  -> apps\genre\data\
#   eda\figures\before + after                         -> apps\genre\eda\figures\
#   models\<bundle> (x3 default)                       -> apps\genre\models\
#
# Path resolution:
#   DeployRepo      = this script's own folder (it lives at repo root)
#   ExperimentsRepo = sibling folder named INFO_698_experiments,
#                     else pass -ExperimentsRepo "D:\path\INFO_698_experiments"
#
#   .\sync_genre_artifacts_mk3.ps1 -WhatIfOnly     # list, copy nothing
#   .\sync_genre_artifacts_mk3.ps1
# =====================================================================
param(
    [string]$ExperimentsRepo = "",
    [string]$DeployRepo      = "",
    [string[]]$Bundles       = @("beardown","beardown_rrm","beardown_3sec"),
    [switch]$WhatIfOnly
)

$ErrorActionPreference = "Stop"

function Say([string]$m, [string]$c = "Cyan") { Write-Host $m -ForegroundColor $c }

# ---- resolve repos ---------------------------------------------------
if (-not $DeployRepo) { $DeployRepo = $PSScriptRoot }
if (-not $ExperimentsRepo) {
    $guess = Join-Path (Split-Path $DeployRepo -Parent) "INFO_698_experiments"
    if (Test-Path $guess) { $ExperimentsRepo = $guess }
}
$src = if ($ExperimentsRepo) { Join-Path $ExperimentsRepo "projects\genre" } else { "" }
$dst = Join-Path $DeployRepo "apps\genre"

Say "=== sync_genre_artifacts_mk3 ===" "Yellow"
Say ("  deploy repo      : " + $DeployRepo)
Say ("  experiments repo : " + $(if ($ExperimentsRepo) { $ExperimentsRepo } else { "(not found)" }))

if (-not (Test-Path (Join-Path $DeployRepo "docker-compose.yml"))) {
    throw ("deploy repo root not found (no docker-compose.yml beside this script): " + $DeployRepo)
}
if (-not $src -or -not (Test-Path $src)) {
    Say ""
    Say "experiments checkout not located. Re-run with the path, e.g.:" "Red"
    Say '  .\sync_genre_artifacts_mk3.ps1 -ExperimentsRepo "C:\path\to\INFO_698_experiments"' "Red"
    throw "experiments genre root not found"
}

$mode = @("/MIR","/NJH","/NJS","/NDL","/NC","/NS","/NP")
if ($WhatIfOnly) { $mode += "/L"; Say "  MODE: WhatIf (list only, nothing copied)" "Magenta" }

$steps = 2 + $Bundles.Count
$step  = 0
$fail  = 0

function Bar([int]$i, [int]$n, [string]$label) {
    $pct = [int](($i / [math]::Max(1,$n)) * 100)
    Write-Progress -Activity "sync_genre_artifacts" -Status $label -PercentComplete $pct
}

function Sync([string]$from, [string]$to, [string]$label, [string[]]$extra = @()) {
    $script:step++
    Bar $script:step $steps $label
    Say ("[" + $script:step + "/" + $steps + "] " + $label)
    if (-not (Test-Path $from)) { Say ("     SKIP - source missing: " + $from) "DarkYellow"; return }
    $xd = @(); for ($i=0; $i -lt $extra.Count; $i++) { if ($extra[$i] -eq "/XD" -and ($i+1) -lt $extra.Count) { $xd += $extra[$i+1] } }
    $srcFiles = Get-ChildItem -Recurse -File $from -ErrorAction SilentlyContinue
    foreach ($x in $xd) { $srcFiles = $srcFiles | Where-Object { $_.FullName -notmatch ("\\" + [regex]::Escape($x) + "\\") } }
    $srcN = ($srcFiles | Measure-Object).Count
    if ($WhatIfOnly) {
        & robocopy $from $to @mode @extra | Out-Null
        if ($LASTEXITCODE -ge 8) { $script:fail++; Say ("     FAIL rc=" + $LASTEXITCODE) "Red" }
        else { Say ("     listed (dry run) - " + $srcN + " files at source") "Green" }
        return
    }
    New-Item -ItemType Directory -Force -Path $to | Out-Null
    & robocopy $from $to @mode @extra | Out-Null
    $rc = $LASTEXITCODE
    if ($rc -ge 8) { $script:fail++; Say ("     FAIL rc=" + $rc) "Red" }
    else {
        $n = (Get-ChildItem -Recurse -File $to -ErrorAction SilentlyContinue | Measure-Object).Count
        Say ("     ok  (" + $n + " files in dest)") "Green"
    }
}

# ---- 1) data: jsons + before/after, raw wavs excluded at copy time ----
Sync (Join-Path $src "data") (Join-Path $dst "data") "data\ (jsons + before/after, raw EXCLUDED)" @("/XD","raw")
# belt+suspenders: if an earlier sync ever landed raw in the deploy copy, drop it
$rawDst = Join-Path $dst "data\raw"
if ((Test-Path $rawDst) -and (-not $WhatIfOnly)) {
    Remove-Item -Recurse -Force $rawDst
    Say "     pruned stale data\raw from deploy copy" "DarkYellow"
}

# ---- 2) figures -------------------------------------------------------
Sync (Join-Path $src "eda\figures") (Join-Path $dst "eda\figures") "eda\figures\ (before + after)"

# ---- 3) model bundles -------------------------------------------------
foreach ($b in $Bundles) {
    Sync (Join-Path $src ("models\" + $b)) (Join-Path $dst ("models\" + $b)) ("models\" + $b)
}

Write-Progress -Activity "sync_genre_artifacts" -Completed
if ($fail -gt 0) { Say ("DONE with " + $fail + " FAILURES") "Red"; exit 1 }
Say "DONE - deploy tree carries data + figures + bundles. Next: docker compose build genre" "Yellow"
exit 0
