<#
lag_probe_mk1.ps1 — forge-observatory latency probe (local container vs public tunnel)

Usage:
  .\lag_probe_mk1.ps1                          # timing table, N=5 per endpoint
  .\lag_probe_mk1.ps1 -N 9
  .\lag_probe_mk1.ps1 -Burst                   # + 8-way parallel burst (serialization check)
  .\lag_probe_mk1.ps1 -Paths "/","/api/forge/meta"

Reading the table:
  loc_ttfb   ~ app compute (container, zero network)
  delta      ~ tunnel + Cloudflare edge + your uplink + TLS  (pub_ttfb - loc_ttfb)
  cf_cache   DYNAMIC = nothing cached at edge; HIT = served from CF, never touched your box
  pub 403/503 while loc is 200 => CF bot-fight/challenge is eating curl — verify in a browser
#>

param(
    [string]$LocalBase  = "http://127.0.0.1:5601",
    [string]$PublicBase = "https://genre.forge-observatory.com",
    [string[]]$Paths    = @("/", "/dashboard", "/api/forge/meta"),
    [int]$N             = 5,
    [switch]$Burst
)

if (-not (Get-Command curl.exe -ErrorAction SilentlyContinue)) {
    Write-Host "[x] curl.exe not found (ships with Win10+). Abort." -ForegroundColor Red
    exit 1
}

$fmt = "%{time_starttransfer} %{time_total} %{http_code} %{size_download}"
$ua  = "lag-probe-mk1"

function Get-Median([double[]]$a) {
    if ($a.Count -eq 0) { return $null }
    $s = $a | Sort-Object
    return [math]::Round($s[[int][math]::Floor($s.Count / 2)], 0)
}

function Probe([string]$base, [string]$path, [string]$tag) {
    $ttfb = @(); $tot = @(); $code = "---"; $bytes = "0"
    for ($i = 1; $i -le $N; $i++) {
        Write-Host (("`r  [{0}/{1}] {2,-6} {3}" -f $i, $N, $tag, $path).PadRight(70)) -NoNewline
        $out = & curl.exe -s -A $ua -o NUL -w $fmt --max-time 30 "$base$path" 2>$null
        if (-not $out) { continue }
        $p = ($out.Trim()) -split "\s+"
        if ($p.Count -lt 4 -or $p[2] -eq "000") { continue }
        $ttfb += [double]$p[0] * 1000
        $tot  += [double]$p[1] * 1000
        $code = $p[2]; $bytes = $p[3]
    }
    Write-Host (("`r").PadRight(70) + "`r") -NoNewline
    if ($ttfb.Count -eq 0) { return $null }
    [pscustomobject]@{
        ttfb  = Get-Median $ttfb
        total = Get-Median $tot
        code  = $code
        kb    = [math]::Round([double]$bytes / 1kb, 1)
    }
}

function Get-CfCache([string]$url) {
    $h = & curl.exe -s -A $ua -D - -o NUL --max-time 30 $url 2>$null
    $line = $h | Where-Object { $_ -match '^cf-cache-status:' } | Select-Object -First 1
    if ($line) { return ($line -split ':\s*')[1].Trim() }
    return "-"
}

Write-Host ""
Write-Host "=== lag_probe_mk1 ===" -ForegroundColor Cyan
Write-Host ("local : {0}" -f $LocalBase)
Write-Host ("public: {0}" -f $PublicBase)
Write-Host ("N     : {0} per endpoint (medians, ms)" -f $N)
Write-Host ""

$rows = @()
$k = 0
foreach ($path in $Paths) {
    $k++
    Write-Host ("[{0}/{1}] {2}" -f $k, $Paths.Count, $path) -ForegroundColor Yellow
    $loc = Probe $LocalBase  $path "local"
    $pub = Probe $PublicBase $path "public"
    $cf  = Get-CfCache "$PublicBase$path"
    $rows += [pscustomobject]@{
        path      = $path
        loc_ttfb  = $(if ($loc) { $loc.ttfb }  else { "DOWN" })
        loc_total = $(if ($loc) { $loc.total } else { "-" })
        pub_ttfb  = $(if ($pub) { $pub.ttfb }  else { "DOWN" })
        pub_total = $(if ($pub) { $pub.total } else { "-" })
        delta     = $(if ($loc -and $pub) { $pub.ttfb - $loc.ttfb } else { "-" })
        cf_cache  = $cf
        kb        = $(if ($pub) { $pub.kb } elseif ($loc) { $loc.kb } else { "-" })
        code      = $(if ($pub) { $pub.code } elseif ($loc) { $loc.code } else { "-" })
    }
}

Write-Host ""
$rows | Format-Table -AutoSize
Write-Host "read: loc_ttfb ~ app compute | delta ~ tunnel+edge+uplink | cf_cache DYNAMIC = no edge caching" -ForegroundColor DarkGray
Write-Host "note: run again from another network (phone hotspot) for the true visitor path." -ForegroundColor DarkGray

if ($Burst) {
    $bp = "/api/forge/meta"
    foreach ($base in @($LocalBase, $PublicBase)) {
        $url = "$base$bp"
        Write-Host ""
        Write-Host ("=== burst: 8x parallel {0} ===" -f $url) -ForegroundColor Cyan
        Write-Host "  [1/2] single baseline..."
        $t0 = Get-Date
        & curl.exe -s -A $ua -o NUL --max-time 30 $url | Out-Null
        $one = [math]::Round(((Get-Date) - $t0).TotalMilliseconds, 0)
        Write-Host "  [2/2] firing 8 parallel..."
        $t0 = Get-Date
        $procs = @()
        1..8 | ForEach-Object {
            $procs += Start-Process curl.exe -ArgumentList @("-s", "-A", $ua, "-o", "NUL", "--max-time", "60", $url) -NoNewWindow -PassThru
        }
        $procs | Wait-Process -ErrorAction SilentlyContinue
        $wall  = [math]::Round(((Get-Date) - $t0).TotalMilliseconds, 0)
        $ratio = if ($one -gt 0) { [math]::Round($wall / $one, 1) } else { "-" }
        Write-Host ("  single = {0} ms | 8-parallel wall = {1} ms | ratio = {2}x" -f $one, $wall, $ratio)
        Write-Host "  ratio ~8x => requests fully serialized (single worker). ~1-2x => concurrent." -ForegroundColor DarkGray
    }
}
