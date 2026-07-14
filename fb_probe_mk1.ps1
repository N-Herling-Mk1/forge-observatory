# fb_probe_mk1.ps1 - who does the edge refuse?
# Fetches the site under different User-Agents over forced TLS 1.2 and prints a verdict.
param([string]$Domain = "forge-observatory.com")

[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Get-Status([string]$Url, [string]$UA) {
  try {
    $r = Invoke-WebRequest -Uri $Url -UserAgent $UA -UseBasicParsing -TimeoutSec 15 -ErrorAction Stop
    return [string]$r.StatusCode
  } catch {
    if ($_.Exception.Response) { return [string][int]$_.Exception.Response.StatusCode }
    return "ERR: " + $_.Exception.Message.Split("`n")[0]
  }
}

$uas = [ordered]@{
  "normal client " = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
  "facebook bot  " = "facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)"
  "twitter bot   " = "Twitterbot/1.0"
}

Write-Host ""
Write-Host ("  FB PROBE mk1  //  {0}  //  {1}" -f $Domain, (Get-Date -Format "HH:mm:ss")) -ForegroundColor White
Write-Host ("  " + "." * 78) -ForegroundColor DarkCyan

$res = @{}
foreach ($k in $uas.Keys) {
  $code = Get-Status ("https://{0}" -f $Domain) $uas[$k]
  $res[$k.Trim()] = $code
  $color = "Green"; if ($code -ne "200") { $color = "Red" }
  Write-Host ("    {0} -> {1}" -f $k, $code) -ForegroundColor $color
}

$rob = Get-Status ("https://{0}/robots.txt" -f $Domain) $uas["normal client "]
Write-Host ("    robots.txt     -> {0}" -f $rob) -ForegroundColor $(if ($rob -eq "200") { "Green" } else { "Yellow" })

Write-Host ("  " + "." * 78) -ForegroundColor DarkCyan
$fb = $res["facebook bot"]; $nc = $res["normal client"]
if ($nc -like "ERR*") {
  Write-Host "    VERDICT: client TLS still failing locally - probe from another network/device." -ForegroundColor Yellow
} elseif ($fb -eq "200") {
  Write-Host "    VERDICT: UA is NOT blocked from your IP -> GitHub blocks Meta's crawler IPs." -ForegroundColor Cyan
  Write-Host "    FIX: orange-cloud both DNS records (set Cloudflare SSL mode to Full first)." -ForegroundColor Cyan
} elseif ($fb -eq "403") {
  Write-Host "    VERDICT: UA-based block at GitHub's edge - affects FB regardless of source IP." -ForegroundColor Cyan
  Write-Host "    OPTIONS: wait it out (known to self-heal) or proxy via Cloudflare with UA rewrite." -ForegroundColor Cyan
} else {
  Write-Host ("    VERDICT: mixed ({0}) - paste full output back." -f $fb) -ForegroundColor Yellow
}
Write-Host ""
