# site_probe_mk2.ps1 - FORGE site diagnostic
# Usage:  .\site_probe_mk2.ps1                      (defaults to forge-observatory.com)
#         .\site_probe_mk2.ps1 -Domain example.com
# PS 5.1 compatible. Read-only: makes no changes anywhere.
param(
  [string]$Domain = "forge-observatory.com",
  [string]$ExpectTarget = "n-herling-mk1.github.io"
)

$W = 100
function Rule { Write-Host ("." * $W) -ForegroundColor DarkCyan }
function Step($n, $t) { Write-Host ""; Write-Host ("  [{0}/6]  {1}" -f $n, $t) -ForegroundColor Cyan }
function OK($m)   { Write-Host ("    [ OK ]  {0}" -f $m) -ForegroundColor Green }
function BAD($m)  { Write-Host ("    [FAIL]  {0}" -f $m) -ForegroundColor Red }
function INFO($m) { Write-Host ("    [ .. ]  {0}" -f $m) -ForegroundColor Gray }

Rule
Write-Host ("  SITE PROBE mk2  //  {0}  //  {1}" -f $Domain, (Get-Date -Format "yyyy-MM-dd HH:mm:ss")) -ForegroundColor White
Rule

$state = @{ dns = $false; tcp = $false; tlsSubject = $null; httpOk = $false; httpsOk = $false }

# ---------------------------------------------------------------- 1. DNS
Step 1 "DNS resolution (default resolver vs public resolvers)"
$resolvers = [ordered]@{ "system default" = $null; "cloudflare 1.1.1.1" = "1.1.1.1"; "google 8.8.8.8" = "8.8.8.8" }
foreach ($name in @($Domain, "www.$Domain")) {
  foreach ($rk in $resolvers.Keys) {
    try {
      $p = @{ Name = $name; Type = "A"; ErrorAction = "Stop"; DnsOnly = $true }
      if ($resolvers[$rk]) { $p.Server = $resolvers[$rk] }
      $ips = (Resolve-DnsName @p | Where-Object Type -eq "A" | Select-Object -Expand IPAddress) -join ", "
      if ($ips) { OK ("{0,-26} via {1,-18} -> {2}" -f $name, $rk, $ips); $state.dns = $true }
      else      { BAD ("{0,-26} via {1,-18} -> no A records" -f $name, $rk) }
    } catch { BAD ("{0,-26} via {1,-18} -> {2}" -f $name, $rk, $_.Exception.Message.Split("`n")[0]) }
  }
}

# ---------------------------------------------------------------- 2. TCP 443
Step 2 "TCP reachability :443"
try {
  $tcp = New-Object Net.Sockets.TcpClient
  $iar = $tcp.BeginConnect($Domain, 443, $null, $null)
  if ($iar.AsyncWaitHandle.WaitOne(5000) -and $tcp.Connected) { OK "TCP connect :443 succeeded"; $state.tcp = $true }
  else { BAD "TCP connect :443 timed out (5s)" }
  $tcp.Close()
} catch { BAD ("TCP connect :443 -> {0}" -f $_.Exception.Message) }

# ---------------------------------------------------------------- 3. TLS handshake matrix
Step 3 "TLS handshake + presented certificate (per protocol)"
foreach ($protoName in @("Tls13", "Tls12")) {
  $proto = $null
  try { $proto = [System.Security.Authentication.SslProtocols]::$protoName } catch {}
  if (-not $proto) { INFO ("{0}: not available in this .NET runtime - skipped" -f $protoName); continue }
  $tcp = $null; $ssl = $null
  try {
    $tcp = New-Object Net.Sockets.TcpClient($Domain, 443)
    $ssl = New-Object Net.Security.SslStream($tcp.GetStream(), $false, { $true })
    $ssl.AuthenticateAsClient($Domain, $null, $proto, $false)
    $c = New-Object Security.Cryptography.X509Certificates.X509Certificate2($ssl.RemoteCertificate)
    OK ("{0}: handshake OK" -f $protoName)
    INFO ("  subject : {0}" -f $c.Subject)
    INFO ("  issuer  : {0}" -f ($c.Issuer -split ",")[0])
    INFO ("  expires : {0}" -f $c.NotAfter)
    try { INFO ("  SAN     : {0}" -f (($c.DnsNameList | Select-Object -Expand Unicode) -join ", ")) } catch {}
    if (-not $state.tlsSubject) { $state.tlsSubject = $c.Subject }
  } catch {
    BAD ("{0}: {1}" -f $protoName, $_.Exception.Message.Split("`n")[0])
  } finally {
    if ($ssl) { $ssl.Dispose() }; if ($tcp) { $tcp.Close() }
  }
}

# ---------------------------------------------------------------- 4. HTTP (no TLS)
Step 4 "HTTP  http://$Domain  (cert-independent content check)"
try {
  $r = Invoke-WebRequest -Uri ("http://{0}" -f $Domain) -UseBasicParsing -TimeoutSec 10 -ErrorAction Stop
  $title = ""
  if ($r.Content -match "<title>(.*?)</title>") { $title = $Matches[1] }
  OK ("HTTP {0}  title: {1}" -f $r.StatusCode, $title)
  $state.httpOk = $true
} catch { BAD ("HTTP -> {0}" -f $_.Exception.Message.Split("`n")[0]) }

# ---------------------------------------------------------------- 5. HTTPS end-to-end
Step 5 "HTTPS  https://$Domain  (full request, cert captured even if invalid)"
$script:capturedCert = $null
$oldCb = [Net.ServicePointManager]::ServerCertificateValidationCallback
$oldSp = [Net.ServicePointManager]::SecurityProtocol
try {
  try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 -bor [Net.SecurityProtocolType]::Tls13 }
  catch { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 }
  [Net.ServicePointManager]::ServerCertificateValidationCallback = {
    param($s, $cert, $chain, $errs)
    $script:capturedCert = New-Object Security.Cryptography.X509Certificates.X509Certificate2($cert)
    $true  # accept, so we can inspect
  }
  $r = Invoke-WebRequest -Uri ("https://{0}" -f $Domain) -UseBasicParsing -TimeoutSec 12 -ErrorAction Stop
  OK ("HTTPS {0} - request completed" -f $r.StatusCode)
  $state.httpsOk = $true
} catch { BAD ("HTTPS -> {0}" -f $_.Exception.Message.Split("`n")[0]) }
finally {
  [Net.ServicePointManager]::ServerCertificateValidationCallback = $oldCb
  [Net.ServicePointManager]::SecurityProtocol = $oldSp
}
if ($script:capturedCert) {
  INFO ("presented cert subject : {0}" -f $script:capturedCert.Subject)
  INFO ("presented cert issuer  : {0}" -f ($script:capturedCert.Issuer -split ",")[0])
  if (-not $state.tlsSubject) { $state.tlsSubject = $script:capturedCert.Subject }
}

# ---------------------------------------------------------------- 6. Verdict
Step 6 "VERDICT"
Rule
if (-not $state.dns) {
  BAD  "DNS is the blocker. Check records exist in the zone table (grey cloud / DNS only)."
} elseif ($state.httpsOk -and $state.tlsSubject -match [regex]::Escape($Domain)) {
  OK   "Certificate ISSUED and serving for $Domain."
  INFO "Action: GitHub -> Settings -> Pages -> tick 'Enforce HTTPS'. Browser warnings now = stale session."
} elseif ($state.tlsSubject -match "github\.io") {
  INFO "Cert still PROVISIONING: edge is presenting the *.github.io fallback."
  INFO "Action: wait; the Pages settings box narrates. Lever (remove/re-add custom domain) only if stuck hours."
} elseif ($state.httpOk -and $state.tcp -and -not $state.tlsSubject) {
  INFO "Site serves over HTTP; edge refuses TLS for this hostname -> mid-provisioning state."
  INFO "Action: wait it out (minutes-to-an-hour typical). Pages settings box is ground truth."
} elseif (-not $state.tcp) {
  BAD  "Nothing listening / reachable on :443 - network-level problem, not certificates."
} else {
  INFO "Mixed state - paste this full output back for interpretation."
}
Rule
